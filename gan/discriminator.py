"""
gan/discriminator.py
====================
Layer 1 — GAN Data Engine | Discriminator Network

Dual-path Discriminator combining:
  • Temporal path  — LSTM captures sequential dependencies.
  • Spectral path  — 1D-CNN captures local patterns and frequency features.

Both paths are fused and projected to a scalar probability: real=1 / fake=0.

Design decisions
----------------
* Dual-path fusion > single path: financial series have both local (candlestick
  patterns) and long-range (trend) structure; each path specialises.
* Spectral normalisation on Linear layers: prevents discriminator from
  overwhelming the generator early (mode collapse prevention).
* Label smoothing compatible: the sigmoid output is raw logit-free, so
  BCEWithLogitsLoss can be used in trainer for numerical stability.
* feature_extract() exposes the penultimate representation for feature-matching
  regularisation during generator training.

cuDNN fix (gradient_penalty)
-----------------------------
gradient_penalty() wraps the interpolated forward pass in
    torch.backends.cudnn.flags(enabled=False)
to prevent NotImplementedError: _cudnn_rnn_backward.
create_graph=True triggers a second backward through the LSTM.
cuDNN does not implement double-backward for RNNs — disabling it for
that one call forces PyTorch's pure-Python LSTM path which does.
All other forward/backward passes remain on cuDNN and are unaffected.

**kwargs on gradient_penalty absorbs the `regime=...` kwarg forwarded
by CondDiscriminator's trainer dispatch (_gp helper in trainer.py)
so the base signature stays forward-compatible without a required override.

_init_weights fix (v4)
-----------------------
spectral_norm() wraps nn.Linear by:
  1. Removing .weight from the module's parameters.
  2. Registering .weight_orig as the actual learnable parameter.
  3. Registering .weight as a computed property (weight_orig / sigma).

Calling nn.init.xavier_uniform_(module.weight) on a SN-wrapped Linear
initialises the *computed* property, not the stored parameter — the
effect is immediately overwritten on the next forward pass when SN
recomputes weight from weight_orig. The fix is to check for weight_orig
and initialise that instead. gain=0.5 halves the initial gradient
magnitude, reducing the probability of the grad-clip firing every step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import yaml

logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "gan_config.yaml"


def _load_cfg() -> dict:
    if not _CFG_PATH.exists():
        logger.warning("Config file not found: %s — using defaults.", _CFG_PATH)
        return {}
    with open(_CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        logger.warning("YAML is empty or invalid — using defaults.")
        return {}
    return cfg


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """
    LSTM-based encoder for capturing sequential (trend) structure.

    Input  : (batch, seq_len, ohlcv_dim)
    Output : (batch, hidden_dim)  — mean-pool + last hidden state fused.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch, hidden_dim)"""
        out, (h_n, _) = self.lstm(x)
        mean_pool  = out.mean(dim=1)   # (batch, hidden_dim)
        last_state = h_n[-1]           # (batch, hidden_dim)
        fused = self.norm(mean_pool + last_state)
        return fused


class SpectralEncoder(nn.Module):
    """
    1D-CNN encoder for capturing local candlestick patterns.

    Input  : (batch, seq_len, ohlcv_dim)
    Output : (batch, cnn_dim)
    """

    def __init__(self, input_dim: int, cnn_dim: int, seq_len: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(cnn_dim // 2),

            nn.Conv1d(cnn_dim // 2, cnn_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(cnn_dim),

            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch, cnn_dim)"""
        x = x.transpose(1, 2)
        x = self.convs(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Dual-path Discriminator: LSTM temporal + CNN spectral paths fused via MLP.
    """

    def __init__(
        self,
        ohlcv_dim: int    = 5,
        hidden_dim: int   = 256,
        cnn_dim: int      = 128,
        seq_len: int      = 30,
        num_layers: int   = 2,
        dropout: float    = 0.2,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.ohlcv_dim   = ohlcv_dim
        self.hidden_dim  = hidden_dim
        self.use_sigmoid = use_sigmoid

        self.temporal = TemporalEncoder(ohlcv_dim, hidden_dim, num_layers, dropout)
        self.spectral  = SpectralEncoder(ohlcv_dim, cnn_dim, seq_len)

        fusion_dim = hidden_dim + cnn_dim
        self.fusion = nn.Sequential(
            spectral_norm(nn.Linear(fusion_dim, fusion_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(fusion_dim // 2, fusion_dim // 4)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = spectral_norm(nn.Linear(fusion_dim // 4, 1))

        self._init_weights()
        logger.info(
            "Discriminator init | ohlcv_dim=%d  hidden=%d  cnn=%d  "
            "seq_len=%d  layers=%d  sigmoid=%s",
            ohlcv_dim, hidden_dim, cnn_dim, seq_len, num_layers, use_sigmoid,
        )

    # ------------------------------------------------------------------
    # Weight init — v4 fix
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Initialise weights correctly for spectral-normed and plain layers.

        KEY FIX: spectral_norm() replaces .weight with a computed property
        and stores the real parameter as .weight_orig. Calling xavier on
        .weight writes to the computed view and is immediately discarded.
        We must check for .weight_orig and initialise that instead.

        gain=0.5 on all xavier calls halves the initial spectral radius,
        which reduces raw gradient magnitudes and stops grad_clip_d from
        firing on every step.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # SN-wrapped Linear: initialise weight_orig, not weight
                if hasattr(module, "weight_orig"):
                    nn.init.xavier_uniform_(module.weight_orig, gain=0.5)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param, gain=0.5)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        # Forget gate bias = 1.0 for better gradient flow
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_feat = self.temporal(x)
        s_feat = self.spectral(x)
        fused  = torch.cat([t_feat, s_feat], dim=-1)
        mid    = self.fusion(fused)
        logit  = self.head(mid)
        return torch.sigmoid(logit) if self.use_sigmoid else logit

    # ------------------------------------------------------------------
    # Feature extraction for feature-matching loss
    # ------------------------------------------------------------------

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        t_feat = self.temporal(x)
        s_feat = self.spectral(x)
        fused  = torch.cat([t_feat, s_feat], dim=-1)
        return self.fusion(fused)

    def temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(x)

    # ------------------------------------------------------------------
    # Gradient penalty (WGAN-GP)
    # ------------------------------------------------------------------

    def gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        device: torch.device,
        lambda_gp: float = 10.0,
        **kwargs,
    ) -> torch.Tensor:
        batch = real.size(0)
        alpha = torch.rand(batch, 1, 1, device=device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        with torch.backends.cudnn.flags(enabled=False):
            d_interp = self.forward(interpolated)

        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]

        grads   = grads.reshape(batch, -1)
        penalty = lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_discriminator(cfg: dict | None = None) -> Discriminator:
    cfg = cfg or _load_cfg() or {}
    d = Discriminator(
        ohlcv_dim   = cfg.get("ohlcv_dim",    5),
        hidden_dim  = cfg.get("hidden_dim",  256),
        cnn_dim     = cfg.get("cnn_dim",     128),
        seq_len     = cfg.get("seq_len",      30),
        num_layers  = cfg.get("disc_layers",   2),
        dropout     = cfg.get("dropout",     0.2),
        use_sigmoid = cfg.get("use_sigmoid", True),
    )
    logger.info("Discriminator ready — %d trainable parameters.", d.num_parameters())
    return d


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    D = build_discriminator().to(device)

    # Verify weight_orig norms are small after init (should be 0.3–1.5)
    print("\n── weight_orig norms after _init_weights ──")
    for name, param in D.named_parameters():
        if "weight_orig" in name or ("weight" in name and "norm" not in name):
            print(f"  {name:45s}  norm={param.norm().item():.4f}")

    batch, seq_len, feats = 8, 30, 5
    x_real = torch.randn(batch, seq_len, feats, device=device)
    x_fake = torch.randn(batch, seq_len, feats, device=device)

    prob_real = D(x_real)
    prob_fake = D(x_fake)
    print(f"\nforward(real): {tuple(prob_real.shape)}  mean={prob_real.mean():.3f}")
    print(f"forward(fake): {tuple(prob_fake.shape)}  mean={prob_fake.mean():.3f}")

    feats_out = D.feature_extract(x_real)
    print(f"features: {tuple(feats_out.shape)}")

    gp = D.gradient_penalty(x_real, x_fake, device=device)
    print(f"gradient_penalty: {gp.item():.4f}  ✓")
    print(f"Parameters: {D.num_parameters():,}")