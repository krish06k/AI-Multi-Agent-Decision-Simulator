"""
gan/generator.py
================
Layer 1 — GAN Data Engine | Generator Network  v4

Key changes from v3
-------------------
* SpectralNorm on LSTM weight matrices — the single biggest stabiliser for
  WGAN-GP. Prevents the generator from dominating the critic early on, which
  was causing W_dist to diverge unboundedly (observed: 9 → 15 → runaway).

* SiLU (Swish) replaces ReLU throughout — smoother gradient landscape, no
  dead-neuron problem, empirically better for financial time-series GANs.

* Temporal self-attention block added after the LSTM stack — captures
  long-range dependencies the LSTM misses (e.g. weekly/monthly correlations
  in a 30-day window). Single nn.MultiheadAttention, num_heads=4.

* Output head deepened slightly: Linear → SiLU → Linear → Tanh. The extra
  nonlinearity before Tanh gives the head more expressive power without
  adding instability.

* _init_weights now skips SpectralNorm-wrapped parameters (they have their
  own internal weight structure — re-initialising them breaks SN invariants).

* extract_features updated to pool from the post-attention hidden state,
  giving the feature-matching loss a richer signal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import yaml

logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "gan_config.yaml"


def _load_cfg() -> dict:
    with open(_CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class ResidualLSTMBlock(nn.Module):
    """
    Single LSTM layer with residual connection and spectral normalisation.

    SpectralNorm is applied to the LSTM's weight_ih and weight_hh matrices.
    This constrains the Lipschitz constant of the generator, preventing it
    from outputting sequences that are too easy for the critic to classify —
    the root cause of early W_dist divergence.

    SiLU on the residual projection (when dims differ) instead of Identity:
    adds a learnable nonlinear shortcut that helps gradients flow in deeper
    stacks without the zero-gradient plateau problem of ReLU.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        lstm_raw = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # Apply spectral norm to the four weight matrices
        # weight_ih: (4*hidden, input), weight_hh: (4*hidden, hidden)
        spectral_norm(lstm_raw, name="weight_ih_l0")
        spectral_norm(lstm_raw, name="weight_hh_l0")
        self.lstm = lstm_raw

        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.residual_proj = (
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.SiLU(),
            )
            if input_dim != hidden_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hidden_state = self.lstm(x, hidden)
        out = self.norm(out + self.residual_proj(x))
        out = self.dropout(out)
        return out, hidden_state


class TemporalAttention(nn.Module):
    """
    Lightweight self-attention over the time dimension.

    Applied once after the full LSTM stack. Lets the generator model
    non-local dependencies — e.g. a volatility spike at t=5 affecting
    the generated distribution at t=25 — which pure LSTM misses.

    Uses pre-norm (LayerNorm before attention) for training stability.
    Residual connection preserves LSTM features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm  = nn.LayerNorm(hidden_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, hidden_dim)"""
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        return x + self.drop(attn_out)   # residual


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    LSTM + Attention Generator for synthetic OHLCV time-series.

    Architecture
    ------------
    z (batch, noise_dim)
        ↓  Linear + SiLU  [noise_dim → hidden_dim * seq_len]
        ↓  Reshape         [(batch, seq_len, hidden_dim)]
        ↓  ResidualLSTMBlock × num_layers  [SpectralNorm, SiLU residual]
        ↓  TemporalAttention               [MultiheadAttention, pre-norm]
        ↓  Linear → SiLU → Linear → Tanh  [hidden_dim → ohlcv_dim]

    Parameters
    ----------
    noise_dim     : Dimension of the input noise vector.
    hidden_dim    : LSTM hidden state / attention embed size.
    seq_len       : Output sequence length (number of time steps).
    num_layers    : Number of stacked LSTM blocks.
    ohlcv_dim     : Output features per time step (5 legacy, 12 full).
    dropout       : Dropout probability inside blocks.
    attn_heads    : Number of attention heads (must divide hidden_dim evenly).
    """

    def __init__(
        self,
        noise_dim: int  = 100,
        hidden_dim: int = 256,
        seq_len: int    = 30,
        num_layers: int = 3,
        ohlcv_dim: int  = 12,
        dropout: float  = 0.1,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.noise_dim  = noise_dim
        self.hidden_dim = hidden_dim
        self.seq_len    = seq_len
        self.num_layers = num_layers
        self.ohlcv_dim  = ohlcv_dim

        # Noise projection — SiLU instead of ReLU
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim * seq_len),
            nn.SiLU(),
        )

        # Stacked residual LSTM blocks (SpectralNorm inside)
        self.lstm_blocks = nn.ModuleList([
            ResidualLSTMBlock(
                hidden_dim, hidden_dim,
                dropout=dropout if i < num_layers - 1 else 0.0,
            )
            for i in range(num_layers)
        ])

        # Temporal self-attention (applied once, after LSTM stack)
        self.attention = TemporalAttention(
            hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
        )

        # Output head — deeper than v3 for more expressive final mapping
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, ohlcv_dim),
            nn.Tanh(),
        )

        self._init_weights()
        logger.info(
            "Generator init | noise_dim=%d  hidden=%d  seq_len=%d  "
            "layers=%d  out_dim=%d  attn_heads=%d  params=%s",
            noise_dim, hidden_dim, seq_len, num_layers, ohlcv_dim,
            attn_heads, f"{self.num_parameters():,}",
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Xavier on Linear, orthogonal on LSTM weight matrices.

        SpectralNorm wraps weights in a _WeightNorm object — we must
        initialise the underlying `weight_orig` tensors, not `weight`
        (which is computed on the fly). Plain isinstance(m, nn.LSTM)
        still works because SN doesn't change the module type.

        We skip bias vectors inside LSTM (SN handles them separately)
        but still set forget-gate bias to 1.0 for gradient flow.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip SN-wrapped linears (none currently, but future-safe)
                if hasattr(module, "weight_orig"):
                    nn.init.xavier_uniform_(module.weight_orig, gain=0.5)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih_l0_orig" in name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif "weight_hh_l0_orig" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name and "orig" not in name:
                        nn.init.zeros_(param)
                        # Forget-gate bias = 1.0 for better gradient flow
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch_size, noise_dim)
        → (batch_size, seq_len, ohlcv_dim)  values in [-1, 1]
        """
        batch = z.size(0)
        x = self.noise_proj(z).view(batch, self.seq_len, self.hidden_dim)

        for block in self.lstm_blocks:
            x, _ = block(x)

        x = self.attention(x)           # temporal context
        return self.output_head(x)      # (batch, seq_len, ohlcv_dim)

    # ------------------------------------------------------------------
    # Feature extraction (feature-matching loss in trainer)
    # ------------------------------------------------------------------

    def extract_features(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean-pooled post-attention hidden state.
        Richer than pre-attention pooling — captures non-local sequence
        structure, giving the FM loss a stronger training signal.

        Returns: (batch, hidden_dim)
        """
        batch = z.size(0)
        x = self.noise_proj(z).view(batch, self.seq_len, self.hidden_dim)
        for block in self.lstm_blocks:
            x, _ = block(x)
        x = self.attention(x)
        return x.mean(dim=1)

    # ------------------------------------------------------------------
    # Sampling helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        z = torch.randn(n, self.noise_dim, device=device) * temperature
        return self.forward(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_generator(cfg: dict | None = None) -> Generator:
    cfg = cfg or _load_cfg()
    g = Generator(
        noise_dim  = cfg.get("noise_dim",   100),
        hidden_dim = cfg.get("hidden_dim",  256),
        seq_len    = cfg.get("seq_len",      30),
        num_layers = cfg.get("num_layers",    3),
        ohlcv_dim  = cfg.get("ohlcv_dim",   12),
        dropout    = cfg.get("dropout",     0.1),
        attn_heads = cfg.get("attn_heads",    4),
    )
    logger.info("Generator ready — %d trainable parameters.", g.num_parameters())
    return g


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = build_generator()
    G = G.to(device)

    z   = torch.randn(8, G.noise_dim, device=device)
    out = G(z)
    print(f"forward()  z:{tuple(z.shape)} → fake_seq:{tuple(out.shape)}")
    assert out.shape == (8, 30, 12), "shape mismatch"
    assert out.min() >= -1.0 and out.max() <= 1.0, "output out of [-1,1]"

    feats = G.extract_features(z)
    print(f"features   shape: {tuple(feats.shape)}")
    assert feats.shape == (8, 256)

    samples = G.sample(16, device=device)
    print(f"sample(16) shape: {tuple(samples.shape)}")
    print(f"Value range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"Parameters: {G.num_parameters():,}")