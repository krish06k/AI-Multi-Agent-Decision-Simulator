"""
gan/conditional_gan.py
======================
Layer 1 — GAN Data Engine | Regime-Conditioned GAN

Extends Generator and Discriminator to accept a one-hot regime vector,
enabling the GAN to generate synthetic OHLCV sequences matching a target
market regime: calm, stressed, or trending.

Architecture
------------
CondGenerator:
    z (noise) concatenated with a learned regime embedding before the base
    Generator's LSTM stack. ReLU in the embedding replaced with SiLU to
    match the base Generator's activation throughout.

    regime_one_hot (3) → Linear(3, embed_dim) → SiLU → embed_dim
    effective input: noise_dim + embed_dim

CondDiscriminator:
    Regime tiled across time and concatenated to each timestep before
    the LSTM/CNN encoders. SpectralNorm applied to all Linear layers to
    enforce Lipschitz constraint alongside the WGAN-GP penalty.

    effective input: ohlcv_dim + NUM_REGIMES = 15

    gradient_penalty() augments real/fake BEFORE interpolation so the
    penalty is computed in the regime-conditioned space. Calls
    Discriminator.forward() directly to avoid double-augmentation.

    feature_extract() routes through the post-attention path on G side
    (base Discriminator.feature_extract on augmented input) so
    feature-matching compares representations in the same space.

Changelog v4
------------
* SiLU replaces ReLU in regime embedding (matches Generator v4 activations).
* SpectralNorm added to CondDiscriminator's inherited Linear layers via
  _apply_spectral_norm() — called once at init after super().__init__().
* gradient_penalty() fixed: interpolation now happens on augmented tensors,
  and Discriminator.forward() is called directly to avoid double-augment.
* build_cond_generator() passes attn_heads from config.
* build_cond_discriminator() correctly uses disc_layers (not num_layers).
* Smoke test uses ohlcv_dim from config, verifies GP runs without error.

cuDNN fix (gradient_penalty)
-----------------------------
gradient_penalty() wraps the Discriminator.forward() call on the
interpolated tensor in torch.backends.cudnn.flags(enabled=False).
create_graph=True triggers a second backward through the LSTM which
cuDNN does not support (NotImplementedError: _cudnn_rnn_backward).
Disabling cuDNN for that one call forces PyTorch's pure-Python LSTM
path which does support double-backward. cuDNN is restored automatically
after the with-block. No effect on any other forward/backward pass.
"""

from __future__ import annotations

import random
import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import yaml

from gan.generator import Generator
from gan.discriminator import Discriminator

logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "gan_config.yaml"

REGIMES     = ["calm", "stressed", "trending"]
NUM_REGIMES = len(REGIMES)

RegimeLabel = Literal["calm", "stressed", "trending"]


def _load_cfg() -> dict:
    with open(_CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


# ---------------------------------------------------------------------------
# Regime encoding utilities
# ---------------------------------------------------------------------------

def sample_regime() -> RegimeLabel:
    return random.choice(REGIMES)


def encode_regime(label: str | int | torch.Tensor) -> torch.Tensor:
    """
    Convert a regime label to a one-hot tensor (NUM_REGIMES,).

    Accepts str ("calm"), int (0), or a Tensor (returned unchanged).
    """
    if isinstance(label, torch.Tensor):
        return label
    if isinstance(label, str):
        idx = REGIMES.index(label)
    elif isinstance(label, int):
        idx = label
    else:
        raise TypeError(f"encode_regime: unsupported type {type(label)}")
    one_hot = torch.zeros(NUM_REGIMES)
    one_hot[idx] = 1.0
    return one_hot


def encode_regime_batch(
    labels: list[str | int],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Encode a list of regime labels → (N, NUM_REGIMES) on device."""
    return torch.stack([encode_regime(l) for l in labels]).to(device)


# ---------------------------------------------------------------------------
# CondGenerator
# ---------------------------------------------------------------------------

class CondGenerator(Generator):
    """
    Regime-conditioned LSTM + Attention Generator.

    Prepends a learned regime embedding to the noise vector z.
    The base Generator's full pipeline (SpectralNorm LSTM blocks +
    TemporalAttention + output head) runs on the concatenated input.

    Parameters
    ----------
    noise_dim        : Raw noise dimension (before embedding concat).
    hidden_dim       : LSTM / attention hidden size.
    seq_len          : Output sequence length.
    num_layers       : Stacked ResidualLSTMBlocks.
    ohlcv_dim        : Output features per timestep.
    dropout          : Dropout probability.
    regime_embed_dim : Regime embedding output size.
    attn_heads       : Attention heads (must divide hidden_dim).
    """

    def __init__(
        self,
        noise_dim: int        = 100,
        hidden_dim: int       = 256,
        seq_len: int          = 30,
        num_layers: int       = 3,
        ohlcv_dim: int        = 12,
        dropout: float        = 0.1,
        regime_embed_dim: int = 32,
        attn_heads: int       = 4,
    ):
        total_input = noise_dim + regime_embed_dim
        super().__init__(
            noise_dim  = total_input,
            hidden_dim = hidden_dim,
            seq_len    = seq_len,
            num_layers = num_layers,
            ohlcv_dim  = ohlcv_dim,
            dropout    = dropout,
            attn_heads = attn_heads,
        )
        self.real_noise_dim = noise_dim

        # SiLU matches base Generator activations (v4 change from ReLU)
        self.regime_embed = nn.Sequential(
            nn.Linear(NUM_REGIMES, regime_embed_dim),
            nn.SiLU(),
        )

        logger.info(
            "CondGenerator ready | noise=%d  embed=%d  total_input=%d  "
            "hidden=%d  seq_len=%d  out_dim=%d  attn_heads=%d",
            noise_dim, regime_embed_dim, total_input,
            hidden_dim, seq_len, ohlcv_dim, attn_heads,
        )

    def forward(
        self,
        z: torch.Tensor,
        regime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z      : (batch, real_noise_dim)
        regime : (batch, NUM_REGIMES) or None → defaults to "calm"
        → (batch, seq_len, ohlcv_dim)
        """
        batch  = z.size(0)
        device = z.device

        if regime is None:
            regime = (
                encode_regime("calm")
                .unsqueeze(0).expand(batch, -1).to(device)
            )

        r_embed = self.regime_embed(regime)            # (batch, embed_dim)
        z_cond  = torch.cat([z, r_embed], dim=-1)      # (batch, total_input)
        return super().forward(z_cond)

    def extract_features(
        self,
        z: torch.Tensor,
        regime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Post-attention mean-pooled features for feature-matching loss.
        Keeps the same embedding path as forward() so features are consistent.
        """
        batch  = z.size(0)
        device = z.device

        if regime is None:
            regime = (
                encode_regime("calm")
                .unsqueeze(0).expand(batch, -1).to(device)
            )

        r_embed = self.regime_embed(regime)
        z_cond  = torch.cat([z, r_embed], dim=-1)
        return super().extract_features(z_cond)


# ---------------------------------------------------------------------------
# CondDiscriminator
# ---------------------------------------------------------------------------

class CondDiscriminator(Discriminator):
    """
    Regime-conditioned Critic (Discriminator).

    Tiles the regime one-hot across the time dimension and concatenates
    it to each timestep before the base LSTM/CNN encoders.
    Effective input: ohlcv_dim + NUM_REGIMES.

    SpectralNorm is applied to all inherited Linear layers after
    super().__init__() via _apply_spectral_norm(). This enforces the
    Lipschitz constraint on D jointly with the WGAN-GP penalty — each
    mechanism alone is weaker than both together.

    gradient_penalty() interpolates in the AUGMENTED space (real_aug,
    fake_aug) so the GP correctly penalises the critic's Lipschitz
    violation on regime-conditioned inputs. Calls Discriminator.forward()
    directly to avoid double-augmentation.

    cuDNN fix: gradient_penalty() wraps Discriminator.forward() on the
    interpolated tensor in torch.backends.cudnn.flags(enabled=False)
    to prevent NotImplementedError: _cudnn_rnn_backward when
    create_graph=True triggers double-backward through the LSTM.
    """

    def __init__(
        self,
        ohlcv_dim: int    = 12,
        hidden_dim: int   = 256,
        cnn_dim: int      = 128,
        seq_len: int      = 30,
        num_layers: int   = 2,
        dropout: float    = 0.2,
        use_sigmoid: bool = False,
    ):
        super().__init__(
            ohlcv_dim   = ohlcv_dim + NUM_REGIMES,
            hidden_dim  = hidden_dim,
            cnn_dim     = cnn_dim,
            seq_len     = seq_len,
            num_layers  = num_layers,
            dropout     = dropout,
            use_sigmoid = use_sigmoid,
        )
        self.input_ohlcv_dim = ohlcv_dim
        self._apply_spectral_norm()

        logger.info(
            "CondDiscriminator ready | input_dim=%d  augmented_dim=%d  "
            "hidden=%d  cnn=%d  seq_len=%d  spectral_norm=True",
            ohlcv_dim, ohlcv_dim + NUM_REGIMES, hidden_dim, cnn_dim, seq_len,
        )

    def _apply_spectral_norm(self) -> None:
        """
        Apply spectral normalisation to Linear layers not already wrapped by the
        base Discriminator. The base class applies SN to fusion + head at init,
        so this is a no-op for those — the guard prevents the double-register crash.
        """
        all_linears = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
        ]
        for name, module in all_linears[:-1]:
            if not hasattr(module, "weight_orig"):
                spectral_norm(module)
                logger.debug("SpectralNorm applied to %s", name)
            else:
                logger.debug("SpectralNorm already present on %s — skipped", name)

    def _augment(
        self,
        x: torch.Tensor,
        regime: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tile regime across time and concatenate to sequence.

        x      : (batch, seq_len, ohlcv_dim)
        regime : (batch, NUM_REGIMES) or (NUM_REGIMES,)
        → (batch, seq_len, ohlcv_dim + NUM_REGIMES)
        """
        batch, seq_len, _ = x.shape
        if regime.dim() == 1:
            regime = regime.unsqueeze(0).expand(batch, -1)
        r_tiled = regime.unsqueeze(1).expand(-1, seq_len, -1)
        return torch.cat([x, r_tiled], dim=-1)

    def _default_regime(self, batch: int, device: torch.device) -> torch.Tensor:
        return (
            encode_regime("calm")
            .unsqueeze(0).expand(batch, -1).to(device)
        )

    def forward(
        self,
        x: torch.Tensor,
        regime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x      : (batch, seq_len, ohlcv_dim)
        regime : (batch, NUM_REGIMES) or None
        → (batch, 1)
        """
        if regime is None:
            regime = self._default_regime(x.size(0), x.device)
        return super().forward(self._augment(x, regime))

    def feature_extract(
        self,
        x: torch.Tensor,
        regime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Penultimate representation for feature-matching loss.
        Routes through the same augmented path as forward().
        """
        if regime is None:
            regime = self._default_regime(x.size(0), x.device)
        return super().feature_extract(self._augment(x, regime))

    def gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        device: torch.device,
        lambda_gp: float = 10.0,
        regime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        WGAN-GP gradient penalty computed in the regime-conditioned space.

        Correctness: real and fake are augmented FIRST, then interpolated,
        so the GP penalises the critic's Lipschitz violation on the same
        (ohlcv_dim + NUM_REGIMES) input space that D actually operates in
        during training. Without this, GP interpolates in a different
        (non-regime-conditioned) space and the penalty is meaningless.

        Double-augmentation prevention: Discriminator.forward() (base class)
        is called directly on the already-augmented interpolated tensor.
        Calling self.forward() (CondDiscriminator) would call _augment()
        again, adding another NUM_REGIMES channels → shape error.

        cuDNN double-backward fix: Discriminator.forward() on the
        interpolated tensor is wrapped in:
            torch.backends.cudnn.flags(enabled=False)
        create_graph=True triggers a second backward through the LSTM.
        cuDNN RNN kernels do not support this and raise:
            NotImplementedError: _cudnn_rnn_backward is not implemented.
        Disabling cuDNN for that one call uses PyTorch's pure-Python LSTM
        which supports double-backward. cuDNN is restored after the block.

        Parameters
        ----------
        real, fake : (batch, seq_len, ohlcv_dim) — RAW, not yet augmented.
        device     : Computation device.
        lambda_gp  : GP weight (default 10, per WGAN-GP paper).
        regime     : (batch, NUM_REGIMES) or None → defaults to "calm".
        """
        batch = real.size(0)
        if regime is None:
            regime = self._default_regime(batch, device)

        # Step 1: augment both sides — GP lives in the conditioned space
        real_aug = self._augment(real, regime)    # (batch, seq_len, ohlcv_dim+3)
        fake_aug = self._augment(fake, regime)    # (batch, seq_len, ohlcv_dim+3)

        # Step 2: random interpolation
        alpha        = torch.rand(batch, 1, 1, device=device)
        interpolated = (
            alpha * real_aug + (1 - alpha) * fake_aug
        ).requires_grad_(True)

        # Step 3: call BASE Discriminator.forward() directly on augmented tensor.
        # self.forward() (CondDiscriminator) would call _augment() again
        # producing (ohlcv_dim + 2*NUM_REGIMES) channels — wrong shape.
        # Disable cuDNN to allow create_graph=True through the LSTM.
        with torch.backends.cudnn.flags(enabled=False):
            d_interp = Discriminator.forward(self, interpolated)

        # Step 4: compute gradient norm penalty
        grads = torch.autograd.grad(
            outputs      = d_interp,
            inputs       = interpolated,
            grad_outputs = torch.ones_like(d_interp),
            create_graph = True,
            retain_graph = True,
        )[0]

        grads   = grads.reshape(batch, -1)
        penalty = lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return penalty


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_cond_generator(cfg: dict | None = None) -> CondGenerator:
    cfg = cfg or _load_cfg()
    return CondGenerator(
        noise_dim        = cfg.get("noise_dim",        100),
        hidden_dim       = cfg.get("hidden_dim",       256),
        seq_len          = cfg.get("seq_len",           30),
        num_layers       = cfg.get("num_layers",         3),
        ohlcv_dim        = cfg.get("ohlcv_dim",         12),
        dropout          = cfg.get("dropout",          0.1),
        regime_embed_dim = cfg.get("regime_embed_dim",  32),
        attn_heads       = cfg.get("attn_heads",         4),
    )


def build_cond_discriminator(cfg: dict | None = None) -> CondDiscriminator:
    cfg = cfg or _load_cfg()
    return CondDiscriminator(
        ohlcv_dim   = cfg.get("ohlcv_dim",      12),
        hidden_dim  = cfg.get("hidden_dim",     256),
        cnn_dim     = cfg.get("cnn_dim",        128),
        seq_len     = cfg.get("seq_len",         30),
        num_layers  = cfg.get("disc_layers",      2),   # disc_layers, not num_layers
        dropout     = cfg.get("dropout",        0.2),
        use_sigmoid = cfg.get("use_sigmoid",   False),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = build_cond_generator(cfg).to(device)
    D = build_cond_discriminator(cfg).to(device)

    ohlcv_dim = cfg.get("ohlcv_dim", 12)
    seq_len   = cfg.get("seq_len",   30)
    batch     = 8

    print(f"Device              : {device}")
    print(f"CondGenerator params: {G.num_parameters():,}")
    print(f"CondDiscriminator   : {sum(p.numel() for p in D.parameters() if p.requires_grad):,}")
    print()

    for r_name in REGIMES:
        r    = encode_regime_batch([r_name] * batch, device=device)
        z    = torch.randn(batch, G.real_noise_dim, device=device)
        fake = G(z, r)
        prob = D(fake, r)
        print(
            f"  {r_name:<10} → fake {tuple(fake.shape)}  "
            f"critic mean {prob.mean().item():.4f}"
        )

    # Verify GP runs without shape errors or cuDNN crash
    real_seq = torch.randn(batch, seq_len, ohlcv_dim, device=device)
    z2       = torch.randn(batch, G.real_noise_dim, device=device)
    r2       = encode_regime_batch(["stressed"] * batch, device=device)
    fake_seq = G(z2, r2).detach()
    gp       = D.gradient_penalty(real_seq, fake_seq, device, regime=r2)
    print(f"\n  GP (stressed, lambda=10): {gp.item():.4f}  ✓")

    # Verify feature-matching shapes align between G and D
    feat_d = D.feature_extract(real_seq, r2)
    feat_g = D.feature_extract(fake_seq, r2)
    assert feat_d.shape == feat_g.shape, "FM feature shape mismatch"
    print(f"  Feature shapes: {tuple(feat_d.shape)}  ✓")