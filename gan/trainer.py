"""
gan/trainer.py
==============
Layer 1 — GAN Data Engine | Adversarial Training Loop

Orchestrates the full GAN training cycle:
  1. Discriminator update  — n_critic steps per G step.
  2. Generator update      — adversarial + optional feature-matching loss.
  3. Gradient penalty      — WGAN-GP Lipschitz regularisation.
  4. Checkpointing         — G + D every save_every epochs.
  5. Loss logging          — CSV + console.

Training modes
--------------
  WGAN-GP     (use_wgan=True)  — Wasserstein loss + gradient penalty.
  Standard GAN (use_wgan=False) — BCE loss with optional label smoothing.
  Conditional  (conditional=True) — regime labels fed to CondG + CondD.

v7 fixes
--------
FIX — deepcopy crash on spectral_norm modules
    spectral_norm registers non-leaf tensors (weight_u, weight_v) that
    do not support the deepcopy protocol, causing a RuntimeError at epoch 9
    when _init_ema_D() was called. Replaced deepcopy with a state_dict
    round-trip: rebuild D from config then load_state_dict(). Fully safe
    with spectral_norm and weight_orig. import copy removed.

FIX — W_dist explodes during warm-up (D unbounded without GP)
    During warm_up_epochs, eff_lambda_gp was 0.0 so D had no Lipschitz
    constraint at all. D weights grew without bound, W_dist hit 226 by
    epoch 8. Fixed by enforcing a minimum GP of min_gp_lambda=1.0 even
    during warm-up inside _train_D. _effective_lambda_gp() still returns
    0.0 for logging; the floor is applied only in the training step.

v6 fixes (retained)
--------------------
FIX 1 — D_gnorm pinned: data_std logged epoch 1, warning if > 2.0.
FIX 2 — W_dist explosion: symptom of FIX 3, resolved by fm_loss control.
FIX 3 — fm_loss unbounded: EMA target + fm_clamp + lambda_fm 0.1 → 0.01.
FIX 4 — GP spikes: warm-up ramp after G unfreezes + clamp at 5×lambda_gp.

v5 changes (retained)
----------------------
* _train_D WGAN: fresh fake inside gradient-enabled block.
* gradient_penalty passes regime via **gp_kwargs.
* EpochMetrics tracks g_adv_loss and g_fm_loss separately.
* Warm-up phase: G frozen, only D updated.
* NaN guard: crash checkpoint + halt.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.conditional_gan import (
    CondGenerator, CondDiscriminator,
    encode_regime_batch, REGIMES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "gan_config.yaml"


def _load_cfg() -> dict:
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    epoch: int
    d_loss: float
    g_loss: float
    g_adv_loss: float
    g_fm_loss: float
    d_real_score: float
    d_fake_score: float
    gp_loss: float
    elapsed_sec: float
    d_grad_norm: float      = 0.0
    g_grad_norm: float      = 0.0
    d_skipped_steps: int    = 0
    wasserstein_dist: float = 0.0
    g_frozen: bool          = False
    data_std: float         = 0.0


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

def _update_ema(ema_model: nn.Module, live_model: nn.Module, decay: float) -> None:
    """In-place exponential moving average update of ema_model parameters."""
    with torch.no_grad():
        for ema_p, live_p in zip(ema_model.parameters(), live_model.parameters()):
            ema_p.data.mul_(decay).add_(live_p.data, alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full adversarial training loop for the OHLCV GAN.

    Parameters
    ----------
    G              : Generator or CondGenerator.
    D              : Discriminator or CondDiscriminator.
    dataloader     : DataLoader yielding real tensors (batch, seq_len, F).
    cfg            : Config dict. If None, loads from gan_config.yaml.
    device         : Computation device.
    regime_sampler : Optional callable(batch_size) → regime tensor (batch, 3).
    """

    def __init__(
        self,
        G: Generator,
        D: Discriminator,
        dataloader: torch.utils.data.DataLoader,
        cfg: dict | None = None,
        device: torch.device | str | None = None,
        regime_sampler: Optional[Callable[[int], torch.Tensor]] = None,
    ):
        self.cfg    = cfg or _load_cfg()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.G = G.to(self.device)
        self.D = D.to(self.device)
        self.dataloader = dataloader

        self._is_cond        = isinstance(G, CondGenerator) or self.cfg.get("conditional", False)
        self._regime_sampler = regime_sampler

        # Hyperparams
        self.epochs           = self.cfg.get("epochs",            200)
        self.lr_g             = float(self.cfg.get("lr_g",       2e-4))
        self.lr_d             = float(self.cfg.get("lr_d",       1e-4))
        self.n_critic         = self.cfg.get("n_critic",             1)
        self.use_wgan         = self.cfg.get("use_wgan",         False)
        self.lambda_gp        = self.cfg.get("lambda_gp",         10.0)
        self.label_smooth     = self.cfg.get("label_smoothing",    0.1)
        self.save_every       = self.cfg.get("save_every",          20)
        self.ckpt_dir         = Path(self.cfg.get("checkpoint_dir", "models/gan_checkpoints"))
        self.log_dir          = Path(self.cfg.get("log_dir",    "data/logs"))
        self.grad_clip_g      = float(self.cfg.get("grad_clip_g",  0.0))
        self.grad_clip_d      = float(self.cfg.get("grad_clip_d",  0.0))
        self.warm_up_epochs   = self.cfg.get("warm_up_epochs",       3)

        # FIX 3: reduced lambda_fm + hard clamp on raw fm_loss
        self.lambda_fm        = self.cfg.get("lambda_fm",          0.01)
        self.fm_clamp         = self.cfg.get("fm_clamp",           10.0)

        # FIX 3: EMA decay for stable feature-matching target
        self.ema_decay        = self.cfg.get("ema_decay",          0.99)
        self._ema_D: Optional[nn.Module] = None

        # FIX 4: GP warm-up ramp and per-step clamp
        self.gp_warmup_epochs = self.cfg.get("gp_warmup_epochs",   10)
        self.gp_clamp_factor  = self.cfg.get("gp_clamp_factor",     5.0)

        # v7: minimum GP always applied during warm-up to bound D weights
        self.min_gp_lambda    = self.cfg.get("min_gp_lambda",       1.0)

        # Accuracy gate — disabled under WGAN-GP
        if self.use_wgan:
            self.d_acc_threshold = 1.0
            logger.info("WGAN-GP: accuracy gate disabled, GP enforces Lipschitz")
        else:
            self.d_acc_threshold = self.cfg.get("d_acc_threshold", 0.85)

        # Noise dim
        if self._is_cond and isinstance(G, CondGenerator):
            self.noise_dim = self.G.real_noise_dim
        else:
            self.noise_dim = self.cfg.get("noise_dim", 100)

        # Adam β1=0.0 for WGAN-GP
        betas = (0.0, 0.9) if self.use_wgan else (0.5, 0.999)
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr_g, betas=betas)
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr_d, betas=betas)

        # No LR schedule under WGAN-GP
        if self.use_wgan:
            self.sched_G = None
            self.sched_D = None
        else:
            self.sched_G = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_G, T_max=self.epochs
            )
            self.sched_D = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_D, T_max=self.epochs
            )

        self.bce = nn.BCELoss()

        self.history: list[EpochMetrics] = []
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trainer ready | device=%s  epochs=%d  lr_g=%.1e  lr_d=%.1e  "
            "WGAN=%s  n_critic=%d  conditional=%s  noise_dim=%d  "
            "clip_g=%.1f  clip_d=%.1f  warm_up=%d  d_acc_gate=%.2f  "
            "lambda_fm=%.4f  fm_clamp=%.1f  ema_decay=%.3f  "
            "gp_warmup=%d  gp_clamp_factor=%.1f  min_gp_lambda=%.1f",
            self.device, self.epochs, self.lr_g, self.lr_d,
            self.use_wgan, self.n_critic, self._is_cond, self.noise_dim,
            self.grad_clip_g, self.grad_clip_d,
            self.warm_up_epochs, self.d_acc_threshold,
            self.lambda_fm, self.fm_clamp, self.ema_decay,
            self.gp_warmup_epochs, self.gp_clamp_factor, self.min_gp_lambda,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_regime(self, batch: int) -> Optional[torch.Tensor]:
        if not self._is_cond:
            return None
        if self._regime_sampler is not None:
            return self._regime_sampler(batch).to(self.device)
        import random
        return encode_regime_batch(
            [random.choice(REGIMES) for _ in range(batch)],
            device=self.device,
        )

    def _D(self, x: torch.Tensor, regime: Optional[torch.Tensor]) -> torch.Tensor:
        return self.D(x, regime) if self._is_cond else self.D(x)

    def _G(self, z: torch.Tensor, regime: Optional[torch.Tensor]) -> torch.Tensor:
        return self.G(z, regime) if self._is_cond else self.G(z)

    def _feat_live(self, x: torch.Tensor, regime: Optional[torch.Tensor]) -> torch.Tensor:
        """Features from the live D."""
        return (
            self.D.feature_extract(x, regime) if self._is_cond
            else self.D.feature_extract(x)
        )

    def _feat_ema(self, x: torch.Tensor, regime: Optional[torch.Tensor]) -> torch.Tensor:
        """
        FIX 3: Features from EMA copy of D — stable target for feature-matching.
        Falls back to live D if EMA not yet initialised.
        """
        if self._ema_D is None:
            return self._feat_live(x, regime)
        if self._is_cond:
            return self._ema_D.feature_extract(x, regime)
        return self._ema_D.feature_extract(x)

    def _gp(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        regime: Optional[torch.Tensor],
        effective_lambda: float,
    ) -> torch.Tensor:
        """FIX 4: GP with effective lambda and post-computation clamp."""
        if self._is_cond:
            gp = self.D.gradient_penalty(
                real, fake, self.device, effective_lambda, regime=regime
            )
        else:
            gp = self.D.gradient_penalty(real, fake, self.device, effective_lambda)

        max_gp = self.gp_clamp_factor * effective_lambda
        return gp.clamp(max=max_gp)

    def _effective_lambda_gp(self, epoch: int) -> float:
        """
        FIX 4 + v7: Logged/displayed lambda_gp value.

        Returns 0 during warm-up (for display) but _train_D always applies
        at least min_gp_lambda regardless, to prevent D from growing unbounded.
        After warm-up: linear ramp from 0 → lambda_gp over gp_warmup_epochs.
        """
        if epoch < self.warm_up_epochs:
            return 0.0
        post_warmup = epoch - self.warm_up_epochs
        if self.gp_warmup_epochs <= 0:
            return self.lambda_gp
        ramp = min(1.0, post_warmup / self.gp_warmup_epochs)
        return self.lambda_gp * ramp

    def _fresh_z(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.noise_dim, device=self.device)

    def _init_ema_D(self) -> None:
        """
        v7 FIX: Build EMA copy of D via state_dict — safe with spectral_norm.

        deepcopy fails on spectral_norm modules because they register
        non-leaf tensors (weight_u, weight_v) which raise:
            RuntimeError: Only Tensors created explicitly by the user
            (graph leaves) support the deepcopy protocol at the moment.

        Solution: rebuild a fresh D from config (same architecture), then
        copy the live D's weights via load_state_dict(). This bypasses
        all deepcopy restrictions while producing an identical parameter copy.
        """
        from gan.conditional_gan import build_cond_discriminator
        from gan.discriminator import build_discriminator

        if self._is_cond:
            self._ema_D = build_cond_discriminator(self.cfg).to(self.device)
        else:
            self._ema_D = build_discriminator(self.cfg).to(self.device)

        self._ema_D.load_state_dict(self.D.state_dict())
        self._ema_D.eval()
        for p in self._ema_D.parameters():
            p.requires_grad_(False)
        logger.info(
            "EMA discriminator initialised via state_dict "
            "(spectral_norm-safe — deepcopy replaced)."
        )

    # ------------------------------------------------------------------
    # Discriminator update
    # ------------------------------------------------------------------

    def _train_D(
        self,
        real_batch: torch.Tensor,
        regime: Optional[torch.Tensor],
        epoch: int,
    ) -> tuple[float, float, float, bool, float]:
        """
        One discriminator update step.

        v7: GP always applied with at least min_gp_lambda even during warm-up.
        This prevents D weights from growing without bound when G is frozen
        and there is no adversarial signal from G to naturally constrain D.

        Returns (d_loss, d_real_score, d_fake_score, was_skipped, gp_val)
        """
        batch = real_batch.size(0)
        real  = real_batch.to(self.device)

        # Instance noise — small perturbation for regularisation
        noise_scale = 0.01 * real.std().clamp(min=1e-6)
        real = real + noise_scale * torch.randn_like(real)

        # Accuracy gate (no_grad)
        with torch.no_grad():
            z_gate      = self._fresh_z(batch)
            fake_gate   = self._G(z_gate, regime)
            d_real_gate = self._D(real, regime)
            d_fake_gate = self._D(fake_gate, regime)

            if not self.use_wgan:
                acc_real = (d_real_gate > 0.5).float().mean().item()
                acc_fake = (d_fake_gate < 0.5).float().mean().item()
                d_acc    = (acc_real + acc_fake) / 2.0
            else:
                d_acc = 0.0   # gate never fires under WGAN

        if d_acc >= self.d_acc_threshold:
            d_real_score = d_real_gate.mean().item()
            d_fake_score = d_fake_gate.mean().item()
            if self.use_wgan:
                d_loss_val = (-d_real_gate.mean() + d_fake_gate.mean()).item()
            else:
                rl = torch.ones(batch, 1, device=self.device) * (1 - self.label_smooth)
                fl = torch.zeros(batch, 1, device=self.device)
                d_loss_val = (
                    self.bce(d_real_gate, rl) + self.bce(d_fake_gate, fl)
                ).item()
            return d_loss_val, d_real_score, d_fake_score, True, 0.0

        # Full D update
        self.opt_D.zero_grad()
        gp_val = 0.0

        if self.use_wgan:
            z_fresh  = self._fresh_z(batch)
            fake     = self._G(z_fresh, regime)
            fake_det = fake.detach()

            d_real = self._D(real, regime)
            d_fake = self._D(fake_det, regime)

            # v7: floor at min_gp_lambda to bound D during warm-up.
            # _effective_lambda_gp returns 0 during warm-up for logging only;
            # actual applied lambda is always >= min_gp_lambda.
            eff_lambda    = self._effective_lambda_gp(epoch)
            actual_lambda = max(eff_lambda, self.min_gp_lambda)

            gp     = self._gp(real, fake_det, regime, actual_lambda)
            gp_val = gp.item()
            d_loss = -d_real.mean() + d_fake.mean() + gp

            d_real_score = d_real.mean().item()
            d_fake_score = d_fake.mean().item()

        else:
            rl = torch.ones(batch, 1, device=self.device) * (1 - self.label_smooth)
            fl = torch.zeros(batch, 1, device=self.device)

            d_real    = self._D(real, regime)
            loss_real = self.bce(d_real, rl)

            z_fresh  = self._fresh_z(batch)
            fake_det = self._G(z_fresh, regime).detach()
            d_fake   = self._D(fake_det, regime)
            loss_fake = self.bce(d_fake, fl)

            d_loss       = loss_real + loss_fake
            d_real_score = d_real.mean().item()
            d_fake_score = d_fake.mean().item()

        d_loss.backward()
        if self.grad_clip_d > 0:
            nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip_d)
        self.opt_D.step()

        # FIX 3: update EMA after each D step
        if self._ema_D is not None:
            _update_ema(self._ema_D, self.D, self.ema_decay)

        return d_loss.item(), d_real_score, d_fake_score, False, gp_val

    # ------------------------------------------------------------------
    # Generator update
    # ------------------------------------------------------------------

    def _train_G(
        self,
        real_batch: torch.Tensor,
        regime: Optional[torch.Tensor],
    ) -> tuple[float, float, float]:
        """
        One generator update step.

        FIX 3: Feature-matching uses EMA D features (stable target).
                fm_loss hard-clamped at fm_clamp before scaling.

        Returns (g_loss, g_adv_loss, fm_loss_raw)
        """
        self.opt_G.zero_grad()
        batch = real_batch.size(0)
        real  = real_batch.to(self.device)

        z    = self._fresh_z(batch)
        fake = self._G(z, regime)

        # Adversarial loss
        if self.use_wgan:
            adv_loss = -self._D(fake, regime).mean()
        else:
            rl       = torch.ones(batch, 1, device=self.device)
            adv_loss = self.bce(self._D(fake, regime), rl)

        # FIX 3: Feature-matching against EMA D (stable target)
        if self.lambda_fm > 0.0:
            with torch.no_grad():
                real_feat = self._feat_ema(real, regime)       # EMA — stable reference
            fake_feat   = self._feat_live(fake, regime)         # live D on fake
            fm_loss_raw = nn.functional.mse_loss(fake_feat, real_feat.detach())

            # Hard clamp before scaling prevents a single bad batch from
            # spiking G_loss and drowning the adversarial signal
            fm_loss_clamped = fm_loss_raw.clamp(max=self.fm_clamp)
            g_loss = adv_loss + self.lambda_fm * fm_loss_clamped
        else:
            fm_loss_raw = torch.tensor(0.0, device=self.device)
            g_loss      = adv_loss

        g_loss.backward()
        if self.grad_clip_g > 0:
            nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip_g)
        self.opt_G.step()

        return g_loss.item(), adv_loss.item(), fm_loss_raw.item()

    # ------------------------------------------------------------------
    # Gradient norm
    # ------------------------------------------------------------------

    def _grad_norm(self, model: nn.Module) -> float:
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        return math.sqrt(total)

    # ------------------------------------------------------------------
    # NaN guard
    # ------------------------------------------------------------------

    def _check_nan(self, metrics: EpochMetrics, epoch: int) -> bool:
        if math.isnan(metrics.d_loss) or math.isnan(metrics.g_loss):
            logger.error(
                "NaN detected at epoch %d — d_loss=%.4f  g_loss=%.4f. "
                "Saving crash checkpoint and halting.",
                epoch + 1, metrics.d_loss, metrics.g_loss,
            )
            self.save_checkpoint(epoch + 1, suffix="crash")
            return True
        return False

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int) -> EpochMetrics:
        self.G.train()
        self.D.train()

        g_frozen = epoch < self.warm_up_epochs

        # FIX 3 + v7: initialise EMA D (via state_dict) on first G-active epoch
        if not g_frozen and self._ema_D is None:
            self._init_ema_D()

        d_losses, g_losses, g_adv_losses, fm_losses, gp_losses = [], [], [], [], []
        d_real_scores, d_fake_scores = [], []
        d_gnorms, g_gnorms = [], []
        d_skipped = 0
        data_stds = []
        t0 = time.time()

        for real_batch in self.dataloader:
            if isinstance(real_batch, (list, tuple)):
                real_batch = real_batch[0]

            batch  = real_batch.size(0)
            regime = self._sample_regime(batch)

            # FIX 1: track data std for normalisation diagnostics
            data_stds.append(real_batch.std().item())

            # Discriminator updates (n_critic steps)
            for _ in range(self.n_critic):
                d_loss, d_real, d_fake, skipped, gp_val = self._train_D(
                    real_batch, regime, epoch
                )
                d_losses.append(d_loss)
                d_real_scores.append(d_real)
                d_fake_scores.append(d_fake)
                gp_losses.append(gp_val)
                if skipped:
                    d_skipped += 1
            d_gnorms.append(self._grad_norm(self.D))

            # Generator update (skip during warm-up)
            if not g_frozen:
                g_loss, g_adv, fm_loss = self._train_G(real_batch, regime)
                g_losses.append(g_loss)
                g_adv_losses.append(g_adv)
                fm_losses.append(fm_loss)
                g_gnorms.append(self._grad_norm(self.G))
            else:
                g_losses.append(0.0)
                g_adv_losses.append(0.0)
                fm_losses.append(0.0)
                g_gnorms.append(0.0)

        if self.sched_G is not None:
            self.sched_G.step()
        if self.sched_D is not None:
            self.sched_D.step()

        def _avg(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        avg_data_std = _avg(data_stds)

        # FIX 1: warn on epoch 1 if data std indicates broken normalisation
        if epoch == 0 and avg_data_std > 2.0:
            logger.warning(
                "Epoch 1 data_std=%.4f > 2.0 — normalisation may be broken. "
                "If D_gnorm still pins at clip from epoch 1, check norm_method "
                "and delete __pycache__ to flush stale .pyc files.",
                avg_data_std,
            )

        avg_d_real = _avg(d_real_scores)
        avg_d_fake = _avg(d_fake_scores)

        return EpochMetrics(
            epoch            = epoch,
            d_loss           = _avg(d_losses),
            g_loss           = _avg(g_losses),
            g_adv_loss       = _avg(g_adv_losses),
            g_fm_loss        = _avg(fm_losses),
            d_real_score     = avg_d_real,
            d_fake_score     = avg_d_fake,
            gp_loss          = _avg(gp_losses),
            elapsed_sec      = time.time() - t0,
            d_grad_norm      = _avg(d_gnorms),
            g_grad_norm      = _avg(g_gnorms),
            d_skipped_steps  = d_skipped,
            wasserstein_dist = avg_d_real - avg_d_fake,
            g_frozen         = g_frozen,
            data_std         = avg_data_std,
        )

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, resume_epoch: int = 0) -> list[EpochMetrics]:
        logger.info(
            "=== GAN Training Started | %d epochs | conditional=%s | WGAN=%s | warm_up=%d ===",
            self.epochs, self._is_cond, self.use_wgan, self.warm_up_epochs,
        )
        log_path = self.log_dir / "gan_losses.csv"
        self._init_log(log_path)

        for epoch in range(resume_epoch, self.epochs):
            metrics = self._run_epoch(epoch)
            self.history.append(metrics)
            self._log_metrics(metrics, log_path)

            if self._check_nan(metrics, epoch):
                break

            if self.use_wgan:
                logger.info(
                    "Epoch %4d/%d | D=%.4f  G=%.4f (adv=%.4f fm=%.4f)  "
                    "W_dist=%.4f  GP=%.4f  "
                    "D_gnorm=%.3f  G_gnorm=%.3f  skip=%d  frozen=%s  "
                    "data_std=%.3f  eff_λgp=%.2f  [%.1fs]",
                    epoch + 1, self.epochs,
                    metrics.d_loss, metrics.g_loss,
                    metrics.g_adv_loss, metrics.g_fm_loss,
                    metrics.wasserstein_dist, metrics.gp_loss,
                    metrics.d_grad_norm, metrics.g_grad_norm,
                    metrics.d_skipped_steps, metrics.g_frozen,
                    metrics.data_std,
                    self._effective_lambda_gp(epoch),
                    metrics.elapsed_sec,
                )
            else:
                logger.info(
                    "Epoch %4d/%d | D=%.4f  G=%.4f (adv=%.4f fm=%.4f)  "
                    "D(real)=%.3f  D(fake)=%.3f  "
                    "D_gnorm=%.3f  G_gnorm=%.3f  skip=%d  frozen=%s  "
                    "data_std=%.3f  [%.1fs]",
                    epoch + 1, self.epochs,
                    metrics.d_loss, metrics.g_loss,
                    metrics.g_adv_loss, metrics.g_fm_loss,
                    metrics.d_real_score, metrics.d_fake_score,
                    metrics.d_grad_norm, metrics.g_grad_norm,
                    metrics.d_skipped_steps, metrics.g_frozen,
                    metrics.data_std,
                    metrics.elapsed_sec,
                )

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(self.epochs, suffix="final")
        logger.info("=== Training Complete ===")
        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, suffix: str = "") -> Path:
        tag  = f"epoch_{epoch:04d}" + (f"_{suffix}" if suffix else "")
        path = self.ckpt_dir / f"gan_{tag}.pt"
        torch.save(
            {
                "epoch":               epoch,
                "generator_state":     self.G.state_dict(),
                "discriminator_state": self.D.state_dict(),
                "opt_G_state":         self.opt_G.state_dict(),
                "opt_D_state":         self.opt_D.state_dict(),
                "cfg":                 self.cfg,
                "conditional":         self._is_cond,
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["generator_state"])
        self.D.load_state_dict(ckpt["discriminator_state"])
        self.opt_G.load_state_dict(ckpt["opt_G_state"])
        self.opt_D.load_state_dict(ckpt["opt_D_state"])
        epoch = ckpt.get("epoch", 0)
        logger.info("Loaded checkpoint from %s (epoch %d)", path, epoch)
        return epoch

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_log(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "d_loss", "g_loss", "g_adv_loss", "g_fm_loss",
                "d_real_score", "d_fake_score", "gp_loss",
                "d_grad_norm", "g_grad_norm",
                "d_skipped_steps", "wasserstein_dist", "g_frozen",
                "data_std", "elapsed_sec",
            ])

    def _log_metrics(self, m: EpochMetrics, path: Path) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                m.epoch,
                round(m.d_loss, 6),       round(m.g_loss, 6),
                round(m.g_adv_loss, 6),   round(m.g_fm_loss, 6),
                round(m.d_real_score, 4), round(m.d_fake_score, 4),
                round(m.gp_loss, 6),
                round(m.d_grad_norm, 4),  round(m.g_grad_norm, 4),
                m.d_skipped_steps,
                round(m.wasserstein_dist, 6),
                int(m.g_frozen),
                round(m.data_std, 6),
                round(m.elapsed_sec, 2),
            ])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_trainer(
    G: Generator,
    D: Discriminator,
    dataloader: torch.utils.data.DataLoader,
    cfg: dict | None = None,
    regime_sampler: Optional[Callable[[int], torch.Tensor]] = None,
) -> Trainer:
    return Trainer(G, D, dataloader, cfg, regime_sampler=regime_sampler)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import random
    from torch.utils.data import TensorDataset, DataLoader
    from gan.generator import build_generator
    from gan.discriminator import build_discriminator
    from gan.conditional_gan import build_cond_generator, build_cond_discriminator
    from gan.data_loader import build_dataset, SP500_TOP_50

    parser = argparse.ArgumentParser(description="Train the OHLCV GAN.")
    parser.add_argument("--ticker", default="SPY",
                        help="SPY | TOP50 | comma-separated list")
    parser.add_argument("--start",       default="2010-01-01")
    parser.add_argument("--end",         default="2023-12-31")
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--conditional", action="store_true")
    args = parser.parse_args()

    cfg             = _load_cfg()
    use_conditional = args.conditional or cfg.get("conditional", False)
    tickers         = (
        SP500_TOP_50
        if args.ticker == "TOP50"
        else [t.strip() for t in args.ticker.split(",")]
    )

    if len(tickers) > 1:
        pt_path = Path("data/raw/SP50.pt")
        if pt_path.exists():
            tensor = torch.load(pt_path, map_location="cpu")
            logger.info("Loaded cached tensor %s", tuple(tensor.shape))
        else:
            result = build_dataset(
                ticker               = tickers,
                start                = args.start,
                end                  = args.end,
                seq_len              = cfg.get("seq_len", 30),
                method               = cfg.get("norm_method", "smart"),
                stride               = cfg.get("stride", 1),
                per_ticker_normalise = True,
                skip_errors          = True,
                save_path            = "data/raw/SP50",
            )
            tensor, _, _ = result
    else:
        pt_path = Path(f"data/raw/{tickers[0]}.pt")
        if pt_path.exists():
            tensor = torch.load(pt_path, map_location="cpu")
            logger.info("Loaded cached tensor %s", tuple(tensor.shape))
        else:
            result = build_dataset(
                ticker    = tickers[0],
                start     = args.start,
                end       = args.end,
                seq_len   = cfg.get("seq_len", 30),
                method    = cfg.get("norm_method", "smart"),
                stride    = cfg.get("stride", 1),
                save_path = f"data/raw/{tickers[0]}",
            )
            tensor, _, _ = result

    loader = DataLoader(
        TensorDataset(tensor),
        batch_size = cfg.get("batch_size", 64),
        shuffle    = True,
        drop_last  = True,
    )

    if use_conditional:
        G = build_cond_generator(cfg)
        D = build_cond_discriminator(cfg)
        regime_sampler = lambda n: encode_regime_batch(
            [random.choice(REGIMES) for _ in range(n)]
        )
    else:
        G = build_generator(cfg)
        D = build_discriminator(cfg)
        regime_sampler = None

    trainer = build_trainer(G, D, loader, cfg, regime_sampler)

    resume = 0
    if args.checkpoint:
        resume = trainer.load_checkpoint(args.checkpoint)

    trainer.run(resume_epoch=resume)