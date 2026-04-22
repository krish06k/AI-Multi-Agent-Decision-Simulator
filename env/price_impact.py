"""
env/price_impact.py
===================
Layer 2 — RL Market Environment | Price Impact Model

Converts net order flow (imbalance) into a price change each tick.

Models supported
----------------
* linear  — ΔP = λ · Q_net                   (simple, transparent)
* sqrt    — ΔP = λ · sign(Q) · √|Q|          (Almgren-Chriss style)
* log     — ΔP = λ · sign(Q) · log(1 + |Q|)  (diminishing impact)
* kyle    — ΔP = λ · Q_net (like Kyle 1985, but with dynamic λ)

All models share:
* Volatility scaling : impact is amplified when market is volatile.
* Decay / mean reversion : residual impact decays each tick.
* Spread enforcement : mid-price kept ≥ min_spread from best quotes.
* Price floor : price cannot go below min_price.

FIX SUMMARY
-----------
FIX 1 — Residual decay reduced from 0.85 → 0.30.
         Old decay of 0.85 meant each tick carried 85% of prior impact
         forward, compounding into permanent upward drift on any net-buy
         imbalance. 0.30 dissipates residual cleanly within ~3 ticks.

FIX 2 — Residual now decays on `raw` impact only, not on `total_impact`.
         Old: self._residual = decay * total_impact  (includes prior residual)
         New: self._residual = decay * raw           (only current tick)
         Prevents the residual from feeding itself into an infinite loop.

FIX 3 — Adaptive lambda vol_baseline corrected.
         Old: vol_baseline = lambda_base * 50  (arbitrary, uncalibrated)
         New: vol_baseline = 0.002             (reasonable EWM vol baseline)
         Prevents vol_scalar from spiking to 5x on normal market moves.

FIX 4 — noise_std default reduced from 0.0002 → 0.0001.
         Microstructure noise was adding small but persistent upward drift
         when combined with a high residual decay factor.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Literal, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "env_config.yaml"
ImpactModel = Literal["linear", "sqrt", "log", "kyle"]


def _load_cfg() -> dict:
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Volatility estimator
# ---------------------------------------------------------------------------

class RolisedVolEstimator:
    """
    Online exponentially-weighted realised volatility estimator.

    Uses log-return EWM to estimate current market volatility.
    This is fed back into the impact model to scale λ dynamically.

    Parameters
    ----------
    window      : Number of ticks for EWM half-life.
    min_vol     : Minimum vol estimate (prevents zero denominator).
    annualise   : If True, scale to annualised vol (assuming daily ticks).
    """

    def __init__(self, window: int = 20, min_vol: float = 1e-4, annualise: bool = False):
        self.window    = window
        self.min_vol   = min_vol
        self.annualise = annualise
        self._prices: Deque[float] = deque(maxlen=window + 1)
        self._ewm_var: float       = 0.0
        self._alpha: float         = 2.0 / (window + 1)

    def update(self, price: float) -> float:
        self._prices.append(price)
        if len(self._prices) < 2:
            return self.min_vol

        log_ret = math.log(self._prices[-1] / max(self._prices[-2], 1e-8))
        self._ewm_var = (self._alpha * log_ret ** 2 +
                         (1 - self._alpha) * self._ewm_var)
        vol = math.sqrt(max(self._ewm_var, self.min_vol ** 2))

        if self.annualise:
            vol *= math.sqrt(252)

        return vol

    def current(self) -> float:
        return math.sqrt(max(self._ewm_var, self.min_vol ** 2))

    def reset(self) -> None:
        self._prices.clear()
        self._ewm_var = 0.0


# ---------------------------------------------------------------------------
# Impact result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImpactResult:
    """Result of one tick's price impact calculation."""
    price_before:     float
    price_after:      float
    delta:            float
    net_order_flow:   float   # signed net qty (+ = net buy)
    raw_impact:       float   # impact before decay/noise
    vol_scalar:       float   # volatility multiplier applied
    depth_scalar:     float   # book depth multiplier applied
    noise:            float   # microstructure noise added
    residual_impact:  float   # carry-over from previous tick


# ---------------------------------------------------------------------------
# Price Impact Engine
# ---------------------------------------------------------------------------

class PriceImpactModel:
    """
    Tick-by-tick price impact engine.

    Parameters
    ----------
    model           : Impact function — "linear" | "sqrt" | "log" | "kyle".
    lambda_base     : Base impact coefficient.
    decay           : Residual impact decay factor per tick (0 < decay < 1).
                      FIX 1: Reduced default from 0.85 → 0.30 to prevent
                      compounding upward drift on net-buy order flow.
    min_price       : Hard floor on price.
    vol_window      : Ticks for volatility estimator.
    noise_std       : Microstructure noise std as fraction of price.
                      FIX 4: Reduced default from 0.0002 → 0.0001.
    spread_bps      : Baseline spread in basis points.
    adaptive_lambda : If True, scale λ by current realised volatility.
    """

    def __init__(
        self,
        model:           ImpactModel = "sqrt",
        lambda_base:     float       = 0.005,
        decay:           float       = 0.30,   # FIX 1: was 0.85
        min_price:       float       = 1.0,
        vol_window:      int         = 20,
        noise_std:       float       = 0.0001, # FIX 4: was 0.0002
        spread_bps:      float       = 10.0,
        adaptive_lambda: bool        = True,
    ):
        self.model           = model
        self.lambda_base     = lambda_base
        self.decay           = decay
        self.min_price       = min_price
        self.noise_std       = noise_std
        self.spread_bps      = spread_bps
        self.adaptive_lambda = adaptive_lambda

        self._vol_est  = RolisedVolEstimator(window=vol_window)
        self._residual: float = 0.0
        self._price:    float = 100.0

        self._impact_history: list[ImpactResult] = []

        logger.info(
            "PriceImpactModel | model=%s  λ=%.4f  decay=%.2f  adaptive=%s",
            model, lambda_base, decay, adaptive_lambda,
        )

    # ------------------------------------------------------------------
    # Core: compute impact
    # ------------------------------------------------------------------

    def _raw_impact(self, net_flow: float, lambda_eff: float) -> float:
        """Apply selected impact function."""
        if net_flow == 0:
            return 0.0
        sign  = 1.0 if net_flow > 0 else -1.0
        abs_q = abs(net_flow)

        if self.model == "linear":
            return lambda_eff * net_flow
        elif self.model == "sqrt":
            return sign * lambda_eff * math.sqrt(abs_q)
        elif self.model == "log":
            return sign * lambda_eff * math.log1p(abs_q)
        elif self.model == "kyle":
            return lambda_eff * net_flow
        else:
            raise ValueError(f"Unknown impact model: {self.model}")

    def step(
        self,
        net_order_flow:   float,
        current_price:    float,
        book_depth:       float = 1.0,
        shock_multiplier: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> ImpactResult:
        """
        Compute price change for one tick.

        Parameters
        ----------
        net_order_flow  : Signed net quantity (positive = net buys).
        current_price   : Price before this tick's impact.
        book_depth      : Normalised book depth scalar (1 = normal).
        shock_multiplier: External multiplier (>1 during shock events).
        rng             : Optional numpy random generator.

        Returns
        -------
        ImpactResult with new price and all intermediate values.
        """
        rng = rng or np.random.default_rng()

        # 1. Update volatility estimate
        vol = self._vol_est.update(current_price)
        self._price = current_price

        # 2. Adaptive lambda
        # FIX 3: use a fixed calibrated baseline instead of lambda_base * 50
        if self.adaptive_lambda:
            vol_baseline = 0.002   # FIX 3: was self.lambda_base * 50 (uncalibrated)
            vol_scalar   = 1.0 + max(0.0, (vol - vol_baseline) / max(vol_baseline, 1e-8))
            vol_scalar   = min(vol_scalar, 3.0)  # reduced cap from 5x → 3x
        else:
            vol_scalar = 1.0

        # 3. Book depth scalar: thin book → larger impact
        depth_scalar = 1.0 / max(book_depth, 0.1)
        depth_scalar = min(depth_scalar, 10.0)

        # 4. Effective lambda
        lambda_eff = self.lambda_base * vol_scalar * depth_scalar * shock_multiplier

        # 5. Raw impact from current flow
        raw = self._raw_impact(net_order_flow, lambda_eff)

        # 6. Add decayed residual from previous tick
        total_impact = raw + self._residual

        # 7. Microstructure noise — zero-mean, does not create directional drift
        noise = rng.normal(0, self.noise_std * current_price)

        # 8. New price
        new_price = max(self.min_price, current_price + total_impact + noise)

        # FIX 2: decay residual on RAW only — not on total_impact.
        # Old: self._residual = self.decay * total_impact
        #      → residual fed itself, growing without bound on sustained imbalance
        # New: self._residual = self.decay * raw
        #      → each tick's carry-over is bounded by that tick's fresh impact
        self._residual = self.decay * raw

        result = ImpactResult(
            price_before    = current_price,
            price_after     = new_price,
            delta           = new_price - current_price,
            net_order_flow  = net_order_flow,
            raw_impact      = raw,
            vol_scalar      = vol_scalar,
            depth_scalar    = depth_scalar,
            noise           = noise,
            residual_impact = self._residual,
        )
        self._impact_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Quote generation
    # ------------------------------------------------------------------

    def generate_quotes(
        self,
        mid_price:        float,
        vol:              Optional[float] = None,
        shock_multiplier: float           = 1.0,
    ) -> tuple[float, float]:
        """
        Derive bid and ask quotes from mid-price.
        Spread widens under high volatility and during shocks.
        """
        vol = vol or self._vol_est.current()
        base_spread_pct = self.spread_bps / 10000.0

        vol_spread_adj       = 1.0 + 5.0 * vol
        effective_spread_pct = base_spread_pct * vol_spread_adj * shock_multiplier
        half_spread          = mid_price * effective_spread_pct / 2.0

        bid = max(self.min_price, mid_price - half_spread)
        ask = mid_price + half_spread
        return round(bid, 4), round(ask, 4)

    # ------------------------------------------------------------------
    # Aggregate flow from order book
    # ------------------------------------------------------------------

    @staticmethod
    def compute_net_flow(buy_volume: int, sell_volume: int) -> float:
        return float(buy_volume - sell_volume)

    @staticmethod
    def volume_weighted_impact(
        trades:          list,
        reference_price: float,
    ) -> float:
        total_vol = sum(t.qty for t in trades)
        if total_vol == 0:
            return 0.0
        vwap = sum(t.price * t.qty for t in trades) / total_vol
        return vwap - reference_price

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, initial_price: float = 100.0) -> None:
        self._vol_est.reset()
        self._residual = 0.0
        self._price    = initial_price
        self._impact_history.clear()
        self._vol_est.update(initial_price)

    @property
    def current_vol(self) -> float:
        return self._vol_est.current()

    @property
    def residual_impact(self) -> float:
        return self._residual

    def history(self) -> list[ImpactResult]:
        return list(self._impact_history)

    def stats(self) -> dict:
        if not self._impact_history:
            return {}
        deltas = [r.delta for r in self._impact_history]
        return {
            "ticks":            len(deltas),
            "mean_delta":       float(np.mean(deltas)),
            "std_delta":        float(np.std(deltas)),
            "max_up":           float(max(deltas)),
            "max_down":         float(min(deltas)),
            "current_vol":      self.current_vol,
            "residual_impact":  self._residual,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_price_impact(cfg: dict | None = None) -> PriceImpactModel:
    """Build PriceImpactModel from env_config.yaml."""
    cfg = cfg or _load_cfg()
    return PriceImpactModel(
        model           = cfg.get("impact_model",       "sqrt"),
        lambda_base     = cfg.get("impact_lambda",       0.005),
        decay           = cfg.get("decay_factor",        0.30),   # FIX 1
        min_price       = cfg.get("min_price",           1.0),
        vol_window      = cfg.get("volatility_window",   20),
        spread_bps      = cfg.get("spread_bps",          10),
        noise_std       = cfg.get("noise_std",           0.0001), # FIX 4
    )


if __name__ == "__main__":
    print("Testing PriceImpactModel (fixed)...\n")

    model = PriceImpactModel()
    price = 100.0

    for step in range(20):
        net_flow = np.random.randint(-200, 200)
        result   = model.step(net_order_flow=net_flow, current_price=price)
        price    = result.price_after
        print(
            f"Step {step:2d} | flow={net_flow:+4d} | "
            f"price={result.price_after:.4f} | "
            f"delta={result.delta:+.5f} | "
            f"residual={result.residual_impact:+.5f}"
        )

    print(f"\nFinal price: {price:.4f}  (started at 100.0)")
    print(f"Drift: {price - 100:.4f}")