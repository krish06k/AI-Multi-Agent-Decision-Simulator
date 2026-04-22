"""
env/shock.py
============
Layer 2 — RL Market Environment | Shock Event System

KEY FIXES vs previous version
------------------------------
FIX 1 — Flash Crash magnitude is now a TOTAL drop spread across ticks,
         not applied at full magnitude every single tick.
         Old: price_force = -0.25 every tick → -750% total over 30 ticks
         New: price_force = total_drop / duration_ticks → correct total drop

FIX 2 — Volatility spike whipsaw amplitude capped and scaled correctly.
         Old: secondary_mag applied raw every tick → vertical spikes
         New: amplitude scaled as fraction of price per tick, capped at 1.5%

FIX 3 — Post-shock dampening added via ShockEffect.post_shock_dampen flag.
         Market env applies a mean-reversion drag for N ticks after shock.

FIX 4 — News shock reprice now a one-time jump (prog < first tick only),
         not repeated every tick during the 10% window.

FIX 5 — (NEW) Random shock direction for NewsShock is now truly 50/50.
         Old: direction defaulted to -1 (always downward news shock).
         New: maybe_random_shock passes a random direction for news_shock
         so upward and downward news is equally probable, preventing
         any systematic downward bias from random shocks.

FIX 6 — (NEW) shock_probability reduced from 0.02 → 0.01 default.
         2% chance per tick = ~20 shocks per 1000-tick episode, which is
         unrealistically frequent. 1% gives ~10 shocks per episode.

Shock types
-----------
* FlashCrash       — sudden large price drop, volume spike, slow bleed recovery
* LiquidityCrisis  — spread widens, book empties, price grinds lower
* VolatilitySpike  — bidirectional whipsaw, no directional bias
* CircuitBreaker   — trading halts when move exceeds threshold
* NewsShock        — sharp directional jump, aftershock volatility tail
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "env_config.yaml"


def _load_cfg() -> dict:
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Shock types
# ---------------------------------------------------------------------------

class ShockType(Enum):
    FLASH_CRASH       = auto()
    LIQUIDITY_CRISIS  = auto()
    VOLATILITY_SPIKE  = auto()
    CIRCUIT_BREAKER   = auto()
    NEWS_SHOCK        = auto()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ShockEvent:
    """Immutable specification of a shock event."""
    shock_type:        ShockType
    start_tick:        int
    duration_ticks:    int
    magnitude:         float          # TOTAL move as fraction (e.g. 0.20 = 20% total drop)
    direction:         int   = -1
    volume_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    depth_reduction:   float = 0.0
    secondary_mag:     float = 0.0   # whipsaw amp / aftershock vol / grind drift
    shock_id:          int   = 0
    label:             str   = ""
    _reprice_done:     bool  = field(default=False, repr=False, compare=False)

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_ticks

    def is_active(self, tick: int) -> bool:
        return self.start_tick <= tick < self.end_tick

    def progress(self, tick: int) -> float:
        elapsed = tick - self.start_tick
        return min(1.0, elapsed / max(1, self.duration_ticks))

    def ticks_elapsed(self, tick: int) -> int:
        return tick - self.start_tick


@dataclass
class ShockEffect:
    """Per-tick effect consumed by market_env.step()."""
    price_shock_pct:   float = 0.0   # fractional price change this tick
    vol_multiplier:    float = 1.0
    spread_multiplier: float = 1.0
    depth_multiplier:  float = 1.0
    volume_multiplier: float = 1.0
    halt_trading:      bool  = False
    post_shock_dampen: bool  = False
    active_shocks:     list[ShockEvent] = field(default_factory=list)

    def is_stressed(self) -> bool:
        return (
            abs(self.price_shock_pct) > 0.001
            or self.vol_multiplier > 1.5
            or self.halt_trading
        )

    def regime_label(self) -> str:
        if self.halt_trading:
            return "circuit_breaker"
        if abs(self.price_shock_pct) > 0.05 or self.vol_multiplier > 3.0:
            return "stressed"
        if self.vol_multiplier > 1.5:
            return "volatile"
        return "calm"


# ---------------------------------------------------------------------------
# Shock Engine
# ---------------------------------------------------------------------------

class ShockEngine:
    def __init__(
        self,
        cfg: dict | None = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg         = cfg or _load_cfg()
        self.rng         = rng or np.random.default_rng()
        self._shock_cfg  = self.cfg.get("shock_types", {})

        # FIX 6: default reduced from 0.02 → 0.01 to prevent over-shocking
        self._shock_prob = self.cfg.get("shock_probability", 0.01)

        self._halt_threshold = (
            self._shock_cfg.get("circuit_breaker", {})
            .get("halt_threshold_pct", 7.0) / 100.0
        )

        self._active_shocks:    list[ShockEvent] = []
        self._scheduled_shocks: list[ShockEvent] = []
        self._shock_history:    list[ShockEvent] = []
        self._shock_id_counter: int = 0
        self._halted_until:     int = -1
        self._post_shock_ticks: int = 0

        logger.info("ShockEngine ready | shock_prob=%.3f", self._shock_prob)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def current_effect(self) -> ShockEffect:
        return getattr(self, "_last_effect", ShockEffect())

    def _next_id(self) -> int:
        self._shock_id_counter += 1
        return self._shock_id_counter

    @staticmethod
    def _ease_in(t: float) -> float:
        return t * t * t

    @staticmethod
    def _ease_out(t: float) -> float:
        return 1.0 - (1.0 - t) ** 3

    @staticmethod
    def _bell(t: float, width: float = 0.08) -> float:
        return math.exp(-((t - 0.5) ** 2) / width)

    # ------------------------------------------------------------------
    # Shock factories
    # ------------------------------------------------------------------

    def make_flash_crash(self, tick: int) -> ShockEvent:
        """
        Flash Crash.
        magnitude = TOTAL price drop spread across duration ticks (FIX 1).
        """
        sc        = self._shock_cfg.get("flash_crash", {})
        drop_min  = sc.get("price_drop_pct_min", 5.0)  / 100.0
        drop_max  = sc.get("price_drop_pct_max", 12.0) / 100.0
        magnitude = float(self.rng.uniform(drop_min, drop_max))
        duration  = sc.get("duration_ticks", 25)
        return ShockEvent(
            shock_type        = ShockType.FLASH_CRASH,
            start_tick        = tick,
            duration_ticks    = duration,
            magnitude         = magnitude,
            direction         = -1,
            volume_multiplier = sc.get("volume_spike", 6.0),
            spread_multiplier = 5.0,
            depth_reduction   = 0.80,
            shock_id          = self._next_id(),
            label             = f"Flash Crash -{magnitude*100:.1f}%",
        )

    def make_liquidity_crisis(self, tick: int) -> ShockEvent:
        sc        = self._shock_cfg.get("liquidity_crisis", {})
        drift_mag = float(self.rng.uniform(0.001, 0.004))
        return ShockEvent(
            shock_type        = ShockType.LIQUIDITY_CRISIS,
            start_tick        = tick,
            duration_ticks    = sc.get("duration_ticks", 20),
            magnitude         = drift_mag,
            direction         = -1,
            spread_multiplier = sc.get("spread_multiplier", 10.0),
            depth_reduction   = sc.get("depth_reduction", 0.88),
            volume_multiplier = 0.15,
            secondary_mag     = drift_mag * 0.3,
            shock_id          = self._next_id(),
            label             = "Liquidity Crisis",
        )

    def make_volatility_spike(self, tick: int) -> ShockEvent:
        """
        Volatility Spike — bidirectional, no net drift (FIX 2).
        """
        sc                = self._shock_cfg.get("volatility_spike", {})
        vol_mult          = sc.get("vol_multiplier", 5.0)
        whipsaw_amplitude = min(float(self.rng.uniform(0.002, 0.006)), 0.008)
        return ShockEvent(
            shock_type        = ShockType.VOLATILITY_SPIKE,
            start_tick        = tick,
            duration_ticks    = sc.get("duration_ticks", 15),
            magnitude         = vol_mult,
            direction         = 0,
            volume_multiplier = sc.get("volume_multiplier", 3.0),
            spread_multiplier = 3.5,
            depth_reduction   = 0.35,
            secondary_mag     = whipsaw_amplitude,
            shock_id          = self._next_id(),
            label             = f"Volatility Spike x{vol_mult:.0f}",
        )

    def make_news_shock(
        self,
        tick:           int,
        direction:      int             = -1,
        magnitude_pct:  Optional[float] = None,
    ) -> ShockEvent:
        """
        News Shock — one-time reprice on first tick only (FIX 4).
        """
        sc             = self._shock_cfg.get("news_shock", {})
        mag_min        = sc.get("magnitude_min_pct", 2.0) / 100.0
        mag_max        = sc.get("magnitude_max_pct", 8.0) / 100.0
        mag            = magnitude_pct or float(self.rng.uniform(mag_min, mag_max))
        aftershock_vol = mag * float(self.rng.uniform(0.15, 0.30))
        label          = f"News Shock {'UP' if direction > 0 else 'DOWN'} {mag*100:.1f}%"
        return ShockEvent(
            shock_type        = ShockType.NEWS_SHOCK,
            start_tick        = tick,
            duration_ticks    = sc.get("duration_ticks", 12),
            magnitude         = mag,
            direction         = direction,
            volume_multiplier = 4.0,
            spread_multiplier = 3.0,
            depth_reduction   = 0.55,
            secondary_mag     = aftershock_vol,
            shock_id          = self._next_id(),
            label             = label,
        )

    def make_circuit_breaker(self, tick: int) -> ShockEvent:
        sc = self._shock_cfg.get("circuit_breaker", {})
        return ShockEvent(
            shock_type     = ShockType.CIRCUIT_BREAKER,
            start_tick     = tick,
            duration_ticks = sc.get("halt_duration_ticks", 3),
            magnitude      = 0.0,
            direction      = 0,
            shock_id       = self._next_id(),
            label          = "CIRCUIT BREAKER Halt",
        )

    # ------------------------------------------------------------------
    # Scheduling & injection
    # ------------------------------------------------------------------

    def schedule(self, shock: ShockEvent) -> None:
        self._scheduled_shocks.append(shock)
        logger.info("Scheduled: %s at tick %d", shock.label, shock.start_tick)

    def inject_now(self, shock_type: str, tick: int, **kwargs) -> ShockEvent:
        makers = {
            "flash_crash":      self.make_flash_crash,
            "liquidity_crisis": self.make_liquidity_crisis,
            "volatility_spike": self.make_volatility_spike,
            "news_shock":       lambda t: self.make_news_shock(t, **kwargs),
            "circuit_breaker":  self.make_circuit_breaker,
        }
        if shock_type not in makers:
            raise ValueError(f"Unknown shock type: '{shock_type}'")
        shock = makers[shock_type](tick)
        self._active_shocks.append(shock)
        self._shock_history.append(shock)
        logger.info("Injected: %s", shock.label)
        return shock

    def maybe_random_shock(self, tick: int) -> Optional[ShockEvent]:
        if self.rng.random() > self._shock_prob:
            return None
        choices    = ["flash_crash", "liquidity_crisis", "volatility_spike", "news_shock"]
        weights    = [0.3, 0.2, 0.3, 0.2]
        shock_type = self.rng.choice(choices, p=weights)

        # FIX 5: randomise news shock direction — was always -1 (always bearish).
        # Equal probability up/down so random shocks have no systematic bias.
        kwargs = {}
        if shock_type == "news_shock":
            kwargs["direction"] = int(self.rng.choice([-1, 1]))

        return self.inject_now(shock_type, tick, **kwargs)

    # ------------------------------------------------------------------
    # Per-tick processing
    # ------------------------------------------------------------------

    def step(
        self,
        tick:          int,
        current_price: float,
        prev_price:    float,
    ) -> ShockEffect:
        # 1. Activate scheduled shocks
        newly_scheduled = [s for s in self._scheduled_shocks if s.start_tick <= tick]
        for shock in newly_scheduled:
            self._active_shocks.append(shock)
            self._shock_history.append(shock)
        self._scheduled_shocks = [s for s in self._scheduled_shocks if s.start_tick > tick]

        # 2. Random shock injection (only if none active)
        if not self._active_shocks:
            self.maybe_random_shock(tick)

        # 3. Auto circuit breaker
        if prev_price > 0:
            move_pct = abs(current_price - prev_price) / prev_price
            if move_pct >= self._halt_threshold and tick > self._halted_until:
                cb = self.make_circuit_breaker(tick)
                self._active_shocks.append(cb)
                self._shock_history.append(cb)
                self._halted_until = cb.end_tick
                logger.warning(
                    "Circuit breaker triggered! Move=%.1f%% at tick %d",
                    move_pct * 100, tick,
                )

        # 4. Track post-shock countdown
        had_active = len(self._active_shocks) > 0

        # 5. Expire finished shocks
        self._active_shocks = [s for s in self._active_shocks if s.is_active(tick)]

        # 6. Start post-shock dampening when shocks just expired
        if had_active and not self._active_shocks:
            self._post_shock_ticks = 15

        if self._post_shock_ticks > 0:
            self._post_shock_ticks -= 1

        # 7. Aggregate effects
        self._last_effect = self._aggregate_effects(tick, current_price)
        if self._post_shock_ticks > 0:
            self._last_effect.post_shock_dampen = True

        return self._last_effect

    # ------------------------------------------------------------------
    # Effect aggregation
    # ------------------------------------------------------------------

    def _aggregate_effects(self, tick: int, price: float) -> ShockEffect:
        effect = ShockEffect(active_shocks=list(self._active_shocks))

        for shock in self._active_shocks:
            prog = shock.progress(tick)

            # ── FLASH CRASH ─────────────────────────────────────────────
            if shock.shock_type == ShockType.FLASH_CRASH:
                dt = shock.duration_ticks
                if prog < 0.20:
                    phase_ticks = max(1, int(dt * 0.20))
                    per_tick    = (shock.magnitude * 0.50) / phase_ticks
                    vol_m       = 5.0
                    vol_mult    = shock.volume_multiplier
                elif prog < 0.60:
                    phase_ticks = max(1, int(dt * 0.40))
                    per_tick    = (shock.magnitude * 0.35) / phase_ticks
                    vol_m       = 3.0
                    vol_mult    = shock.volume_multiplier * 0.5
                else:
                    phase_ticks = max(1, int(dt * 0.40))
                    per_tick    = (shock.magnitude * 0.15) / phase_ticks
                    vol_m       = 1.8
                    vol_mult    = shock.volume_multiplier * 0.2

                effect.price_shock_pct  += shock.direction * per_tick
                effect.vol_multiplier    = max(effect.vol_multiplier, vol_m)
                effect.spread_multiplier = max(effect.spread_multiplier, shock.spread_multiplier)
                effect.depth_multiplier  = min(effect.depth_multiplier, 1 - shock.depth_reduction)
                effect.volume_multiplier = max(effect.volume_multiplier, vol_mult)

            # ── LIQUIDITY CRISIS ─────────────────────────────────────────
            elif shock.shock_type == ShockType.LIQUIDITY_CRISIS:
                if prog < 0.15:
                    ramp        = prog / 0.15
                    spread_m    = 1.0 + (shock.spread_multiplier - 1.0) * ramp
                    depth_red   = shock.depth_reduction * ramp
                    price_drift = 0.0
                    vol_m       = 1.5 + ramp
                    vol_mult    = 0.5
                elif prog < 0.50:
                    spread_m    = shock.spread_multiplier
                    depth_red   = shock.depth_reduction
                    price_drift = shock.direction * shock.magnitude
                    vol_m       = 2.5
                    vol_mult    = shock.volume_multiplier
                elif prog < 0.80:
                    spread_m    = shock.spread_multiplier * 0.60
                    depth_red   = shock.depth_reduction * 0.65
                    price_drift = shock.direction * shock.secondary_mag
                    vol_m       = 1.8
                    vol_mult    = shock.volume_multiplier
                else:
                    decay       = (prog - 0.80) / 0.20
                    spread_m    = shock.spread_multiplier * 0.60 * (1 - decay * 0.5)
                    depth_red   = shock.depth_reduction * 0.65 * (1 - decay * 0.6)
                    price_drift = 0.0
                    vol_m       = max(1.0, 1.8 - decay * 0.6)
                    vol_mult    = shock.volume_multiplier

                effect.price_shock_pct  += price_drift
                effect.vol_multiplier    = max(effect.vol_multiplier, vol_m)
                effect.spread_multiplier = max(effect.spread_multiplier, spread_m)
                effect.depth_multiplier  = min(effect.depth_multiplier, 1 - depth_red)
                effect.volume_multiplier = min(effect.volume_multiplier, vol_mult)

            # ── VOLATILITY SPIKE ─────────────────────────────────────────
            elif shock.shock_type == ShockType.VOLATILITY_SPIKE:
                if prog < 0.25:
                    t         = prog / 0.25
                    vol_m     = shock.magnitude * self._ease_in(t)
                    amplitude = shock.secondary_mag * self._ease_in(t)
                    spread_m  = shock.spread_multiplier * t
                elif prog < 0.75:
                    t         = (prog - 0.25) / 0.50
                    bell      = self._bell(t)
                    vol_m     = shock.magnitude * (0.6 + 0.4 * bell)
                    amplitude = shock.secondary_mag * (0.7 + 0.3 * bell)
                    spread_m  = shock.spread_multiplier
                else:
                    t         = (prog - 0.75) / 0.25
                    vol_m     = shock.magnitude * math.exp(-3.0 * t)
                    amplitude = shock.secondary_mag * math.exp(-3.0 * t)
                    spread_m  = shock.spread_multiplier * max(0.3, 1 - t)

                amplitude = min(amplitude, 0.015)  # hard cap 1.5% per tick (FIX 2)

                # Alternating sign — bidirectional, zero net drift
                whipsaw_sign = 1.0 if (tick % 2 == 0) else -1.0
                effect.price_shock_pct  += whipsaw_sign * amplitude
                effect.vol_multiplier    = max(effect.vol_multiplier, max(1.0, vol_m))
                effect.spread_multiplier = max(effect.spread_multiplier, max(1.0, spread_m))
                depth_red = shock.depth_reduction * max(0.0, 1 - max(0, (prog - 0.75) / 0.25))
                effect.depth_multiplier  = min(effect.depth_multiplier, 1 - depth_red)
                vol_mult_decay = shock.volume_multiplier * max(0.3, 1 - prog)
                effect.volume_multiplier = max(effect.volume_multiplier, vol_mult_decay)

            # ── NEWS SHOCK ───────────────────────────────────────────────
            elif shock.shock_type == ShockType.NEWS_SHOCK:
                # FIX 4: one-time reprice on tick 0 only
                if not shock._reprice_done and shock.ticks_elapsed(tick) == 0:
                    effect.price_shock_pct  += shock.direction * shock.magnitude
                    effect.volume_multiplier = max(effect.volume_multiplier, shock.volume_multiplier)
                    effect.spread_multiplier = max(effect.spread_multiplier, shock.spread_multiplier)
                    effect.depth_multiplier  = min(effect.depth_multiplier, 1 - shock.depth_reduction)
                    effect.vol_multiplier    = max(effect.vol_multiplier, 4.0)
                    shock._reprice_done      = True
                else:
                    t          = max(0, (prog - 0.10) / 0.90)
                    decay      = math.exp(-2.0 * t)
                    noise      = min(shock.secondary_mag * decay, 0.005)
                    noise_sign = 1.0 if (tick % 3 != 0) else -1.0
                    effect.price_shock_pct  += noise_sign * noise
                    effect.spread_multiplier = max(
                        effect.spread_multiplier,
                        max(1.0, shock.spread_multiplier * (1 - t * 0.7))
                    )
                    effect.vol_multiplier = max(
                        effect.vol_multiplier,
                        max(1.0, 3.5 * (1 - t * 0.8))
                    )
                    effect.volume_multiplier = max(
                        effect.volume_multiplier,
                        max(1.0, shock.volume_multiplier * (1 - t * 0.8))
                    )

            # ── CIRCUIT BREAKER ──────────────────────────────────────────
            elif shock.shock_type == ShockType.CIRCUIT_BREAKER:
                effect.halt_trading = True

        # Safety net: total shock move capped at 8% per tick
        effect.price_shock_pct = float(np.clip(effect.price_shock_pct, -0.08, 0.08))
        return effect

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_halted(self, tick: int) -> bool:
        return tick <= self._halted_until

    def active_shocks(self) -> list[ShockEvent]:
        return list(self._active_shocks)

    def shock_history(self) -> list[ShockEvent]:
        return list(self._shock_history)

    def reset(self) -> None:
        self._active_shocks.clear()
        self._scheduled_shocks.clear()
        self._shock_history.clear()
        self._halted_until     = -1
        self._post_shock_ticks = 0
        logger.debug("ShockEngine reset.")

    def summary(self) -> dict:
        return {
            "total_shocks":  len(self._shock_history),
            "active_shocks": len(self._active_shocks),
            "shock_types": {
                st.name: sum(1 for s in self._shock_history if s.shock_type == st)
                for st in ShockType
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_shock_engine(cfg: dict | None = None, seed: int | None = None) -> ShockEngine:
    rng = np.random.default_rng(seed)
    return ShockEngine(cfg=cfg or _load_cfg(), rng=rng)


if __name__ == "__main__":
    import sys
    shock_type = sys.argv[1] if len(sys.argv) > 1 else "flash_crash"
    print(f"\nTesting ShockEngine: {shock_type}\n" + "-" * 50)

    engine = ShockEngine()
    price  = 100.0
    prev   = price
    engine.inject_now(shock_type, tick=0)

    shock = engine.active_shocks()[0]
    print(f"Shock: {shock.label}  |  duration: {shock.duration_ticks} ticks\n")

    for tick in range(shock.duration_ticks + 10):
        effect = engine.step(tick=tick, current_price=price, prev_price=prev)
        if effect.halt_trading:
            print(f"  Tick {tick:2d} | HALTED")
            prev = price
            continue
        price   = price * (1 + effect.price_shock_pct)
        bar_len = int(abs(effect.price_shock_pct) * 800)
        bar     = ("▼" if effect.price_shock_pct < 0 else "▲") * min(bar_len, 30)
        print(
            f"  Tick {tick:2d} | Price: {price:7.3f} | "
            f"d%: {effect.price_shock_pct:+.5f} | "
            f"Vol x{effect.vol_multiplier:.1f} | "
            f"Sprd x{effect.spread_multiplier:.1f} | "
            f"Damp: {effect.post_shock_dampen} | "
            f"{bar}"
        )
        prev = price

    print(f"\nFinal price: {price:.3f}  ({(price/100-1)*100:+.1f}% from start)")