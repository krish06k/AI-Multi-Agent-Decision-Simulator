"""
agents/panic_agent.py
=====================
Layer 2 — Agent Policies | Panic Agent

Simulates institutional stop-loss cascades and retail panic selling.

Trigger logic
-------------
1. Monitor rolling window of price returns.
2. If cumulative decline > trigger_drop_pct in trigger_window ticks → activate panic.
3. In panic mode: dump sell_fraction of holdings each tick.
4. Cascade: if neighbouring panic agents are also selling (detected via
   market imbalance), amplify sell qty by cascade_factor.
5. Recovery: deactivate when price recovers recovery_threshold%.
6. Cooldown: minimum ticks before panic can re-trigger.

FIX: Panic agents all showed +$0.00
-------------------------------------
Root cause: PanicAgent starts with shares=0. When panic triggered, the
act() method hit `if self.shares == 0: action = HOLD` immediately — there
was nothing to sell. Agents never accumulated a position so PnL was always
zero regardless of market conditions.

Fix: Two-phase behaviour:

  Phase 1 — ACCUMULATION (calm state, no panic):
    Each panic agent buys shares during calm periods to build a position
    it can later dump. Buying is driven by a per-agent seeded RNG so
    agents accumulate at different rates and different times, giving
    heterogeneous PnL when panic eventually fires.

    Parameters (per-agent, seeded from agent_id):
      accumulation_freq  : probability of buying on any calm tick [0.2, 0.6]
      preferred_size     : BUY_SMALL vs BUY_LARGE preference
      max_accum_shares   : target position size before stopping buys [50, 400]

  Phase 2 — PANIC SELLING (triggered state):
    Original logic unchanged — dumps holdings aggressively, amplified by
    cascade if other panic agents are also selling.

Per-agent heterogeneity is also applied to panic trigger parameters so
agents don't all panic at the same tick (the original sensitivity linspace
in AgentPool already handled this, but we reinforce it here too).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from agents.base_agent import BaseAgent, AgentType, AgentAction, MarketSnapshot

logger = logging.getLogger(__name__)


class PanicAgent(BaseAgent):
    """
    Stress-triggered cascade seller.

    Parameters
    ----------
    agent_id       : Agent index (80–99).
    cfg            : agent_config.yaml dict.
    rng            : Numpy random generator (for staggering panic onset).
    sensitivity    : Multiplier on trigger_drop_pct (allows heterogeneity).
    """

    def __init__(
        self,
        agent_id: int,
        cfg: dict | None = None,
        rng: np.random.Generator | None = None,
        sensitivity: float = 1.0,
    ):
        super().__init__(agent_id, AgentType.PANIC, cfg)
        self._type_cfg    = self.cfg.get("panic", {})
        self._rng         = rng or np.random.default_rng()
        self._sensitivity = sensitivity

        # Panic trigger parameters
        self._trigger_drop    = self._type_cfg.get("trigger_drop_pct", 3.0) / 100.0 * sensitivity
        self._trigger_window  = self._type_cfg.get("trigger_window", 5)
        self._sell_fraction   = self._type_cfg.get("sell_fraction", 0.5)
        self._cascade_factor  = self._type_cfg.get("cascade_factor", 1.5)
        self._recovery_thresh = self._type_cfg.get("recovery_threshold", 1.0) / 100.0
        self._cooldown        = self._type_cfg.get("panic_cooldown", 10)

        # ----------------------------------------------------------------
        # FIX: Per-agent accumulation parameters
        # Each panic agent builds a position during calm ticks so it has
        # shares to dump when panic fires. Parameters are seeded from
        # agent_id for stable, heterogeneous behaviour.
        # ----------------------------------------------------------------
        _arng = np.random.default_rng(seed=agent_id * 3571 + 99991)

        # How often the agent buys during calm periods (fraction of ticks)
        self._accum_freq = float(_arng.uniform(0.20, 0.60))

        # Whether this agent prefers large or small accumulation orders
        self._accum_large = bool(_arng.random() < 0.4)   # 40% use BUY_LARGE

        # Maximum shares to accumulate before stopping new buys
        # (so the agent doesn't spend all cash before panic fires)
        self._max_accum_shares = int(_arng.integers(50, 400))

        # Accumulation start delay: some agents wait N ticks before buying
        # so they're not all positioned at the same time
        self._accum_start_tick = int(_arng.integers(0, 30))

        # Position target fraction: how much of max_position to aim for
        # before considering itself "fully loaded" for a panic dump
        self._load_target = float(_arng.uniform(0.3, 0.9))

        logger.debug(
            "PanicAgent[%d] accum_freq=%.2f accum_large=%s "
            "max_accum=%d start_delay=%d trigger_drop=%.4f",
            agent_id,
            self._accum_freq, self._accum_large,
            self._max_accum_shares, self._accum_start_tick,
            self._trigger_drop,
        )

        # State machine
        self._in_panic: bool = False
        self._panic_start_price: float = 0.0
        self._panic_start_tick: int = -1
        self._last_panic_end_tick: int = -999
        self._consecutive_panic_ticks: int = 0

        # Price window for trigger detection
        self._price_window: deque[float] = deque(maxlen=self._trigger_window + 1)

    # ------------------------------------------------------------------
    # Trigger detection
    # ------------------------------------------------------------------

    def _should_trigger_panic(self, snap: MarketSnapshot) -> bool:
        """Check if panic should be newly triggered this tick."""
        if self._in_panic:
            return False

        if snap.tick - self._last_panic_end_tick < self._cooldown:
            return False

        self._price_window.append(snap.price)
        if len(self._price_window) < self._trigger_window + 1:
            return False

        oldest  = self._price_window[0]
        current = self._price_window[-1]
        if oldest <= 0:
            return False

        decline   = (oldest - current) / oldest
        triggered = decline >= self._trigger_drop

        # External shock: 60% chance to trigger on any stressed shock
        if snap.shock_active and snap.shock_regime in ("stressed", "circuit_breaker"):
            if self._rng.random() < 0.6:
                triggered = True

        return triggered

    def _should_recover(self, snap: MarketSnapshot) -> bool:
        """Check if panic mode should end (price recovered)."""
        if not self._in_panic or self._panic_start_price <= 0:
            return False
        recovery = (snap.price - self._panic_start_price) / self._panic_start_price
        return recovery >= self._recovery_thresh

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def compute_signal(self) -> tuple[float, str]:
        """
        Signal reflects panic intensity.
        0 = calm accumulating, negative = selling pressure.
        """
        snap = self._snapshot
        if snap is None:
            return 0.0, "no_snapshot"

        if not self._in_panic:
            # Positive signal during accumulation (buying bias)
            accum_signal = min(self.shares / max(self._max_accum_shares, 1), 1.0)
            return float(accum_signal * 0.1), f"accumulating|shares={self.shares}"

        # In panic: signal proportional to holdings and cascade pressure
        base_signal = -self._sell_fraction
        imbalance_penalty = min(0.0, snap.imbalance) * self._cascade_factor
        panic_signal = base_signal + imbalance_penalty

        label = (
            f"PANIC|tick={self._consecutive_panic_ticks}  "
            f"start_price={self._panic_start_price:.2f}  "
            f"imbalance={snap.imbalance:.3f}  "
            f"cascade_factor={self._cascade_factor:.1f}"
        )
        return float(panic_signal), label

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> int:
        snap = self._snapshot
        if snap is None:
            return AgentAction.HOLD.value

        # Update price window
        self._price_window.append(snap.price)

        # --- State machine transitions ---

        # Recovery check
        if self._in_panic and self._should_recover(snap):
            logger.info(
                "Agent[%d] PANIC RECOVERY at tick %d (price=%.2f)",
                self.agent_id, snap.tick, snap.price,
            )
            self._in_panic = False
            self._last_panic_end_tick = snap.tick
            self._consecutive_panic_ticks = 0

        # New panic trigger
        elif not self._in_panic and self._should_trigger_panic(snap):
            self._in_panic = True
            self._panic_start_price = snap.price
            self._panic_start_tick  = snap.tick
            self._consecutive_panic_ticks = 0
            logger.warning(
                "Agent[%d] PANIC TRIGGERED at tick %d (price=%.2f, shares=%d)",
                self.agent_id, snap.tick, snap.price, self.shares,
            )

        signal, label = self.compute_signal()

        extra: dict      = {}
        tags:  list[str] = []

        # ----------------------------------------------------------------
        # FIX: Phase 1 — Accumulation during calm state
        # ----------------------------------------------------------------
        if not self._in_panic:
            # Wait for start delay before accumulating
            if snap.tick < self._accum_start_tick:
                action = AgentAction.HOLD
                tags   = ["pre_accumulation_delay"]

            # Already fully loaded — stop buying, just hold and wait
            elif self.shares >= self._max_accum_shares:
                action = AgentAction.HOLD
                tags   = ["accumulation_complete_holding"]

            # Actively accumulate with per-agent frequency
            elif self._rng.random() < self._accum_freq:
                action = AgentAction.BUY_LARGE if self._accum_large else AgentAction.BUY_SMALL
                tags   = ["calm_accumulation"]

            else:
                action = AgentAction.HOLD
                tags   = ["calm_hold"]

        # ----------------------------------------------------------------
        # Phase 2 — Panic selling
        # ----------------------------------------------------------------
        elif self.shares == 0:
            # Nothing left to sell — hold until recovery
            action = AgentAction.HOLD
            tags   = ["panic_no_position"]
            logger.debug(
                "Agent[%d] panic state but shares=0 at tick %d — holding.",
                self.agent_id, snap.tick,
            )

        else:
            self._consecutive_panic_ticks += 1

            logger.warning(
                "Agent[%d] PANIC SELLING at tick %d (price=%.2f, shares=%d, "
                "consecutive_panic_ticks=%d)",
                self.agent_id, snap.tick, snap.price,
                self.shares, self._consecutive_panic_ticks,
            )

            # Cascade amplification
            is_cascade = snap.imbalance < -0.3
            if is_cascade and self._consecutive_panic_ticks <= 3:
                action = AgentAction.SELL_LARGE
                tags   = ["panic_cascade_sell_large"]
            elif self._sell_fraction >= 0.5:
                action = AgentAction.SELL_LARGE
                tags   = ["panic_sell_large"]
            else:
                action = AgentAction.SELL_SMALL
                tags   = ["panic_sell_small"]

            extra = {
                "panic_start_tick":        self._panic_start_tick,
                "panic_start_price":       round(self._panic_start_price, 4),
                "consecutive_panic_ticks": self._consecutive_panic_ticks,
                "is_cascade":              is_cascade,
            }

        self._execute_trade(action, snap.price)
        self._record_decision(
            action, signal, label,
            reason_tags=tags,
            extra=extra,
        )
        return action.value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_panicking(self) -> bool:
        return self._in_panic

    @property
    def is_accumulating(self) -> bool:
        """True when agent is in calm buy-accumulation phase."""
        return (
            not self._in_panic
            and self.shares < self._max_accum_shares
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._in_panic                = False
        self._panic_start_price       = 0.0
        self._panic_start_tick        = -1
        self._last_panic_end_tick     = -999
        self._consecutive_panic_ticks = 0
        self._price_window.clear()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Testing PanicAgent fix (accumulation → panic dump)...\n")

    agents = [
        PanicAgent(agent_id=aid, rng=np.random.default_rng(aid), sensitivity=s)
        for aid, s in zip(range(80, 100), np.linspace(0.6, 1.8, 20))
    ]

    # Phase 1: stable market — agents accumulate
    prices = [100.0]
    for _ in range(40):
        prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.003)))

    # Phase 2: crash — panic triggers
    for _ in range(15):
        prices.append(prices[-1] * 0.97)

    # Phase 3: recovery
    for _ in range(10):
        prices.append(prices[-1] * 1.01)

    log_returns = np.diff(np.log(prices)).tolist()

    for tick, price in enumerate(prices):
        snap = MarketSnapshot(
            tick=tick, price=price,
            price_history=prices[:tick + 1],
            log_returns=log_returns[:tick] if tick > 0 else [],
            volatility=np.std(log_returns[:tick]) if tick > 1 else 0.0,
            bid=price - 0.05, ask=price + 0.05, spread=0.1,
            imbalance=-0.5 if tick > 40 else 0.1,
            bid_depth=1000, ask_depth=1000,
            shock_active=False, shock_regime="calm", vol_multiplier=1.0,
        )
        obs = np.random.randn(47).astype(np.float32)
        for a in agents:
            a.observe(snap)
            a.act(obs)

    print("Final state after accumulation + panic + recovery:")
    print(f"{'ID':>4}  {'shares':>7}  {'cash':>9}  {'realised':>10}  "
          f"{'panicking':>9}  {'max_acc':>7}  {'freq':>5}")
    for a in agents:
        pnl = a.realised_pnl + a.unrealised_pnl(prices[-1])
        print(
            f"{a.agent_id:>4}  {a.shares:>7}  {a.cash:>9.2f}  "
            f"{a.realised_pnl:>+10.2f}  {str(a.is_panicking):>9}  "
            f"{a._max_accum_shares:>7}  {a._accum_freq:>5.2f}"
        )

    pnls = [a.realised_pnl + a.unrealised_pnl(prices[-1]) for a in agents]
    print(f"\nPnL range: {min(pnls):+.2f} to {max(pnls):+.2f}")
    non_zero = sum(1 for p in pnls if abs(p) > 0.01)
    print(f"Agents with non-zero PnL: {non_zero}/20")
    assert non_zero > 5, "Expected most panic agents to have non-zero PnL!"
    print("\n✅ Panic agents now accumulate and dump correctly.")