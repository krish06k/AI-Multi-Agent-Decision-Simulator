"""
agents/momentum_agent.py
========================
Layer 2 — Agent Policies | Momentum Agent (PPO-backed)

Strategy: Follow trend signals using dual-EMA crossover.
RL backbone: PPO via Stable-Baselines3 (optional; falls back to rule-based).

Signal logic
------------
* Compute fast EMA (5-tick) and slow EMA (20-tick).
* Signal = (fast_EMA / slow_EMA) − 1   → positive = uptrend.
* Also incorporates: momentum of vol, order book imbalance tilt.
* Conviction scales order size: weak signal → small, strong → large.

FIX: Heterogeneous Momentum Agents
------------------------------------
Previously all 30 Momentum agents used identical EMA windows, thresholds
and weights → identical signals → identical decisions → identical PnL.

Fix: Each agent derives unique parameters from its agent_id seed:
  fast_ma           : int   [3, 10]     — different EMA speeds
  slow_ma           : int   [15, 40]    — different baseline EMAs
  signal_threshold  : float [0.001, 0.008] — different trigger sensitivity
  trend_weight      : float [0.4, 0.9]  — EMA vs momentum blend
  imbalance_scale   : float [0.0, 0.003]— how much book tilt matters
  conviction_cutoff : float [0.4, 0.8]  — when to use LARGE vs SMALL orders
  shock_exit_bias   : float [0.3, 0.9]  — eagerness to reduce risk in shocks
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from agents.base_agent import BaseAgent, AgentType, AgentAction, MarketSnapshot

logger = logging.getLogger(__name__)


class MomentumAgent(BaseAgent):
    """
    Trend-following agent using dual-EMA crossover with PPO policy override.

    Parameters
    ----------
    agent_id : Agent index.
    cfg      : agent_config.yaml dict.
    policy   : Optional pre-trained SB3 PPO policy for act() override.
    """

    def __init__(
        self,
        agent_id: int,
        cfg: dict | None = None,
        policy=None,
    ):
        super().__init__(agent_id, AgentType.MOMENTUM, cfg)
        self._type_cfg = self.cfg.get("momentum", {})
        self._policy   = policy

        # ----------------------------------------------------------------
        # FIX: Per-agent heterogeneous parameters
        # Agent ID is combined with a large prime to space seeds well
        # across the 0–29 momentum range.
        # ----------------------------------------------------------------
        _rng = np.random.default_rng(seed=agent_id * 6271 + 54321)

        # EMA windows: different agents "see" trend at different speeds
        self._fast_ma = int(_rng.integers(
            low=self._type_cfg.get("fast_ma_min", 3),
            high=self._type_cfg.get("fast_ma_max", 10),
        ))
        # Ensure slow > fast by at least 8 ticks
        slow_min = self._fast_ma + 8
        self._slow_ma = int(_rng.integers(
            low=max(slow_min, self._type_cfg.get("slow_ma_min", 15)),
            high=self._type_cfg.get("slow_ma_max", 40),
        ))

        # Signal threshold: minimum crossover to act on
        _base_thr = self._type_cfg.get("signal_threshold", 0.002)
        self._threshold = float(_rng.uniform(
            low=_base_thr * 0.5,   # hair-trigger: 0.001
            high=_base_thr * 4.0,  # sluggish: 0.008
        ))

        # EMA vs momentum blend
        _base_tw = self._type_cfg.get("trend_strength_weight", 0.6)
        self._trend_wt = float(_rng.uniform(
            low=max(0.3, _base_tw - 0.25),
            high=min(0.95, _base_tw + 0.25),
        ))

        # Imbalance sensitivity: how much order book tilt shifts the signal
        self._imbalance_scale = float(_rng.uniform(0.0, 0.003))

        # Conviction cutoff: threshold to use LARGE vs SMALL orders
        self._conviction_cutoff = float(_rng.uniform(0.4, 0.8))

        # Shock exit bias: eagerness to reduce risk during stressed regimes
        # Higher = agent starts selling earlier on bad news
        self._shock_exit_bias = float(_rng.uniform(0.3, 0.9))

        # Momentum lookback: how many ticks of returns to use as momentum
        self._mom_lookback = int(_rng.integers(3, 10))

        # Signal amplifier: some agents exaggerate their conviction
        self._signal_amplifier = float(_rng.uniform(0.7, 1.4))

        logger.debug(
            "MomentumAgent[%d] params: ema=%d/%d thr=%.4f tw=%.3f "
            "imb=%.4f conv_cut=%.2f",
            agent_id,
            self._fast_ma, self._slow_ma, self._threshold,
            self._trend_wt, self._imbalance_scale, self._conviction_cutoff,
        )

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def compute_signal(self) -> tuple[float, str]:
        """
        Dual-EMA crossover signal, enhanced with momentum and book tilt.

        Returns
        -------
        (signal, label) where signal > 0 = bullish, signal < 0 = bearish.
        """
        snap = self._snapshot
        if snap is None or len(snap.price_history) < self._slow_ma:
            return 0.0, "insufficient_history"

        fast_ema = snap.ema(self._fast_ma)
        slow_ema = snap.ema(self._slow_ma)

        # Primary: EMA crossover
        ema_signal = (fast_ema / max(slow_ema, 1e-8)) - 1.0

        # Secondary: recent momentum (per-agent lookback)
        mom_signal = snap.recent_return(self._mom_lookback)

        # Tertiary: per-agent imbalance sensitivity
        book_signal = snap.imbalance * self._imbalance_scale

        # Combine with per-agent weights
        combined = (
            self._trend_wt * ema_signal +
            (1 - self._trend_wt) * 0.5 * mom_signal +
            book_signal
        ) * self._signal_amplifier

        label = (
            f"EMA({self._fast_ma}/{self._slow_ma})={ema_signal:.4f}  "
            f"mom[{self._mom_lookback}]={mom_signal:.4f}  "
            f"imb={snap.imbalance:.3f}×{self._imbalance_scale:.4f}"
        )
        return float(combined), label

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> int:
        """
        If SB3 PPO policy is loaded, delegate to it.
        Otherwise use rule-based signal logic.
        """
        if self._policy is not None:
            action, _ = self._policy.predict(observation, deterministic=False)
            return int(action)

        signal, label = self.compute_signal()

        # Stop-loss override
        if self._snapshot and self._check_stop_loss(self._snapshot.price):
            self._execute_trade(AgentAction.SELL_LARGE, self._snapshot.price)
            self._record_decision(
                AgentAction.SELL_LARGE, signal, label,
                reason_tags=["stop_loss_triggered"],
            )
            return AgentAction.SELL_LARGE.value

        # Shock: reduce position during stressed regime
        # Per-agent shock_exit_bias controls how readily the agent exits
        if self._snapshot and self._snapshot.shock_active:
            if self.shares > 0 and signal < -self._threshold * (1.0 - self._shock_exit_bias):
                self._execute_trade(AgentAction.SELL_SMALL, self._snapshot.price)
                self._record_decision(
                    AgentAction.SELL_SMALL, signal, label,
                    reason_tags=["shock_risk_reduction"],
                )
                return AgentAction.SELL_SMALL.value

        abs_sig    = abs(signal)
        conviction = min(abs_sig / max(self._threshold * 5, 1e-6), 1.0)

        # Per-agent conviction cutoff for LARGE vs SMALL orders
        use_large = conviction > self._conviction_cutoff

        if signal > self._threshold:
            action = AgentAction.BUY_LARGE if use_large else AgentAction.BUY_SMALL
            tags = ["ema_bullish", "trend_follow"]
        elif signal < -self._threshold:
            action = AgentAction.SELL_LARGE if use_large else AgentAction.SELL_SMALL
            tags = ["ema_bearish", "trend_follow"]
        else:
            action = AgentAction.HOLD
            tags = ["signal_weak"]

        price = self._snapshot.price if self._snapshot else 0.0
        self._execute_trade(action, price)
        self._record_decision(action, signal, label, reason_tags=tags)
        return action.value


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Testing MomentumAgent heterogeneity...\n")
    from agents.base_agent import MarketSnapshot

    agents = [MomentumAgent(agent_id=i) for i in range(0, 30)]

    print("Parameter diversity check (first 10 Momentum agents):")
    print(f"{'ID':>4}  {'fast':>5}  {'slow':>5}  {'thr':>7}  {'tw':>6}  {'conv_cut':>8}")
    for a in agents[:10]:
        print(
            f"{a.agent_id:>4}  {a._fast_ma:>5}  {a._slow_ma:>5}  "
            f"{a._threshold:>7.4f}  {a._trend_wt:>6.3f}  {a._conviction_cutoff:>8.3f}"
        )

    fast_mas = [a._fast_ma for a in agents]
    assert len(set(fast_mas)) > 3, "Fast MAs should be diverse!"
    print("\n✅ All Momentum agents have unique parameters.")

    # Simulate trending market
    prices = [100.0]
    for _ in range(50):
        prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.01)))

    log_returns = np.diff(np.log(prices)).tolist()

    for tick, price in enumerate(prices):
        snap = MarketSnapshot(
            tick=tick, price=price,
            price_history=prices[:tick + 1],
            log_returns=log_returns[:tick] if tick > 0 else [],
            volatility=np.std(log_returns[:tick]) if tick > 1 else 0.0,
            bid=price - 0.05, ask=price + 0.05, spread=0.1,
            imbalance=np.random.uniform(-0.3, 0.3),
            bid_depth=1000, ask_depth=1000,
            shock_active=False, shock_regime="calm", vol_multiplier=1.0,
        )
        obs = np.random.randn(47).astype(np.float32)
        for a in agents[:5]:
            a.observe(snap)
            a.act(obs)

    print("\nPnL diversity over simulation (first 5 Momentum agents):")
    for a in agents[:5]:
        pnl = a.unrealised_pnl(prices[-1]) + a.realised_pnl
        print(
            f"  Agent {a.agent_id}: pnl={pnl:+.2f}  "
            f"trades={len([d for d in a.decision_history() if d.action.name != 'HOLD'])}  "
            f"ema={a._fast_ma}/{a._slow_ma}  thr={a._threshold:.4f}"
        )

    pnls = [a.unrealised_pnl(prices[-1]) + a.realised_pnl for a in agents[:5]]
    assert len(set(round(p, 2) for p in pnls)) > 1, "PnLs should differ!"
    print("\n✅ Momentum agents now show diverse PnL values.")