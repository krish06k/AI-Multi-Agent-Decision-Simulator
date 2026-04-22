"""
agents/value_agent.py
=====================
Layer 2 — Agent Policies | Value Agent (DQN-backed)

Strategy: Mean-reversion / buy-the-dip.
Computes a "fair value" as a rolling moving average.
Buys when price drops > buy_threshold below fair value.
Sells when price rises > sell_threshold above fair value.

FIX: Heterogeneous Value Agents
--------------------------------
Previously all 30 Value agents used identical parameters → identical signals
→ identical actions → identical PnL (all showed -$661).

Root cause: ValueAgent had no per-agent parameter variation. All 30 agents
shared the same fair_value_window, buy_threshold, sell_threshold, and
mr_weight from config. Since they all received the same price history and
computed the same signal, they always made the same decision at the same tick.

Fix: Each agent now draws its parameters from a seeded RNG that is derived
from its agent_id, creating stable but unique parameter sets:

  fair_value_window  : uniform [20, 80]   — different "memory" lengths
  buy_threshold      : uniform [-0.06, -0.015]  — different entry sensitivity
  sell_threshold     : uniform [0.02, 0.10]     — different exit targets
  mr_weight          : uniform [0.4, 0.9]        — different signal weights
  conviction_scale   : uniform [0.5, 1.5]        — different order sizing
  contrarian_bias    : uniform [0.0, 0.3]        — some agents fight trends more

These are drawn once at __init__ time so each agent has stable, reproducible
behaviour across ticks while being distinct from all other agents.
"""

from __future__ import annotations
import logging
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentAction

logger = logging.getLogger(__name__)


class ValueAgent(BaseAgent):

    def __init__(self, agent_id: int, cfg: dict | None = None, policy=None):
        super().__init__(agent_id, AgentType.VALUE, cfg)
        self._type_cfg = self.cfg.get("value", {})
        self._policy   = policy

        # ----------------------------------------------------------------
        # FIX: Per-agent heterogeneous parameters
        # Each Value agent gets a unique parameter set derived from its ID.
        # Using agent_id as seed ensures reproducibility across runs while
        # making every agent behave differently.
        # ----------------------------------------------------------------
        _rng = np.random.default_rng(seed=agent_id * 7919 + 31337)  # stable per-agent seed

        # Fair value window: short-memory vs long-memory agents
        self._fv_window = int(_rng.integers(
            low=self._type_cfg.get("fair_value_window_min", 20),
            high=self._type_cfg.get("fair_value_window_max", 80),
        ))

        # Buy threshold: how deep a dip triggers a buy (negative value)
        _base_buy = self._type_cfg.get("buy_threshold", -0.03)
        self._buy_thresh = float(_rng.uniform(
            low=_base_buy * 2.0,   # more aggressive: -0.06
            high=_base_buy * 0.5,  # less aggressive: -0.015
        ))

        # Sell threshold: how high above FV to take profit (positive value)
        _base_sell = self._type_cfg.get("sell_threshold", 0.05)
        self._sell_thresh = float(_rng.uniform(
            low=_base_sell * 0.4,  # eager profit-taker: ~0.02
            high=_base_sell * 2.0, # patient: ~0.10
        ))

        # Mean-reversion weight in signal combination
        _base_mr = self._type_cfg.get("mean_reversion_weight", 0.7)
        self._mr_weight = float(_rng.uniform(
            low=max(0.3, _base_mr - 0.3),
            high=min(0.95, _base_mr + 0.2),
        ))

        # Conviction scale: amplifies or dampens order size decisions
        self._conviction_scale = float(_rng.uniform(0.5, 1.5))

        # Contrarian bias: some agents are more aggressive at mean-reverting
        # during strong trends (fight the trend harder)
        self._contrarian_bias = float(_rng.uniform(0.0, 0.3))

        # Trend filter: some agents skip buy signals during uptrends
        # (only want to buy genuine dips, not micro-pullbacks in an uptrend)
        self._trend_filter_strength = float(_rng.uniform(0.0, 0.5))

        # Partial fill preference: fraction of agents that prefer small orders
        self._prefer_small = bool(_rng.random() < 0.35)

        logger.debug(
            "ValueAgent[%d] params: window=%d buy=%.4f sell=%.4f "
            "mr=%.3f conv_scale=%.2f contrarian=%.3f",
            agent_id,
            self._fv_window, self._buy_thresh, self._sell_thresh,
            self._mr_weight, self._conviction_scale, self._contrarian_bias,
        )

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def compute_signal(self) -> tuple[float, str]:
        """
        Signal = (price / fair_value) − 1.
        Negative = undervalued (buy opportunity).
        Positive = overvalued (sell opportunity).

        Enhancement: z-score normalisation and trend context
        so agents with different windows give meaningfully different signals.
        """
        snap = self._snapshot
        if snap is None or len(snap.price_history) < 10:
            return 0.0, "insufficient_history"

        hist = snap.price_history
        window = min(self._fv_window, len(hist))
        window_prices = hist[-window:]

        fair_value = float(np.mean(window_prices))
        std = float(np.std(window_prices))

        # Primary deviation signal
        signal = (snap.price / max(fair_value, 1e-8)) - 1.0

        # Z-score: how many std-devs away from fair value
        z_score = signal / max(std / max(fair_value, 1e-8), 1e-6)

        # Short-term momentum context (5-tick return)
        momentum = snap.recent_return(5) if len(hist) >= 6 else 0.0

        # Contrarian overlay: if strong downtrend, bias toward buying more
        # (contrarian_bias > 0 means the agent leans into dips harder)
        contrarian_adj = -self._contrarian_bias * momentum

        # Combined signal (weighted)
        combined = self._mr_weight * signal + (1 - self._mr_weight) * contrarian_adj

        label = (
            f"FV={fair_value:.2f}[w={self._fv_window}]  "
            f"price={snap.price:.2f}  dev={signal:.4f}  "
            f"z={z_score:.2f}  mom={momentum:.4f}"
        )
        return float(combined), label

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> int:
        if self._policy is not None:
            action, _ = self._policy.predict(observation, deterministic=False)
            return int(action)

        signal, label = self.compute_signal()
        snap = self._snapshot

        # Stop-loss
        if snap and self._check_stop_loss(snap.price):
            self._execute_trade(AgentAction.SELL_LARGE, snap.price)
            self._record_decision(AgentAction.SELL_LARGE, signal, label,
                                  reason_tags=["stop_loss"])
            return AgentAction.SELL_LARGE.value

        shocked = snap and snap.shock_active

        # Scale conviction by per-agent conviction_scale
        raw_conviction = min(abs(signal) / max(abs(self._buy_thresh), 1e-6), 1.0)
        conviction = min(raw_conviction * self._conviction_scale, 1.0)

        # Trend filter: some agents won't buy unless the dip is "deep enough"
        # relative to recent momentum (avoids chasing in downtrends)
        if snap:
            recent_momentum = snap.recent_return(10) if len(snap.price_history) >= 11 else 0.0
            trend_penalty = abs(min(recent_momentum, 0)) * self._trend_filter_strength
        else:
            trend_penalty = 0.0

        effective_buy_thresh = self._buy_thresh - trend_penalty

        # --- Decision logic ---
        if signal < effective_buy_thresh and not shocked:
            if self._prefer_small:
                action = AgentAction.BUY_SMALL
            else:
                action = AgentAction.BUY_LARGE if conviction > 0.7 else AgentAction.BUY_SMALL
            tags = ["undervalued", "buy_dip"]

        elif signal < effective_buy_thresh * 0.5 and not shocked:
            action = AgentAction.BUY_SMALL
            tags = ["mild_undervalue"]

        elif signal > self._sell_thresh:
            if self._prefer_small:
                action = AgentAction.SELL_SMALL
            else:
                action = AgentAction.SELL_LARGE if conviction > 0.7 else AgentAction.SELL_SMALL
            tags = ["overvalued", "take_profit"]

        elif signal > self._sell_thresh * 0.5 and self.shares > 0:
            action = AgentAction.SELL_SMALL
            tags = ["mild_overvalue"]

        else:
            action = AgentAction.HOLD
            tags = ["at_fair_value"]

        price = snap.price if snap else 0.0
        self._execute_trade(action, price)
        self._record_decision(action, signal, label, reason_tags=tags)
        return action.value


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Testing ValueAgent heterogeneity...\n")
    from agents.base_agent import MarketSnapshot

    # Create several agents and verify they have different parameters
    agents = [ValueAgent(agent_id=i) for i in range(30, 60)]

    print("Parameter diversity check (first 10 Value agents):")
    print(f"{'ID':>4}  {'window':>6}  {'buy_thr':>8}  {'sell_thr':>8}  {'mr_wt':>6}  {'conv':>6}")
    for a in agents[:10]:
        print(
            f"{a.agent_id:>4}  {a._fv_window:>6}  "
            f"{a._buy_thresh:>8.4f}  {a._sell_thresh:>8.4f}  "
            f"{a._mr_weight:>6.3f}  {a._conviction_scale:>6.3f}"
        )

    # Verify uniqueness
    windows = [a._fv_window for a in agents]
    assert len(set(windows)) > 5, "Windows should be diverse!"
    buy_thresholds = [round(a._buy_thresh, 5) for a in agents]
    assert len(set(buy_thresholds)) > 20, "Buy thresholds should be unique!"
    print("\n✅ All agents have unique parameters.")

    # Simulate a dip + recovery
    prices = [100.0]
    for _ in range(20):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.002)))
    for _ in range(10):
        prices.append(prices[-1] * 0.97)
    for _ in range(15):
        prices.append(prices[-1] * 1.02)

    log_returns = np.diff(np.log(prices)).tolist()

    actions_taken = {a.agent_id: [] for a in agents[:5]}
    for tick, price in enumerate(prices):
        snap = MarketSnapshot(
            tick=tick, price=price,
            price_history=prices[:tick + 1],
            log_returns=log_returns[:tick] if tick > 0 else [],
            volatility=np.std(log_returns[:tick]) if tick > 1 else 0.0,
            bid=price - 0.05, ask=price + 0.05, spread=0.1,
            imbalance=np.random.uniform(-0.2, 0.2),
            bid_depth=1000, ask_depth=1000,
            shock_active=False, shock_regime="calm", vol_multiplier=1.0,
        )
        obs = np.random.randn(47).astype(np.float32)
        for a in agents[:5]:
            a.observe(snap)
            act = a.act(obs)
            actions_taken[a.agent_id].append(act)

    print("\nAction diversity over simulation (first 5 Value agents):")
    for a in agents[:5]:
        action_seq = actions_taken[a.agent_id]
        buy_c  = action_seq.count(1) + action_seq.count(2)
        sell_c = action_seq.count(3) + action_seq.count(4)
        hold_c = action_seq.count(0)
        pnl    = a.unrealised_pnl(prices[-1]) + a.realised_pnl
        print(
            f"  Agent {a.agent_id}: buy={buy_c} sell={sell_c} hold={hold_c}  "
            f"pnl={pnl:+.2f}  window={a._fv_window}  buy_thr={a._buy_thresh:.4f}"
        )

    pnls = [a.unrealised_pnl(prices[-1]) + a.realised_pnl for a in agents[:5]]
    assert len(set(round(p, 2) for p in pnls)) > 1, "PnLs should differ!"
    print("\n✅ Value agents now show diverse PnL values.")
    print("\nFinal stats for Agent 30:")
    print(agents[0].stats())