"""
agents/base_agent.py
====================
Layer 2 — Agent Policies | Abstract Base Agent

Defines the contract that every agent in the simulation must implement.
Provides shared infrastructure: portfolio tracking, decision logging,
stop-loss logic, and introspection hooks used by the GenAI explainability layer.

All concrete agents (Momentum, Value, Noise, Panic) inherit from BaseAgent
and override: observe(), compute_signal(), act(), update().

FIX SUMMARY (clustering fix)
------------------------------
All agents were showing the same PnL value (e.g. all Momentum = +$308,
all Value = +$243) because:

1. qty_small=10 / qty_large=50 were shared globals from config.
   Two agents both deciding BUY_SMALL buy exactly 10 shares at the same
   price → identical cost basis → identical unrealised PnL = shares × (price − avg_cost).
   Fix: each agent draws unique qty_small ∈ [4,25] and qty_large ∈ [20,120]
   from a seed of (agent_id × 2053 + 7919).

2. All agents executed at exactly the same mid-price.
   Even with different quantities, agents buying at the same price produce
   proportionally identical PnL unless price moves asymmetrically.
   Fix: per-agent slippage (0–15 bps seeded per agent) applied in
   _execution_price() so each agent's effective cost basis diverges.

3. All portfolios started at exactly $10,000.00.
   Tiny early differences were masked. A ±5% cash jitter ensures values
   are visually distinct from the start and compound differently.

Note: runner.py also needs the obs-noise fix (see runner.py FIX comment).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_AGENT_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "agent_config.yaml"


def _load_cfg() -> dict:
    with open(_AGENT_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentType(Enum):
    MOMENTUM = "momentum"
    VALUE    = "value"
    NOISE    = "noise"
    PANIC    = "panic"


class AgentAction(Enum):
    HOLD       = 0
    BUY_SMALL  = 1
    BUY_LARGE  = 2
    SELL_SMALL = 3
    SELL_LARGE = 4


# ---------------------------------------------------------------------------
# Decision record — used by GenAI explainability layer
# ---------------------------------------------------------------------------

@dataclass
class AgentDecision:
    """
    Full audit record of one agent decision, passed to genai/explainer.py.
    """
    agent_id: int
    agent_type: AgentType
    tick: int
    action: AgentAction
    action_int: int
    signal_value: float
    signal_label: str
    price: float
    portfolio_value: float
    cash: float
    shares: int
    unrealised_pnl: float
    drawdown: float
    reason_tags: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_id":        self.agent_id,
            "agent_type":      self.agent_type.value,
            "tick":            self.tick,
            "action":          self.action.name,
            "signal_value":    round(self.signal_value, 6),
            "signal_label":    self.signal_label,
            "price":           round(self.price, 4),
            "portfolio_value": round(self.portfolio_value, 2),
            "cash":            round(self.cash, 2),
            "shares":          self.shares,
            "unrealised_pnl":  round(self.unrealised_pnl, 2),
            "drawdown":        round(self.drawdown, 4),
            "reason_tags":     self.reason_tags,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Market observation snapshot — standardised across all agent types
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """
    Distilled market view given to each agent's observe() method.
    """
    tick: int
    price: float
    price_history: list[float]
    log_returns: list[float]
    volatility: float
    bid: Optional[float]
    ask: Optional[float]
    spread: float
    imbalance: float           # [−1, 1]
    bid_depth: float
    ask_depth: float
    shock_active: bool
    shock_regime: str
    vol_multiplier: float

    @property
    def mid(self) -> float:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.price

    def recent_return(self, n: int = 1) -> float:
        """Log return over last n ticks."""
        if len(self.price_history) < n + 1:
            return 0.0
        p_now  = self.price_history[-1]
        p_then = self.price_history[-(n + 1)]
        if p_then <= 0:
            return 0.0
        return float(np.log(p_now / p_then))

    def ema(self, window: int) -> float:
        """Exponential moving average of price."""
        if len(self.price_history) < 2:
            return self.price
        prices = np.array(self.price_history[-window:], dtype=float)
        alpha  = 2.0 / (len(prices) + 1)
        ema    = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return float(ema)


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base class for all simulation agents.

    Lifecycle per tick
    ------------------
    1. observe(snapshot)        → store market view
    2. compute_signal()         → derive raw trading signal (agent-specific)
    3. act(observation) → int   → return discrete action for env.step()
    4. update(reward, done)     → optional RL update
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        cfg: dict | None = None,
    ):
        self.agent_id   = agent_id
        self.agent_type = agent_type
        self.cfg        = cfg or _load_cfg()
        self._base_cfg  = self.cfg.get("base", {})
        self._type_cfg  = self.cfg.get(agent_type.value, {})

        # ----------------------------------------------------------------
        # FIX 1: Per-agent order sizing
        # All agents previously used qty_small=10, qty_large=50 from config.
        # Two agents both deciding BUY_SMALL → same 10 shares → same cost
        # basis → identical unrealised PnL = shares × (price − avg_cost).
        #
        # Each agent now draws unique sizes from a stable per-agent seed so
        # even agents making the same action type trade different quantities.
        # ----------------------------------------------------------------
        _qty_rng = np.random.default_rng(seed=agent_id * 2053 + 7919)

        base_small = self._base_cfg.get("qty_small", 10)
        base_large = self._base_cfg.get("qty_large", 50)

        # qty_small: range [40% of base, 250% of base], floor 2
        self._qty_small: int = int(_qty_rng.integers(
            low=max(2, int(base_small * 0.4)),
            high=max(3, int(base_small * 2.5)),
        ))
        # qty_large: range [40% of base, 250% of base], floor 10
        self._qty_large: int = int(_qty_rng.integers(
            low=max(10, int(base_large * 0.4)),
            high=max(15, int(base_large * 2.5)),
        ))

        # ----------------------------------------------------------------
        # FIX 2: Per-agent initial cash variance (±5%)
        # All portfolios starting at exactly $10,000.00 means tiny early
        # differences get masked by display rounding. Small jitter ensures
        # portfolio values are visually distinct from tick 0.
        # ----------------------------------------------------------------
        base_cash   = self._base_cfg.get("initial_cash", 10000.0)
        cash_jitter = float(_qty_rng.uniform(-0.05, 0.05))
        self.cash: float = base_cash * (1.0 + cash_jitter)

        # ----------------------------------------------------------------
        # FIX 3: Per-agent execution slippage (0–15 bps)
        # Agents executing at the same tick pay slightly different prices
        # → diverging cost bases → diverging unrealised PnL even when
        # making the same action type.
        # ----------------------------------------------------------------
        self._slippage_bps: float = float(_qty_rng.uniform(0.0, 15.0))

        # Portfolio state
        self.shares: int             = 0
        self.avg_cost: float         = 0.0
        self.peak_value: float       = self.cash
        self.total_pnl: float        = 0.0
        self.realised_pnl: float     = 0.0
        self.total_commission: float = 0.0

        # Config limits
        self.max_position: int     = self._base_cfg.get("max_position", 1000)
        self.max_order_size: int   = self._base_cfg.get("max_order_size", 200)
        self.allow_short: bool     = self._base_cfg.get("allow_short", False)
        self.commission_bps: float = self._base_cfg.get("commission_bps", 5)
        self.stop_loss_pct: float  = self._base_cfg.get("stop_loss_pct", 0.10)

        # State
        self._snapshot: Optional[MarketSnapshot] = None
        self._last_action: AgentAction = AgentAction.HOLD
        self._decision_history: deque[AgentDecision] = deque(maxlen=500)
        self._pnl_history: deque[float] = deque(maxlen=100)
        self._tick: int = 0
        self._is_stopped_out: bool = False

        logger.debug(
            "Agent[%d] %s | qty_small=%d qty_large=%d cash=%.2f slippage=%.1fbps",
            agent_id, agent_type.value,
            self._qty_small, self._qty_large,
            self.cash, self._slippage_bps,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_signal(self) -> tuple[float, str]: ...

    @abstractmethod
    def act(self, observation: np.ndarray) -> int: ...

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def observe(self, snapshot: MarketSnapshot) -> None:
        """Store the current market snapshot. Called before act() each tick."""
        self._snapshot = snapshot
        self._tick     = snapshot.tick

    # ------------------------------------------------------------------
    # RL update hook
    # ------------------------------------------------------------------

    def update(self, reward: float, done: bool, info: dict) -> None:
        self._pnl_history.append(reward)

    # ------------------------------------------------------------------
    # Stop-loss
    # ------------------------------------------------------------------

    def _check_stop_loss(self, price: float) -> bool:
        if self.shares <= 0 or self.avg_cost <= 0:
            return False
        loss_pct = (self.avg_cost - price) / self.avg_cost
        return loss_pct >= self.stop_loss_pct

    # ------------------------------------------------------------------
    # Position sizing helper
    # ------------------------------------------------------------------

    def _size_order(self, conviction: float, base_qty: Optional[int] = None) -> int:
        base = base_qty or self.max_order_size
        qty  = int(base * max(0.0, min(1.0, conviction)))
        return max(1, qty)

    # ------------------------------------------------------------------
    # FIX 3: Per-agent execution price with slippage
    # ------------------------------------------------------------------

    def _execution_price(self, mid_price: float, is_buy: bool) -> float:
        """
        Apply per-agent slippage to the mid price.
        Buys pay slightly more, sells receive slightly less.
        Slippage magnitude is unique per agent (drawn at __init__).
        """
        slip = self._slippage_bps / 10_000.0
        return mid_price * (1.0 + slip) if is_buy else mid_price * (1.0 - slip)

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute_trade(self, action: AgentAction, price: float) -> bool:
        """
        Update agent's own cash / shares / avg_cost / realised_pnl.
        Uses per-agent qty sizes and per-agent execution price (slippage).
        """
        if action == AgentAction.HOLD:
            return False

        commission_rate = self.commission_bps / 10_000.0

        if action in (AgentAction.BUY_SMALL, AgentAction.BUY_LARGE):
            qty        = self._qty_small if action == AgentAction.BUY_SMALL else self._qty_large
            exec_price = self._execution_price(price, is_buy=True)

            qty = min(qty, max(0, self.max_position - self.shares))
            if qty <= 0:
                return False

            cost       = exec_price * qty
            commission = cost * commission_rate
            total_cost = cost + commission

            if total_cost > self.cash:
                affordable = int(self.cash / (exec_price * (1 + commission_rate)))
                qty = min(qty, affordable)
                if qty <= 0:
                    return False
                cost       = exec_price * qty
                commission = cost * commission_rate
                total_cost = cost + commission

            new_shares    = self.shares + qty
            self.avg_cost = (self.avg_cost * self.shares + exec_price * qty) / new_shares
            self.cash          -= total_cost
            self.shares         = new_shares
            self.total_commission += commission
            return True

        if action in (AgentAction.SELL_SMALL, AgentAction.SELL_LARGE):
            qty        = self._qty_small if action == AgentAction.SELL_SMALL else self._qty_large
            exec_price = self._execution_price(price, is_buy=False)

            qty = min(qty, self.shares)
            if qty <= 0:
                return False

            proceeds   = exec_price * qty
            commission = proceeds * commission_rate
            net        = proceeds - commission

            if self.avg_cost > 0:
                self.realised_pnl += (exec_price - self.avg_cost) * qty - commission

            self.cash  += net
            self.shares -= qty
            self.total_commission += commission

            if self.shares == 0:
                self.avg_cost = 0.0

            return True

        return False

    # ------------------------------------------------------------------
    # Decision recording
    # ------------------------------------------------------------------

    def _record_decision(
        self,
        action: AgentAction,
        signal_value: float,
        signal_label: str,
        reason_tags: list[str] | None = None,
        extra: dict | None = None,
    ) -> AgentDecision:
        """
        Create and store an AgentDecision for the explainability layer.
        Call _execute_trade() BEFORE this so portfolio state is current.
        """
        snap  = self._snapshot
        price = snap.price if snap else 0.0
        portfolio_value = self.cash + self.shares * price
        unrealised = (
            self.shares * (price - self.avg_cost)
            if self.avg_cost > 0 and self.shares > 0
            else 0.0
        )
        drawdown = (self.peak_value - portfolio_value) / max(self.peak_value, 1.0)
        self.peak_value = max(self.peak_value, portfolio_value)

        decision = AgentDecision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            tick=self._tick,
            action=action,
            action_int=action.value,
            signal_value=signal_value,
            signal_label=signal_label,
            price=price,
            portfolio_value=portfolio_value,
            cash=self.cash,
            shares=self.shares,
            unrealised_pnl=unrealised,
            drawdown=drawdown,
            reason_tags=reason_tags or [],
            extra=extra or {},
        )
        self._decision_history.append(decision)
        self._last_action = action
        return decision

    # ------------------------------------------------------------------
    # Portfolio sync from env info
    # ------------------------------------------------------------------

    def sync_portfolio(self, info: dict) -> None:
        """
        No longer overwrites agent portfolio from env info (env only tracks
        agent 0). Only used to keep peak_value current for drawdown calcs.
        """
        price = info.get("price", 0.0)
        if price > 0:
            pv = self.cash + self.shares * price
            self.peak_value = max(self.peak_value, pv)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def last_decision(self) -> Optional[AgentDecision]:
        return self._decision_history[-1] if self._decision_history else None

    def decision_history(self) -> list[AgentDecision]:
        return list(self._decision_history)

    def portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def unrealised_pnl(self, price: float) -> float:
        if self.shares == 0 or self.avg_cost == 0:
            return 0.0
        return self.shares * (price - self.avg_cost)

    def sharpe_estimate(self) -> float:
        if len(self._pnl_history) < 5:
            return 0.0
        arr = np.array(self._pnl_history)
        std = np.std(arr)
        return float(np.mean(arr) / (std + 1e-8))

    def win_rate(self) -> float:
        if not self._pnl_history:
            return 0.0
        wins = sum(1 for r in self._pnl_history if r > 0)
        return wins / len(self._pnl_history)

    def stats(self) -> dict:
        return {
            "agent_id":         self.agent_id,
            "type":             self.agent_type.value,
            "cash":             round(self.cash, 2),
            "shares":           self.shares,
            "avg_cost":         round(self.avg_cost, 4),
            "qty_small":        self._qty_small,
            "qty_large":        self._qty_large,
            "slippage_bps":     round(self._slippage_bps, 2),
            "peak_value":       round(self.peak_value, 2),
            "realised_pnl":     round(self.realised_pnl, 2),
            "total_commission": round(self.total_commission, 2),
            "sharpe":           round(self.sharpe_estimate(), 4),
            "win_rate":         round(self.win_rate(), 4),
            "decisions_logged": len(self._decision_history),
            "last_action":      self._last_action.name,
        }

    def reset(self) -> None:
        """Reset agent state for a new episode (preserves per-agent params)."""
        # Recompute the stable per-agent jitter from seed so cash is
        # consistent across episodes for the same agent.
        _rng        = np.random.default_rng(seed=self.agent_id * 2053 + 7919)
        _rng.integers(1, 1000)   # skip past qty_small draw
        _rng.integers(1, 1000)   # skip past qty_large draw
        base_cash   = self._base_cfg.get("initial_cash", 10000.0)
        cash_jitter = float(_rng.uniform(-0.05, 0.05))

        self.cash             = base_cash * (1.0 + cash_jitter)
        self.shares           = 0
        self.avg_cost         = 0.0
        self.peak_value       = self.cash
        self.total_pnl        = 0.0
        self.realised_pnl     = 0.0
        self.total_commission = 0.0
        self._snapshot        = None
        self._last_action     = AgentAction.HOLD
        self._decision_history.clear()
        self._pnl_history.clear()
        self._tick            = 0
        self._is_stopped_out  = False

    def __repr__(self) -> str:
        return f"Agent[{self.agent_id}|{self.agent_type.value}]"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Testing BaseAgent fixes...\n")

    class DummyAgent(BaseAgent):
        def compute_signal(self):
            return 0.5, "dummy_signal"

        def act(self, observation):
            action = AgentAction.BUY_SMALL
            if self._snapshot:
                self._execute_trade(action, self._snapshot.price)
            self._record_decision(action, 0.5, "dummy_signal", ["test"])
            return action.value

    # Verify per-agent qty diversity
    agents = [DummyAgent(agent_id=i, agent_type=AgentType.NOISE) for i in range(10)]
    print("Per-agent qty_small / qty_large / slippage_bps / initial_cash:")
    for a in agents:
        print(
            f"  Agent {a.agent_id:>2}: qty_s={a._qty_small:>3}  "
            f"qty_l={a._qty_large:>4}  slip={a._slippage_bps:>5.2f}bps  "
            f"cash={a.cash:>9.2f}"
        )

    qty_smalls = [a._qty_small for a in agents]
    assert len(set(qty_smalls)) > 3, "qty_small should be diverse!"
    cashes = [round(a.cash, 2) for a in agents]
    assert len(set(cashes)) > 5, "initial cash should be diverse!"
    print("\n✅ All per-agent parameters are unique.")

    # Verify trade execution uses slippage
    snap = MarketSnapshot(
        tick=1, price=100.0,
        price_history=[100.0], log_returns=[],
        volatility=0.01, bid=99.9, ask=100.1,
        spread=0.2, imbalance=0.0,
        bid_depth=1000, ask_depth=1000,
        shock_active=False, shock_regime="calm", vol_multiplier=1.0,
    )
    obs = np.zeros(47, dtype=np.float32)
    for a in agents[:3]:
        a.observe(snap)
        a.act(obs)
        ep = a._execution_price(100.0, is_buy=True)
        print(f"  Agent {a.agent_id}: exec_price={ep:.6f}  avg_cost={a.avg_cost:.6f}")

    avg_costs = [a.avg_cost for a in agents[:3] if a.avg_cost > 0]
    assert len(set(round(c, 4) for c in avg_costs)) > 1, "avg_cost should differ!"
    print("\n✅ Execution prices diverge due to per-agent slippage.")
    print("\nStats for Agent 0:")
    print(agents[0].stats())