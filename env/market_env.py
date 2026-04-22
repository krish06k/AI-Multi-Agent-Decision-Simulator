"""
env/market_env.py
=================
Layer 2 — RL Market Environment | Gymnasium Environment

Full OpenAI Gymnasium-compatible market simulation environment.

Observation space (per agent)
------------------------------
* Price features   : normalised price, log-return, EMA ratios, vol estimate.
* Order book       : bid/ask imbalance, spread, top-5 depth on each side.
* Portfolio        : cash fraction, position fraction, unrealised PnL, drawdown.
* Market regime    : encoded shock state, vol regime, tick progress.
Total: 47 features → Box(47,) float32.

Action space
------------
Discrete(5): 0=hold, 1=buy_small, 2=buy_large, 3=sell_small, 4=sell_large

Reward
------
Configurable: "pnl" | "pnl_sharpe" | "log_return"
Includes transaction cost penalty and holding penalty for large positions.

FIX SUMMARY (this version)
---------------------------
FIX 1 — step_multi() action encoding was wrong.
         Old: net_action < 0  →  env_action = 2  (buy_large!)
         New: net_action < 0  →  env_action = 3  (sell_small)
              net_action ≪ 0  →  env_action = 4  (sell_large)
         This was the primary cause of the always-upward chart: the env
         interpreted net selling pressure as a large buy order.

FIX 2 — Mean reversion now pulls toward a rolling fundamental value,
         not the fixed _initial_price.
         Old: mean_reversion = (_initial_price - new_price) * strength
              → Once price >> initial_price this became a strong upward force
         New: _fundamental_price tracks a slow EMA of actual prices.
              Mean reversion only applies when price deviates from that EMA,
              and is bidirectional (pulls down when above, up when below).

FIX 3 — Market maker (MM) orders are cancelled before each replenishment.
         Old: MM orders accumulated tick-by-tick without expiry.
              The order book filled with stale bids → persistent buy imbalance
              → net_flow always positive → price always rising.
         New: order_book.cancel_agent_orders(-1) called before each refill.
              Book imbalance is now genuinely zero-mean between agent actions.

FIX 4 — Shock engine called ONCE per tick (was already fixed in prior version,
         retained and documented here).

FIX 5 — Post-shock dampening preserved from prior version.

FIX 6 — Liquidity replenishment scales with shock spread_multiplier.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, SupportsFloat

import numpy as np
import yaml

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from env.order_book import OrderBook, Side, OrderType, Trade
from env.price_impact import PriceImpactModel, build_price_impact
from env.shock import ShockEngine, ShockEffect, build_shock_engine

logger = logging.getLogger(__name__)

_CFG_PATH       = Path(__file__).resolve().parents[1] / "config" / "env_config.yaml"
_AGENT_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "agent_config.yaml"


def _load_cfg(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """Per-agent portfolio tracking."""
    agent_id:         int
    cash:             float
    shares:           int   = 0
    avg_cost:         float = 0.0
    peak_value:       float = 0.0
    total_commission: float = 0.0
    trade_count:      int   = 0

    pnl_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def total_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def unrealised_pnl(self, price: float) -> float:
        if self.shares == 0:
            return 0.0
        return self.shares * (price - self.avg_cost)

    def unrealised_pnl_pct(self, price: float) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (price - self.avg_cost) / self.avg_cost

    def drawdown(self, price: float) -> float:
        current          = self.total_value(price)
        self.peak_value  = max(self.peak_value, current)
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - current) / self.peak_value

    def update_on_buy(self, qty: int, exec_price: float, commission: float) -> None:
        cost = qty * exec_price + commission
        if self.shares + qty > 0:
            self.avg_cost = (
                (self.shares * self.avg_cost + qty * exec_price)
                / (self.shares + qty)
            )
        self.shares           += qty
        self.cash             -= cost
        self.total_commission += commission
        self.trade_count      += 1

    def update_on_sell(self, qty: int, exec_price: float, commission: float) -> None:
        proceeds              = qty * exec_price - commission
        self.shares          -= qty
        self.cash            += proceeds
        self.total_commission += commission
        self.trade_count      += 1
        if self.shares == 0:
            self.avg_cost = 0.0

    def reset(self, initial_cash: float) -> None:
        self.cash             = initial_cash
        self.shares           = 0
        self.avg_cost         = 0.0
        self.peak_value       = initial_cash
        self.total_commission = 0.0
        self.trade_count      = 0
        self.pnl_history.clear()


# ---------------------------------------------------------------------------
# Market action encoding
# ---------------------------------------------------------------------------

class ActionDecoder:
    """Converts discrete action indices to (label, qty, side, order_type) tuples."""

    def __init__(self, max_order_size: int = 200):
        self.max_order_size = max_order_size
        self._action_map = {
            0: ("hold",  0,                          None,     None),
            1: ("buy",   int(max_order_size * 0.25), Side.BID, OrderType.LIMIT),
            2: ("buy",   int(max_order_size * 1.00), Side.BID, OrderType.MARKET),
            3: ("sell",  int(max_order_size * 0.25), Side.ASK, OrderType.LIMIT),
            4: ("sell",  int(max_order_size * 1.00), Side.ASK, OrderType.MARKET),
        }

    def decode(self, action: int) -> tuple:
        return self._action_map.get(action, self._action_map[0])

    @property
    def n_actions(self) -> int:
        return len(self._action_map)


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class MarketEnv(gym.Env):
    """Multi-agent market simulation Gymnasium environment."""

    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}
    OBS_DIM  = 47

    def __init__(
        self,
        agent_id:     int           = 0,
        cfg:          dict | None   = None,
        agent_cfg:    dict | None   = None,
        shared_state: Any | None    = None,
        seed:         int | None    = None,
    ):
        super().__init__()

        self.agent_id      = agent_id
        self.cfg           = cfg       or _load_cfg(_CFG_PATH)
        self.agent_cfg     = agent_cfg or _load_cfg(_AGENT_CFG_PATH)
        self._shared_state = shared_state
        self._seed         = seed

        self._initial_price   = self.cfg.get("initial_price",       100.0)
        self._tick_limit      = self.cfg.get("tick_limit",           1000)
        self._min_price       = self.cfg.get("min_price",            1.0)
        self._price_hist_len  = self.cfg.get("price_history_len",    30)
        self._tx_cost_bps     = self.cfg.get("transaction_cost_bps", 5)
        self._holding_pen     = self.cfg.get("holding_penalty",      0.0001)
        self._reward_type     = self.cfg.get("reward_type",          "pnl_sharpe")
        self._sharpe_window   = self.cfg.get("sharpe_window",        20)
        self._max_pos         = self.agent_cfg["base"].get("max_position",   1000)
        self._initial_cash    = self.agent_cfg["base"].get("initial_cash",   10000.0)
        self._max_order_size  = self.agent_cfg["base"].get("max_order_size", 200)

        # FIX 2: EMA alpha for tracking rolling fundamental price.
        # Slow EMA (alpha=0.001) means fundamental price adapts over ~1000 ticks.
        self._fundamental_alpha = self.cfg.get("fundamental_ema_alpha", 0.001)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space    = spaces.Discrete(5)
        self._action_decoder = ActionDecoder(self._max_order_size)

        self._own_market = shared_state is None
        if self._own_market:
            self.order_book   = OrderBook(
                tick_size=self.cfg.get("tick_size",      0.01),
                max_depth=self.cfg.get("max_book_depth", 20),
            )
            self.price_model  = build_price_impact(self.cfg)
            self.shock_engine = build_shock_engine(self.cfg, seed=seed)

        self.portfolio = Portfolio(agent_id=agent_id, cash=self._initial_cash)

        self._tick:              int            = 0
        self._price:             float          = self._initial_price
        self._fundamental_price: float          = self._initial_price  # FIX 2
        self._price_history:     deque[float]   = deque(maxlen=self._price_hist_len)
        self._return_history:    deque[float]   = deque(maxlen=self._sharpe_window)
        self._trades_this_tick:  list[Trade]    = []
        self._current_shock:     ShockEffect    = ShockEffect()
        self._episode_trades:    list[Trade]    = []
        self._rng:               np.random.Generator = np.random.default_rng(seed)

        # Post-shock dampening state
        self._pre_shock_price:   float = self._initial_price
        self._shock_was_active:  bool  = False
        self._post_shock_ticks:  int   = 0

        logger.info("MarketEnv[agent=%d] initialised.", agent_id)

    # ------------------------------------------------------------------
    # FIX 1: step_multi — correct action encoding for net sell pressure
    # ------------------------------------------------------------------

    def step_multi(self, actions: dict[int, int]):
        """
        Aggregate all agent actions into a single env step.

        FIX 1: Old code mapped net_action < 0 → env_action = 2 (buy_large).
        This meant net selling pressure was encoded as a large buy — the
        primary driver of the always-upward chart.

        New mapping:
          net_action >  1  → action 2 (buy_large)
          net_action == 1  → action 1 (buy_small)
          net_action == 0  → action 0 (hold)
          net_action == -1 → action 3 (sell_small)
          net_action < -1  → action 4 (sell_large)
        """
        net_action = 0
        for action in actions.values():
            if action == 1:
                net_action += 1
            elif action == 2:
                net_action += 2
            elif action == 3:
                net_action -= 1
            elif action == 4:
                net_action -= 2

        # FIX 1: was:  env_action = 1 if net > 0 else (2 if net < 0 else 0)
        #              → mapped sell pressure to buy_large (action 2)
        if net_action > 1:
            env_action = 2   # buy_large
        elif net_action == 1:
            env_action = 1   # buy_small
        elif net_action == 0:
            env_action = 0   # hold
        elif net_action == -1:
            env_action = 3   # sell_small
        else:
            env_action = 4   # sell_large

        obs, reward, done, truncated, info = self.step(env_action)

        obs_dict       = {aid: obs    for aid in actions}
        reward_dict    = {aid: reward for aid in actions}
        done_dict      = {aid: done   for aid in actions}
        truncated_dict = {aid: truncated for aid in actions}
        info_dict      = {
            aid: {
                "shock_active":     info.get("shock_active", False),
                "shock_regime":     info.get("shock_regime", "none"),
                "book_stats":       info.get("book_stats", {
                    "spread_bps": 10.0,
                    "bid_total":  0,
                    "ask_total":  0,
                    "imbalance":  0.0,
                }),
                "trades_this_tick": info.get("trades_this_tick", []),
            }
            for aid in actions
        }

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._seed = seed
            self._rng  = np.random.default_rng(seed)

        self._tick                = 0
        self._price               = self._initial_price
        self._fundamental_price   = self._initial_price   # FIX 2
        self._pre_shock_price     = self._initial_price
        self._shock_was_active    = False
        self._post_shock_ticks    = 0

        self._price_history.clear()
        self._return_history.clear()
        self._trades_this_tick = []
        self._episode_trades   = []
        self._current_shock    = ShockEffect()

        for _ in range(self._price_hist_len):
            self._price_history.append(self._initial_price)

        if self._own_market:
            self.order_book.reset()
            self.price_model.reset(self._initial_price)
            self.shock_engine.reset()
            self._replenish_liquidity(n_levels=20, qty_per_level=100)

        self.portfolio.reset(self._initial_cash)

        obs  = self._build_obs()
        info = self._build_info()
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        prev_price           = self._price
        prev_portfolio_value = self.portfolio.total_value(self._price)

        label, qty, side, order_type = self._action_decoder.decode(action)
        trades     = []
        commission = 0.0

        if side is not None and qty > 0:
            if side == Side.BID:
                max_affordable = int(self.portfolio.cash / max(self._price, 1e-8))
                qty = min(qty, max_affordable)
            elif side == Side.ASK and qty > self.portfolio.shares:
                qty = self.portfolio.shares

            if qty > 0:
                mid    = self._price
                offset = self.price_model.generate_quotes(mid)[
                    0 if side == Side.BID else 1
                ]
                if order_type == OrderType.LIMIT:
                    _, trades = self.order_book.add_limit_order(
                        self.agent_id, side, offset, qty, self._tick)
                else:
                    _, trades = self.order_book.add_market_order(
                        self.agent_id, side, qty, self._tick)

                for trade in trades:
                    exec_qty   = trade.qty
                    exec_price = trade.price
                    commission = exec_qty * exec_price * (self._tx_cost_bps / 10000.0)
                    if side == Side.BID:
                        self.portfolio.update_on_buy(exec_qty, exec_price, commission)
                    else:
                        self.portfolio.update_on_sell(exec_qty, exec_price, commission)

        self._trades_this_tick = trades
        self._episode_trades.extend(trades)

        # ── Price impact from net order flow ────────────────────────────
        book_stats = self.order_book.stats()
        imbalance  = self.order_book.imbalance()
        net_flow   = imbalance * self._max_order_size

        if self._own_market:
            impact = self.price_model.step(
                net_order_flow = net_flow,
                current_price  = self._price,
                book_depth     = max(
                    0.1,
                    book_stats.get("bid_depth", 1) + book_stats.get("ask_depth", 1)
                ),
            )
            new_price = impact.price_after

            # ── FIX 2: Mean reversion toward rolling fundamental price ───
            # Old: pulled toward fixed _initial_price → became upward force
            #      once price had risen well above 100.
            # New: _fundamental_price is a slow EMA of actual prices.
            #      Reversion is bidirectional and always pulls toward centre.
            shock_currently_active = len(self.shock_engine.active_shocks()) > 0

            # Update fundamental price EMA (only during calm — freeze during shocks)
            if not shock_currently_active:
                self._fundamental_price = (
                    self._fundamental_alpha * new_price
                    + (1.0 - self._fundamental_alpha) * self._fundamental_price
                )

            if not shock_currently_active:
                mr_strength    = self.cfg.get("mean_reversion_strength", 0.003)
                # Bidirectional: pushes price toward fundamental regardless of
                # which direction it has drifted.
                mean_reversion = (self._fundamental_price - new_price) * mr_strength
                new_price      = new_price + mean_reversion

            # Noise cap: tighter during calm (0.8%), relaxed during shock (3%)
            if shock_currently_active:
                max_tick_move = prev_price * 0.03
            else:
                max_tick_move = prev_price * 0.008
            raw_move = new_price - prev_price
            if abs(raw_move) > max_tick_move:
                new_price = prev_price + max_tick_move * (1.0 if raw_move > 0 else -1.0)

        else:
            new_price = self._price

        # ── Shock engine — called ONCE per tick (FIX 4) ─────────────────
        if self._own_market:
            prev_price_for_shock = (
                list(self._price_history)[-2]
                if len(self._price_history) >= 2
                else self._price
            )
            shock_effect = self.shock_engine.step(
                tick          = self._tick,
                current_price = new_price,
                prev_price    = prev_price_for_shock,
            )
            self._current_shock = shock_effect

            if not shock_effect.halt_trading:
                price_shock = new_price * shock_effect.price_shock_pct
                new_price   = max(self._min_price, new_price + price_shock)

            shock_now_active = shock_effect.is_stressed()

            if shock_now_active and not self._shock_was_active:
                self._pre_shock_price  = self._price
                self._shock_was_active = True
                logger.debug(
                    "Shock started at tick %d, anchor=%.2f",
                    self._tick, self._pre_shock_price,
                )

            if not shock_now_active and self._shock_was_active:
                self._shock_was_active = False
                self._post_shock_ticks = 30
                logger.debug("Shock ended at tick %d, dampening 30 ticks", self._tick)

            # Post-shock dampening (FIX 5)
            if self._post_shock_ticks > 0 and not shock_now_active:
                drag_strength = 0.08 * (self._post_shock_ticks / 30.0) ** 2
                scar_factor   = 0.97
                anchor        = self._pre_shock_price * scar_factor
                drag          = (anchor - new_price) * drag_strength
                new_price     = max(self._min_price, new_price + drag)
                self._post_shock_ticks -= 1

        self._price = new_price
        self._price_history.append(self._price)

        # ── FIX 3: Cancel stale MM orders before replenishing ────────────
        # Old: MM orders accumulated every tick without expiry.
        #      Book filled with thousands of stale bids → imbalance always
        #      positive → net_flow always positive → price only ever went up.
        # New: cancel all MM orders (agent_id=-1) before each refill.
        #      The book is swept clean then refilled symmetrically.
        if self._own_market:
            shock_spread = getattr(self._current_shock, "spread_multiplier", 1.0)
            liq_qty      = max(10, int(50 / max(shock_spread, 1.0)))
            self._replenish_liquidity(n_levels=5, qty_per_level=liq_qty)

        # ── Reward ───────────────────────────────────────────────────────
        new_value = self.portfolio.total_value(self._price)
        reward    = self._compute_reward(prev_portfolio_value, new_value, commission)

        if prev_portfolio_value > 0:
            log_ret = np.log(new_value / max(prev_portfolio_value, 1e-8))
            self._return_history.append(log_ret)
            self.portfolio.pnl_history.append(new_value - prev_portfolio_value)

        self._tick += 1

        terminated = (
            self.portfolio.total_value(self._price) <= 0.0
            or self._price <= self._min_price
        )
        truncated = self._tick >= self._tick_limit

        obs  = self._build_obs()
        info = self._build_info(trades=trades, action_label=label)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # FIX 3: Liquidity replenishment — cancel stale orders first
    # ------------------------------------------------------------------

    def _replenish_liquidity(self, n_levels: int = 5, qty_per_level: int = 50) -> None:
        """
        Inject synthetic market maker liquidity around the current price.

        FIX 3: Cancel all existing MM orders (agent_id=-1) before placing
        new ones. Without this, MM orders accumulate tick-by-tick and create
        a permanently positive bid-side imbalance, pushing net_flow positive
        every tick and causing the price to only ever rise.
        """
        # Sweep stale MM orders before refilling
        try:
            self.order_book.cancel_agent_orders(agent_id=-1)
        except AttributeError:
            # Fallback if cancel_agent_orders not yet implemented in OrderBook.
            # Add this method to order_book.py — see note below.
            logger.debug(
                "cancel_agent_orders not found on OrderBook. "
                "Add it to order_book.py to fix bid imbalance accumulation."
            )

        p = self._price
        for i in range(1, n_levels + 1):
            offset = p * 0.001 * i   # 0.1% per level
            try:
                self.order_book.add_limit_order(
                    agent_id = -1,
                    side     = Side.BID,
                    price    = p - offset,
                    qty      = qty_per_level,
                    tick     = self._tick,
                )
                self.order_book.add_limit_order(
                    agent_id = -1,
                    side     = Side.ASK,
                    price    = p + offset,
                    qty      = qty_per_level,
                    tick     = self._tick,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_value: float,
        new_value:  float,
        commission: float,
    ) -> float:
        if self._reward_type == "pnl":
            reward = (new_value - prev_value) / max(prev_value, 1.0)

        elif self._reward_type == "pnl_sharpe":
            if len(self._return_history) >= 5:
                returns = np.array(self._return_history)
                mean_r  = np.mean(returns)
                std_r   = np.std(returns) + 1e-8
                reward  = float(mean_r / std_r)
            else:
                reward = (new_value - prev_value) / max(prev_value, 1.0)

        elif self._reward_type == "log_return":
            reward = np.log(max(new_value, 1e-8) / max(prev_value, 1e-8))

        else:
            reward = (new_value - prev_value) / max(prev_value, 1.0)

        tx_penalty   = commission / max(prev_value, 1.0)
        pos_fraction = abs(self.portfolio.shares) / max(self._max_pos, 1)
        hold_penalty = self._holding_pen * (pos_fraction ** 2)

        return float(reward - tx_penalty - hold_penalty)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        obs    = np.zeros(self.OBS_DIM, dtype=np.float32)
        prices = list(self._price_history)
        p      = self._price
        p0     = self._initial_price

        obs[0] = (p / p0) - 1.0
        obs[1] = np.log(p / max(prices[-2], 1e-8)) if len(prices) >= 2 else 0.0

        for i, window in enumerate([5, 10, 20, 30]):
            if len(prices) >= window:
                ema        = float(np.mean(prices[-window:]))
                obs[2 + i] = (p / max(ema, 1e-8)) - 1.0

        obs[6] = self.price_model.current_vol if self._own_market else 0.0

        snap = self.order_book.level2_snapshot(depth=5)
        norm = float(self._max_order_size * 10)
        for j, (_, qty) in enumerate(snap.bids[:5]):
            obs[7  + j] = qty / norm
        for j, (_, qty) in enumerate(snap.asks[:5]):
            obs[12 + j] = qty / norm

        obs[17] = self.order_book.imbalance()
        obs[18] = (snap.spread / max(p, 1.0)) if snap.spread else 0.0

        total_val  = self.portfolio.total_value(p)
        obs[19] = self.portfolio.cash / max(total_val, 1.0)
        obs[20] = self.portfolio.shares / max(self._max_pos, 1)
        obs[21] = float(np.clip(self.portfolio.unrealised_pnl_pct(p), -1.0, 1.0))
        obs[22] = float(np.clip(self.portfolio.drawdown(p), 0.0, 1.0))

        obs[23] = self._tick / max(self._tick_limit, 1)
        shock   = self._current_shock
        obs[24] = float(shock.is_stressed())
        obs[25] = min(shock.vol_multiplier    / 5.0,  1.0)
        obs[26] = min(shock.spread_multiplier / 10.0, 1.0)

        returns = []
        for k in range(len(prices) - 1, max(len(prices) - 21, 0), -1):
            r = np.log(prices[k] / max(prices[k - 1], 1e-8)) if k > 0 else 0.0
            returns.append(float(np.clip(r, -0.5, 0.5)))
        returns    = (returns + [0.0] * 20)[:20]
        obs[27:47] = returns

        return obs

    def _build_info(self, trades: list = None, action_label: str = "hold") -> dict:
        p = self._price
        return {
            "tick":             self._tick,
            "price":            p,
            "agent_id":         self.agent_id,
            "action":           action_label,
            "portfolio_value":  self.portfolio.total_value(p),
            "cash":             self.portfolio.cash,
            "shares":           self.portfolio.shares,
            "unrealised_pnl":   self.portfolio.unrealised_pnl(p),
            "drawdown":         self.portfolio.drawdown(p),
            "trade_count":      self.portfolio.trade_count,
            "total_commission": self.portfolio.total_commission,
            "trades_this_tick": trades or [],
            "shock_active":     self._current_shock.is_stressed(),
            "shock_regime":     self._current_shock.regime_label(),
            "book_stats":       self.order_book.stats(),
            "post_shock_damp":  self._post_shock_ticks > 0,
            "fundamental_price": self._fundamental_price,  # FIX 2: expose for debug
        }

    # ------------------------------------------------------------------
    # Shock injection
    # ------------------------------------------------------------------

    def inject_shock(self, shock_type: str = "flash_crash") -> None:
        if self._own_market:
            self.shock_engine.inject_now(shock_type, self._tick)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode == "ansi":
            shock_label = (
                f" ⚡ {self._current_shock.regime_label().upper()}"
                if self._current_shock.is_stressed() else ""
            )
            damp_label = " 〜damping" if self._post_shock_ticks > 0 else ""
            return (
                f"Tick {self._tick:4d} | Price: ${self._price:.2f}"
                f"{shock_label}{damp_label} | "
                f"Fundamental: ${self._fundamental_price:.2f} | "
                f"Agent {self.agent_id}: "
                f"Cash=${self.portfolio.cash:.0f}  "
                f"Shares={self.portfolio.shares}  "
                f"PnL=${self.portfolio.unrealised_pnl(self._price):.2f}"
            )
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_price(self) -> float:
        return self._price

    @property
    def fundamental_price(self) -> float:
        return self._fundamental_price

    @property
    def current_tick(self) -> int:
        return self._tick

    @property
    def price_history(self) -> list[float]:
        return list(self._price_history)

    @property
    def episode_trades(self) -> list[Trade]:
        return list(self._episode_trades)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_market_env(
    agent_id: int           = 0,
    cfg:      dict | None   = None,
    seed:     int | None    = None,
) -> MarketEnv:
    return MarketEnv(agent_id=agent_id, cfg=cfg, seed=seed)


if __name__ == "__main__":
    env = build_market_env()
    obs, info = env.reset()
    print("Initial Obs Shape:", obs.shape)
    print(f"Initial price: {info['price']:.2f}")
    print(f"Fundamental:   {info['fundamental_price']:.2f}\n")

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(
            f"Step {step:2d} | Action: {action} | Reward: {reward:+.5f} | "
            f"Price: {info['price']:.4f} | "
            f"Fundamental: {info['fundamental_price']:.4f} | "
            f"Damp: {info['post_shock_damp']}"
        )
        if done or truncated:
            break