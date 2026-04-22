"""
simulation/runner.py
====================
Layer 3 — Simulation Loop | Episode Runner

Ties together all three layers into a single, controllable simulation:

    Layer 1 (GAN)       → synthetic OHLCV seed data + regime conditioning
    Layer 2 (RL Market) → 100-agent market environment with order book
    Layer 3 (GenAI)     → per-tick narration + per-agent explanations

The Runner exposes:
* run_episode()         — synchronous full-episode loop (CLI / training)
* step()                — single-tick advance (dashboard real-time mode)
* inject_shock()        — trigger shock event mid-episode
* get_dashboard_state() — snapshot of current state for the UI

FIX SUMMARY
-----------
v2 fixes:
1. Price continuity between episodes: new GAN sequence is scaled to start
   from the last price of the previous episode, eliminating the 300→400
   jump when auto-reset fires.

2. Agent 0 order submission: removed `if aid == 0` exclusion in step() so
   all 100 agents including agent 0 submit orders to the order book.

3. Runner-owned price history persists across ticks (from v1).

4. panic_count and shocks_occurred added to dashboard state (from v1).

5. stop() method for dashboard Stop button (from v1).

v3 fixes:
6. FIX BUG 1 — Agents all show same PnL (+$0.00 or same value):
   step() was passing self.env.price_history (short, resets every episode)
   to _build_snapshot(). Momentum/Value agents need >= 20-50 ticks of
   history to compute signals. With a short env history they always return
   "insufficient_history" → always HOLD → PnL stays $0.00.
   Fix: pass self._dashboard_price_history (persistent, always growing)
   to _build_snapshot() so agents always have a rich price history.

7. FIX BUG 1b — Per-agent rewards were all the same scalar (env reward
   for agent 0 only). Now each agent gets its own portfolio-delta reward
   so _pnl_history tracks individual performance correctly.

v4 fixes (order book display):
8. FIX ORDER BOOK 1 — book_stats was called BEFORE orders were submitted.
   Moved book_stats = self.env.order_book.stats() to AFTER all agent
   orders are submitted so the snapshot is always populated.

9. FIX ORDER BOOK 2 — Added _last_book_stats and _last_order_book_snap
   caches on the runner. get_dashboard_state() uses these so the UI never
   flashes an empty order book between ticks or during the warm-up period
   when agents mostly Hold.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FILTERED_DIR = Path("data/filtered")


# ---------------------------------------------------------------------------
# Filtered data loader
# ---------------------------------------------------------------------------

class SyntheticDataLoader:
    """
    Loads and serves filtered GAN-generated sequences.
    On each episode, picks a random CSV from data/filtered/ and returns
    its Close price series as the price trajectory for that episode.
    """

    def __init__(self, data_dir: str | Path = _FILTERED_DIR, seed: int | None = None):
        self.data_dir = Path(data_dir)
        self._rng = random.Random(seed)
        self._files: list[Path] = []
        self._load_index()

    def _load_index(self) -> None:
        if not self.data_dir.exists():
            logger.warning(
                "Filtered data directory not found: %s — "
                "runner will use env's internal price generation.",
                self.data_dir,
            )
            return
        self._files = sorted(self.data_dir.glob("*.csv"))
        if self._files:
            logger.info(
                "SyntheticDataLoader: found %d sequences in %s",
                len(self._files), self.data_dir,
            )
        else:
            logger.warning(
                "No CSV files found in %s — "
                "runner will use env's internal price generation.",
                self.data_dir,
            )

    def available(self) -> bool:
        return len(self._files) > 0

    def sample(self) -> tuple[list[float], dict]:
        if not self._files:
            return [], {}
        path = self._rng.choice(self._files)
        try:
            df = pd.read_csv(path, index_col=0)
            prices = df["Close"].dropna().tolist()
            meta = {
                "file": path.name,
                "n_ticks": len(prices),
                "start_price": prices[0] if prices else 100.0,
                "end_price": prices[-1] if prices else 100.0,
                "vol": float(pd.Series(prices).pct_change().std()),
            }
            return prices, meta
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return [], {}

    def reload(self) -> None:
        self._load_index()


# ---------------------------------------------------------------------------
# Tick result
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    tick: int
    price: float
    prev_price: float
    actions: dict[int, int]
    rewards: dict[int, float]
    terminated: dict[int, bool]
    infos: dict[int, dict]
    shock_active: bool
    shock_regime: str
    narrator_event: Any = None
    explain_results: list = field(default_factory=list)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode_id: int
    total_ticks: int
    final_price: float
    initial_price: float
    price_return: float
    shocks_occurred: int
    cascade_events: int
    portfolio_summary: dict
    sequence_file: str = ""
    tick_history: list[TickResult] = field(default_factory=list)
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Full three-layer simulation runner.
    """

    def __init__(
        self,
        market_env,
        agent_pool,
        narrator=None,
        explainer=None,
        logger_obj=None,
        enable_genai: bool = True,
        genai_every: int = 1,
        explain_top_n: int = 3,
        data_dir: str | Path = _FILTERED_DIR,
        seed: int | None = None,
    ):
        self.env = market_env
        self.pool = agent_pool
        self.narrator = narrator
        self.explainer = explainer
        self.sim_logger = logger_obj
        self.enable_genai = enable_genai
        self.genai_every = genai_every
        self.explain_top_n = explain_top_n
        self._rng = np.random.default_rng(seed)
        self._episode_id = 0

        self._data_loader = SyntheticDataLoader(data_dir=data_dir, seed=seed)

        self._price_series: list[float] = []
        self._price_series_idx: int = 0
        self._sequence_file: str = ""

        self._current_tick: int = 0
        self._current_price: float = 0.0
        self._prev_price: float = 0.0
        self._running: bool = False
        self._stopped: bool = False
        self._shock_count: int = 0
        self._cascade_count: int = 0
        self._pending_genai: list[asyncio.Task] = []

        # Persistent price history owned by the runner.
        # env.price_history resets every episode. This list grows
        # continuously and is what the dashboard chart reads.
        self._dashboard_price_history: list[float] = []

        # FIX ORDER BOOK 1 & 2: caches for order book data.
        # book_stats is now populated AFTER orders are submitted in step(),
        # and cached here so get_dashboard_state() always has fresh data
        # even between ticks or during the Hold-heavy warm-up period.
        self._last_book_stats: dict = {}
        self._last_order_book_snap: dict | None = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict:
        self._episode_id += 1
        self._current_tick = 0
        self._shock_count = 0
        self._cascade_count = 0
        self._running = True
        self._stopped = False

        # FIX v2: capture last price BEFORE clearing anything
        # so we can bridge the new sequence to it
        last_price = self._current_price if self._current_price > 0 else 0.0

        # Do NOT clear _dashboard_price_history here —
        # the chart should show continuous price across episodes.
        # Only clear it on a true manual restart (stop → start).

        self._price_series = []
        self._price_series_idx = 0
        self._sequence_file = ""

        if self._data_loader.available():
            prices, meta = self._data_loader.sample()
            if prices:
                # FIX v2: scale new sequence to start from last_price
                # so there is no discontinuous jump in the chart
                if last_price > 0 and prices[0] > 0:
                    scale = last_price / prices[0]
                    prices = [p * scale for p in prices]
                    logger.info(
                        "Episode %d: bridging price %.2f → %.2f (scale=%.4f)",
                        self._episode_id, prices[0] / scale, prices[0], scale,
                    )

                self._price_series = prices
                self._sequence_file = meta.get("file", "")
                self.env._initial_price = prices[0]
                self.env._price = prices[0]
                logger.info(
                    "Episode %d: loaded sequence '%s' | "
                    "%d ticks | price %.2f→%.2f | vol=%.4f",
                    self._episode_id,
                    meta.get("file", "?"),
                    len(prices),
                    prices[0],
                    prices[-1],
                    meta.get("vol", 0),
                )
        else:
            logger.info(
                "Episode %d: no synthetic data — using env internal price generation.",
                self._episode_id,
            )

        obs, info = self.env.reset(seed=seed)
        self.pool.reset_all()

        if self.narrator:
            try:
                self.narrator.reset_history()
            except AttributeError:
                pass

        self._current_price = self.env.current_price
        self._prev_price = self._current_price

        # Seed the history with the initial price
        self._dashboard_price_history.append(self._current_price)

        if self.sim_logger:
            self.sim_logger.new_episode(self._episode_id)

        logger.info(
            "Episode %d reset. Initial price=%.4f | sequence=%s",
            self._episode_id, self._current_price,
            self._sequence_file or "internal",
        )

        return {aid: obs for aid in range(len(self.pool))}

    def full_reset(self, seed: int | None = None) -> dict:
        """
        Hard reset — clears price history too.
        Call this when the user presses Stop then Start fresh.
        """
        self._current_price = 0.0
        self._dashboard_price_history = []
        self._last_book_stats = {}
        self._last_order_book_snap = None
        return self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Price series injection
    # ------------------------------------------------------------------

    def _get_next_series_price(self) -> float | None:
        if not self._price_series:
            return None
        if self._price_series_idx >= len(self._price_series):
            return None
        price = self._price_series[self._price_series_idx]
        self._price_series_idx += 1
        return price

    def _inject_series_price(self) -> None:
        next_price = self._get_next_series_price()
        if next_price is not None and next_price > 0:
            self.env._price = float(next_price)
            self.env._price_history.append(float(next_price))

    # ------------------------------------------------------------------
    # Single tick step
    # ------------------------------------------------------------------

    def step(
        self,
        on_narrator_ready: Optional[Callable] = None,
    ) -> TickResult:
        t0 = time.perf_counter()
        tick = self._current_tick
        prev_price = self._current_price

        self._inject_series_price()

        # FIX ORDER BOOK 1: book_stats moved to AFTER order submission below.
        # Previously it was called here (before any orders were submitted this
        # tick), so the dashboard always saw a one-tick-stale or empty book.

        # FIX v3 BUG 1: use persistent dashboard history (always growing)
        # instead of self.env.price_history (resets every episode, stays short).
        # Momentum agents need slow_ma=20 ticks, Value agents need fv_window=50.
        # With a short env history they always return "insufficient_history"
        # → always HOLD → all agents show +$0.00.
        prices = (
            self._dashboard_price_history
            if self._dashboard_price_history
            else self.env.price_history
        )

        # Build snapshot using a preliminary stats read for signal computation.
        # This is only used for agent observation — the authoritative book_stats
        # for the dashboard is captured after orders are submitted (see below).
        _pre_book_stats = self._last_book_stats or self.env.order_book.stats()
        snapshot = self._build_snapshot(tick, prices, _pre_book_stats)

        self.pool.observe_all(snapshot)

        shared_obs = self.env._build_obs()
        observations = {aid: shared_obs.copy() for aid in range(len(self.pool))}

        actions = self.pool.act_all(observations)

        # Primary env step with agent 0's action
        obs, reward, terminated, truncated, info = self.env.step(actions.get(0, 0))

        # FIX v2: submit orders for ALL agents including agent 0
        # Previously `if aid == 0` excluded agent 0 from order book
        all_trades: list = []
        for aid, action in actions.items():
            if action == 0:   # skip Hold actions only, not agent 0
                continue
            label, qty, side, order_type = self.env._action_decoder.decode(action)
            if side is None or qty <= 0:
                continue
            current_price = self.env.current_price
            try:
                if order_type.name == "MARKET":
                    _, trades = self.env.order_book.add_market_order(
                        aid, side, qty, self.env.current_tick
                    )
                else:
                    _, trades = self.env.order_book.add_limit_order(
                        aid, side, current_price, qty, self.env.current_tick
                    )
                all_trades.extend(trades)
            except Exception as _order_exc:
                logger.debug("Order submission agent=%d failed: %s", aid, _order_exc)

        info["trades_this_tick"] = info.get("trades_this_tick", []) + all_trades

        # FIX ORDER BOOK 1: capture book_stats NOW — after all orders have been
        # submitted this tick — so the snapshot reflects the actual book state.
        book_stats = self.env.order_book.stats()
        self._last_book_stats = book_stats

        # FIX ORDER BOOK 2: cache the level-2 snapshot so get_dashboard_state()
        # always has a non-empty book to show even if called between ticks.
        _snap = self.env.order_book.level2_snapshot(depth=10)
        _bids = [(p, q) for p, q in _snap.bids[:10]]
        _asks = [(p, q) for p, q in _snap.asks[:10]]
        if _bids or _asks:
            self._last_order_book_snap = {
                "bids":      _bids,
                "asks":      _asks,
                "spread":    _snap.spread,
                "imbalance": self.env.order_book.imbalance(),
            }

        # FIX v3 BUG 1b: give each agent its OWN portfolio-delta reward
        # instead of stamping agent-0's env scalar onto all 100 agents.
        # This ensures _pnl_history tracks each agent's individual performance.
        current_price = self.env.current_price
        rewards: dict[int, float] = {}
        for aid, agent in self.pool._agents.items():
            initial_cash = agent._base_cfg.get("initial_cash", 10_000.0)
            rewards[aid] = agent.portfolio_value(current_price) - initial_cash

        dones   = {aid: bool(terminated or truncated) for aid in range(len(self.pool))}
        infos   = {aid: dict(info, agent_id=aid) for aid in range(len(self.pool))}
        self.pool.update_all(rewards, dones, infos)

        self._current_price = self.env.current_price
        self._current_tick = self.env.current_tick

        # Append to runner-owned history every tick
        self._dashboard_price_history.append(self._current_price)

        shock_active = info.get("shock_active", False)
        shock_regime = info.get("shock_regime", "normal")
        if shock_active:
            self._shock_count += 1
        if self.pool.is_cascade_active():
            self._cascade_count += 1

        if self.sim_logger:
            self.sim_logger.log_tick(tick, self._current_price, info, self.pool)

        narrator_event = None
        explain_results = []

        if self.enable_genai and (tick % self.genai_every == 0):
            tick_state = self._build_tick_state(
                tick, self._current_price, prev_price, info, book_stats
            )
            narrator_event = self._fire_narrator(tick_state, on_narrator_ready)
            explain_results = self._fire_explains(tick, info)

        elapsed = (time.perf_counter() - t0) * 1000

        result = TickResult(
            tick=tick,
            price=self._current_price,
            prev_price=prev_price,
            actions=actions,
            rewards=rewards,
            terminated={aid: terminated for aid in range(len(self.pool))},
            infos=infos,
            shock_active=shock_active,
            shock_regime=shock_regime,
            narrator_event=narrator_event,
            explain_results=explain_results,
            elapsed_ms=elapsed,
        )

        sequence_done = (
            self._price_series and
            self._price_series_idx >= len(self._price_series)
        )
        if terminated or truncated or sequence_done:
            self._running = False
            if sequence_done:
                logger.info(
                    "Episode %d: synthetic sequence exhausted at tick %d — "
                    "auto-resetting for continuous dashboard mode.",
                    self._episode_id, tick,
                )
                # Auto-reset so dashboard keeps running continuously.
                # Uses soft reset (preserves price history, bridges price).
                self.reset()

        return result

    # ------------------------------------------------------------------
    # Full episode loop
    # ------------------------------------------------------------------

    def run_episode(
        self,
        seed: int | None = None,
        on_tick: Optional[Callable[[TickResult], None]] = None,
        max_ticks: Optional[int] = None,
    ) -> EpisodeResult:
        t_start = time.perf_counter()
        self.reset(seed=seed)
        initial_price = self._current_price
        tick_history: list[TickResult] = []

        if self._price_series:
            limit = len(self._price_series)
        else:
            limit = max_ticks or getattr(self.env, '_tick_limit', 1000)

        logger.info("Episode %d starting. Tick limit=%d", self._episode_id, limit)

        for _ in range(limit):
            result = self.step()
            tick_history.append(result)
            if on_tick:
                on_tick(result)
            if not self._running:
                break

        elapsed = time.perf_counter() - t_start
        final_price = self._current_price
        price_return = (final_price - initial_price) / max(initial_price, 1e-8)
        portfolio_summary = self.pool.portfolio_summary(final_price)

        ep_result = EpisodeResult(
            episode_id=self._episode_id,
            total_ticks=self._current_tick,
            final_price=final_price,
            initial_price=initial_price,
            price_return=price_return,
            shocks_occurred=self._shock_count,
            cascade_events=self._cascade_count,
            portfolio_summary=portfolio_summary,
            sequence_file=self._sequence_file,
            tick_history=tick_history,
            elapsed_sec=elapsed,
        )

        logger.info(
            "Episode %d done | %d ticks | price %.4f→%.4f (%+.2f%%) | "
            "shocks=%d cascades=%d | seq=%s | %.1fs",
            self._episode_id, self._current_tick,
            initial_price, final_price, price_return * 100,
            self._shock_count, self._cascade_count,
            self._sequence_file or "internal",
            elapsed,
        )
        return ep_result

    # ------------------------------------------------------------------
    # Shock injection
    # ------------------------------------------------------------------

    def inject_shock(self, shock_type: str = "flash_crash") -> None:
        self.env.inject_shock(shock_type)
        logger.info("Shock injected: %s at tick %d", shock_type, self._current_tick)

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the simulation cleanly (called by dashboard Stop button)."""
        self._running = False
        self._stopped = True
        logger.info("SimulationRunner stopped at tick %d.", self._current_tick)

    # ------------------------------------------------------------------
    # Dashboard state snapshot
    # ------------------------------------------------------------------

    def get_dashboard_state(self) -> dict:
        """
        Return a complete state snapshot for the dashboard UI.
        price_history comes from _dashboard_price_history (runner-owned,
        persistent across ticks and episodes).

        FIX ORDER BOOK 2: order_book data now comes from _last_order_book_snap
        (cached after orders are submitted in step()) so the UI never shows a
        blank book between ticks or during the warm-up period.
        """
        price = self._current_price

        # Use cached book stats (populated after order submission in step()).
        # Fall back to a live read only if the cache is empty (e.g. before
        # the very first step() call).
        book_stats = self._last_book_stats or self.env.order_book.stats()

        # Use cached order book snapshot so the UI is never empty.
        # If nothing cached yet, do a live read and cache it.
        if self._last_order_book_snap is not None:
            ob = self._last_order_book_snap
        else:
            snap = self.env.order_book.level2_snapshot(depth=10)
            bids = [(p, q) for p, q in snap.bids[:10]]
            asks = [(p, q) for p, q in snap.asks[:10]]
            ob = {
                "bids":      bids,
                "asks":      asks,
                "spread":    snap.spread,
                "imbalance": self.env.order_book.imbalance(),
            }
            if bids or asks:
                self._last_order_book_snap = ob

        seq_len = len(self._price_series) if self._price_series else 0
        seq_idx = self._price_series_idx
        seq_pct = (seq_idx / seq_len * 100) if seq_len else 0.0

        narrator_feed: list[str] = []
        if self.narrator and hasattr(self.narrator, "recent"):
            try:
                entries = self.narrator.recent(10)
                narrator_feed = [getattr(e, "text", str(e)) for e in entries]
            except Exception:
                pass

        _portfolio = self.pool.portfolio_summary(price)
        _shock = getattr(self.env, "_current_shock", None)

        return {
            "tick":           self._current_tick,
            "price":          price,
            "price_history":  list(self._dashboard_price_history),
            "shock_active":   _shock.is_stressed() if _shock else False,
            "shock_regime":   _shock.regime_label() if _shock else None,
            "portfolio_summary": _portfolio,
            "panic_count":    self.pool.panic_count,
            "panic_fraction": self.pool.panic_fraction,
            "cascade_active": self.pool.is_cascade_active(),
            "order_book": {
                "bids":      ob["bids"],
                "asks":      ob["asks"],
                "spread":    ob.get("spread", 0.0),
                "imbalance": ob.get("imbalance", self.env.order_book.imbalance()),
            },
            "book_stats":     book_stats,
            "episode_id":     self._episode_id,
            "running":        self._running,
            "shocks_occurred": self._shock_count,
            "narrator_feed":  narrator_feed,
            "sequence_file":  self._sequence_file,
            "sequence_progress": (
                f"{seq_idx}/{seq_len} ({seq_pct:.0f}%)" if seq_len else "internal"
            ),
            "_running": self._running and not self._stopped,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_snapshot(self, tick: int, prices: list, book_stats: dict):
        try:
            from agents.base_agent import MarketSnapshot
        except ImportError:
            return None

        p = self._current_price
        snap = self.env.order_book.level2_snapshot(depth=1)

        log_returns = []
        for i in range(1, min(len(prices), 11)):
            r = np.log(prices[-i] / max(prices[-(i+1)], 1e-8)) if len(prices) > i else 0.0
            log_returns.append(float(r))

        return MarketSnapshot(
            tick=tick,
            price=p,
            price_history=list(prices),
            log_returns=log_returns,
            volatility=getattr(self.env.price_model, 'current_vol', 0.001),
            bid=snap.bids[0][0] if snap.bids else None,
            ask=snap.asks[0][0] if snap.asks else None,
            spread=snap.spread or 0.0,
            imbalance=self.env.order_book.imbalance(),
            bid_depth=float(book_stats.get("bid_depth", 0)),
            ask_depth=float(book_stats.get("ask_depth", 0)),
            shock_active=self.env._current_shock.is_stressed(),
            shock_regime=self.env._current_shock.regime_label(),
            vol_multiplier=self.env._current_shock.vol_multiplier,
        )

    def _build_tick_state(
        self,
        tick: int,
        price: float,
        prev_price: float,
        info: dict,
        book_stats: dict,
    ) -> dict:
        notable = self.pool.notable_decisions(tick, self.explain_top_n)
        return {
            "tick":           tick,
            "price":          price,
            "prev_price":     prev_price,
            "volatility":     getattr(self.env.price_model, 'current_vol', 0.001),
            "spread":         info.get("book_stats", {}).get("spread", 0.0),
            "imbalance":      self.env.order_book.imbalance(),
            "shock_active":   info.get("shock_active", False),
            "shock_regime":   info.get("shock_regime", "normal"),
            "vol_multiplier": self.env._current_shock.vol_multiplier,
            "panic_fraction": self.pool.panic_fraction,
            "cascade_active": self.pool.is_cascade_active(),
            "sequence_file":  self._sequence_file,
            "notable_decisions": [
                {
                    "agent_id":    d.agent_id,
                    "agent_type":  d.agent_type.value,
                    "action":      d.action.name,
                    "signal_label": d.signal_label,
                }
                for d in notable
            ],
            "book_stats":        book_stats,
            "portfolio_summary": self.pool.portfolio_summary(price),
        }

    def _fire_narrator(self, tick_state: dict, callback: Optional[Callable] = None):
        if not self.narrator:
            return None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.ensure_future(
                    self._async_narrate(tick_state, callback)
                )
                self._pending_genai.append(task)
            else:
                return loop.run_until_complete(self.narrator.narrate_async(tick_state))
        except Exception as exc:
            logger.debug("Narrator fire failed: %s", exc)
        return None

    async def _async_narrate(self, tick_state: dict, callback: Optional[Callable]):
        event = await self.narrator.narrate_async(tick_state)
        if callback:
            callback(event)
        return event

    def _fire_explains(self, tick: int, info: dict) -> list:
        if not self.explainer:
            return []
        notable = self.pool.notable_decisions(tick, self.explain_top_n)
        if not notable:
            return []

        results = []
        for decision in notable:
            try:
                explanation = self.explainer.explain_decision(decision)
                # Store explanation back onto the agent so the dashboard can read it
                agent = self.pool._agents.get(decision.agent_id)
                if agent is not None:
                    agent._latest_explanation = explanation
                results.append({
                    "agent_id":    decision.agent_id,
                    "explanation": explanation,
                })
                logger.debug(
                    "Explained agent %d: %s",
                    decision.agent_id,
                    explanation[:80],
                )
            except Exception as exc:
                logger.debug(
                    "explain_decision failed for agent %d: %s",
                    decision.agent_id, exc,
                )
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_runner(
    market_env,
    agent_pool,
    narrator=None,
    explainer=None,
    logger_obj=None,
    enable_genai: bool = True,
    data_dir: str | Path = _FILTERED_DIR,
    seed: int | None = None,
) -> SimulationRunner:
    return SimulationRunner(
        market_env=market_env,
        agent_pool=agent_pool,
        narrator=narrator,
        explainer=explainer,
        logger_obj=logger_obj,
        enable_genai=enable_genai,
        data_dir=data_dir,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    from env.market_env import build_market_env
    from agents.agent_pool import build_agent_pool

    print("🚀 Testing SimulationRunner with synthetic data...\n")

    env  = build_market_env(agent_id=0, seed=42)
    pool = build_agent_pool(seed=42, load_policies=False)
    runner = build_runner(
        market_env=env,
        agent_pool=pool,
        enable_genai=False,
        seed=42,
    )

    if runner._data_loader.available():
        print(f"✅ Synthetic data loaded: {len(runner._data_loader._files)} sequences\n")
    else:
        print("⚠️  No synthetic data found — using internal price generation\n")

    try:
        print("▶ Running 1 full episode...\n")
        result = runner.run_episode(seed=42)
        print(
            f"  Episode {result.episode_id} done | "
            f"{result.total_ticks} ticks | "
            f"price {result.initial_price:.2f}→{result.final_price:.2f} "
            f"({result.price_return*100:+.2f}%) | "
            f"shocks={result.shocks_occurred} | "
            f"seq={result.sequence_file or 'internal'} | "
            f"{result.elapsed_sec:.1f}s"
        )

        print("\n🔄 Testing step-by-step mode (5 ticks)...\n")
        runner.reset(seed=0)
        for _ in range(5):
            tick_result = runner.step()
            dash = runner.get_dashboard_state()
            print(
                f"  Tick {tick_result.tick:4d} | "
                f"Price: {tick_result.price:.4f} | "
                f"Seq: {dash['sequence_progress']} | "
                f"Shock: {tick_result.shock_active} ({tick_result.shock_regime}) | "
                f"History len: {len(dash['price_history'])} | "
                f"OB bids: {len(dash['order_book']['bids'])} "
                f"asks: {len(dash['order_book']['asks'])}"
            )

        print("\n⚡ Injecting flash_crash shock...")
        runner.reset(seed=1)
        runner.inject_shock("flash_crash")
        for _ in range(3):
            runner.step()
        dash = runner.get_dashboard_state()
        print(f"  Shock active: {dash['shock_active']}  regime: {dash['shock_regime']}")
        print(f"  Panic count: {dash['panic_count']}/20")
        print(f"  Price history length: {len(dash['price_history'])}")
        print(f"  Order book bids: {len(dash['order_book']['bids'])}  asks: {len(dash['order_book']['asks'])}")

        print("\n✅ All smoke tests passed.")

    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        traceback.print_exc()