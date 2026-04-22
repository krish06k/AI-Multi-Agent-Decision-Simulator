"""
simulation/logger.py
====================
Layer 2/3 — Simulation | Structured Event Logger

Writes structured JSON-lines event logs for the simulation, enabling:
  * Offline analysis and replay
  * Dashboard chart data export
  * Training data for future RL iterations
  * Audit trail for agent decisions and shock events

Log files
---------
  data/logs/sim/events_{episode:04d}.jsonl     — all tick events
  data/logs/sim/decisions_{episode:04d}.jsonl  — agent decision audit trail
  data/logs/sim/shocks_{episode:04d}.jsonl     — shock event records
  data/logs/sim/episode_{episode:04d}.json     — end-of-episode summary

Architecture
------------
  SimLogger         — main class; wraps Python logging + JSONL writers
  EventType         — enum of loggable event categories
  LogEntry          — typed dict written to JSONL

Public API
----------
  SimLogger.log_tick(snapshot)
  SimLogger.log_decision(decision)
  SimLogger.log_shock(shock_record)
  SimLogger.log_episode(episode_stats, agent_records)
  SimLogger.close()

Called by
---------
  simulation/runner.py after each tick, decision, shock, and episode end.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, TextIO

from simulation.state import TickSnapshot, ShockRecord, EpisodeStats, AgentRecord

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_LOG_DIR = Path("data/logs/sim")
_MAX_TICK_HISTORY = 2000   # max ticks to hold in memory per episode


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    TICK       = "tick"
    DECISION   = "decision"
    SHOCK      = "shock"
    CASCADE    = "cascade"
    REGIME     = "regime_change"
    EPISODE    = "episode_end"
    RESET      = "episode_reset"
    ERROR      = "error"


# ---------------------------------------------------------------------------
# SimLogger
# ---------------------------------------------------------------------------

class SimLogger:
    """
    Structured event logger for the simulation.

    Writes JSON-lines files for real-time events and a JSON summary at
    episode end.  Also maintains an in-memory buffer for the dashboard
    to query recent events without touching disk.

    Parameters
    ----------
    log_dir    : Directory for log files.
    episode    : Starting episode number.
    enabled    : Set False to suppress all file I/O (test mode).
    console_level : Python logging level for console output.
    """

    def __init__(
        self,
        log_dir: Path | str = _DEFAULT_LOG_DIR,
        episode: int = 0,
        enabled: bool = True,
        console_level: int = logging.INFO,
    ):
        self._log_dir  = Path(log_dir)
        self._episode  = episode
        self._enabled  = enabled
        self._started  = False

        # In-memory buffers (for dashboard)
        self._tick_buffer: list[dict]     = []
        self._decision_buffer: list[dict] = []
        self._shock_buffer: list[dict]    = []
        self._event_buffer: list[dict]    = []  # all events

        # File handles (opened in _open_files)
        self._tick_fh:     Optional[TextIO] = None
        self._decision_fh: Optional[TextIO] = None
        self._shock_fh:    Optional[TextIO] = None

        # Console logger
        logging.basicConfig(
            level   = console_level,
            format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
        )

        if self._enabled:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._open_files(episode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_files(self, episode: int) -> None:
        """Open JSONL files for a new episode."""
        self._close_files()
        ep = f"{episode:04d}"
        try:
            self._tick_fh     = open(self._log_dir / f"events_{ep}.jsonl",    "a", encoding="utf-8")
            self._decision_fh = open(self._log_dir / f"decisions_{ep}.jsonl", "a", encoding="utf-8")
            self._shock_fh    = open(self._log_dir / f"shocks_{ep}.jsonl",    "a", encoding="utf-8")
            self._started = True
            _log.info("SimLogger: opened log files for episode %d in %s", episode, self._log_dir)
        except OSError as exc:
            _log.error("SimLogger: failed to open log files — %s", exc)
            self._enabled = False

    def _close_files(self) -> None:
        for fh in (self._tick_fh, self._decision_fh, self._shock_fh):
            if fh:
                try:
                    fh.close()
                except OSError:
                    pass
        self._tick_fh = self._decision_fh = self._shock_fh = None

    def _write(self, fh: Optional[TextIO], record: dict) -> None:
        if fh and not fh.closed:
            try:
                fh.write(json.dumps(record, default=str) + "\n")
                fh.flush()
            except OSError as exc:
                _log.warning("SimLogger write error: %s", exc)

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _push_event(self, record: dict) -> None:
        self._event_buffer.append(record)
        if len(self._event_buffer) > _MAX_TICK_HISTORY:
            self._event_buffer = self._event_buffer[-_MAX_TICK_HISTORY:]

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def on_reset(self, episode: int, initial_price: float, gan_regime: Optional[str] = None) -> None:
        """
        Call at the start of each new episode.
        Rotates log files and resets in-memory buffers.
        """
        self._episode = episode
        self._tick_buffer.clear()
        self._decision_buffer.clear()
        self._shock_buffer.clear()
        self._event_buffer.clear()

        if self._enabled:
            self._open_files(episode)

        record = {
            "event": EventType.RESET.value,
            "ts": self._ts(),
            "episode": episode,
            "initial_price": initial_price,
            "gan_regime": gan_regime,
        }
        self._write(self._tick_fh, record)
        self._push_event(record)
        _log.info("Episode %d started — initial_price=%.2f  regime=%s",
                    episode, initial_price, gan_regime or "random")

    # ------------------------------------------------------------------
    # Per-tick logging
    # ------------------------------------------------------------------

    def log_tick(self, snapshot: TickSnapshot) -> None:
        """
        Log a completed TickSnapshot.  Called every tick.

        Parameters
        ----------
        snapshot : TickSnapshot from simulation/state.py.
        """
        record = {
            "event":              EventType.TICK.value,
            "ts":                 self._ts(),
            "episode":            self._episode,
            "tick":               snapshot.tick,
            "price":              round(snapshot.price, 4),
            "log_return":         round(snapshot.log_return, 6),
            "spread_bps":         round(snapshot.spread_bps, 2),
            "bid_depth":          snapshot.bid_depth,
            "ask_depth":          snapshot.ask_depth,
            "imbalance":          round(snapshot.imbalance, 4),
            "buy_volume":         snapshot.buy_volume,
            "sell_volume":        snapshot.sell_volume,
            "total_trades":       snapshot.total_trades,
            "vol_estimate":       round(snapshot.vol_estimate, 6),
            "shock_active":       snapshot.shock_active,
            "shock_regime":       snapshot.shock_regime,
            "dominant_action":    snapshot.dominant_action,
            "dominant_type":      snapshot.dominant_agent_type,
            "narrator":           snapshot.narrator_comment,
        }

        self._tick_buffer.append(record)
        if len(self._tick_buffer) > _MAX_TICK_HISTORY:
            self._tick_buffer = self._tick_buffer[-_MAX_TICK_HISTORY:]

        self._write(self._tick_fh, record)
        self._push_event(record)

        if snapshot.tick % 100 == 0:
            _log.info(
                "Tick %4d | price=%.2f  vol=%.4f  shock=%s",
                snapshot.tick, snapshot.price, snapshot.vol_estimate,
                snapshot.shock_regime if snapshot.shock_active else "none",
            )

    # ------------------------------------------------------------------
    # Agent decision logging
    # ------------------------------------------------------------------

    def log_decision(self, decision: Any) -> None:
        """
        Log an AgentDecision from agents/base_agent.py.

        Parameters
        ----------
        decision : AgentDecision dataclass.
        """
        record = {
            "event":          EventType.DECISION.value,
            "ts":             self._ts(),
            "episode":        self._episode,
            "tick":           decision.tick,
            "agent_id":       decision.agent_id,
            "agent_type":     decision.agent_type.value,
            "action":         decision.action.name,
            "action_int":     decision.action_int,
            "signal_value":   round(float(decision.signal_value), 6),
            "signal_label":   decision.signal_label,
            "price":          round(decision.price, 4),
            "portfolio_value":round(decision.portfolio_value, 2),
            "cash":           round(decision.cash, 2),
            "shares":         decision.shares,
            "unrealised_pnl": round(decision.unrealised_pnl, 2),
            "drawdown":       round(decision.drawdown, 4),
            "reason_tags":    decision.reason_tags,
        }

        self._decision_buffer.append(record)
        if len(self._decision_buffer) > _MAX_TICK_HISTORY * 2:
            self._decision_buffer = self._decision_buffer[-_MAX_TICK_HISTORY * 2:]

        self._write(self._decision_fh, record)
        self._push_event(record)

        _log.debug(
            "Decision | agent=%d (%s) tick=%d action=%s signal=%.4f",
            decision.agent_id, decision.agent_type.value,
            decision.tick, decision.action.name, decision.signal_value,
        )

    # ------------------------------------------------------------------
    # Shock logging
    # ------------------------------------------------------------------

    def log_shock(self, shock: ShockRecord) -> None:
        """
        Log a ShockRecord from simulation/state.py.

        Parameters
        ----------
        shock : ShockRecord dataclass.
        """
        record = {
            "event":              EventType.SHOCK.value,
            "ts":                 self._ts(),
            "episode":            self._episode,
            "tick":               shock.tick,
            "shock_type":         shock.shock_type,
            "price_before":       round(shock.price_before, 4),
            "price_after":        round(shock.price_after, 4),
            "spread_before_bps":  round(shock.spread_before_bps, 2),
            "spread_after_bps":   round(shock.spread_after_bps, 2),
            "volume_spike":       round(shock.volume_spike, 2),
            "duration_ticks":     shock.duration_ticks,
            "agents_affected":    shock.agents_affected,
            "explanation":        shock.explanation,
        }

        self._shock_buffer.append(record)
        self._write(self._shock_fh, record)
        self._push_event(record)

        _log.warning(
            "SHOCK | %s at tick %d | price %.2f → %.2f  spread %.1f→%.1f bps",
            shock.shock_type, shock.tick,
            shock.price_before, shock.price_after,
            shock.spread_before_bps, shock.spread_after_bps,
        )

    # ------------------------------------------------------------------
    # Cascade logging
    # ------------------------------------------------------------------

    def log_cascade(self, cascade_data: dict) -> None:
        """Log a panic cascade event."""
        record = {
            "event":   EventType.CASCADE.value,
            "ts":      self._ts(),
            "episode": self._episode,
            **cascade_data,
        }
        self._write(self._tick_fh, record)
        self._push_event(record)
        _log.warning(
            "CASCADE | trigger tick=%d  peak panic agents=%d  price %.2f → %.2f",
            cascade_data.get("trigger_tick", 0),
            cascade_data.get("peak_panic_agents", 0),
            cascade_data.get("price_at_trigger", 0),
            cascade_data.get("price_trough", 0),
        )

    # ------------------------------------------------------------------
    # Regime change logging
    # ------------------------------------------------------------------

    def log_regime_change(self, old_regime: str, new_regime: str, tick: int) -> None:
        record = {
            "event":       EventType.REGIME.value,
            "ts":          self._ts(),
            "episode":     self._episode,
            "tick":        tick,
            "old_regime":  old_regime,
            "new_regime":  new_regime,
        }
        self._write(self._tick_fh, record)
        self._push_event(record)
        _log.info("Regime change: %s → %s at tick %d", old_regime, new_regime, tick)

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------

    def log_episode_end(
        self,
        stats: EpisodeStats,
        agent_records: dict[int, AgentRecord],
    ) -> None:
        """
        Write the end-of-episode JSON summary and log to console.

        Parameters
        ----------
        stats          : EpisodeStats from simulation/state.py.
        agent_records  : All AgentRecord instances keyed by agent_id.
        """
        agent_summary = [
            {
                "agent_id":       r.agent_id,
                "agent_type":     r.agent_type,
                "portfolio_value":round(r.portfolio_value, 2),
                "unrealised_pnl": round(r.unrealised_pnl, 2),
                "realised_pnl":   round(r.realised_pnl, 2),
                "drawdown":       round(r.drawdown, 4),
                "trade_count":    r.trade_count,
                "total_commission":round(r.total_commission, 2),
            }
            for r in agent_records.values()
        ]

        summary = {
            "event":          EventType.EPISODE.value,
            "ts":             self._ts(),
            "episode":        stats.episode,
            "total_ticks":    stats.total_ticks,
            "total_trades":   stats.total_trades,
            "initial_price":  round(stats.initial_price, 4),
            "final_price":    round(stats.final_price, 4),
            "peak_price":     round(stats.peak_price, 4),
            "trough_price":   round(stats.trough_price, 4),
            "price_return":   round(stats.price_return(), 6),
            "total_buy_vol":  stats.total_buy_volume,
            "total_sell_vol": stats.total_sell_volume,
            "shock_count":    stats.shock_count,
            "cascade_count":  stats.cascade_count,
            "gan_regime":     stats.gan_regime,
            "best_type":      stats.best_type(),
            "worst_type":     stats.worst_type(),
            "pnl_by_type": {
                "momentum": round(stats.momentum_total_pnl, 2),
                "value":    round(stats.value_total_pnl, 2),
                "noise":    round(stats.noise_total_pnl, 2),
                "panic":    round(stats.panic_total_pnl, 2),
            },
            "agents": agent_summary,
        }

        # Write JSON summary file
        if self._enabled:
            summary_path = self._log_dir / f"episode_{stats.episode:04d}.json"
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=str)
                _log.info("Episode summary written → %s", summary_path)
            except OSError as exc:
                _log.error("Failed to write episode summary: %s", exc)

        self._write(self._tick_fh, summary)
        self._push_event(summary)

        _log.info(
            "Episode %d complete | ticks=%d  trades=%d  "
            "price %.2f→%.2f (%.2f%%)  shocks=%d  best=%s  worst=%s",
            stats.episode, stats.total_ticks, stats.total_trades,
            stats.initial_price, stats.final_price,
            stats.price_return() * 100,
            stats.shock_count, stats.best_type(), stats.worst_type(),
        )

    # ------------------------------------------------------------------
    # Dashboard query helpers
    # ------------------------------------------------------------------

    def recent_ticks(self, n: int = 100) -> list[dict]:
        """Return the last n tick records (for chart streaming)."""
        return self._tick_buffer[-n:]

    def recent_decisions(self, n: int = 50, agent_id: Optional[int] = None) -> list[dict]:
        """Return the last n decision records, optionally filtered by agent."""
        buf = self._decision_buffer
        if agent_id is not None:
            buf = [d for d in buf if d.get("agent_id") == agent_id]
        return buf[-n:]

    def recent_shocks(self) -> list[dict]:
        return list(self._shock_buffer)

    def recent_events(self, n: int = 200) -> list[dict]:
        """Return the last n events of any type (for event timeline)."""
        return self._event_buffer[-n:]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all file handles."""
        self._close_files()
        _log.info("SimLogger closed for episode %d.", self._episode)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_logger(
    log_dir: Path | str = _DEFAULT_LOG_DIR,
    episode: int = 0,
    enabled: bool = True,
    console_level: int = logging.INFO,
) -> SimLogger:
    """
    Build a SimLogger from default configuration.

    Parameters
    ----------
    log_dir       : Directory for log files.
    episode       : Starting episode number.
    enabled       : False = no file I/O (test mode).
    console_level : Python logging level.
    """
    return SimLogger(
        log_dir       = log_dir,
        episode       = episode,
        enabled       = enabled,
        console_level = console_level,
    )

if __name__ == "__main__":
    print("🔍 Testing SimLogger...\n")

    from types import SimpleNamespace

    # 🔥 Use different name (IMPORTANT)
    sim_logger = build_logger(enabled=False)

    # -------------------------------
    # 1. Episode Reset
    # -------------------------------
    sim_logger.on_reset(
        episode=1,
        initial_price=100.0,
        gan_regime="calm"
    )

    # -------------------------------
    # 2. Fake Tick Snapshot
    # -------------------------------
    for tick in range(1, 6):
        snapshot = SimpleNamespace(
            tick=tick,
            price=100 + tick,
            log_return=0.001 * tick,
            spread_bps=10.0,
            bid_depth=1000,
            ask_depth=1000,
            imbalance=0.1,
            buy_volume=20000,
            sell_volume=15000,
            total_trades=50,
            vol_estimate=0.02,
            shock_active=False,
            shock_regime="none",
            dominant_action="buy",
            dominant_agent_type="momentum",
            narrator_comment="Market stable"
        )

        sim_logger.log_tick(snapshot)

    print("✅ Tick logging done")

    # -------------------------------
    # 3. Fake Decision
    # -------------------------------
    decision = SimpleNamespace(
        tick=5,
        agent_id=1,
        agent_type=SimpleNamespace(value="momentum"),
        action=SimpleNamespace(name="BUY_LARGE"),
        action_int=2,
        signal_value=0.034,
        signal_label="EMA bullish",
        price=105.0,
        portfolio_value=10200,
        cash=8000,
        shares=20,
        unrealised_pnl=200,
        drawdown=0.01,
        reason_tags=["trend_follow"]
    )

    sim_logger.log_decision(decision)
    print("✅ Decision logging done")

    # -------------------------------
    # 4. Fake Shock
    # -------------------------------
    shock = SimpleNamespace(
        tick=6,
        shock_type="flash_crash",
        price_before=105,
        price_after=95,
        spread_before_bps=10,
        spread_after_bps=50,
        volume_spike=3.5,
        duration_ticks=5,
        agents_affected=80,
        explanation="Sudden liquidity collapse"
    )

    sim_logger.log_shock(shock)
    print("⚡ Shock logged")

    # -------------------------------
    # 5. Cascade Event
    # -------------------------------
    sim_logger.log_cascade({
        "trigger_tick": 6,
        "price_at_trigger": 105,
        "price_trough": 90,
        "peak_panic_agents": 20
    })

    # -------------------------------
    # 6. Regime Change
    # -------------------------------
    sim_logger.log_regime_change("calm", "stressed", tick=7)

    # -------------------------------
    # 7. Episode Summary
    # -------------------------------
    stats = SimpleNamespace(
        episode=1,
        total_ticks=100,
        total_trades=500,
        initial_price=100,
        final_price=110,
        peak_price=115,
        trough_price=95,
        total_buy_volume=100000,
        total_sell_volume=90000,
        shock_count=1,
        cascade_count=1,
        gan_regime="calm",
        momentum_total_pnl=500,
        value_total_pnl=200,
        noise_total_pnl=0,
        panic_total_pnl=-300,
        price_return=lambda: 0.10,
        best_type=lambda: "momentum",
        worst_type=lambda: "panic"
    )

    agent_records = {
        1: SimpleNamespace(
            agent_id=1,
            agent_type="momentum",
            portfolio_value=10500,
            unrealised_pnl=500,
            realised_pnl=200,
            drawdown=0.02,
            trade_count=10,
            total_commission=15
        )
    }

    sim_logger.log_episode_end(stats, agent_records)

    print("\n📊 Recent Events:")
    for e in sim_logger.recent_events(5):
        print(e["event"], "| tick:", e.get("tick"))

    # -------------------------------
    # Cleanup
    # -------------------------------
    sim_logger.close()

    print("\n🔥 SimLogger test complete")