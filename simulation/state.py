
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SimStatus(Enum):
    IDLE       = auto()   # not yet started
    RUNNING    = auto()   # episode in progress
    PAUSED     = auto()   # manually paused by dashboard
    FINISHED   = auto()   # episode completed normally
    ERROR      = auto()   # terminated due to exception


class RegimeLabel(Enum):
    CALM      = "calm"
    STRESSED  = "stressed"
    TRENDING  = "trending"
    CRISIS    = "crisis"   # active shock of severity >= flash_crash


# ---------------------------------------------------------------------------
# Per-tick snapshot (immutable history entry)
# ---------------------------------------------------------------------------

@dataclass
class TickSnapshot:
    """
    Immutable record of one completed simulation tick.
    Written by runner.py; read by dashboard charts.

    All monetary values are in simulation USD.
    """
    tick: int
    price: float
    log_return: float               # log(price_t / price_{t-1})
    spread_bps: float
    bid_depth: float
    ask_depth: float
    imbalance: float                # [-1, 1]
    buy_volume: int                 # total shares bought this tick
    sell_volume: int                # total shares sold this tick
    total_trades: int               # number of matched orders
    shock_active: bool
    shock_regime: str               # "none" | "flash_crash" | etc.
    vol_estimate: float             # realised vol (rolling window)
    narrator_comment: str = ""      # live narrator sentence for this tick
    dominant_action: str = "hold"
    dominant_agent_type: str = "noise"


# ---------------------------------------------------------------------------
# Per-agent record (mutable — updated each tick)
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    """
    Dashboard-facing summary of a single agent's current state.
    Updated in-place by runner.py after each tick.
    """
    agent_id: int
    agent_type: str                 # "momentum" | "value" | "noise" | "panic"
    portfolio_value: float = 10_000.0
    cash: float            = 10_000.0
    shares: int            = 0
    unrealised_pnl: float  = 0.0
    realised_pnl: float    = 0.0
    drawdown: float        = 0.0
    trade_count: int       = 0
    total_commission: float= 0.0
    last_action: str       = "hold"
    last_signal: float     = 0.0
    is_panic: bool         = False   # True when panic agent is in panic mode
    latest_explanation: str = ""     # last LLM explanation for this agent


# ---------------------------------------------------------------------------
# Shock log entry
# ---------------------------------------------------------------------------

@dataclass
class ShockRecord:
    """Log entry written when a shock event fires."""
    tick: int
    shock_type: str
    price_before: float
    price_after: float
    spread_before_bps: float
    spread_after_bps: float
    volume_spike: float
    duration_ticks: int
    agents_affected: int
    explanation: str = ""           # populated by explainer.py


# ---------------------------------------------------------------------------
# Episode-level statistics (accumulated during run)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """
    Cumulative statistics for the current episode.
    Reset on each env reset.
    """
    episode: int = 0
    total_ticks: int = 0
    total_trades: int = 0
    initial_price: float = 100.0
    final_price: float   = 100.0
    peak_price: float    = 100.0
    trough_price: float  = 100.0
    total_buy_volume: int  = 0
    total_sell_volume: int = 0
    shock_count: int       = 0
    cascade_count: int     = 0
    gan_regime: Optional[str] = None

    # Per-type PnL at episode end
    momentum_total_pnl: float = 0.0
    value_total_pnl: float    = 0.0
    noise_total_pnl: float    = 0.0
    panic_total_pnl: float    = 0.0

    def price_return(self) -> float:
        """Episode price return as a decimal fraction."""
        return (self.final_price - self.initial_price) / max(self.initial_price, 1e-8)

    def best_type(self) -> str:
        pnls = {
            "momentum": self.momentum_total_pnl,
            "value":    self.value_total_pnl,
            "noise":    self.noise_total_pnl,
            "panic":    self.panic_total_pnl,
        }
        return max(pnls, key=pnls.get)

    def worst_type(self) -> str:
        pnls = {
            "momentum": self.momentum_total_pnl,
            "value":    self.value_total_pnl,
            "noise":    self.noise_total_pnl,
            "panic":    self.panic_total_pnl,
        }
        return min(pnls, key=pnls.get)


# ---------------------------------------------------------------------------
# Top-level simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """
    Complete mutable state of the running simulation.

    Runner.py holds the single authoritative instance and updates it
    in-place after each tick. The dashboard reads it (thread-safely via
    a copy) to render charts and agent cards.

    Fields
    ------
    status          : Current simulation lifecycle status.
    tick            : Current tick number (0 at episode start).
    price           : Latest mid-price.
    regime          : Current market regime label.
    episode_stats   : Accumulated episode-level stats.
    tick_history    : Rolling deque of TickSnapshot records.
    agent_records   : Dict of agent_id → AgentRecord.
    shock_log       : List of all ShockRecord events this episode.
    narrator_feed   : List of the last N narrator comments (raw strings).
    cascade_active  : True when ≥2 panic agents are simultaneously dumping.
    panic_count     : Current number of active panic agents.
    error_message   : Non-empty when status == ERROR.
    meta            : Arbitrary key→value store for dashboard extras.
    """

    # Lifecycle
    status: SimStatus = SimStatus.IDLE
    error_message: str = ""

    # Current tick state
    tick: int    = 0
    price: float = 100.0
    regime: RegimeLabel = RegimeLabel.CALM

    # Market microstructure (latest tick)
    spread_bps: float  = 10.0
    bid_depth: float   = 0.0
    ask_depth: float   = 0.0
    imbalance: float   = 0.0
    vol_estimate: float= 0.0

    # Shock state
    shock_active: bool = False
    shock_regime: str  = "none"
    cascade_active: bool = False
    panic_count: int     = 0

    # Episode metadata
    episode_stats: EpisodeStats = field(default_factory=EpisodeStats)

    # History (rolling window — size controlled by runner)
    tick_history: list[TickSnapshot] = field(default_factory=list)

    # Agent state
    agent_records: dict[int, AgentRecord] = field(default_factory=dict)

    # Event logs
    shock_log: list[ShockRecord] = field(default_factory=list)
    narrator_feed: list[str]     = field(default_factory=list)

    # Dashboard extras
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: int) -> Optional[AgentRecord]:
        return self.agent_records.get(agent_id)

    def agents_by_type(self, agent_type: str) -> list[AgentRecord]:
        return [a for a in self.agent_records.values() if a.agent_type == agent_type]

    def price_series(self) -> list[float]:
        """Ordered list of mid-prices from tick_history."""
        return [s.price for s in self.tick_history]

    def return_series(self) -> list[float]:
        return [s.log_return for s in self.tick_history]

    def volume_series(self) -> list[tuple[int, int]]:
        """List of (buy_volume, sell_volume) per tick."""
        return [(s.buy_volume, s.sell_volume) for s in self.tick_history]

    def latest_narrator(self) -> str:
        return self.narrator_feed[-1] if self.narrator_feed else ""

    def total_portfolio_value(self) -> float:
        return sum(a.portfolio_value for a in self.agent_records.values())

    def is_running(self) -> bool:
        return self.status == SimStatus.RUNNING

    def snapshot_dict(self) -> dict:
        """
        Lightweight JSON-serialisable summary for dashboard polling.
        Does NOT include full tick_history (too large for every poll).
        """
        return {
            "status":          self.status.name,
            "tick":            self.tick,
            "price":           round(self.price, 4),
            "regime":          self.regime.value,
            "shock_active":    self.shock_active,
            "shock_regime":    self.shock_regime,
            "cascade_active":  self.cascade_active,
            "panic_count":     self.panic_count,
            "spread_bps":      round(self.spread_bps, 2),
            "imbalance":       round(self.imbalance, 4),
            "vol_estimate":    round(self.vol_estimate, 6),
            "total_portfolio": round(self.total_portfolio_value(), 2),
            "latest_comment":  self.latest_narrator(),
            "episode":         self.episode_stats.episode,
            "error":           self.error_message,
        }

    # ------------------------------------------------------------------
    # Mutation helpers (called by runner.py)
    # ------------------------------------------------------------------

    def reset_episode(self, episode: int, initial_price: float, gan_regime: Optional[str] = None) -> None:
        """Reset mutable state for a new episode."""
        self.tick          = 0
        self.price         = initial_price
        self.regime        = RegimeLabel.CALM
        self.shock_active  = False
        self.shock_regime  = "none"
        self.cascade_active= False
        self.panic_count   = 0
        self.spread_bps    = 10.0
        self.bid_depth     = 0.0
        self.ask_depth     = 0.0
        self.imbalance     = 0.0
        self.vol_estimate  = 0.0
        self.tick_history  = []
        self.shock_log     = []
        self.narrator_feed = []
        self.episode_stats = EpisodeStats(
            episode       = episode,
            initial_price = initial_price,
            final_price   = initial_price,
            peak_price    = initial_price,
            trough_price  = initial_price,
            gan_regime    = gan_regime,
        )
        self.status = SimStatus.RUNNING

    def push_tick(self, snapshot: TickSnapshot) -> None:
        """Append a TickSnapshot and update episode stats."""
        self.tick_history.append(snapshot)
        stats = self.episode_stats
        stats.total_ticks    += 1
        stats.total_trades   += snapshot.total_trades
        stats.total_buy_volume  += snapshot.buy_volume
        stats.total_sell_volume += snapshot.sell_volume
        stats.final_price    = snapshot.price
        if snapshot.price > stats.peak_price:
            stats.peak_price = snapshot.price
        if snapshot.price < stats.trough_price:
            stats.trough_price = snapshot.price
        if snapshot.shock_active:
            pass  # shock_count incremented by runner on onset only

    def push_narrator(self, text: str, max_feed: int = 50) -> None:
        self.narrator_feed.append(text)
        if len(self.narrator_feed) > max_feed:
            self.narrator_feed = self.narrator_feed[-max_feed:]

    def push_shock(self, record: ShockRecord) -> None:
        self.shock_log.append(record)
        self.episode_stats.shock_count += 1

    def update_agent(self, record: AgentRecord) -> None:
        self.agent_records[record.agent_id] = record

    def mark_error(self, message: str) -> None:
        self.status = SimStatus.ERROR
        self.error_message = message


if __name__ == "__main__":
    print("🔍 Testing SimulationState...\n")

    # -------------------------------
    # 1. Create state
    # -------------------------------
    state = SimulationState()

    # Reset episode
    state.reset_episode(
        episode=1,
        initial_price=100.0,
        gan_regime="calm"
    )

    print("✅ Episode started")
    print("Initial snapshot:", state.snapshot_dict())

    # -------------------------------
    # 2. Simulate ticks
    # -------------------------------
    import numpy as np

    price = 100.0

    for tick in range(1, 6):
        new_price = price * (1 + np.random.normal(0, 0.01))
        log_ret = np.log(new_price / price)

        snapshot = TickSnapshot(
            tick=tick,
            price=new_price,
            log_return=log_ret,
            spread_bps=10.0,
            bid_depth=1000,
            ask_depth=1000,
            imbalance=np.random.uniform(-1, 1),
            buy_volume=20000,
            sell_volume=15000,
            total_trades=50,
            shock_active=False,
            shock_regime="none",
            vol_estimate=0.02,
            narrator_comment="Market stable",
            dominant_action="buy",
            dominant_agent_type="momentum"
        )

        state.push_tick(snapshot)
        state.push_narrator(f"Tick {tick} commentary")

        # Update price
        state.price = new_price
        state.tick = tick

    print("\n📈 Price Series:", state.price_series())
    print("📊 Return Series:", state.return_series())
    print("📦 Volume Series:", state.volume_series())

    # -------------------------------
    # 3. Add agents
    # -------------------------------
    agent = AgentRecord(
        agent_id=1,
        agent_type="momentum",
        portfolio_value=10500,
        cash=8000,
        shares=20,
        unrealised_pnl=500,
        trade_count=5,
        last_action="BUY"
    )

    state.update_agent(agent)

    print("\n👤 Agent:", state.get_agent(1))
    print("💰 Total Portfolio:", state.total_portfolio_value())

    # -------------------------------
    # 4. Shock event
    # -------------------------------
    shock = ShockRecord(
        tick=6,
        shock_type="flash_crash",
        price_before=105,
        price_after=90,
        spread_before_bps=10,
        spread_after_bps=50,
        volume_spike=3.0,
        duration_ticks=3,
        agents_affected=80,
        explanation="Sudden panic selling"
    )

    state.push_shock(shock)

    print("\n⚡ Shock Log:", state.shock_log)

    # -------------------------------
    # 5. Snapshot summary
    # -------------------------------
    print("\n📊 Snapshot Dict:")
    print(state.snapshot_dict())

    # -------------------------------
    # 6. Error test
    # -------------------------------
    state.mark_error("Test error occurred")

    print("\n❌ Error State:")
    print(state.snapshot_dict())

    print("\n🔥 SimulationState test complete")