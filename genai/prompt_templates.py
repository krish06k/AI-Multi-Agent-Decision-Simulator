"""
genai/prompt_templates.py
=========================
Layer 3 — GenAI Explainability | Prompt Template Library

All LLM prompt construction lives here so that explainer.py and narrator.py
stay clean and only handle API calls and post-processing.

Template philosophy
-------------------
* Every function returns a plain str ready for the LLM.
* Templates are intentionally *structured but conversational* — the LLM
  should sound like a knowledgeable market analyst, not a log file.
* We embed only the data the model actually needs; keep prompts lean to
  reduce latency and cost.

Naming convention
-----------------
  build_<target>_prompt(...)  → str

Exported symbols
----------------
  build_decision_prompt        — single agent decision explanation
  build_portfolio_prompt       — per-agent portfolio narrative
  build_market_regime_prompt   — regime / macro state summary
  build_shock_prompt           — shock event explanation
  build_episode_summary_prompt — end-of-episode performance recap
  build_cascade_prompt         — panic cascade analysis
  build_narrator_prompt        — tick-by-tick live commentary
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal fraction as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def _fmt_price(value: float) -> str:
    return f"${value:,.2f}"


def _agent_type_description(agent_type: str) -> str:
    descriptions = {
        "momentum": "trend-following (dual-EMA crossover, PPO-backed)",
        "value":    "mean-reversion / buy-the-dip (DQN-backed)",
        "noise":    "random liquidity provider (uninformed retail)",
        "panic":    "stress-triggered cascade seller (stop-loss / institutional)",
    }
    return descriptions.get(agent_type.lower(), agent_type)


# ---------------------------------------------------------------------------
# 1. Single-agent decision explanation
# ---------------------------------------------------------------------------

def build_decision_prompt(
    agent_id: int,
    agent_type: str,
    action: str,
    signal_value: float,
    signal_label: str,
    price: float,
    portfolio_value: float,
    cash: float,
    shares: int,
    unrealised_pnl: float,
    drawdown: float,
    reason_tags: list[str],
    shock_active: bool = False,
    shock_regime: str = "none",
    tick: int = 0,
) -> str:
    """
    Construct a prompt asking the LLM to explain why a specific agent
    took a specific action at a given tick.

    Returns
    -------
    str  — fully-formed user prompt (no system prompt; caller adds that).
    """
    shock_ctx = (
        f"\nNOTE: A {shock_regime} shock is currently active — this is an "
        f"exogenous stress event affecting all agents."
        if shock_active else ""
    )

    return f"""You are a financial market simulation analyst. Explain the following trading decision in 2-3 concise sentences, as if briefing a portfolio risk manager.

AGENT CONTEXT
  Agent ID   : {agent_id}
  Agent type : {_agent_type_description(agent_type)}
  Simulation tick : {tick}

MARKET STATE
  Mid-price  : {_fmt_price(price)}
  Signal     : {signal_value:+.5f}  ({signal_label}){shock_ctx}

ACTION TAKEN
  Action     : {action.upper()}
  Reason tags: {', '.join(reason_tags) if reason_tags else 'none'}

PORTFOLIO AT DECISION TIME
  Total value      : {_fmt_price(portfolio_value)}
  Cash             : {_fmt_price(cash)}
  Shares held      : {shares}
  Unrealised PnL   : {_fmt_price(unrealised_pnl)}
  Peak drawdown    : {_fmt_pct(drawdown)}

Explain the agent's reasoning, referencing its strategy type and the signal that drove this decision. Keep your answer factual and succinct."""


# ---------------------------------------------------------------------------
# 2. Per-agent portfolio narrative
# ---------------------------------------------------------------------------

def build_portfolio_prompt(
    agent_id: int,
    agent_type: str,
    portfolio_value: float,
    initial_cash: float,
    cash: float,
    shares: int,
    unrealised_pnl: float,
    realised_pnl: float,
    drawdown: float,
    trade_count: int,
    total_commission: float,
    tick: int,
    price: float,
) -> str:
    """
    Prompt for a narrative summary of a single agent's portfolio performance
    over the current episode so far.
    """
    net_return = (portfolio_value - initial_cash) / max(initial_cash, 1.0)
    return f"""You are a simulation performance analyst. Write a short 3-sentence portfolio narrative for the following agent, suitable for a live dashboard tooltip.

AGENT
  ID   : {agent_id}
  Type : {_agent_type_description(agent_type)}
  Tick : {tick}

PERFORMANCE
  Starting cash       : {_fmt_price(initial_cash)}
  Current value       : {_fmt_price(portfolio_value)}
  Net return          : {_fmt_pct(net_return)}
  Cash remaining      : {_fmt_price(cash)}
  Shares held         : {shares}
  Unrealised PnL      : {_fmt_price(unrealised_pnl)}
  Realised PnL        : {_fmt_price(realised_pnl)}
  Max drawdown        : {_fmt_pct(drawdown)}
  Trade count         : {trade_count}
  Total commissions   : {_fmt_price(total_commission)}
  Current price       : {_fmt_price(price)}

Describe: (1) overall performance, (2) the agent's current positioning, (3) one risk or opportunity given the current market state. Use plain English — no bullet points."""


# ---------------------------------------------------------------------------
# 3. Market regime summary
# ---------------------------------------------------------------------------

def build_market_regime_prompt(
    tick: int,
    price: float,
    price_change_pct: float,
    volatility: float,
    spread_bps: float,
    bid_depth: float,
    ask_depth: float,
    imbalance: float,
    shock_active: bool,
    shock_regime: str,
    vol_multiplier: float,
    gan_regime: str | None,
    active_panic_agents: int,
    total_agents: int,
) -> str:
    """
    Prompt for a human-readable summary of current market conditions.
    Used in the narrator's regime-change commentary.
    """
    regime_ctx = (
        f"GAN-generated price path regime: {gan_regime}." if gan_regime else
        "GAN regime: randomly sampled per episode."
    )

    shock_ctx = (
        f"ACTIVE SHOCK: {shock_regime} (vol multiplier ×{vol_multiplier:.1f})"
        if shock_active else "No active shock event."
    )

    return f"""You are a real-time market commentator in a simulated multi-agent market. Summarise the current market conditions in 2 sentences for a risk dashboard.

TICK {tick}
  Price            : {_fmt_price(price)}  ({price_change_pct:+.2f}% change)
  Realised vol     : {_fmt_pct(volatility)} (annualised estimate)
  Bid-ask spread   : {spread_bps:.1f} bps
  Book imbalance   : {imbalance:+.3f}  (−1=all asks, +1=all bids)
  Bid depth        : {bid_depth:,.0f} shares
  Ask depth        : {ask_depth:,.0f} shares
  {shock_ctx}
  {regime_ctx}
  Agents in panic  : {active_panic_agents} / {total_agents}

Describe the current regime (calm / stressed / trending / crisis) and the dominant market dynamic visible in the data."""


# ---------------------------------------------------------------------------
# 4. Shock event explanation
# ---------------------------------------------------------------------------

def build_shock_prompt(
    shock_type: str,
    tick: int,
    price_before: float,
    price_after: float,
    spread_before_bps: float,
    spread_after_bps: float,
    volume_spike: float,
    duration_ticks: int,
    agents_affected: int,
) -> str:
    """
    Prompt for a one-paragraph explanation of a shock event's market impact.
    Displayed in the dashboard event log.
    """
    price_chg = (price_after - price_before) / max(price_before, 1e-8) * 100
    shock_descriptions = {
        "flash_crash":      "a sudden, severe price drop driven by automated sell orders",
        "liquidity_crisis": "a collapse in market depth — buyers vanish and spreads widen dramatically",
        "volatility_spike": "a burst of heightened two-way volatility with no clear directional bias",
        "circuit_breaker":  "a trading halt triggered when price movement exceeded the circuit-breaker threshold",
    }
    desc = shock_descriptions.get(shock_type, shock_type.replace("_", " "))

    return f"""You are a market risk analyst. Write one concise paragraph explaining the following shock event to a trader watching a live simulation dashboard.

SHOCK EVENT
  Type          : {shock_type.upper().replace('_', ' ')}
  Description   : {desc}
  Trigger tick  : {tick}
  Duration      : {duration_ticks} ticks

IMPACT METRICS
  Price before  : {_fmt_price(price_before)}
  Price after   : {_fmt_price(price_after)}  ({price_chg:+.2f}%)
  Spread before : {spread_before_bps:.1f} bps
  Spread after  : {spread_after_bps:.1f} bps
  Volume spike  : ×{volume_spike:.1f}
  Agents affected : {agents_affected}

Explain what happened, why it is significant, and what market participants typically do in response to this type of event. Write for a sophisticated but non-specialist audience."""


# ---------------------------------------------------------------------------
# 5. End-of-episode summary
# ---------------------------------------------------------------------------

def build_episode_summary_prompt(
    episode: int,
    total_ticks: int,
    initial_price: float,
    final_price: float,
    total_trades: int,
    shock_events: list[dict],
    agent_stats: list[dict],
    gan_regime: str | None,
) -> str:
    """
    Prompt for a full narrative recap of a completed simulation episode.
    Used in the post-episode report panel.
    """
    price_return = (final_price - initial_price) / max(initial_price, 1e-8) * 100

    # Aggregate by type
    by_type: dict[str, list] = {}
    for a in agent_stats:
        t = a.get("agent_type", "unknown")
        by_type.setdefault(t, []).append(a)

    type_summaries = []
    for atype, agents in by_type.items():
        avg_pnl = sum(a.get("unrealised_pnl", 0) for a in agents) / max(len(agents), 1)
        profitable = sum(1 for a in agents if a.get("unrealised_pnl", 0) > 0)
        type_summaries.append(
            f"  {atype.capitalize():10s}: {len(agents)} agents, "
            f"avg PnL {_fmt_price(avg_pnl)}, {profitable}/{len(agents)} profitable"
        )

    shock_summary = (
        "\n".join(
            f"  Tick {s.get('tick', '?'):4d}: {s.get('type', '?').replace('_', ' ').upper()}"
            for s in shock_events[:10]
        )
        if shock_events else "  None"
    )

    regime_line = f"GAN regime: {gan_regime}" if gan_regime else "GAN regime: randomly sampled"

    return f"""You are a quantitative simulation researcher. Write a 4-5 sentence post-episode debrief for episode {episode}, suitable for a research log.

EPISODE OVERVIEW
  Total ticks     : {total_ticks}
  {regime_line}
  Price: {_fmt_price(initial_price)} → {_fmt_price(final_price)}  ({price_return:+.2f}%)
  Total trades    : {total_trades}

SHOCK EVENTS
{shock_summary}

AGENT PERFORMANCE BY TYPE
{chr(10).join(type_summaries)}

Cover: (1) price path behaviour and any notable events, (2) which agent type performed best/worst and why, (3) lessons for improving agent policy or risk management. Be analytical and specific."""


# ---------------------------------------------------------------------------
# 6. Panic cascade analysis
# ---------------------------------------------------------------------------

def build_cascade_prompt(
    trigger_tick: int,
    price_at_trigger: float,
    peak_panic_agents: int,
    total_panic_agents: int,
    price_trough: float,
    trough_tick: int,
    recovery_tick: int | None,
    total_sell_volume: int,
    cascade_factor: float,
) -> str:
    """
    Prompt for an explanation of a panic cascade event — used in the
    simulation event log when multiple panic agents fire simultaneously.
    """
    drop_pct = (price_at_trigger - price_trough) / max(price_at_trigger, 1e-8) * 100
    recovered = recovery_tick is not None

    return f"""You are a systemic-risk analyst. In 2-3 sentences, explain the following panic cascade event observed in a multi-agent market simulation.

CASCADE DETAILS
  Trigger tick       : {trigger_tick}
  Price at trigger   : {_fmt_price(price_at_trigger)}
  Peak panic agents  : {peak_panic_agents} / {total_panic_agents} panic agents fired
  Price trough       : {_fmt_price(price_trough)} at tick {trough_tick}  (−{abs(drop_pct):.2f}%)
  Recovery           : {'tick ' + str(recovery_tick) if recovered else 'NOT YET RECOVERED'}
  Total sell volume  : {total_sell_volume:,} shares
  Cascade factor     : ×{cascade_factor:.2f}

Describe the cascade mechanics, the severity of the price impact, and whether market structure (book depth, other agent types) provided any stabilisation. Reference real market analogues briefly if helpful."""


# ---------------------------------------------------------------------------
# 7. Live narrator tick commentary
# ---------------------------------------------------------------------------

def build_narrator_prompt(
    tick: int,
    price: float,
    price_change: float,
    dominant_action: str,
    dominant_agent_type: str,
    buy_volume: int,
    sell_volume: int,
    hold_count: int,
    shock_active: bool,
    shock_regime: str,
    notable_events: list[str],
    prior_commentary: str = "",
) -> str:
    """
    Prompt for a single-sentence live tick commentary — like a trading-floor
    announcer calling market action in real time.

    Parameters
    ----------
    prior_commentary : Last sentence of previous commentary for continuity.
    """
    event_lines = (
        "\n".join(f"  - {e}" for e in notable_events)
        if notable_events else "  - None"
    )

    continuity = (
        f'\nCONTINUITY (last comment): "{prior_commentary}"\n'
        if prior_commentary else ""
    )

    shock_note = f"  ⚡ SHOCK ACTIVE: {shock_regime}" if shock_active else ""

    return f"""You are a real-time market simulation narrator. Write exactly ONE sentence of live commentary for tick {tick}. The sentence must be vivid, precise, and under 25 words.{continuity}

TICK {tick} DATA
  Price       : {_fmt_price(price)}  ({price_change:+.4f} change)
  Buy volume  : {buy_volume:,} shares
  Sell volume : {sell_volume:,} shares
  Holding     : {hold_count} agents{shock_note}
  Lead action : {dominant_action.upper()} by {dominant_agent_type} agents

NOTABLE EVENTS THIS TICK
{event_lines}

Respond with ONE sentence only. No prefix, no labels, no punctuation beyond the sentence itself."""

if __name__ == "__main__":
    print("🔍 Testing Prompt Templates...\n")

    # -------------------------------
    # 1. Decision Prompt
    # -------------------------------
    print("---- Decision Prompt ----")
    decision_prompt = build_decision_prompt(
        agent_id=1,
        agent_type="momentum",
        action="BUY_LARGE",
        signal_value=0.0345,
        signal_label="EMA crossover bullish",
        price=102.5,
        portfolio_value=10500,
        cash=8000,
        shares=25,
        unrealised_pnl=500,
        drawdown=0.02,
        reason_tags=["trend_follow", "strong_signal"],
        shock_active=False,
        shock_regime="calm",
        tick=10,
    )
    print(decision_prompt)

    # -------------------------------
    # 2. Portfolio Prompt
    # -------------------------------
    print("\n---- Portfolio Prompt ----")
    portfolio_prompt = build_portfolio_prompt(
        agent_id=1,
        agent_type="value",
        portfolio_value=9800,
        initial_cash=10000,
        cash=5000,
        shares=50,
        unrealised_pnl=-200,
        realised_pnl=100,
        drawdown=0.05,
        trade_count=12,
        total_commission=15,
        tick=20,
        price=98.0,
    )
    print(portfolio_prompt)

    # -------------------------------
    # 3. Shock Prompt
    # -------------------------------
    print("\n---- Shock Prompt ----")
    shock_prompt = build_shock_prompt(
        shock_type="flash_crash",
        tick=30,
        price_before=100,
        price_after=85,
        spread_before_bps=10,
        spread_after_bps=50,
        volume_spike=3.5,
        duration_ticks=5,
        agents_affected=80,
    )
    print(shock_prompt)

    # -------------------------------
    # 4. Episode Summary Prompt
    # -------------------------------
    print("\n---- Episode Summary Prompt ----")
    episode_prompt = build_episode_summary_prompt(
        episode=1,
        total_ticks=100,
        initial_price=100,
        final_price=110,
        total_trades=500,
        shock_events=[{"tick": 30, "type": "flash_crash"}],
        agent_stats=[
            {"agent_type": "momentum", "unrealised_pnl": 200},
            {"agent_type": "value", "unrealised_pnl": -50},
        ],
        gan_regime="trending",
    )
    print(episode_prompt)

    # -------------------------------
    # 5. Cascade Prompt
    # -------------------------------
    print("\n---- Cascade Prompt ----")
    cascade_prompt = build_cascade_prompt(
        trigger_tick=40,
        price_at_trigger=100,
        peak_panic_agents=15,
        total_panic_agents=20,
        price_trough=80,
        trough_tick=45,
        recovery_tick=60,
        total_sell_volume=50000,
        cascade_factor=1.8,
    )
    print(cascade_prompt)

    # -------------------------------
    # 6. Narrator Prompt
    # -------------------------------
    print("\n---- Narrator Prompt ----")
    narrator_prompt = build_narrator_prompt(
        tick=50,
        price=102.5,
        price_change=0.25,
        dominant_action="buy",
        dominant_agent_type="momentum",
        buy_volume=20000,
        sell_volume=15000,
        hold_count=65,
        shock_active=False,
        shock_regime="none",
        notable_events=["Momentum surge", "Low volatility"],
        prior_commentary="Market was stable last tick.",
    )
    print(narrator_prompt)