"""
dashboard/app.py
================
MarketSim live dashboard — connects SimulationRunner to a Dash UI.

FIX SUMMARY (overlay fix)
--------------------------
FIX OVERLAY 1 — Layout:
    The close button is now a permanent top-level layout element (never
    inside the overlay's children). Previously it lived inside the
    overlay children, so when the callback replaced children the
    registered Input disappeared and Dash silently broke the callback.

FIX OVERLAY 2 — _build_analytics_panel:
    Removed id="close-analytics-btn" from the ✕ button inside the panel.
    Changed to pattern-match id {"type":"panel-close-btn","index":0} so
    there is never a duplicate ID conflict with the layout close button.

FIX OVERLAY 3 — select_agent callback:
    Now listens to both the layout close button AND the panel-close-btn
    pattern-match input so either button clears the selection.

FIX OVERLAY 4 — render_analytics_overlay split into TWO callbacks:
    render_analytics_content  → Output("agent-analytics-overlay","children")
    toggle_overlay_style      → Output("agent-analytics-overlay","style")
    A single callback updating both caused race conditions where style
    could be set to "display:flex" before children were populated,
    showing a blank/empty overlay momentarily or not at all.

FIX OVERLAY 5 (NEW) — Ghost click prevention in select_agent:
    Every 800ms refresh() recreates all _agent_row divs with n_clicks=0.
    Dash diffs the DOM, sees "new" components, and the pattern-match
    callback fires — jumping the panel to Agent 0 or reopening it.
    Fix: check ctx.triggered value > 0 before accepting any agent-row
    click. Also changed stray `return None` to `return dash.no_update`
    so unknown triggers never accidentally clear the selection.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import dash
from dash import Input, Output, State, callback, dcc, html, ctx
import plotly.graph_objects as go

from env.market_env import build_market_env
from agents.agent_pool import build_agent_pool
from simulation.runner import build_runner, SimulationRunner

try:
    from genai.narrator import build_narrator
    _narrator_available = True
except ImportError:
    _narrator_available = False

# ─────────────────────────────────────────────────────────────────────────────
# Theme palettes
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(
    bg       = "#080b11",
    surface  = "#0e1118",
    card     = "#10141d",
    border   = "#1a2035",
    nav      = "#0b0e16",
    topbar   = "#0b0e16",
    accent   = "#3b82f6",
    green    = "#22c55e",
    red      = "#ef4444",
    orange   = "#f97316",
    yellow   = "#eab308",
    purple   = "#a855f7",
    muted    = "#4b5675",
    text     = "#e8edf5",
    dim      = "#8b95af",
    shadow   = "rgba(0,0,0,0.5)",
    chart_bg = "#080b11",
    grid     = "#141926",
    input_bg = "#12161f",
)
LIGHT = dict(
    bg       = "#f0f2f8",
    surface  = "#ffffff",
    card     = "#ffffff",
    border   = "#dde2ef",
    nav      = "#fafbff",
    topbar   = "#ffffff",
    accent   = "#2563eb",
    green    = "#16a34a",
    red      = "#dc2626",
    orange   = "#ea580c",
    yellow   = "#ca8a04",
    purple   = "#9333ea",
    muted    = "#94a3b8",
    text     = "#0f172a",
    dim      = "#475569",
    shadow   = "rgba(0,0,0,0.08)",
    chart_bg = "#f8fafc",
    grid     = "#e8ecf5",
    input_bg = "#f1f5fd",
)

FONT_UI   = "'Space Grotesk', 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', 'Fira Mono', monospace"

TYPE_COLOR_DARK = {
    "momentum": DARK["accent"],
    "value":    DARK["green"],
    "noise":    DARK["muted"],
    "panic":    DARK["red"],
}
TYPE_COLOR_LIGHT = {
    "momentum": LIGHT["accent"],
    "value":    LIGHT["green"],
    "noise":    LIGHT["muted"],
    "panic":    LIGHT["red"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Gemini API key — used by both narrator and explainer
# ─────────────────────────────────────────────────────────────────────────────
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "AIzaSyAIjv_Vdv6HgfbiLcvrPMwCRE78FiEjn0Y"

_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

print("\n" + "=" * 60)
print("GenAI API Key Status")
print("=" * 60)
if _GEMINI_KEY:
    print(f"  GEMINI_API_KEY : SET ({len(_GEMINI_KEY)} chars)")
    print("  Narrator  : ENABLED (Gemini)")
    print("  Explainer : ENABLED (Gemini)")
else:
    print("  GEMINI_API_KEY : NOT SET — Narrator and Explainer disabled")
print("=" * 60 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Build singleton runner
# ─────────────────────────────────────────────────────────────────────────────
_env  = build_market_env(agent_id=0, seed=42)
_pool = build_agent_pool(seed=42, load_policies=False)

if _narrator_available:
    _narrator = build_narrator(interval=5, enabled=True)
else:
    _narrator = None

try:
    from genai.explainer import build_explainer
    _explainer = build_explainer(enabled=bool(_GEMINI_KEY))
except ImportError:
    _explainer = None

_genai_enabled = bool(_GEMINI_KEY)

runner: SimulationRunner = build_runner(
    market_env   = _env,
    agent_pool   = _pool,
    narrator     = _narrator,
    explainer    = _explainer,
    enable_genai = _genai_enabled,
    seed         = 42,
)
runner.reset(seed=42)

_sim_lock   = threading.Lock()
_sim_speed  = 1
_sim_paused = False

def _sim_loop():
    while True:
        if not _sim_paused:
            with _sim_lock:
                try:
                    for _ in range(_sim_speed):
                        runner.step()
                except Exception as exc:
                    print(f"[sim_loop] error: {exc}")
        time.sleep(0.5)

threading.Thread(target=_sim_loop, daemon=True).start()


def get_state() -> dict:
    with _sim_lock:
        state = runner.get_dashboard_state()
        return state


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers  (theme-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _theme(is_dark: bool) -> dict:
    return DARK if is_dark else LIGHT

def _type_color(agent_type: str, is_dark: bool) -> str:
    m = TYPE_COLOR_DARK if is_dark else TYPE_COLOR_LIGHT
    C = _theme(is_dark)
    return m.get(agent_type, C["muted"])


def _badge(text: str, color: str) -> html.Span:
    return html.Span(text, style=dict(
        fontSize=9, fontWeight=700, borderRadius=20, padding="2px 10px",
        color=color, border=f"1px solid {color}55",
        background=f"{color}18", textTransform="uppercase",
        letterSpacing=".08em", fontFamily=FONT_UI,
    ))


def _spd_btn(label: str, btn_id: str, C: dict, active: bool = False) -> html.Button:
    return html.Button(label, id=btn_id, n_clicks=0, style=dict(
        background=C["accent"] if active else "transparent",
        border=f"1px solid {C['accent'] if active else C['border']}",
        color="#fff" if active else C["dim"],
        fontFamily=FONT_UI, fontSize=10, fontWeight=600,
        borderRadius=20, padding="3px 14px", cursor="pointer",
        transition="all .2s",
    ))


def _rcard(title: str, children, C: dict, mb: int = 10) -> html.Div:
    return html.Div([
        html.Div(title, style=dict(
            fontSize=9, color=C["muted"], textTransform="uppercase",
            letterSpacing=".12em", marginBottom=10, fontFamily=FONT_UI,
            fontWeight=600,
        )),
        *children,
    ], style=dict(
        background=C["card"],
        border=f"1px solid {C['border']}",
        borderRadius=12, padding="12px 14px", marginBottom=mb,
        boxShadow=f"0 1px 6px {C['shadow']}",
        transition="background .3s, border .3s",
    ))


def _stat_cell(label: str, cell_id: str, C: dict, color: str = None, last: bool = False) -> html.Div:
    val_color = color or C["text"]
    return html.Div([
        html.Div(label, style=dict(
            fontSize=8, color=C["muted"], textTransform="uppercase",
            letterSpacing=".1em", marginBottom=2, fontFamily=FONT_UI, fontWeight=600,
        )),
        html.Div("—", id=cell_id, style=dict(
            fontSize=14, fontWeight=700, color=val_color,
            fontVariantNumeric="tabular-nums", fontFamily=FONT_MONO,
        )),
    ], style=dict(
        flex=1, padding="7px 10px",
        borderRight="none" if last else f"1px solid {C['border']}",
    ))


def _section_header(text: str, C: dict) -> html.Div:
    return html.Div(text, style=dict(
        padding="7px 12px 4px", fontSize=8, color=C["muted"],
        textTransform="uppercase", letterSpacing=".14em", fontWeight=700,
        borderBottom=f"1px solid {C['border']}", fontFamily=FONT_UI,
    ))


def _collapsible_section_header(
    label: str,
    count: int,
    toggle_id: str,
    arrow_id: str,
    dot_color: str,
    C: dict,
) -> html.Div:
    return html.Div(
        id=toggle_id,
        n_clicks=0,
        style=dict(
            display="flex", alignItems="center", justifyContent="space-between",
            padding="6px 12px 5px",
            borderBottom=f"1px solid {C['border']}",
            cursor="pointer",
            userSelect="none",
            transition="background .15s",
        ),
        children=[
            html.Div([
                html.Div(style=dict(
                    width=6, height=6, borderRadius="50%",
                    background=dot_color, flexShrink=0,
                )),
                html.Span(label, style=dict(
                    fontSize=8, color=C["muted"],
                    textTransform="uppercase", letterSpacing=".14em",
                    fontWeight=700, fontFamily=FONT_UI,
                )),
                html.Span(str(count), style=dict(
                    fontSize=7, fontWeight=700,
                    color=dot_color,
                    background=f"{dot_color}18",
                    border=f"1px solid {dot_color}44",
                    borderRadius=20, padding="0px 5px",
                    fontFamily=FONT_MONO,
                )),
            ], style=dict(display="flex", alignItems="center", gap=5)),
            html.Span("▼", id=arrow_id, style=dict(
                fontSize=7, color=C["muted"],
                transition="transform .2s",
            )),
        ],
    )


def _agent_row(agent_id: int, agent_type: str, pnl: float, C: dict, is_dark: bool) -> html.Div:
    color = C["green"] if pnl >= 0 else C["red"]
    dot   = _type_color(agent_type, is_dark)
    if abs(pnl) >= 100:
        pnl_s = f"{'+' if pnl >= 0 else ''}${pnl:,.0f}"
    elif abs(pnl) >= 0.01:
        pnl_s = f"{'+' if pnl >= 0 else ''}${pnl:.2f}"
    else:
        pnl_s = "+$0.00"
    return html.Div(
        id={"type": "agent-row", "index": agent_id},
        n_clicks=0,
        style=dict(
            display="flex", alignItems="center", justifyContent="space-between",
            padding="4px 12px", cursor="pointer",
            transition="background .15s",
        ),
        className="agent-row-clickable",
        children=[
            html.Div([
                html.Div(style=dict(
                    width=6, height=6, borderRadius="50%",
                    background=dot, flexShrink=0, marginTop=1,
                )),
                html.Div(f"Agent {agent_id}", style=dict(
                    fontSize=10, color=C["text"], fontFamily=FONT_UI,
                )),
            ], style=dict(display="flex", alignItems="flex-start", gap=6)),
            html.Span(pnl_s, style=dict(
                fontSize=10, fontWeight=700, color=color, fontFamily=FONT_MONO,
            )),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Analytics Panel
# ─────────────────────────────────────────────────────────────────────────────

def _build_analytics_panel(agent_data: dict | None, C: dict) -> html.Div:
    is_dark = C is DARK

    if agent_data is None:
        return html.Div()

    agent_id   = agent_data.get("agent_id", "?")
    agent_type = agent_data.get("agent_type", "noise")
    dot_color  = _type_color(agent_type, is_dark)

    realised   = agent_data.get("realised_pnl", 0.0)
    unrealised = agent_data.get("unrealised_pnl", 0.0)
    total_pnl  = realised + unrealised
    portfolio_val = agent_data.get("portfolio_value", 100_000)
    last_action   = agent_data.get("last_action", "hold").upper()
    last_signal   = agent_data.get("last_signal", 0.0)
    explanation   = agent_data.get("latest_explanation", "")
    episode_id    = agent_data.get("episode_id", 1)
    tick          = agent_data.get("current_tick", 0)

    trade_history = agent_data.get("trade_history", [])
    wins   = [t for t in trade_history if t.get("pnl", 0) > 0]
    losses = [t for t in trade_history if t.get("pnl", 0) < 0]
    total_trades = len(trade_history)
    win_rate  = (len(wins)  / total_trades * 100) if total_trades else 0
    loss_rate = (len(losses) / total_trades * 100) if total_trades else 0
    avg_win   = (sum(t["pnl"] for t in wins)   / len(wins))   if wins   else 0
    avg_loss  = (sum(t["pnl"] for t in losses) / len(losses)) if losses else 0

    actions    = [t.get("action", "hold") for t in trade_history]
    buy_count  = actions.count("buy")
    sell_count = actions.count("sell")
    hold_count = actions.count("hold")
    max_action = max(buy_count + sell_count + hold_count, 1)

    pnl_curve     = agent_data.get("pnl_history", [0])
    pnl_up        = total_pnl >= 0
    pnl_color_str = C["green"] if pnl_up else C["red"]
    pnl_pct       = (total_pnl / (portfolio_val - total_pnl) * 100) if (portfolio_val - total_pnl) else 0

    pnl_color = C["green"] if total_pnl >= 0 else C["red"]
    act_color = {"BUY": C["green"], "SELL": C["red"]}.get(last_action, C["muted"])

    def _micro_stat(label, value, color=None):
        return html.Div([
            html.Div(label, style=dict(
                fontSize=8, color=C["muted"], textTransform="uppercase",
                letterSpacing=".12em", fontWeight=700, marginBottom=5,
                fontFamily=FONT_UI,
            )),
            html.Div(value, style=dict(
                fontSize=17, fontWeight=800,
                fontFamily=FONT_MONO, letterSpacing="-.5px",
                color=color or C["text"],
            )),
        ], style=dict(
            background=C["input_bg"],
            border=f"1px solid {C['border']}",
            borderRadius=10, padding="10px 12px",
        ))

    def _action_bar_row(label, count, total, color):
        w = f"{count / total * 100:.0f}%" if total else "0%"
        return html.Div([
            html.Div(label, style=dict(
                fontSize=9, color=C["dim"], width=32, paddingTop=2,
                fontFamily=FONT_UI,
            )),
            html.Div(style=dict(
                flex=1, background=C["border"], borderRadius=99,
                height=14, overflow="hidden",
            ), children=[
                html.Div(style=dict(
                    height=14, width=w, borderRadius=99,
                    background=f"{color}30",
                    display="flex", alignItems="center", paddingLeft=7,
                ), children=[
                    html.Span(str(count), style=dict(
                        fontSize=8, fontWeight=700, color=color,
                        fontFamily=FONT_MONO,
                    )),
                ]),
            ]),
        ], style=dict(display="flex", gap=6, marginBottom=6))

    def _trade_pill(action):
        a = action.upper()
        if a == "BUY":
            return html.Span("BUY", style=dict(
                borderRadius=20, padding="1px 7px", fontSize=8, fontWeight=700,
                textTransform="uppercase", letterSpacing=".05em",
                background=f"{C['green']}18", color=C["green"],
                border=f"1px solid {C['green']}40",
            ))
        elif a == "SELL":
            return html.Span("SELL", style=dict(
                borderRadius=20, padding="1px 7px", fontSize=8, fontWeight=700,
                textTransform="uppercase", letterSpacing=".05em",
                background=f"{C['red']}18", color=C["red"],
                border=f"1px solid {C['red']}40",
            ))
        else:
            return html.Span("HOLD", style=dict(
                borderRadius=20, padding="1px 7px", fontSize=8, fontWeight=700,
                textTransform="uppercase", letterSpacing=".05em",
                background=f"{C['muted']}25", color=C["dim"],
                border=f"1px solid {C['muted']}40",
            ))

    mini_fig = go.Figure(go.Scatter(
        y=pnl_curve,
        mode="lines",
        line=dict(color=pnl_color_str, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.06)" if pnl_up else "rgba(239,68,68,0.06)",
        hoverinfo="y",
        hovertemplate="$%{y:.0f}<extra></extra>",
    ))
    mini_fig.update_layout(
        paper_bgcolor=C["input_bg"],
        plot_bgcolor=C["input_bg"],
        margin=dict(l=0, r=44, t=4, b=0),
        showlegend=False,
        height=80,
        xaxis=dict(visible=False),
        yaxis=dict(
            side="right", showgrid=True, gridcolor=C["grid"], gridwidth=0.5,
            tickfont=dict(color=C["muted"], size=8, family="JetBrains Mono"),
            tickprefix="$", tickformat=".0f", nticks=3,
        ),
        font=dict(family=FONT_MONO),
    )

    recent_trades = trade_history[-5:] if trade_history else []
    trade_rows = []
    for i, t in enumerate(reversed(recent_trades)):
        pnl_val = t.get("pnl", None)
        pnl_td  = "—"
        pnl_col = C["muted"]
        if pnl_val is not None:
            pnl_td  = f"{'+' if pnl_val >= 0 else ''}${pnl_val:.0f}"
            pnl_col = C["green"] if pnl_val >= 0 else C["red"]
        sig_val = t.get("signal", 0.0)
        sig_col = C["green"] if sig_val > 0 else C["red"] if sig_val < 0 else C["muted"]
        trade_rows.append(html.Tr([
            html.Td(str(t.get("trade_num", total_trades - i)),
                    style=dict(color=C["text"])),
            html.Td(_trade_pill(t.get("action", "hold"))),
            html.Td(str(t.get("tick", "—")), style=dict(color=C["dim"])),
            html.Td(f"${t.get('price', 0):.2f}", style=dict(color=C["dim"])),
            html.Td(f"{sig_val:+.4f}", style=dict(color=sig_col)),
            html.Td(pnl_td, style=dict(color=pnl_col, textAlign="right")),
        ], style=dict(borderBottom=f"1px solid {C['border']}20")))

    if not trade_rows:
        trade_rows = [html.Tr([
            html.Td("No trades yet", colSpan=6, style=dict(
                color=C["muted"], fontSize=9, padding="6px 0", textAlign="center",
            ))
        ])]

    panel_content = html.Div([

        # Header
        html.Div([
            html.Div([
                html.Div(style=dict(
                    width=9, height=9, borderRadius="50%",
                    background=dot_color, marginRight=8,
                )),
                html.Span(f"Agent {agent_id}", style=dict(
                    fontSize=15, fontWeight=700, color=C["text"],
                    fontFamily=FONT_UI,
                )),
                html.Span(agent_type.upper(), style=dict(
                    fontSize=9, color=C["accent"], fontWeight=600,
                    background=f"{C['accent']}18",
                    border=f"1px solid {C['accent']}40",
                    borderRadius=20, padding="2px 8px",
                    textTransform="uppercase", letterSpacing=".1em",
                    marginLeft=8, fontFamily=FONT_UI,
                )),
            ], style=dict(display="flex", alignItems="center")),
            html.Div([
                html.Span(f"EP {episode_id} · TICK {tick}", style=dict(
                    fontSize=9, color=C["muted"], fontFamily=FONT_MONO,
                    marginRight=8,
                )),
                html.Button("✕",
                    id={"type": "panel-close-btn", "index": 0},
                    n_clicks=0,
                    style=dict(
                        background="transparent",
                        border=f"1px solid {C['border']}",
                        color=C["muted"], borderRadius=6,
                        width=26, height=26, cursor="pointer", fontSize=13,
                        display="flex", alignItems="center", justifyContent="center",
                        fontFamily="sans-serif",
                    )
                ),
            ], style=dict(display="flex", alignItems="center")),
        ], style=dict(
            background=C["nav"],
            borderBottom=f"1px solid {C['border']}",
            padding="14px 18px",
            display="flex", alignItems="center", justifyContent="space-between",
        )),

        # Body
        html.Div([

            # Stat grid
            html.Div([
                _micro_stat("Total PnL",
                            f"{'+' if total_pnl >= 0 else ''}${total_pnl:,.0f}",
                            pnl_color),
                _micro_stat("Portfolio",
                            f"${portfolio_val:,.0f}",
                            C["dim"]),
                _micro_stat("Total Trades",
                            str(total_trades),
                            C["accent"]),
                html.Div([
                    html.Div("Last Action", style=dict(
                        fontSize=8, color=C["muted"], textTransform="uppercase",
                        letterSpacing=".12em", fontWeight=700, marginBottom=5,
                        fontFamily=FONT_UI,
                    )),
                    html.Div(last_action, style=dict(
                        fontSize=13, fontWeight=800, color=act_color,
                        fontFamily=FONT_MONO, marginTop=2,
                    )),
                    html.Div(f"Signal {last_signal:+.4f}", style=dict(
                        fontSize=8, color=C["muted"], fontFamily=FONT_MONO,
                        marginTop=2,
                    )),
                ], style=dict(
                    background=C["input_bg"],
                    border=f"1px solid {C['border']}",
                    borderRadius=10, padding="10px 12px",
                )),
            ], style=dict(
                display="grid",
                gridTemplateColumns="repeat(4, minmax(0,1fr))",
                gap=8, marginBottom=14,
            )),

            # PnL Curve
            html.Div([
                html.Div([
                    html.Span("PnL Curve", style=dict(
                        fontSize=8, color=C["muted"], textTransform="uppercase",
                        letterSpacing=".12em", fontWeight=700, fontFamily=FONT_UI,
                    )),
                    html.Span(f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%", style=dict(
                        fontSize=9, color=pnl_color, fontFamily=FONT_MONO, fontWeight=700,
                    )),
                ], style=dict(
                    display="flex", justifyContent="space-between",
                    alignItems="center", marginBottom=10,
                )),
                dcc.Graph(
                    id="agent-pnl-mini-chart",
                    figure=mini_fig,
                    config=dict(displayModeBar=False),
                    style=dict(height=80),
                ),
            ], style=dict(
                background=C["input_bg"],
                border=f"1px solid {C['border']}",
                borderRadius=10, padding="12px 14px",
                marginBottom=14,
            )),

            # Win/Loss + Action Breakdown
            html.Div([

                # Win/Loss card
                html.Div([
                    html.Div("Win / Loss Ratio", style=dict(
                        fontSize=8, color=C["muted"], textTransform="uppercase",
                        letterSpacing=".12em", fontWeight=700, marginBottom=10,
                        fontFamily=FONT_UI,
                    )),
                    html.Div(style=dict(
                        height=6, borderRadius=99, overflow="hidden",
                        background=C["border"], display="flex", marginBottom=8,
                    ), children=[
                        html.Div(style=dict(
                            height=6, width=f"{win_rate:.1f}%",
                            background=C["green"], borderRadius="99px 0 0 99px",
                        )),
                        html.Div(style=dict(
                            height=6, width=f"{loss_rate:.1f}%",
                            background=C["red"], borderRadius="0 99px 99px 0",
                        )),
                    ]),
                    html.Div([
                        html.Div([
                            html.Div(f"{win_rate:.1f}%", style=dict(
                                fontSize=18, fontWeight=800, color=C["green"],
                                fontFamily=FONT_MONO,
                            )),
                            html.Div("Win Rate", style=dict(
                                fontSize=8, color=C["muted"], textTransform="uppercase",
                                letterSpacing=".1em", fontWeight=700, marginTop=2,
                                fontFamily=FONT_UI,
                            )),
                            html.Div(f"{len(wins)} trades", style=dict(
                                fontSize=9, color=C["muted"], fontFamily=FONT_MONO,
                                marginTop=1,
                            )),
                        ], style=dict(textAlign="center")),
                        html.Div(style=dict(width=1, background=C["border"], margin="0 8px")),
                        html.Div([
                            html.Div(f"{loss_rate:.1f}%", style=dict(
                                fontSize=18, fontWeight=800, color=C["red"],
                                fontFamily=FONT_MONO,
                            )),
                            html.Div("Loss Rate", style=dict(
                                fontSize=8, color=C["muted"], textTransform="uppercase",
                                letterSpacing=".1em", fontWeight=700, marginTop=2,
                                fontFamily=FONT_UI,
                            )),
                            html.Div(f"{len(losses)} trades", style=dict(
                                fontSize=9, color=C["muted"], fontFamily=FONT_MONO,
                                marginTop=1,
                            )),
                        ], style=dict(textAlign="center")),
                    ], style=dict(display="flex", justifyContent="space-around")),
                    html.Div([
                        html.Div([
                            html.Div("Avg Win", style=dict(
                                fontSize=8, color=C["muted"], textTransform="uppercase",
                                letterSpacing=".1em", fontWeight=700,
                                fontFamily=FONT_UI,
                            )),
                            html.Div(f"+${avg_win:.1f}", style=dict(
                                fontSize=11, fontWeight=700, color=C["green"],
                                fontFamily=FONT_MONO,
                            )),
                        ]),
                        html.Div([
                            html.Div("Avg Loss", style=dict(
                                fontSize=8, color=C["muted"], textTransform="uppercase",
                                letterSpacing=".1em", fontWeight=700, textAlign="right",
                                fontFamily=FONT_UI,
                            )),
                            html.Div(f"-${abs(avg_loss):.1f}", style=dict(
                                fontSize=11, fontWeight=700, color=C["red"],
                                fontFamily=FONT_MONO, textAlign="right",
                            )),
                        ]),
                    ], style=dict(
                        display="flex", justifyContent="space-between",
                        marginTop=10, paddingTop=8,
                        borderTop=f"1px solid {C['border']}",
                    )),
                ], style=dict(
                    background=C["input_bg"],
                    border=f"1px solid {C['border']}",
                    borderRadius=10, padding="12px 14px",
                )),

                # Action Breakdown card
                html.Div([
                    html.Div("Action Breakdown", style=dict(
                        fontSize=8, color=C["muted"], textTransform="uppercase",
                        letterSpacing=".12em", fontWeight=700, marginBottom=10,
                        fontFamily=FONT_UI,
                    )),
                    _action_bar_row("Buy",  buy_count,  max_action, C["green"]),
                    _action_bar_row("Sell", sell_count, max_action, C["red"]),
                    _action_bar_row("Hold", hold_count, max_action, C["muted"]),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span("Realised PnL", style=dict(
                                    fontSize=8, color=C["muted"], textTransform="uppercase",
                                    letterSpacing=".1em", fontWeight=700, fontFamily=FONT_UI,
                                )),
                                html.Span(
                                    f"{'+' if realised >= 0 else ''}${realised:.0f}",
                                    style=dict(
                                        fontSize=11, fontWeight=700,
                                        color=C["green"] if realised >= 0 else C["red"],
                                        fontFamily=FONT_MONO,
                                    )
                                ),
                            ], style=dict(display="flex", justifyContent="space-between")),
                            html.Div([
                                html.Span("Unrealised PnL", style=dict(
                                    fontSize=8, color=C["muted"], textTransform="uppercase",
                                    letterSpacing=".1em", fontWeight=700, fontFamily=FONT_UI,
                                )),
                                html.Span(
                                    f"{'+' if unrealised >= 0 else ''}${unrealised:.0f}",
                                    style=dict(
                                        fontSize=11, fontWeight=700,
                                        color=C["green"] if unrealised >= 0 else C["red"],
                                        fontFamily=FONT_MONO,
                                    )
                                ),
                            ], style=dict(
                                display="flex", justifyContent="space-between", marginTop=4,
                            )),
                        ]),
                    ], style=dict(
                        borderTop=f"1px solid {C['border']}",
                        marginTop=8, paddingTop=8,
                    )),
                ], style=dict(
                    background=C["input_bg"],
                    border=f"1px solid {C['border']}",
                    borderRadius=10, padding="12px 14px",
                )),

            ], style=dict(
                display="grid", gridTemplateColumns="1fr 1fr",
                gap=8, marginBottom=14,
            )),

            # Trade History
            html.Div([
                html.Div("Recent Trade History", style=dict(
                    fontSize=8, color=C["muted"], textTransform="uppercase",
                    letterSpacing=".12em", fontWeight=700, marginBottom=10,
                    fontFamily=FONT_UI,
                )),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(col, style=dict(
                            color=C["muted"], textTransform="uppercase",
                            letterSpacing=".1em", fontWeight=700, fontSize=9,
                            padding="0 0 7px",
                            textAlign="right" if col == "PnL" else "left",
                            borderBottom=f"1px solid {C['border']}",
                            fontFamily=FONT_UI,
                        )) for col in ["#", "Action", "Tick", "Price", "Signal", "PnL"]
                    ])),
                    html.Tbody(trade_rows, style=dict(fontFamily=FONT_MONO, fontSize=9)),
                ], style=dict(width="100%", borderCollapse="collapse")),
            ], style=dict(
                background=C["input_bg"],
                border=f"1px solid {C['border']}",
                borderRadius=10, padding="12px 14px",
            )),

            # AI Explanation
            *([] if not explanation else [
                html.Div([
                    html.Div("AI Explanation", style=dict(
                        fontSize=8, color=C["muted"], textTransform="uppercase",
                        letterSpacing=".12em", fontWeight=700, marginBottom=6,
                        fontFamily=FONT_UI,
                    )),
                    html.Div(explanation, style=dict(
                        fontSize=9, color=C["dim"], lineHeight=1.6,
                        fontFamily=FONT_UI,
                    )),
                ], style=dict(
                    background=C["input_bg"],
                    border=f"1px solid {C['border']}",
                    borderRadius=10, padding="12px 14px",
                    marginTop=8,
                )),
            ]),

        ], style=dict(padding="14px 18px", overflowY="auto",
                      maxHeight="calc(100vh - 160px)")),

    ], style=dict(
        background=C["surface"],
        border=f"1px solid {C['border']}",
        borderRadius=14, overflow="hidden",
        width="100%", maxWidth=660,
        fontFamily=FONT_UI,
        boxShadow=f"0 20px 60px {C['shadow']}",
    ))

    return panel_content


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="MarketSim",
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap"
    ],
)

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100vh; width: 100vw; overflow: hidden; }
  body > #react-entry-point,
  body > #react-entry-point > #_dash-app-content { height: 100%; }
  .js-plotly-plot, .plot-container, .plotly { height: 100% !important; }
  .agent-section-toggle:hover { background: rgba(255,255,255,0.03) !important; }
  .agent-row-clickable:hover {
    background: rgba(59,130,246,0.06) !important;
    border-radius: 6px;
  }
  #left-nav, #right-panel {
    scrollbar-width: thin;
    scrollbar-color: #1e2640 transparent;
  }
  #left-nav::-webkit-scrollbar,
  #right-panel::-webkit-scrollbar { width: 3px; }
  #left-nav::-webkit-scrollbar-track,
  #right-panel::-webkit-scrollbar-track { background: transparent; }
  #left-nav::-webkit-scrollbar-thumb,
  #right-panel::-webkit-scrollbar-thumb {
    background: #1e2640; border-radius: 99px; transition: background .2s;
  }
  #left-nav:hover::-webkit-scrollbar-thumb,
  #right-panel:hover::-webkit-scrollbar-thumb { background: #2e3d5e; }
  #left-nav::-webkit-scrollbar-thumb:hover,
  #right-panel::-webkit-scrollbar-thumb:hover { background: #3b82f6; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

_TOPBAR_H   = 44
_PRICEROW_H = 46
_STATBAR_H  = 48
_BOTTOM_H   = 155
_CHART_H    = f"calc(100vh - {_TOPBAR_H + _PRICEROW_H + _STATBAR_H + _BOTTOM_H}px)"

C0 = DARK

_AGENT_GROUPS = [
    ("Momentum", "toggle-momentum", "arrow-momentum", "collapse-momentum", "accent",  30),
    ("Value",    "toggle-value",    "arrow-value",    "collapse-value",    "green",   30),
    ("Noise",    "toggle-noise",    "arrow-noise",    "collapse-noise",    "muted",   20),
    ("Panic",    "toggle-panic",    "arrow-panic",    "collapse-panic",    "red",     20),
]

app.layout = html.Div(id="root", style=dict(
    background=C0["bg"], color=C0["text"],
    fontFamily=FONT_UI,
    height="100vh", width="100vw",
    overflow="hidden",
    display="flex", flexDirection="column",
    transition="background .3s, color .3s",
), children=[

    dcc.Interval(id="poll", interval=800),
    dcc.Store(id="theme-store", data="dark"),
    dcc.Store(id="speed-store", data=1),
    dcc.Store(id="selected-agent-id", data=None),

    dcc.Store(id="collapse-momentum-state", data=False),
    dcc.Store(id="collapse-value-state",    data=False),
    dcc.Store(id="collapse-noise-state",    data=False),
    dcc.Store(id="collapse-panic-state",    data=False),

    # TOP BAR
    html.Div(id="topbar", style=dict(
        background=C0["topbar"],
        borderBottom=f"1px solid {C0['border']}",
        padding="0 16px", display="flex", alignItems="center",
        gap=10, height=44, flexShrink=0,
        boxShadow=f"0 1px 0 {C0['border']}",
        transition="background .3s, border .3s",
    ), children=[
        html.Div([
            html.Span("MS", style=dict(
                background=C0["accent"], color="#fff", fontSize=10,
                fontWeight=800, borderRadius=6, padding="2px 6px",
                letterSpacing="-.5px",
            )),
            html.Span("MarketSim", style=dict(
                fontSize=12, fontWeight=700, color=C0["text"],
                marginLeft=6, letterSpacing="-.3px",
            )),
        ], style=dict(display="flex", alignItems="center")),

        html.Div(style=dict(width=1, height=20, background=C0["border"], margin="0 4px")),

        html.Span("SYN-1", style=dict(
            fontSize=11, fontWeight=600, color=C0["dim"],
            letterSpacing=".05em",
        )),
        html.Span(id="tb-price", style=dict(
            fontSize=13, fontWeight=800, color=C0["green"],
            fontFamily=FONT_MONO,
        )),
        html.Span(id="tb-chg", style=dict(fontSize=10, color=C0["green"], fontFamily=FONT_MONO)),

        html.Div(style=dict(width=1, height=20, background=C0["border"], margin="0 4px")),

        html.Div([
            html.Span("EP", style=dict(
                fontSize=8, color=C0["muted"], fontWeight=700,
                letterSpacing=".12em", marginRight=4,
            )),
            html.Span(id="tb-ep-num", style=dict(
                fontSize=11, color=C0["dim"], fontFamily=FONT_MONO, fontWeight=600,
            )),
            html.Span("·", style=dict(color=C0["border"], margin="0 4px")),
            html.Span("TICK", style=dict(
                fontSize=8, color=C0["muted"], fontWeight=700,
                letterSpacing=".12em", marginRight=4,
            )),
            html.Span(id="tb-tick-num", style=dict(
                fontSize=11, color=C0["dim"], fontFamily=FONT_MONO, fontWeight=600,
            )),
        ], title=(
            "Episode: one complete market simulation run.\n"
            "Resets when price diverges beyond safety bounds or max ticks reached.\n"
            "Each episode starts fresh — new prices, reset agent portfolios.\n"
            "Tick: the current step within this episode."
        ), style=dict(
            display="flex", alignItems="center",
            cursor="help",
            borderBottom=f"1px dashed {C0['muted']}",
            paddingBottom=1,
        )),

        html.Div(id="tb-regime", children=[_badge("calm", C0["accent"])]),

        html.Div([
            html.Span("SPEED", style=dict(
                fontSize=8, color=C0["muted"], fontWeight=700,
                letterSpacing=".1em", marginRight=6,
            )),
            html.Div([
                _spd_btn("1×", "spd-1", C0, active=True),
                _spd_btn("2×", "spd-2", C0),
                _spd_btn("3×", "spd-3", C0),
            ], style=dict(display="flex", gap=3,
                background=C0["input_bg"], borderRadius=20, padding=3,
                border=f"1px solid {C0['border']}")),
        ], style=dict(marginLeft="auto", display="flex", alignItems="center")),

        html.Button(
            "☀", id="theme-toggle", n_clicks=0,
            title="Toggle light / dark mode",
            style=dict(
                background="transparent",
                border=f"1px solid {C0['border']}",
                color=C0["dim"], fontSize=14, borderRadius=8,
                width=32, height=32, cursor="pointer",
                display="flex", alignItems="center", justifyContent="center",
                marginLeft=8, transition="all .2s",
                fontFamily="sans-serif",
            ),
        ),
    ]),

    # BODY
    html.Div(id="body", style=dict(
        display="flex",
        height=f"calc(100vh - {_TOPBAR_H}px)",
        overflow="hidden",
        flexShrink=0,
    ), children=[

        # LEFT NAV
        html.Div(id="left-nav", style=dict(
            width=172, background=C0["nav"],
            borderRight=f"1px solid {C0['border']}",
            display="flex", flexDirection="column",
            height=f"calc(100vh - {_TOPBAR_H}px)",
            overflowY="auto", overflowX="hidden",
            flexShrink=0,
            transition="background .3s, border .3s",
        ), children=[
            _section_header("Global Market", C0),

            html.Div([
                html.Div([
                    html.Div(style=dict(
                        width=8, height=8, borderRadius="50%",
                        background=C0["accent"],
                    )),
                    html.Div([
                        html.Div("Overview", style=dict(
                            fontSize=10, color=C0["text"], fontWeight=600,
                        )),
                        html.Div("100 agents", style=dict(fontSize=8, color=C0["muted"])),
                    ]),
                ], style=dict(display="flex", alignItems="center", gap=7)),
                html.Div(id="gm-pct", children="+0.0%", style=dict(
                    fontSize=8, fontWeight=700, borderRadius=20, padding="1px 7px",
                    color=C0["green"], border=f"1px solid {C0['green']}44",
                    background=f"{C0['green']}12", fontFamily=FONT_MONO,
                )),
            ], style=dict(
                display="flex", alignItems="center", justifyContent="space-between",
                padding="5px 12px",
                borderLeft=f"2px solid {C0['accent']}",
                background=f"{C0['accent']}08",
            )),

            _collapsible_section_header(
                "Momentum", 30, "toggle-momentum", "arrow-momentum", C0["accent"], C0,
            ),
            html.Div(id="collapse-momentum", style={"display": "block"},
                     children=[html.Div(id="agents-momentum")]),

            _collapsible_section_header(
                "Value", 30, "toggle-value", "arrow-value", C0["green"], C0,
            ),
            html.Div(id="collapse-value", style={"display": "block"},
                     children=[html.Div(id="agents-value")]),

            _collapsible_section_header(
                "Noise", 20, "toggle-noise", "arrow-noise", C0["muted"], C0,
            ),
            html.Div(id="collapse-noise", style={"display": "block"},
                     children=[html.Div(id="agents-noise")]),

            _collapsible_section_header(
                "Panic", 20, "toggle-panic", "arrow-panic", C0["red"], C0,
            ),
            html.Div(id="collapse-panic", style={"display": "block"},
                     children=[html.Div(id="agents-panic")]),
        ]),

        # CENTER
        html.Div(id="center-col", style=dict(
            flex=1, display="flex", flexDirection="column",
            minWidth=0,
            height=f"calc(100vh - {_TOPBAR_H}px)",
            overflow="hidden",
        ), children=[

            html.Div(style=dict(
                padding="0 18px",
                display="flex", alignItems="center", gap=10,
                height=_PRICEROW_H, flexShrink=0,
                borderBottom=f"1px solid {C0['border']}",
            ), children=[
                html.Span(id="ph-price", style=dict(
                    fontSize=28, fontWeight=800, color=C0["text"],
                    letterSpacing="-1.5px", fontVariantNumeric="tabular-nums",
                    fontFamily=FONT_MONO,
                )),
                html.Span(id="ph-chg", style=dict(
                    fontSize=12, fontWeight=700, color=C0["green"],
                    fontFamily=FONT_MONO,
                )),
                html.Span(id="ph-meta", style=dict(
                    fontSize=9, color=C0["muted"], marginLeft=6,
                    fontFamily=FONT_MONO,
                )),
            ]),

            html.Div(id="chart-wrap", style=dict(
                height=_CHART_H,
                flexShrink=0,
                padding="4px 18px 4px",
                overflow="hidden",
            ), children=[
                dcc.Graph(
                    id="main-chart",
                    config=dict(displayModeBar=False),
                    style=dict(height="100%", width="100%"),
                ),
            ]),

            html.Div(id="stat-bar", style=dict(
                display="flex", flexShrink=0,
                height=_STATBAR_H,
                borderTop=f"1px solid {C0['border']}",
                borderBottom=f"1px solid {C0['border']}",
                background=C0["surface"],
                transition="background .3s",
            ), children=[
                _stat_cell("Volume",   "sv-trades", C0),
                _stat_cell("Bid Depth","sv-buy",    C0, color=C0["green"]),
                _stat_cell("Ask Depth","sv-sell",   C0, color=C0["red"]),
                _stat_cell("Panic",    "sv-panic",  C0, color=C0["orange"]),
                _stat_cell("Shocks",   "sv-shocks", C0),
                _stat_cell("Spread",   "sv-spread", C0, last=True),
            ]),

            html.Div(id="bottom-row", style=dict(
                display="flex",
                height=_BOTTOM_H,
                flexShrink=0,
                overflow="hidden",
                borderTop=f"1px solid {C0['border']}",
            ), children=[
                html.Div(style=dict(
                    flex=1, padding="8px 14px", overflow="hidden",
                    borderRight=f"1px solid {C0['border']}",
                ), children=[
                    html.Div("Order Book", style=dict(
                        fontSize=8, color=C0["muted"], textTransform="uppercase",
                        letterSpacing=".12em", marginBottom=6, fontWeight=700,
                    )),
                    html.Div(id="order-book"),
                ]),
                html.Div(style=dict(
                    flex=1, padding="8px 14px", overflow="hidden",
                ), children=[
                    html.Div([
                        html.Span("GenAI Narrator", style=dict(
                            fontSize=8, color=C0["muted"], textTransform="uppercase",
                            letterSpacing=".12em", fontWeight=700,
                        )),
                        html.Span("Gemini ✦", style=dict(
                            background=C0["purple"], color="#fff", fontSize=8,
                            fontWeight=700, borderRadius=4, padding="1px 6px",
                            marginLeft=6,
                        )),
                    ], style=dict(
                        display="flex", alignItems="center", marginBottom=6,
                    )),
                    html.Div(id="narrator-feed"),
                ]),
            ]),
        ]),

        # RIGHT PANEL
        html.Div(id="right-panel", style=dict(
            width=215, background=C0["nav"],
            borderLeft=f"1px solid {C0['border']}",
            padding="10px",
            height=f"calc(100vh - {_TOPBAR_H}px)",
            overflowY="auto", flexShrink=0,
            transition="background .3s, border .3s",
        ), children=[

            _rcard("Simulation", [
                html.Div([
                    html.Button("⏸ Pause", id="btn-pause", n_clicks=0, style=dict(
                        flex=1, background="transparent",
                        border=f"1px solid {C0['border']}",
                        color=C0["dim"], fontFamily=FONT_UI, fontSize=10,
                        fontWeight=600, borderRadius=8, padding="6px 0",
                        cursor="pointer", transition="all .2s",
                    )),
                    html.Button("⏹ Stop", id="btn-stop", n_clicks=0, style=dict(
                        flex=1, background=f"{C0['red']}10",
                        border=f"1px solid {C0['red']}44",
                        color=C0["red"], fontFamily=FONT_UI, fontSize=10,
                        fontWeight=600, borderRadius=8, padding="6px 0",
                        cursor="pointer", transition="all .2s",
                    )),
                ], style=dict(display="flex", gap=6)),
            ], C=C0),

            _rcard("Inject Shock", [
                html.Div([
                    html.Button("💥 Flash Crash", id="shock-flash", n_clicks=0, style=dict(
                        background=f"{C0['red']}10", border=f"1px solid {C0['red']}44",
                        color=C0["red"], fontFamily=FONT_UI, fontSize=9, fontWeight=600,
                        borderRadius=8, padding="5px 8px", cursor="pointer", transition="all .2s")),
                    html.Button("💧 Liquidity", id="shock-liq", n_clicks=0, style=dict(
                        background=f"{C0['orange']}10", border=f"1px solid {C0['orange']}44",
                        color=C0["orange"], fontFamily=FONT_UI, fontSize=9, fontWeight=600,
                        borderRadius=8, padding="5px 8px", cursor="pointer", transition="all .2s")),
                    html.Button("📈 Vol Spike", id="shock-vol", n_clicks=0, style=dict(
                        background=f"{C0['yellow']}10", border=f"1px solid {C0['yellow']}44",
                        color=C0["yellow"], fontFamily=FONT_UI, fontSize=9, fontWeight=600,
                        borderRadius=8, padding="5px 8px", cursor="pointer", transition="all .2s")),
                    html.Button("📰 News Shock", id="shock-news", n_clicks=0, style=dict(
                        background=f"{C0['purple']}10", border=f"1px solid {C0['purple']}44",
                        color=C0["purple"], fontFamily=FONT_UI, fontSize=9, fontWeight=600,
                        borderRadius=8, padding="5px 8px", cursor="pointer", transition="all .2s")),
                ], style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap=6)),
            ], C=C0),

            _rcard("Cascade Meter", [
                html.Div([
                    html.Div([
                        html.Span("😱 Panic agents", style=dict(fontSize=10, color=C0["dim"])),
                        html.Span(id="rp-panic", style=dict(
                            fontSize=10, fontWeight=700, color=C0["red"],
                            fontFamily=FONT_MONO,
                        )),
                    ], style=dict(
                        display="flex", justifyContent="space-between", marginBottom=8,
                    )),
                    html.Div(style=dict(
                        background=C0["input_bg"], borderRadius=99,
                        height=6, overflow="hidden",
                    ), children=[
                        html.Div(id="rp-pm", style=dict(
                            height=6, background=C0["muted"],
                            borderRadius=99, width="0%",
                            transition="width .4s, background .4s",
                        )),
                    ]),
                ]),
            ], C=C0),

            _rcard("Portfolio · by type", [
                html.Div(id="portfolio-bars"),
            ], C=C0),

            _rcard("Top Agent", [
                html.Div(id="agent-detail"),
            ], C=C0, mb=0),
        ]),
    ]),

    # Permanent hidden close button — never inside overlay children
    html.Button(
        "",
        id="close-analytics-btn",
        n_clicks=0,
        style=dict(display="none"),
    ),
    html.Div(
        id="agent-analytics-overlay",
        style=dict(display="none"),
        children=[],
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# Collapse toggle callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _make_toggle_callback(group: str, color_key: str):
    toggle_id   = f"toggle-{group}"
    store_id    = f"collapse-{group}-state"
    collapse_id = f"collapse-{group}"
    arrow_id    = f"arrow-{group}"

    @app.callback(
        Output(store_id,    "data"),
        Output(collapse_id, "style"),
        Output(arrow_id,    "children"),
        Input(toggle_id,    "n_clicks"),
        State(store_id,     "data"),
        prevent_initial_call=True,
    )
    def _toggle(n_clicks, is_collapsed):
        now_collapsed = not is_collapsed
        style = {"display": "none"} if now_collapsed else {"display": "block"}
        arrow = "▶" if now_collapsed else "▼"
        return now_collapsed, style, arrow

    _toggle.__name__ = f"_toggle_{group}"
    return _toggle


for _grp, _toggle_id, _arrow_id, _collapse_id, _color_key, _count in _AGENT_GROUPS:
    _make_toggle_callback(_grp.lower(), _color_key)


# ─────────────────────────────────────────────────────────────────────────────
# select_agent — FIXED: ghost click guard + no_update instead of None
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("selected-agent-id", "data"),
    Input({"type": "agent-row",       "index": dash.ALL}, "n_clicks"),
    Input({"type": "panel-close-btn", "index": dash.ALL}, "n_clicks"),
    Input("close-analytics-btn", "n_clicks"),
    prevent_initial_call=True,
)
def select_agent(row_clicks, panel_close_clicks, close_clicks):
    triggered = ctx.triggered_id

    triggered_value = 0
    for t in ctx.triggered:
        if t.get("value") and t["value"] > 0:
            triggered_value = t["value"]
            break

    if triggered == "close-analytics-btn":
        if triggered_value > 0:
            return None
        return dash.no_update

    if isinstance(triggered, dict) and triggered.get("type") == "panel-close-btn":
        if triggered_value > 0:
            return None
        return dash.no_update

    if isinstance(triggered, dict) and triggered.get("type") == "agent-row":
        if triggered_value > 0:
            return triggered["index"]
        return dash.no_update

    return dash.no_update


# ─────────────────────────────────────────────────────────────────────────────
# Two separate callbacks for overlay content and style (avoids race condition)
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("agent-analytics-overlay", "children"),
    Input("selected-agent-id", "data"),
    Input("theme-store", "data"),
    Input("poll", "n_intervals"),
)
def render_analytics_content(agent_id, theme, _poll):
    C = DARK if theme == "dark" else LIGHT

    if agent_id is None:
        return []

    state      = get_state()
    agents_all = state.get("portfolio_summary", {}).get("agents", [])

    try:
        agent_id = int(agent_id)
    except (TypeError, ValueError):
        return []

    agent_data = next(
        (a for a in agents_all if a.get("agent_id") == agent_id),
        None,
    )

    if agent_data is None:
        return []

    agent_data = dict(agent_data)
    agent_data["episode_id"]   = state.get("episode_id", 1)
    agent_data["current_tick"] = state.get("tick", 0)

    return [_build_analytics_panel(agent_data, C)]


@app.callback(
    Output("agent-analytics-overlay", "style"),
    Input("selected-agent-id", "data"),
)
def toggle_overlay_style(agent_id):
    if agent_id is None:
        return dict(display="none")
    return dict(
        position="fixed", top=0, left=0, right=0, bottom=0,
        background="rgba(0,0,0,0.65)",
        display="flex", alignItems="center", justifyContent="center",
        zIndex=9999, padding=16,
        backdropFilter="blur(4px)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Theme toggle
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n, current):
    return "light" if current == "dark" else "dark"


# ─────────────────────────────────────────────────────────────────────────────
# Speed buttons
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("speed-store", "data"),
    Output("poll", "interval"),
    Output("spd-1", "style"),
    Output("spd-2", "style"),
    Output("spd-3", "style"),
    Input("spd-1", "n_clicks"),
    Input("spd-2", "n_clicks"),
    Input("spd-3", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def set_speed(n1, n2, n3, theme):
    global _sim_speed
    C = DARK if theme == "dark" else LIGHT
    triggered = ctx.triggered_id
    speed = {"spd-1": 1, "spd-2": 2, "spd-3": 3}.get(triggered, 1)
    _sim_speed = speed
    interval = {1: 800, 2: 500, 3: 300}[speed]

    def _s(active):
        return dict(
            background=C["accent"] if active else "transparent",
            border=f"1px solid {C['accent'] if active else C['border']}",
            color="#fff" if active else C["dim"],
            fontFamily=FONT_UI, fontSize=10, fontWeight=600,
            borderRadius=20, padding="3px 14px", cursor="pointer",
            transition="all .2s",
        )
    return speed, interval, _s(speed==1), _s(speed==2), _s(speed==3)


# ─────────────────────────────────────────────────────────────────────────────
# Pause / Stop
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("btn-pause", "children"),
    Output("btn-pause", "style"),
    Input("btn-pause", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_pause(n, theme):
    global _sim_paused
    C = DARK if theme == "dark" else LIGHT
    _sim_paused = not _sim_paused
    if _sim_paused:
        return "▶ Resume", dict(
            flex=1, background=f"{C['green']}10",
            border=f"1px solid {C['green']}55", color=C["green"],
            fontFamily=FONT_UI, fontSize=10, fontWeight=600,
            borderRadius=8, padding="6px 0", cursor="pointer", transition="all .2s",
        )
    return "⏸ Pause", dict(
        flex=1, background="transparent",
        border=f"1px solid {C['border']}", color=C["dim"],
        fontFamily=FONT_UI, fontSize=10, fontWeight=600,
        borderRadius=8, padding="6px 0", cursor="pointer", transition="all .2s",
    )


@app.callback(
    Output("btn-stop", "children"),
    Input("btn-stop", "n_clicks"),
    prevent_initial_call=True,
)
def stop_sim(n):
    global _sim_paused
    _sim_paused = True
    runner.stop()
    return "⏹ Stopped"


# ─────────────────────────────────────────────────────────────────────────────
# Shock injection
# ─────────────────────────────────────────────────────────────────────────────

_SHOCK_MAP = {
    "shock-flash": "flash_crash",
    "shock-liq":   "liquidity_crisis",
    "shock-vol":   "volatility_spike",
    "shock-news":  "news_shock",
}

@app.callback(
    Output("shock-flash", "style"),
    Output("shock-liq",   "style"),
    Output("shock-vol",   "style"),
    Output("shock-news",  "style"),
    Input("shock-flash",  "n_clicks"),
    Input("shock-liq",    "n_clicks"),
    Input("shock-vol",    "n_clicks"),
    Input("shock-news",   "n_clicks"),
    State("theme-store",  "data"),
    prevent_initial_call=True,
)
def inject_shock(nf, nl, nv, nn, theme):
    C = DARK if theme == "dark" else LIGHT
    triggered = ctx.triggered_id
    if triggered and triggered in _SHOCK_MAP:
        with _sim_lock:
            runner.inject_shock(_SHOCK_MAP[triggered])

    shock_styles = {
        "shock-flash": (C["red"],    f"{C['red']}10",    f"{C['red']}44"),
        "shock-liq":   (C["orange"], f"{C['orange']}10", f"{C['orange']}44"),
        "shock-vol":   (C["yellow"], f"{C['yellow']}10", f"{C['yellow']}44"),
        "shock-news":  (C["purple"], f"{C['purple']}10", f"{C['purple']}44"),
    }

    def _s(btn_id):
        color, bg_a, border_a = shock_styles[btn_id]
        active = (btn_id == triggered)
        return dict(
            background=bg_a if active else f"{color}08",
            border=f"1px solid {border_a if active else color + '30'}",
            color=color, fontFamily=FONT_UI, fontSize=9, fontWeight=600,
            borderRadius=8, padding="5px 8px", cursor="pointer", transition="all .2s",
        )
    return _s("shock-flash"), _s("shock-liq"), _s("shock-vol"), _s("shock-news")


# ─────────────────────────────────────────────────────────────────────────────
# Main poll callback
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("topbar",          "style"),
    Output("tb-price",        "children"),
    Output("tb-chg",          "children"),
    Output("tb-chg",          "style"),
    Output("tb-ep-num",       "children"),
    Output("tb-tick-num",     "children"),
    Output("tb-regime",       "children"),
    Output("root",            "style"),
    Output("left-nav",        "style"),
    Output("right-panel",     "style"),
    Output("ph-price",        "children"),
    Output("ph-chg",          "children"),
    Output("ph-chg",          "style"),
    Output("ph-meta",         "children"),
    Output("main-chart",      "figure"),
    Output("stat-bar",        "style"),
    Output("sv-trades",       "children"),
    Output("sv-buy",          "children"),
    Output("sv-sell",         "children"),
    Output("sv-panic",        "children"),
    Output("sv-shocks",       "children"),
    Output("sv-spread",       "children"),
    Output("order-book",      "children"),
    Output("narrator-feed",   "children"),
    Output("gm-pct",          "children"),
    Output("gm-pct",          "style"),
    Output("agents-momentum", "children"),
    Output("agents-value",    "children"),
    Output("agents-noise",    "children"),
    Output("agents-panic",    "children"),
    Output("rp-panic",        "children"),
    Output("rp-pm",           "style"),
    Output("portfolio-bars",  "children"),
    Output("agent-detail",    "children"),
    Output("theme-toggle",    "children"),
    Input("poll",        "n_intervals"),
    Input("theme-store", "data"),
)
def refresh(_n, theme):
    C      = DARK if theme == "dark" else LIGHT
    is_dark = theme == "dark"
    TYPE_C  = TYPE_COLOR_DARK if is_dark else TYPE_COLOR_LIGHT

    state = get_state()

    tick        = state["tick"]
    price       = state["price"]
    history     = state["price_history"]
    regime      = state.get("shock_regime", "calm")
    book_stats  = state.get("book_stats", {})
    order_book  = state.get("order_book", {"bids": [], "asks": []})
    portfolio   = state.get("portfolio_summary", {})
    panic_count = state.get("panic_count", 0)
    panic_frac  = state.get("panic_fraction", 0.0)
    episode_id  = state.get("episode_id", 1)
    shocks      = state.get("shocks_occurred", 0)

    prev      = history[-2] if len(history) > 1 else price
    chg       = price - prev
    pct       = (chg / prev * 100) if prev else 0.0
    up        = chg >= 0
    chg_color = C["green"] if up else C["red"]
    chg_str   = f"{'+' if up else ''}{chg:.2f} ({'+' if up else ''}{pct:.2f}%)"
    tb_style  = dict(fontSize=10, color=chg_color, fontFamily=FONT_MONO)

    regime_color = {
        "calm":             C["accent"],
        "stressed":         C["orange"],
        "trending":         C["green"],
        "crisis":           C["red"],
        "flash_crash":      C["red"],
        "liquidity_crisis": C["orange"],
        "volatility_spike": C["yellow"],
        "news_shock":       C["purple"],
    }.get(regime, C["muted"])
    regime_badge = [_badge(regime, regime_color)]

    topbar_style = dict(
        background=C["topbar"],
        borderBottom=f"1px solid {C['border']}",
        padding="0 16px", display="flex", alignItems="center",
        gap=10, height=44, flexShrink=0,
        boxShadow=f"0 1px 0 {C['border']}",
        transition="background .3s, border .3s",
    )
    root_style = dict(
        background=C["bg"], color=C["text"],
        fontFamily=FONT_UI,
        height="100vh", width="100vw",
        overflow="hidden",
        display="flex", flexDirection="column",
        transition="background .3s, color .3s",
    )
    left_style = dict(
        width=172, background=C["nav"],
        borderRight=f"1px solid {C['border']}",
        display="flex", flexDirection="column",
        height=f"calc(100vh - {_TOPBAR_H}px)",
        overflowY="auto", flexShrink=0,
        transition="background .3s, border .3s",
    )
    right_style = dict(
        width=215, background=C["nav"],
        borderLeft=f"1px solid {C['border']}",
        padding="10px",
        height=f"calc(100vh - {_TOPBAR_H}px)",
        overflowY="auto", flexShrink=0,
        transition="background .3s, border .3s",
    )

    MAX_CHART_POINTS = 300
    display_history = history[-MAX_CHART_POINTS:] if len(history) > MAX_CHART_POINTS else history

    line_color = C["green"] if up else C["red"]
    fill_color = (
        "rgba(34,197,94,0.07)"  if (is_dark and up)      else
        "rgba(239,68,68,0.07)"  if (is_dark and not up)  else
        "rgba(22,163,74,0.06)"  if up                    else
        "rgba(220,38,38,0.06)"
    )

    fig = go.Figure(go.Scatter(
        y=display_history,
        mode="lines",
        line=dict(color=line_color, width=1.8),
        fill="tozeroy",
        fillcolor=fill_color,
        hoverinfo="y",
        hovertemplate="$%{y:.2f}<extra></extra>",
    ))

    if len(display_history) > 1:
        y_min = min(display_history)
        y_max = max(display_history)
        y_pad = max((y_max - y_min) * 0.12, y_max * 0.02)
        y_range = [y_min - y_pad, y_max + y_pad]
    else:
        y_range = [price * 0.95, price * 1.05]

    fig.update_layout(
        paper_bgcolor=C["chart_bg"],
        plot_bgcolor=C["chart_bg"],
        margin=dict(l=0, r=55, t=4, b=0),
        showlegend=False,
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(
            side="right",
            showgrid=True,
            gridcolor=C["grid"],
            gridwidth=0.5,
            tickfont=dict(color=C["muted"], size=9, family=FONT_MONO),
            tickprefix="$",
            tickformat=".2f",
            range=y_range,
        ),
        hovermode="x",
        font=dict(family=FONT_MONO),
    )

    if state.get("shock_active"):
        fig.add_vline(
            x=len(display_history) - 1,
            line_color=C["red"],
            line_dash="dot",
            line_width=1.2,
        )

    total_vol  = book_stats.get("total_volume") or 0
    bid_depth  = book_stats.get("bid_depth")    or 0
    ask_depth  = book_stats.get("ask_depth")    or 0
    spread_val = book_stats.get("spread")       or 0.0
    best_bid   = book_stats.get("best_bid")     or 0.0
    best_ask   = book_stats.get("best_ask")     or 0.0

    mid = book_stats.get("mid_price") or price
    spread_bps_val = (
        (best_ask - best_bid) / mid * 10000
        if (best_ask and best_bid and mid)
        else float(spread_val or 0)
    )

    shocks_raw = int(state.get("shocks_occurred", shocks))
    sv_trades = f"{int(total_vol):,}"
    sv_buy    = f"{int(bid_depth):,}"
    sv_sell   = f"{int(ask_depth):,}"
    sv_panic  = f"{panic_count}/20"
    sv_shocks = str(shocks_raw)
    sv_spread = f"{spread_bps_val:.1f} bps"

    stat_bar_style = dict(
        display="flex", flexShrink=0,
        height=_STATBAR_H,
        borderTop=f"1px solid {C['border']}",
        borderBottom=f"1px solid {C['border']}",
        background=C["surface"],
        transition="background .3s",
    )

    bids = order_book.get("bids", [])[:5]
    asks = order_book.get("asks", [])[:5]

    if bids or asks:
        max_q = max(
            max((q for _, q in bids), default=1),
            max((q for _, q in asks), default=1),
            1,
        )
        def _ob_row(bp, bq, ap, aq):
            bw = f"{bq / max_q * 100:.0f}%"
            aw = f"{aq / max_q * 100:.0f}%"
            return html.Div([
                html.Span(f"{bp:.2f}", style=dict(
                    fontSize=10, fontWeight=700, color=C["green"],
                    width=46, display="inline-block", fontFamily=FONT_MONO,
                )),
                html.Div(style=dict(
                    flex=1, height=11, borderRadius=2,
                    background=C["input_bg"], overflow="hidden",
                ), children=[
                    html.Div(style=dict(
                        height=11, width=bw,
                        background=f"{C['green']}25", float="right",
                    )),
                ]),
                html.Span(f"{bq}", style=dict(
                    fontSize=9, color=C["green"], width=28, textAlign="right",
                    display="inline-block", fontFamily=FONT_MONO,
                )),
                html.Span("|", style=dict(
                    fontSize=9, color=C["border"],
                    width=10, textAlign="center", display="inline-block",
                )),
                html.Span(f"{aq}", style=dict(
                    fontSize=9, color=C["red"], width=28,
                    display="inline-block", fontFamily=FONT_MONO,
                )),
                html.Div(style=dict(
                    flex=1, height=11, borderRadius=2,
                    background=C["input_bg"], overflow="hidden",
                ), children=[
                    html.Div(style=dict(
                        height=11, width=aw,
                        background=f"{C['red']}25",
                    )),
                ]),
                html.Span(f"{ap:.2f}", style=dict(
                    fontSize=10, fontWeight=700, color=C["red"],
                    width=46, textAlign="right", display="inline-block",
                    fontFamily=FONT_MONO,
                )),
            ], style=dict(
                display="flex", alignItems="center",
                gap=4, marginBottom=3,
            ))

        ob_rows = [_ob_row(bp, bq, ap, aq)
                   for (bp, bq), (ap, aq) in zip(bids, asks)]
    else:
        ob_rows = [html.Div("Awaiting orders…",
                            style=dict(fontSize=9, color=C["muted"], marginTop=4))]

    narrator_items = []
    if _narrator and hasattr(_narrator, "recent"):
        entries = _narrator.recent(3)
        border_map = {
            "shock":         C["red"],
            "regime_change": C["green"],
            "auto":          C["accent"],
            "manual":        C["yellow"],
        }
        for entry in reversed(entries):
            bc = border_map.get(getattr(entry, "trigger", "auto"), C["muted"])
            narrator_items.append(html.Div([
                html.Div(
                    f"tick {entry.tick} · {entry.trigger}",
                    style=dict(fontSize=8, fontWeight=700, color=C["text"],
                               marginBottom=2, fontFamily=FONT_UI),
                ),
                html.Div(entry.text, style=dict(
                    fontSize=9, color=C["dim"], lineHeight=1.5,
                    fontFamily=FONT_UI,
                )),
            ], style=dict(
                borderLeft=f"2px solid {bc}",
                background=C["card"], borderRadius="0 8px 8px 0",
                padding="5px 8px", marginBottom=5,
            )))
    elif state.get("narrator_feed"):
        for line in reversed(state["narrator_feed"][-3:]):
            narrator_items.append(html.Div(line, style=dict(
                fontSize=9, color=C["dim"], lineHeight=1.5,
                borderLeft=f"2px solid {C['accent']}",
                padding="4px 8px", marginBottom=5,
                background=C["card"], borderRadius="0 8px 8px 0",
                fontFamily=FONT_UI,
            )))
    if not narrator_items:
        narrator_items = [html.Div("Waiting for market events…",
                                   style=dict(fontSize=9, color=C["muted"],
                                              fontFamily=FONT_UI))]

    agents_all = portfolio.get("agents", [])
    agents_by_type: dict = {t: [] for t in ("momentum", "value", "noise", "panic")}
    for ag in agents_all:
        t = ag.get("agent_type", "noise")
        if t in agents_by_type:
            agents_by_type[t].append(ag)

    def _agent_section(agents: list) -> list:
        rows = []
        for ag in sorted(
            agents,
            key=lambda a: -(a.get("unrealised_pnl", 0) + a.get("realised_pnl", 0))
        ):
            pnl = ag.get("unrealised_pnl", 0) + ag.get("realised_pnl", 0)
            rows.append(_agent_row(ag["agent_id"], ag["agent_type"], pnl, C, is_dark))

        return rows or [html.Div("No data", style=dict(
            fontSize=9, color=C["muted"], padding="4px 12px",
        ))]

    gm_color = C["green"] if up else C["red"]
    gm_style = dict(
        fontSize=8, fontWeight=700, borderRadius=20, padding="1px 7px",
        color=gm_color, border=f"1px solid {gm_color}44",
        background=f"{gm_color}12", fontFamily=FONT_MONO,
    )
    gm_pct = f"{'+' if up else ''}{pct:.1f}%"

    pm_pct   = min(panic_frac * 100, 100)
    pm_color = (
        C["red"]    if panic_frac > 0.5 else
        C["orange"] if panic_frac > 0.2 else
        C["muted"]
    )
    pm_style = dict(
        height=6, background=pm_color, borderRadius=99,
        width=f"{pm_pct:.0f}%", transition="width .4s, background .4s",
    )

    type_pnl_pct = portfolio.get("by_type_pct", {})

    def _bar_row(label: str, t_key: str, bar_color: str) -> html.Div:
        raw   = type_pnl_pct.get(t_key, 0.0)
        bar_w = min(abs(raw) * 5, 100)
        val_s = f"{'+' if raw >= 0 else ''}{raw:.2f}%"
        return html.Div([
            html.Div([
                html.Span(label, style=dict(fontSize=10, color=C["dim"], fontFamily=FONT_UI)),
                html.Span(val_s, style=dict(
                    fontSize=10, fontWeight=700, color=bar_color, fontFamily=FONT_MONO,
                )),
            ], style=dict(display="flex", justifyContent="space-between", marginBottom=3)),
            html.Div(style=dict(
                background=C["input_bg"], borderRadius=99,
                height=4, overflow="hidden", marginBottom=8,
            ), children=[
                html.Div(style=dict(
                    height=4, width=f"{bar_w:.0f}%",
                    background=bar_color, borderRadius=99,
                    transition="width .5s",
                )),
            ]),
        ])

    portfolio_bars = [
        _bar_row("Momentum", "momentum", C["accent"]),
        _bar_row("Value",    "value",    C["green"]),
        _bar_row("Noise",    "noise",    C["muted"]),
        _bar_row("Panic",    "panic",    C["red"]),
    ]

    top_ag = max(
        agents_all,
        key=lambda a: a.get("unrealised_pnl", 0) + a.get("realised_pnl", 0),
        default=None,
    )
    if top_ag:
        pnl_top   = top_ag.get("unrealised_pnl", 0) + top_ag.get("realised_pnl", 0)
        pnl_color = C["green"] if pnl_top >= 0 else C["red"]
        pnl_str   = f"{'+' if pnl_top >= 0 else ''}${pnl_top:.0f}"
        sig       = top_ag.get("last_signal", 0.0)
        exp       = top_ag.get("latest_explanation") or (
            f"Signal {sig:+.4f} — {top_ag.get('last_action', 'hold')}"
        )
        dot_color = TYPE_C.get(top_ag.get("agent_type", "noise"), C["muted"])

        def _detail_row(k, v, v_color=None):
            return html.Div([
                html.Span(k, style=dict(fontSize=9, color=C["muted"], fontFamily=FONT_UI)),
                html.Span(v, style=dict(fontSize=9, fontWeight=700,
                                        color=v_color or C["text"], fontFamily=FONT_MONO)),
            ], style=dict(display="flex", justifyContent="space-between", marginBottom=5))

        agent_detail = html.Div([
            html.Div([
                html.Div(style=dict(
                    width=8, height=8, borderRadius="50%",
                    background=dot_color, marginRight=6,
                )),
                html.Span(f"Agent {top_ag['agent_id']}", style=dict(
                    fontSize=11, fontWeight=700, color=C["text"],
                )),
                html.Span(top_ag.get("agent_type", ""), style=dict(
                    fontSize=8, color=dot_color, marginLeft=6,
                    textTransform="uppercase", letterSpacing=".08em", fontWeight=600,
                )),
            ], style=dict(display="flex", alignItems="center", marginBottom=10)),
            _detail_row("Portfolio Value", f"${top_ag.get('portfolio_value', 0):,.0f}"),
            _detail_row("PnL", pnl_str, pnl_color),
            _detail_row("Last Action", top_ag.get("last_action", "hold")),
            html.Div(style=dict(
                borderTop=f"1px solid {C['border']}", marginTop=4,
                paddingTop=6, marginBottom=4,
            )),
            html.Div("AI Explanation", style=dict(
                fontSize=8, color=C["muted"], textTransform="uppercase",
                letterSpacing=".1em", fontWeight=600, marginBottom=4,
            )),
            html.Div(exp, style=dict(
                fontSize=9, color=C["dim"], lineHeight=1.5, fontFamily=FONT_UI,
            )),
        ])
    else:
        agent_detail = html.Div("No agent data yet",
                                style=dict(fontSize=9, color=C["muted"]))

    ph_meta      = f"Vol 0.0000 · Ep {episode_id}"
    ph_chg_style = dict(fontSize=12, fontWeight=700, color=chg_color, fontFamily=FONT_MONO)
    theme_icon   = "🌙" if is_dark else "☀"

    return (
        topbar_style,
        f"${price:.2f}", chg_str, tb_style,
        str(episode_id), str(tick),
        regime_badge,
        root_style, left_style, right_style,
        f"${price:.2f}", chg_str, ph_chg_style, ph_meta,
        fig,
        stat_bar_style,
        sv_trades, sv_buy, sv_sell, sv_panic, sv_shocks, sv_spread,
        ob_rows, narrator_items,
        gm_pct, gm_style,
        _agent_section(agents_by_type["momentum"]),
        _agent_section(agents_by_type["value"]),
        _agent_section(agents_by_type["noise"]),
        _agent_section(agents_by_type["panic"]),
        f"{panic_count}/20", pm_style,
        portfolio_bars, agent_detail,
        theme_icon,
    )


if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            _ = torch.zeros(1, device="cuda")
    except Exception:
        pass

    app.run(debug=False, port=8050)