"""
genai/explainer.py
==================
AI-powered event explainer — fires only on meaningful market events
(shock, cascade, regime change, episode summary, portfolio review).
Routine per-tick agent decisions use fast rule-based text instead.

Model: gemini-2.0-flash-lite  (free tier, uses new google.genai SDK)
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from google import genai
from google.genai import types

from genai.prompt_templates import (
    build_decision_prompt,
    build_portfolio_prompt,
    build_shock_prompt,
    build_episode_summary_prompt,
    build_cascade_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL  = "gemini-3.1-flash-lite"
_DEFAULT_TOKENS = 256
_CACHE_MAX_SIZE = 500
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF  = 1.5

SYSTEM_PROMPT = (
    "You are a concise, expert financial-market-simulation analyst. "
    "You explain agent behaviours and market dynamics clearly and accurately. "
    "Avoid jargon where possible. Never fabricate data. "
    "If data is missing, say so briefly."
)


# ---------------------------------------------------------------------------
# Rule-based decision text (no API call — instant, free, unlimited)
# ---------------------------------------------------------------------------

def _rule_based_decision(
    agent_id: int,
    agent_type: str,
    action: str,
    price: float,
    signal_value: float,
    signal_label: str,
    portfolio_value: float,
    unrealised_pnl: float,
    drawdown: float,
    shock_active: bool,
    shock_regime: str,
    reason_tags: list[str],
) -> str:
    action_map = {
        "BUY":  "entered a long position",
        "SELL": "exited and went short",
        "HOLD": "held their position",
    }
    action_str = action_map.get(action.upper(), action.lower())

    shock_note = ""
    if shock_active:
        shock_note = f" during an active {shock_regime} shock event,"

    pnl_note = ""
    if unrealised_pnl > 0:
        pnl_note = f" with unrealised gains of ${unrealised_pnl:+.2f}"
    elif unrealised_pnl < 0:
        pnl_note = f" carrying unrealised losses of ${unrealised_pnl:+.2f}"

    drawdown_note = ""
    if drawdown > 0.05:
        drawdown_note = f" Current drawdown stands at {drawdown:.1%}."

    tags_note = ""
    if reason_tags:
        tags_note = f" Key signals: {', '.join(reason_tags)}."

    return (
        f"Agent {agent_id} ({agent_type}){shock_note} {action_str} at ${price:.2f}"
        f" based on a {signal_label} reading of {signal_value:.4f}"
        f"{pnl_note}. Portfolio value: ${portfolio_value:.2f}.{drawdown_note}{tags_note}"
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class ExplanationCache:
    def __init__(self, max_size: int = _CACHE_MAX_SIZE):
        self._store: OrderedDict[Any, str] = OrderedDict()
        self._max_size = max_size
        self.hits   = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[str]:
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, key: Any, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size":     self.size,
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": self.hits / max(total, 1),
        }


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class Explainer:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_TOKENS,
        cache: Optional[ExplanationCache] = None,
        api_key: Optional[str] = None,
        enabled: bool = True,
    ):
        self._model      = model
        self._max_tokens = max_tokens
        self._cache      = cache or ExplanationCache()
        self._enabled    = enabled

        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )

        if self._enabled and not self._api_key:
            logger.error("Explainer DISABLED: GEMINI_API_KEY is not set.")
            print("[Explainer] ERROR: GEMINI_API_KEY not set — explainer disabled.")
            self._enabled = False

        if self._enabled:
            try:
                self._client = genai.Client(api_key=self._api_key)
                logger.info("Explainer: Gemini client initialised (model=%s).", model)
                print(f"[Explainer] OK: Gemini client ready (model={model})")
            except Exception as e:
                logger.error("Explainer: Gemini client init FAILED: %s", e)
                print(f"[Explainer] ERROR: Gemini init failed: {e}")
                self._enabled = False
                self._client  = None
        else:
            self._client = None

    # -----------------------------------------------------------------------
    # Internal AI call (used ONLY for events)
    # -----------------------------------------------------------------------

    def _call(self, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        if not self._enabled or self._client is None:
            return "[Explainer disabled — set GEMINI_API_KEY]"

        tokens   = max_tokens or self._max_tokens
        last_err: Exception | None = None

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                response = self._client.models.generate_content(
                    model    = self._model,
                    contents = user_prompt,
                    config   = types.GenerateContentConfig(
                        system_instruction = SYSTEM_PROMPT,
                        max_output_tokens  = tokens,
                        temperature        = 0.4,
                    ),
                )
                text = response.text.strip()
                logger.debug("Explainer API call succeeded (attempt %d).", attempt)
                return text

            except Exception as exc:
                last_err = exc
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Explainer API call FAILED (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, _RETRY_ATTEMPTS, exc, wait,
                )
                print(f"[Explainer] API ERROR attempt {attempt}/{_RETRY_ATTEMPTS}: {exc}")
                time.sleep(wait)

        logger.error("Explainer: all retries exhausted. Last error: %s", last_err)
        print(f"[Explainer] FAILED after {_RETRY_ATTEMPTS} retries: {last_err}")
        return f"[Explanation unavailable: {last_err}]"

    # -----------------------------------------------------------------------
    # PUBLIC: per-decision — RULE-BASED, no API call
    # -----------------------------------------------------------------------

    def explain_decision(self, decision: Any) -> str:
        """
        Fast rule-based text for every agent decision.
        No API call — zero latency, zero cost, unlimited rate.
        """
        cache_key = ("decision", decision.tick, decision.agent_id)
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = _rule_based_decision(
            agent_id        = decision.agent_id,
            agent_type      = decision.agent_type.value,
            action          = decision.action.name,
            price           = decision.price,
            signal_value    = decision.signal_value,
            signal_label    = decision.signal_label,
            portfolio_value = decision.portfolio_value,
            unrealised_pnl  = decision.unrealised_pnl,
            drawdown        = decision.drawdown,
            shock_active    = getattr(decision, "shock_active", False),
            shock_regime    = getattr(decision, "shock_regime", "none"),
            reason_tags     = decision.reason_tags,
        )

        self._cache.set(cache_key, result)
        return result

    # -----------------------------------------------------------------------
    # PUBLIC: events — AI-powered
    # -----------------------------------------------------------------------

    def explain_shock(self, shock_data: dict) -> str:
        """AI explains a market shock event."""
        cache_key = ("shock", shock_data.get("shock_type"), shock_data.get("tick"))
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        prompt = build_shock_prompt(
            shock_type        = shock_data.get("shock_type", "unknown"),
            tick              = shock_data.get("tick", 0),
            price_before      = shock_data.get("price_before", 0),
            price_after       = shock_data.get("price_after", 0),
            spread_before_bps = shock_data.get("spread_before_bps", 0),
            spread_after_bps  = shock_data.get("spread_after_bps", 0),
            volume_spike      = shock_data.get("volume_spike", 1.0),
            duration_ticks    = shock_data.get("duration_ticks", 1),
            agents_affected   = shock_data.get("agents_affected", 0),
        )

        result = self._call(prompt, max_tokens=300)
        self._cache.set(cache_key, result)
        return result

    def explain_cascade(self, cascade_data: dict) -> str:
        """AI explains a panic cascade event."""
        cache_key = ("cascade", cascade_data.get("trigger_tick"))
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        prompt = build_cascade_prompt(
            trigger_tick       = cascade_data.get("trigger_tick", 0),
            price_at_trigger   = cascade_data.get("price_at_trigger", 0),
            peak_panic_agents  = cascade_data.get("peak_panic_agents", 0),
            total_panic_agents = cascade_data.get("total_panic_agents", 20),
            price_trough       = cascade_data.get("price_trough", 0),
            trough_tick        = cascade_data.get("trough_tick", 0),
            recovery_tick      = cascade_data.get("recovery_tick"),
            total_sell_volume  = cascade_data.get("total_sell_volume", 0),
            cascade_factor     = cascade_data.get("cascade_factor", 1.5),
        )

        result = self._call(prompt)
        self._cache.set(cache_key, result)
        return result

    def explain_episode(self, episode_data: dict) -> str:
        """AI explains end-of-episode summary."""
        cache_key = ("episode", episode_data.get("episode"))
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        prompt = build_episode_summary_prompt(
            episode       = episode_data.get("episode", 0),
            total_ticks   = episode_data.get("total_ticks", 0),
            initial_price = episode_data.get("initial_price", 100.0),
            final_price   = episode_data.get("final_price", 100.0),
            total_trades  = episode_data.get("total_trades", 0),
            shock_events  = episode_data.get("shock_events", []),
            agent_stats   = episode_data.get("agent_stats", []),
            gan_regime    = episode_data.get("gan_regime"),
        )

        result = self._call(prompt, max_tokens=400)
        self._cache.set(cache_key, result)
        return result

    def explain_portfolio(self, agent_stats: dict, price: float, initial_cash: float = 10_000.0) -> str:
        """AI explains a portfolio review (on-demand only, not per-tick)."""
        cache_key = ("portfolio", agent_stats.get("agent_id"), agent_stats.get("tick"))
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        prompt = build_portfolio_prompt(
            agent_id        = agent_stats.get("agent_id", 0),
            agent_type      = agent_stats.get("agent_type", "unknown"),
            portfolio_value = agent_stats.get("portfolio_value", 0),
            initial_cash    = initial_cash,
            cash            = agent_stats.get("cash", 0),
            shares          = agent_stats.get("shares", 0),
            unrealised_pnl  = agent_stats.get("unrealised_pnl", 0),
            realised_pnl    = agent_stats.get("realised_pnl", 0),
            drawdown        = agent_stats.get("drawdown", 0),
            trade_count     = agent_stats.get("trade_count", 0),
            total_commission= agent_stats.get("total_commission", 0),
            tick            = agent_stats.get("tick", 0),
            price           = price,
        )

        result = self._call(prompt)
        self._cache.set(cache_key, result)
        return result

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def cache_stats(self) -> dict:
        return self._cache.stats()

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Explainer cache cleared.")


# ---------------------------------------------------------------------------
# Batch helper — rule-based only (no API)
# ---------------------------------------------------------------------------

@dataclass
class ExplainBatch:
    explainer: Explainer
    max_per_type: int = 5

    def run(self, decisions: list[Any]) -> dict[int, str]:
        results: dict[int, str] = {}
        seen_types: dict[str, int] = {}
        for decision in decisions:
            atype = decision.agent_type.value
            count = seen_types.get(atype, 0)
            if count >= self.max_per_type:
                continue
            # Rule-based — no API call
            explanation = self.explainer.explain_decision(decision)
            results[decision.agent_id] = explanation
            seen_types[atype] = count + 1
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_explainer(
    enabled: bool = True,
    cache_size: int = _CACHE_MAX_SIZE,
    max_tokens: int = _DEFAULT_TOKENS,
) -> Explainer:
    cache = ExplanationCache(max_size=cache_size)
    return Explainer(
        model=_DEFAULT_MODEL,
        max_tokens=max_tokens,
        cache=cache,
        enabled=enabled,
    )