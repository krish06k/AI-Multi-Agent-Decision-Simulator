"""
agents/agent_pool.py
====================
Layer 2 — Agent Policies | Agent Pool Manager

Spawns and manages the full heterogeneous population of 100 agents:
  * 30 Momentum agents (trend following, PPO)
  * 30 Value agents    (mean reversion, DQN)
  * 20 Noise agents    (random liquidity)
  * 20 Panic agents    (stress cascades)

Responsibilities
----------------
* Pool creation with correct ID ranges and config slices.
* Per-tick observe→act→update dispatch loop.
* Aggregated statistics for dashboard portfolio view.
* Policy loading: restores trained SB3 weights from rl_policies/.
* Panic agent coordination: exposes panic count for cascade detection.
* Agent lookup by ID, type filter, and batch queries.

FIX SUMMARY
-----------
v2 fixes:
FIX BUG 2 — Agent analytics overlay (click agent → portfolio panel) showed
all-zero action breakdown bars (Buy/Sell/Hold all 0).

Root cause: trade_history stored action as d.action.name which produces
uppercase strings like "BUY_SMALL", "BUY_LARGE", "SELL_SMALL", "SELL_LARGE".
The dashboard app.py counts with:
    buy_count  = actions.count("buy")    # lowercase, never matched
    sell_count = actions.count("sell")   # lowercase, never matched

Fix: normalise action to lowercase "buy", "sell", or "hold" when building
trade_history so dashboard counting always works correctly.
"""

from __future__ import annotations
from simulation.state import AgentRecord
import logging
from pathlib import Path
from typing import Iterator, Optional
from collections import defaultdict
import numpy as np
import yaml

from agents.base_agent import BaseAgent, AgentType, AgentDecision, MarketSnapshot
from agents.momentum_agent import MomentumAgent
from agents.value_agent import ValueAgent
from agents.noise_agent import NoiseAgent
from agents.panic_agent import PanicAgent

logger = logging.getLogger(__name__)

_AGENT_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "agent_config.yaml"
_RL_POLICY_DIR  = Path(__file__).resolve().parents[1] / "models" / "rl_policies"


def _load_cfg() -> dict:
    with open(_AGENT_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ID range constants
# ---------------------------------------------------------------------------
MOMENTUM_IDS = range(0, 30)
VALUE_IDS    = range(30, 60)
NOISE_IDS    = range(60, 80)
PANIC_IDS    = range(80, 100)


# ---------------------------------------------------------------------------
# Agent Pool
# ---------------------------------------------------------------------------

class AgentPool:
    """
    Full heterogeneous population of 100 simulation agents.

    Parameters
    ----------
    cfg  : agent_config.yaml dict. If None, loads from file.
    seed : Master RNG seed for reproducibility across noise/panic agents.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        seed: int | None = None,
    ):
        self.cfg  = cfg or _load_cfg()
        self._rng = np.random.default_rng(seed)
        self._pool_cfg = self.cfg.get("pool", {})

        # Build all 100 agents
        self._agents: dict[int, BaseAgent] = {}
        self._build_pool()

        logger.info(
            "AgentPool ready | %d agents "
            "[Momentum:%d  Value:%d  Noise:%d  Panic:%d]",
            len(self._agents),
            len(self.by_type(AgentType.MOMENTUM)),
            len(self.by_type(AgentType.VALUE)),
            len(self.by_type(AgentType.NOISE)),
            len(self.by_type(AgentType.PANIC)),
        )

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_pool(self) -> None:
        """Instantiate all agents with type-appropriate configs."""
        # --- Momentum (0–29) ---
        for aid in MOMENTUM_IDS:
            self._agents[aid] = MomentumAgent(agent_id=aid, cfg=self.cfg)

        # --- Value (30–59) ---
        for aid in VALUE_IDS:
            self._agents[aid] = ValueAgent(agent_id=aid, cfg=self.cfg)

        # --- Noise (60–79) ---
        for aid in NOISE_IDS:
            agent_rng = np.random.default_rng(self._rng.integers(0, 2**32))
            self._agents[aid] = NoiseAgent(
                agent_id=aid, cfg=self.cfg, rng=agent_rng
            )

        # --- Panic (80–99) ---
        # Heterogeneous sensitivities: some panic early (high sensitivity),
        # some late (low sensitivity). This creates realistic staggered cascades.
        sensitivities = np.linspace(0.6, 1.8, 20)
        for i, aid in enumerate(PANIC_IDS):
            agent_rng = np.random.default_rng(self._rng.integers(0, 2**32))
            self._agents[aid] = PanicAgent(
                agent_id=aid,
                cfg=self.cfg,
                rng=agent_rng,
                sensitivity=float(sensitivities[i]),
            )

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def load_policies(self, policy_dir: str | Path | None = None) -> dict[str, bool]:
        """
        Load pre-trained SB3 RL policies for Momentum (PPO) and Value (DQN) agents.

        Parameters
        ----------
        policy_dir : Directory containing saved .zip policy files.
                     Expected files: momentum_ppo.zip, value_dqn.zip

        Returns
        -------
        Dict mapping policy name → successfully loaded (bool).
        """
        policy_dir = Path(policy_dir or _RL_POLICY_DIR)
        results: dict[str, bool] = {}

        try:
            from stable_baselines3 import PPO, DQN

            # Momentum PPO
            ppo_path = policy_dir / "momentum_ppo.zip"
            if ppo_path.exists():
                ppo = PPO.load(ppo_path)
                for aid in MOMENTUM_IDS:
                    self._agents[aid]._policy = ppo.policy
                results["momentum_ppo"] = True
                logger.info("Loaded PPO policy for %d momentum agents.", len(MOMENTUM_IDS))
            else:
                results["momentum_ppo"] = False
                logger.warning("PPO policy not found at %s.", ppo_path)

            # Value DQN
            dqn_path = policy_dir / "value_dqn.zip"
            if dqn_path.exists():
                dqn = DQN.load(dqn_path)
                for aid in VALUE_IDS:
                    self._agents[aid]._policy = dqn.policy
                results["value_dqn"] = True
                logger.info("Loaded DQN policy for %d value agents.", len(VALUE_IDS))
            else:
                results["value_dqn"] = False
                logger.warning("DQN policy not found at %s.", dqn_path)

        except ImportError:
            logger.warning(
                "stable_baselines3 not installed. "
                "All agents will use rule-based fallback."
            )
            results = {"momentum_ppo": False, "value_dqn": False}

        return results

    # ------------------------------------------------------------------
    # Per-tick dispatch
    # ------------------------------------------------------------------

    def observe_all(self, snapshot: MarketSnapshot) -> None:
        """Broadcast the current market snapshot to all agents."""
        for agent in self._agents.values():
            agent.observe(snapshot)

    def act_all(
        self,
        observations: dict[int, np.ndarray],
    ) -> dict[int, int]:
        """
        Collect actions from all agents simultaneously.

        Parameters
        ----------
        observations : Dict mapping agent_id → 47-dim obs vector.

        Returns
        -------
        Dict mapping agent_id → discrete action int.
        """
        actions: dict[int, int] = {}
        for aid, agent in self._agents.items():
            obs = observations.get(aid, np.zeros(47, dtype=np.float32))
            try:
                actions[aid] = agent.act(obs)
            except Exception as exc:
                logger.error("Agent[%d] act() failed: %s", aid, exc)
                actions[aid] = 0   # HOLD on error
        return actions

    def update_all(
        self,
        rewards: dict[int, float],
        dones: dict[int, bool],
        infos: dict[int, dict],
    ) -> None:
        """Update all agents with their respective reward signals."""
        for aid, agent in self._agents.items():
            reward = rewards.get(aid, 0.0)
            done   = dones.get(aid, False)
            info   = infos.get(aid, {})
            agent.update(reward, done, info)
            agent.sync_portfolio(info)

    # ------------------------------------------------------------------
    # Decisions (for GenAI explainability)
    # ------------------------------------------------------------------

    def collect_decisions(self, tick: int) -> list[AgentDecision]:
        """
        Collect the most recent decision from every agent.

        Returns
        -------
        List of AgentDecision objects for this tick, sorted by agent_id.
        """
        decisions = []
        for agent in self._agents.values():
            d = agent.last_decision()
            if d is not None and d.tick == tick:
                decisions.append(d)
        decisions.sort(key=lambda d: d.agent_id)
        return decisions

    def notable_decisions(self, tick: int, top_n: int = 5) -> list[AgentDecision]:
        """
        Return the most notable decisions this tick (large orders, panic triggers).
        Used by the GenAI narrator to highlight key events.
        """
        all_d = self.collect_decisions(tick)
        def score(d: AgentDecision) -> float:
            s = {"BUY_LARGE": 3, "SELL_LARGE": 3, "BUY_SMALL": 1, "SELL_SMALL": 1, "HOLD": 0}.get(
                d.action.name, 0
            )
            if any("panic" in t for t in d.reason_tags):
                s += 5
            if any("stop_loss" in t for t in d.reason_tags):
                s += 4
            if any("cascade" in t for t in d.reason_tags):
                s += 6
            return s

        return sorted(all_d, key=score, reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Panic state queries
    # ------------------------------------------------------------------

    @property
    def panic_count(self) -> int:
        """Number of currently panicking agents."""
        return sum(
            1 for aid in PANIC_IDS
            if isinstance(self._agents[aid], PanicAgent)
            and self._agents[aid].is_panicking
        )

    @property
    def panic_fraction(self) -> float:
        """Fraction of panic agents currently in panic mode."""
        n_panic = len(list(PANIC_IDS))
        return self.panic_count / max(n_panic, 1)

    def is_cascade_active(self) -> bool:
        """True if ≥ 30% of panic agents are simultaneously selling."""
        return self.panic_fraction >= 0.3

    # ------------------------------------------------------------------
    # Lookup & filtering
    # ------------------------------------------------------------------
    def all_agents(self) -> list[BaseAgent]:
        """Return all agents (for runner compatibility)."""
        return list(self._agents.values())

    def __getitem__(self, agent_id: int) -> BaseAgent:
        return self._agents[agent_id]

    def __iter__(self) -> Iterator[BaseAgent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)

    def by_type(self, agent_type: AgentType) -> list[BaseAgent]:
        """Return all agents of a given type."""
        return [a for a in self._agents.values() if a.agent_type == agent_type]

    def by_ids(self, ids: list[int]) -> list[BaseAgent]:
        return [self._agents[i] for i in ids if i in self._agents]

    # ------------------------------------------------------------------
    # Aggregated statistics
    # ------------------------------------------------------------------

    def portfolio_summary(self, price: float) -> dict:
        summary: dict = {
            "total_agents": len(self._agents),
            "price": price,
            "by_type": {},
            "overall": {},
        }

        all_values, all_pnl = [], []

        for atype in AgentType:
            agents = self.by_type(atype)
            if not agents:
                continue
            values = [a.portfolio_value(price) for a in agents]
            pnl    = [a.unrealised_pnl(price) for a in agents]
            shares = [a.shares for a in agents]
            summary["by_type"][atype.value.lower()] = {
                "count": len(agents),
                "total_value": round(sum(values), 2),
                "mean_value": round(float(np.mean(values)), 2),
                "total_shares": sum(shares),
                "mean_pnl": round(float(np.mean(pnl)), 2),
                "positive_pnl_count": sum(1 for p in pnl if p > 0),
            }
            all_values.extend(values)
            all_pnl.extend(pnl)

        summary["overall"] = {
            "total_portfolio_value": round(sum(all_values), 2),
            "mean_portfolio_value": round(float(np.mean(all_values)), 2),
            "total_unrealised_pnl": round(sum(all_pnl), 2),
            "pct_agents_profitable": round(
                sum(1 for p in all_pnl if p > 0) / max(len(all_pnl), 1), 4
            ),
            "panic_count": self.panic_count,
            "cascade_active": self.is_cascade_active(),
        }

        type_pnl    = defaultdict(float)
        type_counts = defaultdict(int)
        agent_records = []

        for agent in self._agents.values():
            t   = agent.agent_type.value.lower()
            pnl = agent.unrealised_pnl(price) + agent.realised_pnl

            type_pnl[t]    += pnl
            type_counts[t] += 1

            last_dec      = agent.last_decision()
            initial_cash  = agent._base_cfg.get("initial_cash", 10_000.0)
            decision_list = agent.decision_history()

            pnl_history: list[float] = (
                [round(d.portfolio_value - initial_cash, 2) for d in decision_list]
                if decision_list else [0.0]
            )

            trade_history: list[dict] = []
            for i, d in enumerate(decision_list):
                if d.action.name == "HOLD":
                    continue
                prev_pv   = decision_list[i - 1].portfolio_value if i > 0 else initial_cash
                trade_pnl = round(d.portfolio_value - prev_pv, 2)

                # FIX BUG 2: normalise action to lowercase "buy"/"sell"/"hold"
                # so dashboard app.py action counts work correctly.
                # d.action.name produces "BUY_SMALL"/"BUY_LARGE"/"SELL_SMALL"/
                # "SELL_LARGE" — app.py uses actions.count("buy") which never
                # matched the uppercase strings, making all bars show 0.
                action_name = d.action.name.upper()
                if "BUY" in action_name:
                    normalised_action = "buy"
                elif "SELL" in action_name:
                    normalised_action = "sell"
                else:
                    normalised_action = "hold"

                trade_history.append({
                    "trade_num": len(trade_history) + 1,
                    "action":    normalised_action,
                    "tick":      d.tick,
                    "price":     round(d.price, 4),
                    "signal":    round(d.signal_value, 6),
                    "pnl":       trade_pnl,
                })

            drawdown = last_dec.drawdown if last_dec else 0.0
            is_panic = isinstance(agent, PanicAgent) and agent.is_panicking

            latest_explanation = ""
            if last_dec and last_dec.reason_tags:
                latest_explanation = (
                    last_dec.signal_label + " | tags: " +
                    ", ".join(last_dec.reason_tags)
                )

            agent_records.append({
                "agent_id":           agent.agent_id,
                "agent_type":         t,
                "portfolio_value":    round(agent.portfolio_value(price), 2),
                "cash":               round(agent.cash, 2),
                "shares":             agent.shares,
                "unrealised_pnl":     round(agent.unrealised_pnl(price), 2),
                "realised_pnl":       round(agent.realised_pnl, 2),
                "drawdown":           round(drawdown, 4),
                "trade_count":        len(trade_history),
                "total_commission":   round(getattr(agent, "total_commission", 0.0), 2),
                "last_action":        last_dec.action.name.lower() if last_dec else "HOLD",
                "last_signal":        round(last_dec.signal_value, 6) if last_dec else 0.0,
                "is_panic":           is_panic,
                "latest_explanation": latest_explanation,
                "pnl_history":        pnl_history,
                "trade_history":      trade_history,
            })

        capital_per_type = {t: type_counts[t] * 10_000 for t in type_counts}
        summary["by_type_pct"] = {
            t: (type_pnl[t] / max(capital_per_type[t], 1)) * 100
            for t in type_pnl
        }

        summary["agents"] = agent_records
        return summary

    def all_stats(self) -> list[dict]:
        """Return stats dict for every agent (for dashboard portfolio cards)."""
        return [a.stats() for a in self._agents.values()]

    def reset_all(self) -> None:
        """Reset all agents for a new episode."""
        for agent in self._agents.values():
            agent.reset()
        logger.info("All %d agents reset.", len(self._agents))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_agent_pool(
    cfg: dict | None = None,
    seed: int | None = None,
    load_policies: bool = False,
) -> AgentPool:
    """
    Build the full 100-agent pool from config.

    Parameters
    ----------
    cfg            : Optional config override.
    seed           : RNG seed for reproducibility.
    load_policies  : If True, attempt to load saved RL policies from disk.

    Returns
    -------
    AgentPool instance.
    """
    pool = AgentPool(cfg=cfg, seed=seed)
    if load_policies:
        results = pool.load_policies()
        loaded = [k for k, v in results.items() if v]
        logger.info("Loaded policies: %s", loaded or "none")
    return pool


if __name__ == "__main__":
    print("🔍 Testing AgentPool...\n")

    # Build pool
    pool = AgentPool()

    print(f"Total Agents: {len(pool)}")

    # Check type distribution
    print("\nAgent Type Counts:")
    print("Momentum:", len(pool.by_type(AgentType.MOMENTUM)))
    print("Value   :", len(pool.by_type(AgentType.VALUE)))
    print("Noise   :", len(pool.by_type(AgentType.NOISE)))
    print("Panic   :", len(pool.by_type(AgentType.PANIC)))

    # Create fake observation (47-dim like your env)
    observations = {
        aid: np.random.uniform(-1, 1, 47).astype(np.float32)
        for aid in range(len(pool))
    }

    # Simulate one step
    print("\n--- Running one step ---")

    actions = pool.act_all(observations)

    for aid in list(actions.keys())[:10]:
        print(f"Agent {aid} → Action: {actions[aid]}")

    # -------------------------------
    # MULTI-STEP LOOP
    # -------------------------------
    print("\n--- Running multiple steps ---")

    for step in range(5):
        print(f"\n--- Step {step} ---")

        observations = {
            aid: np.random.randn(47).astype(np.float32)
            for aid in range(len(pool))
        }

        for agent in pool.by_type(AgentType.PANIC):
            if hasattr(agent, "_panic_state"):
                agent._panic_state = True

        actions = pool.act_all(observations)

        for aid in list(actions.keys())[:5]:
            print(f"Agent {aid} → Action: {actions[aid]}")

        for agent in pool.by_type(AgentType.PANIC):
            agent._is_panicking = True

        for aid in list(actions.keys())[:10]:
            print(f"Agent {aid} → Action: {actions[aid]}")

        rewards = {aid: np.random.randn() for aid in range(len(pool))}
        dones = {aid: False for aid in range(len(pool))}
        infos = {aid: {} for aid in range(len(pool))}

        pool.update_all(rewards, dones, infos)

        summary = pool.portfolio_summary(price=100.0)

        print("\n--- Portfolio Summary ---")
        print(summary["overall"])

        print("\n--- Panic System ---")
        print("Panic Count:", pool.panic_count)
        print("Cascade Active:", pool.is_cascade_active())