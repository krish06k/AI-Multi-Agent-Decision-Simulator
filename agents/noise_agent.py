"""
agents/noise_agent.py
=====================
Layer 2 — Agent Policies | Noise Agent

Injects random orders to simulate uninformed retail participants.
Provides market liquidity and prevents the order book from going empty.
No RL backbone — purely probabilistic with configurable bias.
"""

from __future__ import annotations
import logging
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentAction

logger = logging.getLogger(__name__)


class NoiseAgent(BaseAgent):

    def __init__(self, agent_id: int, cfg: dict | None = None, rng: np.random.Generator | None = None):
        super().__init__(agent_id, AgentType.NOISE, cfg)
        self._type_cfg   = self.cfg.get("noise", {})
        self._order_freq = self._type_cfg.get("order_frequency", 0.4)
        self._side_bias  = self._type_cfg.get("side_bias", 0.5)       # 0.5 = unbiased
        self._rng        = rng or np.random.default_rng()

    def compute_signal(self) -> tuple[float, str]:
        r = float(self._rng.uniform(-1, 1))
        return r, f"random_noise={r:.4f}"

    def act(self, observation: np.ndarray) -> int:
        # With probability (1 - order_freq), hold
        if self._rng.random() > self._order_freq:
            self._execute_trade(AgentAction.HOLD, self._snapshot.price if self._snapshot else 0.0)
            self._record_decision(AgentAction.HOLD, 0.0, "no_order_this_tick",
                                  reason_tags=["noise_hold"])
            return AgentAction.HOLD.value

        # Random side, random size
        buy = self._rng.random() < self._side_bias
        large = self._rng.random() < 0.3   # 30% chance of large order

        if buy:
            action = AgentAction.BUY_LARGE if large else AgentAction.BUY_SMALL
            tags = ["noise_buy"]
        else:
            if self.shares == 0:   # can't sell what you don't have
                action = AgentAction.HOLD
                tags = ["noise_hold_no_pos"]
            else:
                action = AgentAction.SELL_LARGE if large else AgentAction.SELL_SMALL
                tags = ["noise_sell"]

        signal, label = self.compute_signal()
        self._execute_trade(action, self._snapshot.price if self._snapshot else 0.0)
        self._record_decision(action, signal, label, reason_tags=tags)
        return action.value

if __name__ == "__main__":
    print("🔍 Testing NoiseAgent...\n")

    # Create agent
    agent = NoiseAgent(agent_id=0)

    # Fake observation (not really used, but required)
    obs = np.random.randn(47).astype(np.float32)

    # Run multiple steps to see randomness
    for step in range(10):
        print(f"\n--- Step {step} ---")

        action = agent.act(obs)

        decision = agent.last_decision()
        if decision:
            print("Action:", action)
            print("Decision:", decision.to_dict())

    # Stats after simulation
    print("\nAgent Stats:")
    print(agent.stats())