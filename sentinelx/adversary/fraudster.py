"""Adaptive adversary that learns from agent detection patterns across episodes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .strategies import (
    STRATEGY_REGISTRY,
    BehavioralMimicryStrategy,
    DeviceSpoofingStrategy,
    FraudStrategy,
    HighVelocityStrategy,
    Scenario,
    ShellNetworkStrategy,
    StructuredPaymentsStrategy,
)


class FraudsterAgent:
    """Stateful opponent that escalates tactics as the agent detects it.

    Detection patterns are accumulated across episodes. Each call to
    ``adapt(detected_via)`` shifts the strategy toward harder-to-detect
    methods — mirroring real-world fraud evolution.
    """

    TASK_DEFAULT_STRATEGIES: Dict[str, str] = {
        "stolen-card-easy": "high_velocity",
        "account-takeover-medium": "behavioral_mimicry",
        "money-laundering-hard": "structured_payments",
    }

    ESCALATION_MAP: Dict[str, Dict[str, str]] = {
        # stolen card escalations
        "high_velocity": {"velocity": "device_spoofing"},
        "device_spoofing": {"device": "behavioral_mimicry"},
        # money laundering escalations
        "structured_payments": {"network_analysis": "shell_network"},
        "shell_network": {},  # already at apex
        # ATO escalations
        "behavioral_mimicry": {},  # apex tactic
    }

    def __init__(
        self,
        task_id: str,
        episode_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.task_id = task_id
        self.detection_methods_seen: List[str] = []
        self.strategy_name: str = self.TASK_DEFAULT_STRATEGIES.get(task_id, "high_velocity")

        # Replay past episodes to restore adversary state
        if episode_history:
            for ep in episode_history:
                method = ep.get("detected_via")
                if method:
                    self._do_adapt(method)

    # ------------------------------------------------------------------

    def adapt(self, detected_via: str) -> None:
        """Call when the agent successfully catches fraud, specifying how."""
        self._do_adapt(detected_via)

    def _do_adapt(self, method: str) -> None:
        self.detection_methods_seen.append(method)
        escalations = self.ESCALATION_MAP.get(self.strategy_name, {})
        for trigger, new_strategy in escalations.items():
            if trigger in method:
                self.strategy_name = new_strategy
                break

    def generate_scenario(self, seed: int, user_profile: Dict[str, Any]) -> Scenario:
        """Produce a fraud scenario using the current (possibly escalated) strategy."""
        strategy_cls = STRATEGY_REGISTRY.get(self.strategy_name, HighVelocityStrategy)
        strategy: FraudStrategy = strategy_cls()
        scenario = strategy.generate(self.task_id, seed, user_profile)
        scenario.__dict__["fraudster_strategy"] = self.strategy_name
        return scenario

    # ------------------------------------------------------------------

    @classmethod
    def from_history(
        cls,
        task_id: str,
        episode_history: Optional[List[Dict[str, Any]]] = None,
    ) -> "FraudsterAgent":
        return cls(task_id=task_id, episode_history=episode_history)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy_name": self.strategy_name,
            "detection_methods_seen": list(self.detection_methods_seen),
        }
