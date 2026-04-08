"""SentinelX EnvClient — translates typed models ↔ WebSocket JSON wire format.

Usage:
    from sentinelx.client import SentinelXEnv, FraudAction

    with SentinelXEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="stolen-card-easy", seed=42)
        print(result.observation.evidence_summary)
        result = env.step(FraudAction(
            action_type="check_device_history",
            reasoning="Amount and location match but merchant is high-risk"
        ))
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.env_client import EnvClient  # type: ignore
from openenv.core.client_types import StepResult  # type: ignore

from sentinelx.models import FraudAction, FraudObservation, FraudInvestigationState


class SentinelXEnv(EnvClient[FraudAction, FraudObservation, FraudInvestigationState]):
    """WebSocket client for the SentinelX fraud investigation environment."""

    # ------------------------------------------------------------------ #
    # Abstract method implementations required by EnvClient
    # ------------------------------------------------------------------ #

    def _step_payload(self, action: FraudAction) -> Dict[str, Any]:
        """Serialise FraudAction to JSON for the WebSocket wire."""
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """Parse JSON response into a typed StepResult[FraudObservation]."""
        obs_data: Dict[str, Any] = payload.get("observation", {})

        observation = FraudObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            transaction=obs_data.get("transaction", {}),
            user_profile=obs_data.get("user_profile", {}),
            velocity_data=obs_data.get("velocity_data"),
            device_history=obs_data.get("device_history"),
            ip_reputation=obs_data.get("ip_reputation"),
            network_connections=obs_data.get("network_connections"),
            behavioral_biometrics=obs_data.get("behavioral_biometrics"),
            active_sessions=obs_data.get("active_sessions"),
            temporal_pattern=obs_data.get("temporal_pattern"),
            business_registration=obs_data.get("business_registration"),
            available_actions=obs_data.get("available_actions", []),
            time_remaining=obs_data.get("time_remaining", 0),
            evidence_summary=obs_data.get("evidence_summary", ""),
            last_action_result=obs_data.get("last_action_result"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FraudInvestigationState:
        """Parse JSON response into a typed FraudInvestigationState."""
        return FraudInvestigationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            fraud_type=payload.get("fraud_type", "unknown"),
            fraudster_strategy=payload.get("fraudster_strategy", "basic"),
            agent_detection_patterns=payload.get("agent_detection_patterns", []),
            investigation_ticks=payload.get("investigation_ticks", 0),
            fraud_in_progress=payload.get("fraud_in_progress", True),
            sar_filed=payload.get("sar_filed", False),
            ctr_filed=payload.get("ctr_filed", False),
            final_action=payload.get("final_action"),
        )
