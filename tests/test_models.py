"""Tests for Pydantic model serialisation and validation."""
import pytest
from sentinelx.models import (
    FraudAction,
    FraudObservation,
    FraudInvestigationState,
    INVESTIGATION_ACTIONS,
    INTERVENTION_ACTIONS,
    REGULATORY_ACTIONS,
)


class TestFraudAction:
    def test_valid_investigation_action(self):
        a = FraudAction(action_type="check_device_history", reasoning="testing")
        assert a.action_type == "check_device_history"
        assert a.parameters == {}
        assert a.reasoning == "testing"

    def test_valid_intervention_action(self):
        a = FraudAction(
            action_type="block_transaction",
            reasoning="High risk score",
        )
        assert a.action_type == "block_transaction"

    def test_action_with_parameters(self):
        a = FraudAction(
            action_type="query_linked_accounts",
            parameters={"depth": 2},
            reasoning="Checking network depth",
        )
        assert a.parameters == {"depth": 2}

    def test_invalid_action_type_raises(self):
        with pytest.raises(Exception):
            FraudAction(action_type="do_nothing_invalid")

    def test_round_trip_json(self):
        a = FraudAction(
            action_type="file_sar",
            parameters={},
            reasoning="Structuring detected",
        )
        dumped = a.model_dump()
        restored = FraudAction(**dumped)
        assert restored.action_type == a.action_type
        assert restored.reasoning == a.reasoning

    def test_action_sets_are_disjoint(self):
        """Investigation, intervention, and regulatory sets must not overlap."""
        assert not (INVESTIGATION_ACTIONS & INTERVENTION_ACTIONS)
        assert not (INVESTIGATION_ACTIONS & REGULATORY_ACTIONS)
        assert not (INTERVENTION_ACTIONS & REGULATORY_ACTIONS)


class TestFraudObservation:
    def _make_obs(self, **kwargs):
        defaults = dict(
            done=False,
            reward=None,
            transaction={"amount": 100.0, "merchant": "Test", "location": "NY", "timestamp": "2024-01-01T00:00:00Z"},
            user_profile={"user_id": "U001", "account_age_days": 100},
            available_actions=["approve_transaction", "block_transaction"],
            time_remaining=15,
            evidence_summary="No evidence gathered yet.",
        )
        defaults.update(kwargs)
        return FraudObservation(**defaults)

    def test_default_evidence_fields_are_none(self):
        obs = self._make_obs()
        assert obs.velocity_data is None
        assert obs.device_history is None
        assert obs.ip_reputation is None
        assert obs.behavioral_biometrics is None
        assert obs.active_sessions is None
        assert obs.network_connections is None

    def test_revealed_evidence_field(self):
        obs = self._make_obs(velocity_data={"past_24_hours_count": 12, "anomaly_score": 0.9})
        assert obs.velocity_data is not None
        assert obs.velocity_data["anomaly_score"] == 0.9

    def test_round_trip_json(self):
        obs = self._make_obs(done=True, reward=0.3)
        dumped = obs.model_dump()
        restored = FraudObservation(**dumped)
        assert restored.done is True
        assert restored.reward == pytest.approx(0.3)

    def test_done_false_by_default(self):
        obs = self._make_obs()
        assert obs.done is False


class TestFraudInvestigationState:
    def test_defaults(self):
        state = FraudInvestigationState()
        assert state.step_count == 0
        assert state.task_id == ""
        assert state.fraud_type == "unknown"
        assert state.sar_filed is False
        assert state.ctr_filed is False

    def test_fields_settable(self):
        state = FraudInvestigationState(
            episode_id="ep-001",
            task_id="stolen-card-easy",
            fraud_type="stolen_card",
            step_count=3,
            investigation_ticks=3,
            fraud_in_progress=True,
        )
        assert state.task_id == "stolen-card-easy"
        assert state.step_count == 3
