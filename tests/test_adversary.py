"""Tests for FraudsterAgent adaptive logic and strategy escalation."""
import pytest
from sentinelx.adversary.fraudster import FraudsterAgent
from sentinelx.adversary.strategies import (
    HighVelocityStrategy,
    DeviceSpoofingStrategy,
    BehavioralMimicryStrategy,
    StructuredPaymentsStrategy,
    ShellNetworkStrategy,
    STRATEGY_REGISTRY,
)


USER_PROFILE = {
    "user_id": "U001",
    "account_age_days": 892,
    "typical_location": "Seattle, WA",
    "typical_transaction_size": 45.00,
    "card_present_ratio": 0.85,
    "primary_device": "iPhone 14",
    "primary_browser": "Safari on iOS",
    "typical_login_hours": [8, 9, 12, 18],
    "typing_wpm_baseline": 52,
    "mouse_pattern": "human_curved",
    "average_session_minutes": 8,
}


class TestFraudsterAgentInit:
    def test_default_strategy_stolen_card(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        assert agent.strategy_name == "high_velocity"

    def test_default_strategy_ato(self):
        agent = FraudsterAgent(task_id="account-takeover-medium")
        assert agent.strategy_name == "behavioral_mimicry"

    def test_default_strategy_ml(self):
        agent = FraudsterAgent(task_id="money-laundering-hard")
        assert agent.strategy_name == "structured_payments"

    def test_from_history_class_method(self):
        agent = FraudsterAgent.from_history("stolen-card-easy")
        assert agent.task_id == "stolen-card-easy"


class TestFraudsterAdaptation:
    def test_velocity_detection_escalates_to_device_spoofing(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        assert agent.strategy_name == "high_velocity"
        agent.adapt("blocked_via_velocity")
        assert agent.strategy_name == "device_spoofing"

    def test_device_detection_escalates_to_behavioral(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        agent.adapt("blocked_via_velocity")     # → device_spoofing
        agent.adapt("blocked_via_device")       # → behavioral_mimicry
        assert agent.strategy_name == "behavioral_mimicry"

    def test_apex_strategy_does_not_change(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        agent.strategy_name = "behavioral_mimicry"
        agent.adapt("blocked_via_behavioral")
        assert agent.strategy_name == "behavioral_mimicry"  # apex

    def test_ml_network_analysis_escalates(self):
        agent = FraudsterAgent(task_id="money-laundering-hard")
        agent.adapt("blocked_via_network_analysis")
        assert agent.strategy_name == "shell_network"

    def test_detection_history_accumulates(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        agent.adapt("blocked_via_velocity")
        agent.adapt("blocked_via_device")
        assert len(agent.detection_methods_seen) == 2

    def test_history_replay_restores_state(self):
        history = [{"detected_via": "blocked_via_velocity"}]
        agent = FraudsterAgent(task_id="stolen-card-easy", episode_history=history)
        assert agent.strategy_name == "device_spoofing"

    def test_multi_episode_history_replay(self):
        history = [
            {"detected_via": "blocked_via_velocity"},
            {"detected_via": "blocked_via_device"},
        ]
        agent = FraudsterAgent(task_id="stolen-card-easy", episode_history=history)
        assert agent.strategy_name == "behavioral_mimicry"


class TestScenarioGeneration:
    def test_generates_scenario(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        scenario = agent.generate_scenario(seed=42, user_profile=USER_PROFILE)
        assert scenario is not None
        assert scenario.fraud_type == "stolen_card"
        assert "amount" in scenario.transaction

    def test_scenario_reproducible_same_seed(self):
        agent1 = FraudsterAgent(task_id="stolen-card-easy")
        agent2 = FraudsterAgent(task_id="stolen-card-easy")
        s1 = agent1.generate_scenario(seed=99, user_profile=USER_PROFILE)
        s2 = agent2.generate_scenario(seed=99, user_profile=USER_PROFILE)
        assert s1.transaction["amount"] == s2.transaction["amount"]

    def test_different_seeds_different_amounts(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        s1 = agent.generate_scenario(seed=1, user_profile=USER_PROFILE)
        s2 = agent.generate_scenario(seed=2, user_profile=USER_PROFILE)
        # Amounts will differ (not guaranteed but statistically certain)
        assert s1.transaction["amount"] != s2.transaction["amount"]

    def test_state_dict(self):
        agent = FraudsterAgent(task_id="stolen-card-easy")
        agent.adapt("blocked_via_velocity")
        d = agent.state_dict()
        assert d["task_id"] == "stolen-card-easy"
        assert d["strategy_name"] == "device_spoofing"
        assert "blocked_via_velocity" in d["detection_methods_seen"]


class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        expected = {"high_velocity", "device_spoofing", "behavioral_mimicry",
                    "structured_payments", "shell_network"}
        assert expected == set(STRATEGY_REGISTRY.keys())

    def test_each_strategy_generates_scenario(self):
        user = USER_PROFILE
        for name, cls in STRATEGY_REGISTRY.items():
            strategy = cls()
            scenario = strategy.generate("stolen-card-easy", seed=42, user_profile=user)
            assert scenario.transaction
            assert scenario.fraud_type
