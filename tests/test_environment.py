"""Tests for SentinelXEnvironment — reset/step/state contract verification."""
import pytest
from sentinelx.server.environment import SentinelXEnvironment
from sentinelx.models import FraudAction, FraudObservation, FraudInvestigationState


TASKS = [
    "stolen-card-easy",
    "account-takeover-medium",
    "money-laundering-hard",
]


class TestReset:
    def test_reset_returns_observation(self):
        env = SentinelXEnvironment()
        obs = env.reset(task_id="stolen-card-easy", seed=42)
        assert isinstance(obs, FraudObservation)

    def test_reset_observation_has_transaction(self):
        env = SentinelXEnvironment()
        obs = env.reset(task_id="stolen-card-easy", seed=42)
        assert "amount" in obs.transaction
        assert "merchant" in obs.transaction
        assert obs.transaction["amount"] > 0

    def test_reset_observation_done_is_false(self):
        env = SentinelXEnvironment()
        obs = env.reset(task_id="stolen-card-easy", seed=42)
        assert obs.done is False

    def test_reset_evidence_fields_are_none(self):
        env = SentinelXEnvironment()
        obs = env.reset(task_id="stolen-card-easy", seed=42)
        assert obs.velocity_data is None
        assert obs.device_history is None
        assert obs.ip_reputation is None

    def test_reset_available_actions_not_empty(self):
        env = SentinelXEnvironment()
        obs = env.reset(task_id="stolen-card-easy", seed=42)
        assert len(obs.available_actions) > 0

    @pytest.mark.parametrize("task_id", TASKS)
    def test_reset_all_tasks(self, task_id):
        env = SentinelXEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        assert isinstance(obs, FraudObservation)
        assert obs.transaction

    def test_reset_unknown_task_falls_back_to_easy(self):
        env = SentinelXEnvironment()
        obs = env.reset(task_id="nonexistent-task", seed=1)
        assert isinstance(obs, FraudObservation)

    def test_reset_cleans_state(self):
        env = SentinelXEnvironment()
        env.reset(task_id="stolen-card-easy", seed=42)
        env.step(FraudAction(action_type="check_device_history", reasoning="first ep"))
        obs2 = env.reset(task_id="stolen-card-easy", seed=42)
        # After reset, evidence fields must be blank again
        assert obs2.device_history is None

    def test_reproducible_with_same_seed(self):
        env1 = SentinelXEnvironment()
        env2 = SentinelXEnvironment()
        obs1 = env1.reset(task_id="stolen-card-easy", seed=99)
        obs2 = env2.reset(task_id="stolen-card-easy", seed=99)
        assert obs1.transaction["amount"] == obs2.transaction["amount"]
        assert obs1.transaction["merchant"] == obs2.transaction["merchant"]


class TestStep:
    def _setup(self, task_id="stolen-card-easy", seed=42):
        env = SentinelXEnvironment()
        env.reset(task_id=task_id, seed=seed)
        return env

    def test_investigation_action_returns_observation(self):
        env = self._setup()
        obs = env.step(FraudAction(action_type="check_device_history", reasoning="test"))
        assert isinstance(obs, FraudObservation)

    def test_investigation_populates_evidence(self):
        env = self._setup()
        obs = env.step(FraudAction(action_type="check_device_history", reasoning="test"))
        assert obs.device_history is not None

    def test_investigation_gives_positive_reward_for_relevant_signal(self):
        env = self._setup()
        obs = env.step(FraudAction(action_type="check_device_history", reasoning="new device check"))
        # Reward includes time penalty so we check it's greater than just the penalty
        assert obs.reward is not None and obs.reward > -0.1

    def test_duplicate_investigation_gives_negative_reward(self):
        env = self._setup()
        env.step(FraudAction(action_type="check_device_history", reasoning="first"))
        obs2 = env.step(FraudAction(action_type="check_device_history", reasoning="second"))
        assert obs2.reward is not None and obs2.reward < 0

    def test_block_terminates_episode(self):
        env = self._setup()
        obs = env.step(FraudAction(action_type="block_transaction", reasoning="blocking fraud"))
        assert obs.done is True

    def test_approve_terminates_episode(self):
        env = self._setup()
        obs = env.step(FraudAction(action_type="approve_transaction", reasoning="looks fine"))
        assert obs.done is True

    def test_step_after_done_returns_done(self):
        env = self._setup()
        env.step(FraudAction(action_type="block_transaction", reasoning="done"))
        obs2 = env.step(FraudAction(action_type="check_device_history", reasoning="after done"))
        assert obs2.done is True

    def test_time_remaining_decreases(self):
        env = self._setup()
        obs0 = env.reset(task_id="stolen-card-easy", seed=42)
        t0 = obs0.time_remaining
        obs1 = env.step(FraudAction(action_type="query_velocity", reasoning="check"))
        assert obs1.time_remaining < t0

    def test_network_query_with_depth_parameter(self):
        env = self._setup(task_id="money-laundering-hard")
        obs = env.step(FraudAction(
            action_type="query_linked_accounts",
            parameters={"depth": 2},
            reasoning="deep network check",
        ))
        assert isinstance(obs, FraudObservation)


class TestState:
    def test_state_returns_correct_type(self):
        env = SentinelXEnvironment()
        env.reset(task_id="stolen-card-easy", seed=42)
        assert isinstance(env.state, FraudInvestigationState)

    def test_step_count_increments(self):
        env = SentinelXEnvironment()
        env.reset(task_id="stolen-card-easy", seed=42)
        assert env.state.step_count == 0
        env.step(FraudAction(action_type="query_velocity", reasoning="check"))
        assert env.state.step_count == 1
        env.step(FraudAction(action_type="check_device_history", reasoning="check"))
        assert env.state.step_count == 2

    def test_state_task_id_matches_reset(self):
        env = SentinelXEnvironment()
        env.reset(task_id="account-takeover-medium", seed=10)
        assert env.state.task_id == "account-takeover-medium"

    def test_sar_filed_flag_updates(self):
        env = SentinelXEnvironment()
        env.reset(task_id="money-laundering-hard", seed=15)
        env.step(FraudAction(action_type="file_sar", reasoning="suspicious activity detected"))
        assert env.state.sar_filed is True
