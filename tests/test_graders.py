"""Tests for per-task graders — determinism, score bounds, and edge cases."""
import pytest
from sentinelx.server.graders import (
    EpisodeRecord,
    grade_stolen_card,
    grade_account_takeover,
    grade_money_laundering,
    grade_episode,
)


def _make_record(task_id="stolen-card-easy", fraud_type="stolen_card", is_fraud=True, **kwargs):
    rec = EpisodeRecord(task_id=task_id, fraud_type=fraud_type, is_fraud=is_fraud)
    for k, v in kwargs.items():
        setattr(rec, k, v)
    return rec


# ---------------------------------------------------------------------------
# Task 1: Stolen Card
# ---------------------------------------------------------------------------

class TestGradeStolenCard:
    def test_perfect_score(self):
        rec = _make_record(
            correctly_blocked=True,
            evidence_gathered=["velocity_data", "device_history"],
            total_steps=2,
            final_action="block_transaction",
        )
        score = grade_stolen_card(rec)
        assert score >= 0.80  # 0.40 + 0.20 + 0.20 + speed bonus

    def test_false_negative_is_penalised(self):
        rec = _make_record(false_negative=True)
        score = grade_stolen_card(rec)
        assert score <= 0.0

    def test_false_positive_is_penalised(self):
        rec = _make_record(is_fraud=False, false_positive=True)
        score = grade_stolen_card(rec)
        assert score <= 0.0

    def test_3ds_gives_bonus(self):
        rec = _make_record(
            correctly_blocked=True,
            final_action="request_3ds",
            evidence_gathered=["velocity_data"],
        )
        score_3ds = grade_stolen_card(rec)
        rec2 = _make_record(
            correctly_blocked=True,
            final_action="block_transaction",
            evidence_gathered=["velocity_data"],
        )
        score_block = grade_stolen_card(rec2)
        assert score_3ds > score_block

    def test_score_bounds(self):
        for _ in range(20):
            rec = _make_record(correctly_blocked=True, evidence_gathered=["velocity_data", "device_history"])
            score = grade_stolen_card(rec)
            assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        rec = _make_record(correctly_blocked=True, evidence_gathered=["velocity_data"])
        s1 = grade_stolen_card(rec)
        s2 = grade_stolen_card(rec)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Task 2: Account Takeover
# ---------------------------------------------------------------------------

class TestGradeAccountTakeover:
    def test_correct_ato_with_all_evidence(self):
        rec = _make_record(
            task_id="account-takeover-medium",
            fraud_type="account_takeover",
            final_action="force_password_reset",
            correctly_blocked=True,
            evidence_gathered=["behavioral_biometrics", "active_sessions", "device_history"],
            actions_taken=["check_behavioral_biometrics", "check_active_sessions", "check_device_history", "force_password_reset"],
            total_steps=4,
        )
        score = grade_account_takeover(rec)
        assert score >= 0.70  # 0.40 + 0.15 + 0.15 + 0.10 + speed

    def test_biometrics_gives_partial_score(self):
        rec = _make_record(
            task_id="account-takeover-medium",
            fraud_type="account_takeover",
            final_action="force_password_reset",
            correctly_blocked=True,
            evidence_gathered=["behavioral_biometrics"],
            actions_taken=["check_behavioral_biometrics", "force_password_reset"],
            total_steps=2,
        )
        score = grade_account_takeover(rec)
        assert score > 0.40

    def test_approving_ato_is_catastrophic(self):
        rec = _make_record(
            task_id="account-takeover-medium",
            fraud_type="account_takeover",
            is_fraud=True,
            false_negative=True,
        )
        score = grade_account_takeover(rec)
        assert score < 0.0

    def test_score_bounds(self):
        rec = _make_record(
            task_id="account-takeover-medium",
            fraud_type="account_takeover",
            correctly_blocked=True,
            evidence_gathered=["behavioral_biometrics", "active_sessions"],
        )
        score = grade_account_takeover(rec)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        rec = _make_record(
            task_id="account-takeover-medium",
            fraud_type="account_takeover",
            correctly_blocked=True,
            evidence_gathered=["behavioral_biometrics"],
        )
        assert grade_account_takeover(rec) == grade_account_takeover(rec)


# ---------------------------------------------------------------------------
# Task 3: Money Laundering
# ---------------------------------------------------------------------------

class TestGradeMoneyLaundering:
    def test_full_investigation_and_sar(self):
        rec = _make_record(
            task_id="money-laundering-hard",
            fraud_type="money_laundering",
            sar_filed=True,
            network_depth_checked=2,
            temporal_pattern_checked=True,
            business_registration_checked=True,
            shell_companies_identified=4,
            total_laundered_amount=456700.0,
            structuring_detected=True,
            total_steps=9,
        )
        score = grade_money_laundering(rec)
        assert score >= 0.70

    def test_premature_sar_penalised(self):
        rec_good = _make_record(
            task_id="money-laundering-hard",
            fraud_type="money_laundering",
            sar_filed=True,
            network_depth_checked=2,
            shell_companies_identified=3,
            structuring_detected=True,
        )
        rec_bad = _make_record(
            task_id="money-laundering-hard",
            fraud_type="money_laundering",
            sar_filed=True,
            network_depth_checked=1,  # premature — filed before depth-2 check
            shell_companies_identified=3,
            structuring_detected=True,
        )
        assert grade_money_laundering(rec_good) > grade_money_laundering(rec_bad)

    def test_missed_sar_on_structuring_penalised(self):
        rec = _make_record(
            task_id="money-laundering-hard",
            fraud_type="money_laundering",
            sar_filed=False,
            structuring_detected=True,
            network_depth_checked=2,
        )
        score = grade_money_laundering(rec)
        assert score < 0.0

    def test_too_many_steps_penalised(self):
        rec_fast = _make_record(
            task_id="money-laundering-hard",
            fraud_type="money_laundering",
            sar_filed=True,
            network_depth_checked=2,
            shell_companies_identified=3,
            total_steps=10,
        )
        rec_slow = _make_record(
            task_id="money-laundering-hard",
            fraud_type="money_laundering",
            sar_filed=True,
            network_depth_checked=2,
            shell_companies_identified=3,
            total_steps=20,
        )
        assert grade_money_laundering(rec_fast) > grade_money_laundering(rec_slow)

    def test_score_within_extended_bounds(self):
        rec = _make_record(task_id="money-laundering-hard", fraud_type="money_laundering")
        score = grade_money_laundering(rec)
        assert -1.0 <= score <= 1.0

    def test_deterministic(self):
        rec = _make_record(
            task_id="money-laundering-hard",
            fraud_type="money_laundering",
            sar_filed=True,
            network_depth_checked=2,
            shell_companies_identified=3,
        )
        assert grade_money_laundering(rec) == grade_money_laundering(rec)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestGradeEpisodeDispatcher:
    def test_dispatches_to_correct_grader(self):
        rec = _make_record(task_id="stolen-card-easy", correctly_blocked=True)
        score = grade_episode(rec)
        assert 0.0 <= score <= 1.0

    def test_unknown_task_raises(self):
        rec = _make_record(task_id="nonexistent")
        with pytest.raises(ValueError):
            grade_episode(rec)
