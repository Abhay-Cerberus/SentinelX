"""Per-task deterministic graders for SentinelX.

Each function receives an EpisodeRecord and returns a score in [0.0, 1.0].
Graders are fully deterministic: same episode + same actions = same score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Episode record — built up by the environment, passed to grader at episode end
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    task_id: str
    fraud_type: str                           # ground truth
    is_fraud: bool

    actions_taken: List[str] = field(default_factory=list)
    evidence_gathered: List[str] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)

    final_action: Optional[str] = None
    final_action_params: Dict[str, Any] = field(default_factory=dict)

    # Outcome flags set by environment after terminal action
    correctly_blocked: bool = False
    correctly_approved: bool = False
    false_positive: bool = False
    false_negative: bool = False

    sar_filed: bool = False
    ctr_filed: bool = False
    escalated: bool = False

    # Money laundering specifics
    network_depth_checked: int = 0
    temporal_pattern_checked: bool = False
    business_registration_checked: bool = False
    shell_companies_identified: int = 0
    total_laundered_amount: float = 0.0
    structuring_detected: bool = False

    total_steps: int = 0
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Task 1: Stolen Card
# ---------------------------------------------------------------------------

def grade_stolen_card(ep: EpisodeRecord) -> float:
    """
    Score breakdown:
      +0.40  correct block
      +0.30  correct approval (legitimate)
      +0.20  gathered velocity before deciding
      +0.20  gathered device before deciding
      -0.30  false positive
      -0.50  false negative
    Max achievable: 0.80 (block + velocity + device) — extra reward given for 3DS
    """
    score = 0.0

    # Core decision
    if ep.correctly_blocked:
        score += 0.40
    elif ep.correctly_approved:
        score += 0.30
    elif ep.false_positive:
        score -= 0.30
    elif ep.false_negative:
        score -= 0.50

    # Investigation quality
    if "velocity_data" in ep.evidence_gathered:
        score += 0.20
    if "device_history" in ep.evidence_gathered:
        score += 0.20

    # Bonus: 3DS instead of hard block
    if ep.final_action == "request_3ds" and ep.is_fraud:
        score += 0.10

    # Speed bonus: decisive within 4 steps
    if ep.total_steps <= 4 and (ep.correctly_blocked or ep.final_action == "request_3ds"):
        score += 0.10

    return _clamp(score)


# ---------------------------------------------------------------------------
# Task 2: Account Takeover
# ---------------------------------------------------------------------------

def grade_account_takeover(ep: EpisodeRecord) -> float:
    """
    Score breakdown:
      +0.40  correct decision (force_password_reset or send_push_notification or freeze)
      +0.15  checked behavioral biometrics
      +0.15  checked active sessions
      +0.10  checked device fingerprint
      +0.20  speed bonus (≤ 3 evidence steps before decision)
      -0.50  approved an ATO
      -0.30  false positive reset on legitimate user
    """
    score = 0.0

    # Core decision
    good_ato_actions = {"force_password_reset", "send_push_notification", "temporarily_freeze_account"}
    if ep.correctly_blocked or ep.final_action in good_ato_actions:
        if ep.is_fraud:
            score += 0.40
    elif ep.false_negative:
        score -= 0.50
    elif ep.false_positive:
        score -= 0.30

    # Investigation quality
    if "behavioral_biometrics" in ep.evidence_gathered:
        score += 0.15
    if "active_sessions" in ep.evidence_gathered:
        score += 0.15
    if "device_history" in ep.evidence_gathered:
        score += 0.10

    # Speed: efficient if ≤ 3 investigation steps
    investigation_steps = sum(
        1 for a in ep.actions_taken
        if a in {
            "query_velocity", "check_device_history", "lookup_ip_reputation",
            "check_behavioral_biometrics", "check_active_sessions",
        }
    )
    if investigation_steps <= 3 and (ep.correctly_blocked or ep.final_action in good_ato_actions):
        score += 0.20

    return _clamp(score)


# ---------------------------------------------------------------------------
# Task 3: Money Laundering
# ---------------------------------------------------------------------------

def grade_money_laundering(ep: EpisodeRecord) -> float:
    """
    Score breakdown (complex):
      +0.30  identified network AND filed SAR
      +0.20  identified all shell companies  (≥ 3)
      +0.15  correctly calculated total laundered amount (within 20%)
      +0.15  checked depth-2 network
      +0.10  analyzed temporal patterns
      +0.10  verified business registrations
      -0.10  took > 15 steps
      -0.20  premature SAR (filed before depth-2 check)
      -0.30  amount > $10K and did not file CTR (regulatory violation)
      -0.40  structuring detected but no SAR filed
    """
    score = 0.0

    # Core: identify + report
    if ep.shell_companies_identified >= 3 and ep.sar_filed:
        score += 0.30
    elif ep.shell_companies_identified >= 1 and ep.sar_filed:
        score += 0.15

    # Shell company completeness
    if ep.shell_companies_identified >= 3:
        score += 0.20
    elif ep.shell_companies_identified >= 1:
        score += 0.10

    # Amount accuracy (within 20% of true value)
    if ep.total_laundered_amount > 0:
        true_amount = 456700.0
        pct_error = abs(ep.total_laundered_amount - true_amount) / true_amount
        if pct_error <= 0.20:
            score += 0.15

    # Thoroughness
    if ep.network_depth_checked >= 2:
        score += 0.15
    if ep.temporal_pattern_checked:
        score += 0.10
    if ep.business_registration_checked:
        score += 0.10

    # Penalties
    if ep.total_steps > 15:
        score -= 0.10

    # Premature SAR: filed before doing depth-2 investigation
    if ep.sar_filed and ep.network_depth_checked < 2:
        score -= 0.20

    # Regulatory violations
    txn_amount = ep.final_action_params.get("transaction_amount", 0)
    if txn_amount >= 10000 and not ep.ctr_filed:
        score -= 0.30

    if ep.structuring_detected and not ep.sar_filed:
        score -= 0.40

    return _clamp(score, lo=-1.0, hi=1.0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "stolen-card-easy": grade_stolen_card,
    "account-takeover-medium": grade_account_takeover,
    "money-laundering-hard": grade_money_laundering,
}


def grade_episode(ep: EpisodeRecord) -> float:
    grader = GRADERS.get(ep.task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task_id={ep.task_id!r}")
    return grader(ep)
