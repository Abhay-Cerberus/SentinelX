"""
SentinelX — Pydantic type contracts.

All classes inherit from openenv-core primitives so the framework's
WebSocket serialisation/deserialisation works without extra plumbing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from openenv.core.env_server import Action, Observation, State  # type: ignore


# ---------------------------------------------------------------------------
# Inner domain types (not OpenEnv primitives)
# ---------------------------------------------------------------------------

class Transaction:
    """Plain dataclass-style object (not a Pydantic model itself) for
    cleaner nesting inside FraudObservation."""

    def __init__(
        self,
        transaction_id: str,
        amount: float,
        merchant: str,
        location: str,
        timestamp: str,
        currency: str = "USD",
    ) -> None:
        self.transaction_id = transaction_id
        self.amount = amount
        self.merchant = merchant
        self.location = location
        self.timestamp = timestamp
        self.currency = currency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "amount": self.amount,
            "merchant": self.merchant,
            "location": self.location,
            "timestamp": self.timestamp,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transaction":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class UserProfile:
    def __init__(
        self,
        user_id: str,
        account_age_days: int,
        typical_location: str,
        typical_transaction_size: float,
        card_present_ratio: float,
        name: Optional[str] = None,
    ) -> None:
        self.user_id = user_id
        self.account_age_days = account_age_days
        self.typical_location = typical_location
        self.typical_transaction_size = typical_transaction_size
        self.card_present_ratio = card_present_ratio
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "account_age_days": self.account_age_days,
            "typical_location": self.typical_location,
            "typical_transaction_size": self.typical_transaction_size,
            "card_present_ratio": self.card_present_ratio,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# OpenEnv action model
# ---------------------------------------------------------------------------

ACTION_TYPES = Literal[
    # --- Investigation ---
    "query_velocity",
    "check_device_history",
    "lookup_ip_reputation",
    "check_behavioral_biometrics",
    "check_active_sessions",
    "query_linked_accounts",
    "analyze_temporal_pattern",
    "check_business_registration",
    "request_kyc_documents",
    # --- Intervention ---
    "approve_transaction",
    "block_transaction",
    "request_3ds",
    "send_push_notification",
    "force_password_reset",
    "temporarily_freeze_account",
    # --- Regulatory ---
    "file_sar",
    "file_ctr",
    "escalate_to_compliance",
    # --- Special ---
    "monitor_only",
    "request_additional_info",
]

INVESTIGATION_ACTIONS = {
    "query_velocity",
    "check_device_history",
    "lookup_ip_reputation",
    "check_behavioral_biometrics",
    "check_active_sessions",
    "query_linked_accounts",
    "analyze_temporal_pattern",
    "check_business_registration",
    "request_kyc_documents",
}

INTERVENTION_ACTIONS = {
    "approve_transaction",
    "block_transaction",
    "request_3ds",
    "send_push_notification",
    "force_password_reset",
    "temporarily_freeze_account",
}

REGULATORY_ACTIONS = {"file_sar", "file_ctr", "escalate_to_compliance"}

TERMINAL_ACTIONS = INTERVENTION_ACTIONS | REGULATORY_ACTIONS


class FraudAction(Action):
    """Action sent by the agent each step."""

    action_type: ACTION_TYPES
    parameters: Dict[str, Any] = {}
    reasoning: str = ""


# ---------------------------------------------------------------------------
# OpenEnv observation model
# ---------------------------------------------------------------------------

class FraudObservation(Observation):
    """Partial view of environment state returned after each step.

    Fields beyond *transaction* and *user_profile* are None until
    the agent queries them with the appropriate investigation action.
    """

    # Always visible
    transaction: Dict[str, Any]
    user_profile: Dict[str, Any]

    # Revealed progressively by investigation actions
    velocity_data: Optional[Dict[str, Any]] = None
    device_history: Optional[Dict[str, Any]] = None
    ip_reputation: Optional[Dict[str, Any]] = None
    network_connections: Optional[List[Dict[str, Any]]] = None
    behavioral_biometrics: Optional[Dict[str, Any]] = None
    active_sessions: Optional[List[Dict[str, Any]]] = None
    temporal_pattern: Optional[Dict[str, Any]] = None
    business_registration: Optional[Dict[str, Any]] = None

    # Navigation
    available_actions: List[str] = []
    time_remaining: int = 20
    evidence_summary: str = ""
    last_action_result: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenEnv state model
# ---------------------------------------------------------------------------

class FraudInvestigationState(State):
    """Full episode metadata (visible to the framework; partially hidden from agent)."""

    task_id: str = ""
    fraud_type: str = "unknown"          # "stolen_card" | "account_takeover" | "money_laundering" | "legitimate"
    fraudster_strategy: str = "basic"    # Current adversary strategy (hidden from agent)
    agent_detection_patterns: List[str] = []   # Signals agent has relied on historically
    investigation_ticks: int = 0
    fraud_in_progress: bool = True
    sar_filed: bool = False
    ctr_filed: bool = False
    final_action: Optional[str] = None
