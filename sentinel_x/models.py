from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, List, Optional
from openenv.core import Action as BaseAction
from openenv.core import Observation as BaseObservation
from openenv.core import State as BaseState

class Transaction(BaseModel):
    id: str
    amount: float
    merchant: str
    location: str
    time: str

class UserProfile(BaseModel):
    account_age_days: int
    typical_location: str
    typical_transaction_size: float
    card_present_ratio: float = 0.0
    business_type: str = "Individual"

class Evidence(BaseModel):
    type: str # e.g., 'velocity', 'device', 'ip', 'network', 'behavioral'
    data: Dict[str, Any]
    summary: str

class AccountLink(BaseModel):
    source: str
    target: str
    relation: str

class Observation(BaseObservation):
    transaction: Transaction
    user_profile: UserProfile
    
    velocity_data: Optional[Dict[str, Any]] = None
    device_history: Optional[Dict[str, Any]] = None
    ip_reputation: Optional[Dict[str, Any]] = None
    network_connections: Optional[List[AccountLink]] = None
    behavioral_biometrics: Optional[Dict[str, Any]] = None
    active_sessions: Optional[List[Dict[str, Any]]] = None
    
    available_actions: List[str]
    time_remaining: int
    evidence_summary: str

class Action(BaseAction):
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = ""

class FraudsterStrategy(BaseModel):
    method: str
    adaptation_level: int
    history: List[str] = Field(default_factory=list)

class State(BaseState):
    current_transaction: Transaction
    gathered_evidence: Dict[str, Evidence] = Field(default_factory=dict)
    user_profile: UserProfile
    visible_network: Dict[str, Any] = Field(default_factory=dict)
    
    investigation_ticks: int = 0
    max_ticks: int = 20
    fraud_in_progress: bool = True
    
    fraudster_strategy: FraudsterStrategy
    true_fraud_type: str
    
    agent_detection_patterns: List[str] = Field(default_factory=list)
    is_terminal: bool = False

class TaskConfig(BaseModel):
    task_name: str
    difficulty: str
    max_steps: int = 20

class RewardBreakdown(BaseModel):
    total: float = 0.0
    investigation_bonus: float = 0.0
    decision_reward: float = 0.0
    time_penalty: float = 0.0
    compliance_reward: float = 0.0
