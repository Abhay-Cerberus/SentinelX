from typing import Dict, Any, Optional
from openenv.core import Environment
import uuid
from sentinel_x.models import State, Action, Observation, RewardBreakdown
from sentinel_x.server.tasks.easy import EasyTask
from sentinel_x.server.tasks.medium import MediumTask
from sentinel_x.server.tasks.hard import HardTask
from sentinel_x.server.grader import grade_action

class FraudInvestigationEnv(Environment[Action, Observation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state: Optional[State] = None
        self.current_task = None
    
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_name: str = "easy", **kwargs: Any) -> Observation:
        if episode_id is None:
            episode_id = str(uuid.uuid4())
            
        if task_name == "easy":
            self.current_task = EasyTask()
        elif task_name == "medium":
            self.current_task = MediumTask()
        elif task_name == "hard":
            self.current_task = HardTask()
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        self._state = self.current_task.initialize_state()
        self._state.episode_id = episode_id
        self._state.step_count = 0
        
        return self._get_observation(self._state)
        
    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if self._state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self._state.is_terminal:
            raise ValueError("Episode is already terminal. Call reset().")
            
        self._state.step_count += 1
            
        # 1. Apply action to transition state
        next_state = self.current_task.transition(self._state, action)
        
        # 2. Grade action to get reward and terminal info
        score, reward_breakdown, success, reason = grade_action(
            self.current_task.name, self._state, action, next_state
        )
        
        # Determine if action is terminal directly (e.g. decision actions)
        TERMINAL_ACTIONS = [
            "approve_transaction", "block_transaction", "force_password_reset",
            "temporarily_freeze_account", "file_sar"
        ]
        if action.action_type in TERMINAL_ACTIONS or next_state.investigation_ticks >= next_state.max_ticks:
            next_state.is_terminal = True
            if next_state.investigation_ticks >= next_state.max_ticks and not success:
                reason = "Time elapsed. Fraud was not prevented."
            
        self._state = next_state
        
        obs = self._get_observation(self._state)
        obs.reward = score
        obs.done = self._state.is_terminal
        obs.metadata = {
            "reward_breakdown": reward_breakdown.model_dump(),
            "success": success,
            "reason": reason
        }
        
        return obs
        
    @property
    def state(self) -> State:
        if self._state is None:
            raise ValueError("Environment not initialized.")
        return self._state

    def _get_observation(self, state: State) -> Observation:
        # Construct Observation from State based on what is visible
        def get_ev_data(key: str):
            ev = state.gathered_evidence.get(key)
            return ev.data if ev else None

        return Observation(
            transaction=state.current_transaction,
            user_profile=state.user_profile,
            velocity_data=get_ev_data('velocity'),
            device_history=get_ev_data('device'),
            ip_reputation=get_ev_data('ip'),
            network_connections=get_ev_data('network'),
            behavioral_biometrics=get_ev_data('behavioral'),
            active_sessions=get_ev_data('sessions'),
            available_actions=self.current_task.get_available_actions(state),
            time_remaining=state.max_ticks - state.investigation_ticks,
            evidence_summary="; ".join([e.summary for e in state.gathered_evidence.values()]) or "No evidence gathered yet.",
            done=state.is_terminal,
            reward=0.0
        )
