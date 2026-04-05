from sentinel_x.models import State, Action, RewardBreakdown
from sentinel_x.server.reward import calculate_reward

def grade_action(task_name: str, state: State, action: Action, next_state: State) -> tuple[float, RewardBreakdown, bool, str]:
    """
    Evaluates the action and computes the reward.
    Returns (score, breakdown, success, reason)
    """
    reward = 0.0
    success = False
    reason = "Ongoing investigation"
    
    breakdown = RewardBreakdown()
    
    # Check if action is a terminal decision
    is_terminal = False
    
    if task_name == "easy":
        if action.action_type == "block_transaction":
            is_terminal = True
            success = True
            reason = "Successfully blocked stolen card fraud."
        elif action.action_type == "approve_transaction":
            is_terminal = True
            success = False
            reason = "Failed. Approved stolen card."
    
    elif task_name == "medium":
        if action.action_type == "force_password_reset":
            if "behavioral" in state.gathered_evidence:
                is_terminal = True
                success = True
                reason = "Successfully caught ATO with sufficient evidence."
            else:
                is_terminal = True
                success = False
                reason = "Guessed ATO without checking behavioral biometrics. Reckless action."
        elif action.action_type == "approve_transaction":
            is_terminal = True
            success = False
            reason = "Failed. Approved hijacked account transaction."
            
    elif task_name == "hard":
        if action.action_type == "file_sar":
            if "network" in state.gathered_evidence and len(state.visible_network.get("links", [])) > 1:
                is_terminal = True
                success = True
                reason = "Successfully filed SAR with strong network evidence backing it."
            else:
                is_terminal = True
                success = False
                reason = "Filed SAR with insufficient evidence. Regulatory penalty."
        elif action.action_type == "approve_transaction":
            is_terminal = True
            success = False
            reason = "Failed. Facilitated money laundering."

    score, breakdown = calculate_reward(action, state, next_state, is_terminal, success)
    
    return score, breakdown, success, reason
