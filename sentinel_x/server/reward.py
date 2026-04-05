from sentinel_x.models import State, Action, RewardBreakdown

def calculate_reward(action: Action, state: State, next_state: State, is_terminal: bool, success: bool) -> tuple[float, RewardBreakdown]:
    breakdown = RewardBreakdown()
    
    # 1. Investigation Bonus (Dense shaping)
    # Reward for gathering new types of evidence
    new_evidence = len(next_state.gathered_evidence) - len(state.gathered_evidence)
    if new_evidence > 0:
        breakdown.investigation_bonus += (new_evidence * 0.2)
        
    # Check if reasoning is provided
    if action.reasoning and len(action.reasoning.strip()) > 5:
        breakdown.investigation_bonus += 0.1
        
    # 2. Time Penalty
    # Small penalty for taking too long to decide
    breakdown.time_penalty -= 0.05
    
    # 3. Decision Reward
    if is_terminal:
        if success:
            breakdown.decision_reward += 1.0
            
            # Compliance Reward - did they gather enough evidence before deciding?
            evidence_count = len(state.gathered_evidence)
            if evidence_count >= 2:
                breakdown.compliance_reward += 0.5
        else:
            breakdown.decision_reward -= 1.0 # Penalty for failure
            
    breakdown.total = breakdown.investigation_bonus + breakdown.decision_reward + breakdown.time_penalty + breakdown.compliance_reward
    
    return breakdown.total, breakdown
