from sentinel_x.models import State, Action, Evidence, Transaction, UserProfile, FraudsterStrategy

class MediumTask:
    name = "medium"
    
    def initialize_state(self) -> State:
        txn = Transaction(
            id="TX-2005",
            amount=150.00,
            merchant="Steam Games",
            location="Seattle, WA",  # IP spoofed to match user profile
            time="2023-10-27T03:15:00Z"
        )
        
        profile = UserProfile(
            account_age_days=450,
            typical_location="Seattle, WA",
            typical_transaction_size=30.0,
            card_present_ratio=0.9
        )
        
        fraud_strategy = FraudsterStrategy(
            method="account_takeover",
            adaptation_level=3
        )
        
        return State(
            current_transaction=txn,
            user_profile=profile,
            true_fraud_type="account_takeover",
            fraudster_strategy=fraud_strategy,
            max_ticks=15
        )
        
    def transition(self, state: State, action: Action) -> State:
        state.investigation_ticks += 1
        
        if action.action_type == "check_behavioral_biometrics":
            state.gathered_evidence["behavioral"] = Evidence(
                type="behavioral",
                data={"typing_speed_variance": 2.5, "mouse_movement": "robotic"},
                summary="Typing patterns entirely inconsistent with historical baseline."
            )
        elif action.action_type == "query_active_sessions":
            state.gathered_evidence["sessions"] = Evidence(
                type="sessions",
                data=[{"ip": "104.28.19.1", "device": "MacBook Pro"}, {"ip": "185.153.19.2", "device": "Linux Desktop"}],
                summary="Multiple active sessions from disparate IPs."
            )
            
        return state

    def get_available_actions(self, state: State) -> list[str]:
        return ["query_active_sessions", "check_behavioral_biometrics", "approve_transaction", "force_password_reset"]
