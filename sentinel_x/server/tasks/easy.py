from sentinel_x.models import State, Action, Evidence, Transaction, UserProfile, FraudsterStrategy
import random

class EasyTask:
    name = "easy"
    
    def initialize_state(self) -> State:
        txn = Transaction(
            id="TX-1001",
            amount=850.50,
            merchant="BestBuy Electronics",
            location="Miami, FL",
            time="2023-10-27T14:32:00Z"
        )
        
        profile = UserProfile(
            account_age_days=1200,
            typical_location="Seattle, WA",
            typical_transaction_size=45.0,
            card_present_ratio=0.8
        )
        
        fraud_strategy = FraudsterStrategy(
            method="stolen_card_info",
            adaptation_level=1
        )
        
        return State(
            current_transaction=txn,
            user_profile=profile,
            true_fraud_type="stolen_card",
            fraudster_strategy=fraud_strategy,
            max_ticks=10
        )
        
    def transition(self, state: State, action: Action) -> State:
        state.investigation_ticks += 1
        
        if action.action_type == "query_velocity":
            state.gathered_evidence["velocity"] = Evidence(
                type="velocity",
                data={"tx_count_1hr": 5, "tx_count_24hr": 12},
                summary="High velocity of transactions in the last hour."
            )
        elif action.action_type == "check_device":
            state.gathered_evidence["device"] = Evidence(
                type="device",
                data={"device_id": "new_device_998", "is_known": False},
                summary="Transaction originated from an unknown device."
            )
            
        return state

    def get_available_actions(self, state: State) -> list[str]:
        actions = ["query_velocity", "check_device", "approve_transaction", "block_transaction"]
        return [a for a in actions if a not in [ev.type for ev in state.gathered_evidence.values()]]
