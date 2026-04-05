from sentinel_x.models import State, Action, Evidence, Transaction, UserProfile, FraudsterStrategy, AccountLink

class HardTask:
    name = "hard"
    
    def initialize_state(self) -> State:
        txn = Transaction(
            id="TX-9099",
            amount=9900.00, # Just under 10k reporting limit
            merchant="Consulting Services LLC",
            location="Miami, FL",
            time="2023-10-27T10:00:00Z"
        )
        
        profile = UserProfile(
            account_age_days=60, # Relatively new shell corp
            typical_location="Miami, FL",
            typical_transaction_size=8000.0,
            card_present_ratio=0.0,
            business_type="Consulting"
        )
        
        fraud_strategy = FraudsterStrategy(
            method="money_laundering_layering",
            adaptation_level=5
        )
        
        return State(
            current_transaction=txn,
            user_profile=profile,
            true_fraud_type="money_laundering",
            fraudster_strategy=fraud_strategy,
            max_ticks=20
        )
        
    def transition(self, state: State, action: Action) -> State:
        state.investigation_ticks += 1
        
        if action.action_type == "query_linked_accounts":
            depth = action.parameters.get("depth", 1)
            links = []
            if depth >= 1:
                links.append(AccountLink(source="Current_Account", target="Offshore_Holdings_A", relation="Transfer_Target"))
            if depth >= 2:
                links.append(AccountLink(source="Offshore_Holdings_A", target="Crypto_Exchange_X", relation="Liquidation"))
            
            state.gathered_evidence["network"] = Evidence(
                type="network",
                data={"links": [link.model_dump() for link in links]},
                summary=f"Found suspicious corporate network links at depth {depth}."
            )
            state.visible_network["links"] = links
            
        elif action.action_type == "query_velocity":
            state.gathered_evidence["velocity"] = Evidence(
                type="velocity",
                data={"tx_count_30d": 45, "total_volume": 445000.0},
                summary="High volume of transactions just below reporting thresholds."
            )
            
        return state

    def get_available_actions(self, state: State) -> list[str]:
        return ["query_velocity", "query_linked_accounts", "approve_transaction", "file_sar"]
