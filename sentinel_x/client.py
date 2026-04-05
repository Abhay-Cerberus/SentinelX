import httpx
import json

def test():
    client = httpx.Client()
    
    print("--- Resetting Environment (Easy Task) ---")
    try:
        res = client.post("http://localhost:8000/reset", json={"task_name": "easy"})
        print(json.dumps(res.json(), indent=2))
        
        print("\n--- Stepping Environment (query_velocity) ---")
        action = {"action_type": "query_velocity", "parameters": {}, "reasoning": "Need more info"}
        res = client.post("http://localhost:8000/step", json={"action": action})
        print(json.dumps(res.json(), indent=2))
        
        print("\n--- Stepping Environment (block_transaction) ---")
        action = {"action_type": "block_transaction", "parameters": {}, "reasoning": "Fraud detected"}
        res = client.post("http://localhost:8000/step", json={"action": action})
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"Connection failed. Please run server -> {e}")

if __name__ == "__main__":
    test()
