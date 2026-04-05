import os
import httpx
import json

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

def run_episode(task_name: str, max_steps: int = 15):
    print(f"[START] task={task_name} env=sentinel_x model={MODEL_NAME}")
    
    with httpx.Client() as client:
        try:
            reset_resp = client.post(f"{API_BASE_URL}/reset", json={"task_name": task_name})
            if reset_resp.status_code != 200:
                print(f"Error connecting to env: {reset_resp.text}")
                return
            reset_data = reset_resp.json()
            obs = reset_data["observation"]
        except Exception as e:
            print(f"Error connecting to env: {e}")
            return
            
        step_idx = 0
        total_reward = 0.0
        done = False
        success = False
        
        while not done and step_idx < max_steps:
            print(f"[STEP] {step_idx}")
            
            available_actions = obs.get("available_actions", [])
            # Heuristic Baseline
            if "query_velocity" in available_actions and step_idx == 0:
                action = {"action_type": "query_velocity", "parameters": {}, "reasoning": "checking velocity"}
            elif "check_behavioral_biometrics" in available_actions and step_idx == 0:
                action = {"action_type": "check_behavioral_biometrics", "parameters": {}, "reasoning": "checking behavior"}
            elif "query_linked_accounts" in available_actions and step_idx == 0:
                action = {"action_type": "query_linked_accounts", "parameters": {"depth": 2}, "reasoning": "checking net"}
            else:
                if task_name == "easy":
                    action = {"action_type": "block_transaction", "parameters": {}, "reasoning": "blocking"}
                elif task_name == "medium":
                    action = {"action_type": "force_password_reset", "parameters": {}, "reasoning": "resetting"}
                elif task_name == "hard":
                    action = {"action_type": "file_sar", "parameters": {}, "reasoning": "filing SAR"}
                else:
                    action = {"action_type": "monitor_only", "parameters": {}, "reasoning": "do nothing"}
            
            step_resp = client.post(f"{API_BASE_URL}/step", json={"action": action})
            if step_resp.status_code != 200:
                print(f"Step Error: {step_resp.text}")
                break
                
            step_data = step_resp.json()
            reward = step_data.get("reward") or 0.0
            done = step_data.get("done", True)
            obs = step_data.get("observation", {})
            metadata = obs.get("metadata", {})
            
            total_reward += reward
            if done:
                success = metadata.get("success", False)
                
            step_idx += 1
            
        print(f"[END] success={success} steps={step_idx} rewards={total_reward} final_score={total_reward}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_episode(task)
