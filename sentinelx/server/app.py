"""FastAPI server for SentinelX."""
import sys
import os
import json
from typing import Optional

# Ensure the project root is on the path when run from this directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from sentinelx.server.environment import SentinelXEnvironment
from sentinelx.models import FraudAction, FraudObservation, FraudInvestigationState

app = FastAPI(title="SentinelX", version="1.0.0")

# Global environment instance (lazy-loaded to avoid import-time initialization)
_env = None

def get_env():
    """Get or create the environment instance."""
    global _env
    if _env is None:
        _env = SentinelXEnvironment()
    return _env

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

# ============================================================================
# Reset Endpoint
# ============================================================================

@app.post("/reset")
async def reset(task_id: str = "stolen-card-easy", seed: Optional[int] = None):
    """Reset the environment and start a new episode."""
    try:
        env = get_env()
        result = env.reset(task_id=task_id, seed=seed)
        obs_dict = result.model_dump() if hasattr(result, "model_dump") else result.__dict__
        return {
            "observation": obs_dict,
            "done": False,
            "reward": None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Step Endpoint
# ============================================================================

@app.post("/step")
async def step(action_data: dict):
    """Execute one step in the environment."""
    try:
        env = get_env()
        action = FraudAction(
            action_type=action_data.get("action_type", "monitor_only"),
            parameters=action_data.get("parameters", {}),
            reasoning=action_data.get("reasoning", ""),
        )
        result = env.step(action)
        obs_dict = result.model_dump() if hasattr(result, "model_dump") else result.__dict__
        return {
            "observation": obs_dict,
            "reward": float(result.reward or 0.0),
            "done": bool(result.done),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# State Endpoint
# ============================================================================

@app.get("/state")
async def state():
    """Get the current environment state."""
    try:
        env = get_env()
        s = env.state
        state_dict = s.model_dump() if hasattr(s, "model_dump") else s.__dict__
        return state_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Web UI (Simple HTML)
# ============================================================================

@app.get("/web")
async def web_ui():
    """Simple web interface for playing the environment."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SentinelX — Fraud Investigation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #1e40af; }
            .section { margin: 20px 0; padding: 15px; background: #f9fafb; border-left: 4px solid #1e40af; }
            button { background: #1e40af; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            button:hover { background: #1e3a8a; }
            input, select { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
            .observation { background: #f0f9ff; padding: 15px; border-radius: 4px; margin: 10px 0; max-height: 400px; overflow-y: auto; }
            .reward { font-size: 18px; font-weight: bold; color: #059669; }
            .error { color: #dc2626; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ SentinelX — Fraud Investigation Environment</h1>
            
            <div class="section">
                <h2>Start New Episode</h2>
                <label>Task:</label>
                <select id="taskSelect">
                    <option value="stolen-card-easy">Stolen Card (Easy)</option>
                    <option value="account-takeover-medium">Account Takeover (Medium)</option>
                    <option value="money-laundering-hard">Money Laundering (Hard)</option>
                </select>
                <label>Seed:</label>
                <input type="number" id="seedInput" placeholder="Leave blank for random" />
                <button onclick="resetEnv()">Reset</button>
            </div>

            <div class="section">
                <h2>Current Observation</h2>
                <div id="observation" class="observation">
                    <p>No episode started yet. Click "Reset" to begin.</p>
                </div>
            </div>

            <div class="section">
                <h2>Take Action</h2>
                <label>Action Type:</label>
                <select id="actionSelect">
                    <option value="query_velocity">Query Velocity</option>
                    <option value="check_device_history">Check Device History</option>
                    <option value="lookup_ip_reputation">Lookup IP Reputation</option>
                    <option value="check_behavioral_biometrics">Check Behavioral Biometrics</option>
                    <option value="check_active_sessions">Check Active Sessions</option>
                    <option value="query_linked_accounts">Query Linked Accounts</option>
                    <option value="analyze_temporal_pattern">Analyze Temporal Pattern</option>
                    <option value="check_business_registration">Check Business Registration</option>
                    <option value="request_kyc_documents">Request KYC Documents</option>
                    <option value="approve_transaction">Approve Transaction</option>
                    <option value="block_transaction">Block Transaction</option>
                    <option value="request_3ds">Request 3DS</option>
                    <option value="send_push_notification">Send Push Notification</option>
                    <option value="force_password_reset">Force Password Reset</option>
                    <option value="temporarily_freeze_account">Temporarily Freeze Account</option>
                    <option value="file_sar">File SAR</option>
                    <option value="file_ctr">File CTR</option>
                    <option value="escalate_to_compliance">Escalate to Compliance</option>
                    <option value="monitor_only">Monitor Only</option>
                </select>
                <label>Reasoning:</label>
                <input type="text" id="reasoningInput" placeholder="Explain your action" />
                <button onclick="takeAction()">Execute Action</button>
            </div>

            <div class="section">
                <h2>Result</h2>
                <div id="result">
                    <p>No action taken yet.</p>
                </div>
            </div>
        </div>

        <script>
            async function resetEnv() {
                const task = document.getElementById('taskSelect').value;
                const seed = document.getElementById('seedInput').value || null;
                try {
                    const response = await fetch('/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ task_id: task, seed: seed ? parseInt(seed) : null })
                    });
                    const data = await response.json();
                    document.getElementById('observation').innerHTML = '<pre>' + JSON.stringify(data.observation, null, 2) + '</pre>';
                    document.getElementById('result').innerHTML = '<p>Episode reset. Ready for actions.</p>';
                } catch (e) {
                    document.getElementById('result').innerHTML = '<p class="error">Error: ' + e.message + '</p>';
                }
            }

            async function takeAction() {
                const actionType = document.getElementById('actionSelect').value;
                const reasoning = document.getElementById('reasoningInput').value;
                try {
                    const response = await fetch('/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ action_type: actionType, parameters: {}, reasoning: reasoning })
                    });
                    const data = await response.json();
                    document.getElementById('observation').innerHTML = '<pre>' + JSON.stringify(data.observation, null, 2) + '</pre>';
                    const resultHtml = `
                        <p><strong>Action:</strong> ${actionType}</p>
                        <p><strong>Reward:</strong> <span class="reward">${data.reward.toFixed(2)}</span></p>
                        <p><strong>Done:</strong> ${data.done}</p>
                    `;
                    document.getElementById('result').innerHTML = resultHtml;
                } catch (e) {
                    document.getElementById('result').innerHTML = '<p class="error">Error: ' + e.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ============================================================================
# OpenAPI Docs
# ============================================================================

@app.get("/docs", include_in_schema=False)
async def docs():
    """Redirect to OpenAPI docs."""
    return {"message": "OpenAPI docs available at /openapi.json"}
