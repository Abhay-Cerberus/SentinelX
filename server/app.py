"""FastAPI server for SentinelX — OpenEnv standard layout."""
import sys
import os
import json
import logging
from typing import Optional
from pathlib import Path
from uuid import uuid4

# Ensure the project root is on the path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sentinelx.server.environment import SentinelXEnvironment
from sentinelx.models import FraudAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinelx-app")

app = FastAPI(title="SentinelX", version="1.0.0", description="Financial Fraud Investigation Environment")

# Per-session environments (one per user session)
_sessions: dict[str, SentinelXEnvironment] = {}

def get_session_env(session_id: Optional[str] = None) -> tuple[str, SentinelXEnvironment]:
    """Get or create an environment for a session."""
    if session_id is None:
        session_id = str(uuid4())
    
    if session_id not in _sessions:
        logger.info(f"Creating new environment for session {session_id}")
        _sessions[session_id] = SentinelXEnvironment()
    
    return session_id, _sessions[session_id]


# ============================================================================
# Root / Landing Page
# ============================================================================

@app.get("/")
async def root():
    """Landing page with links to available endpoints."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SentinelX — Fraud Investigation Environment</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container { 
                background: white; 
                border-radius: 12px; 
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 800px;
                padding: 50px;
                text-align: center;
            }
            h1 { 
                font-size: 3em; 
                color: #667eea;
                margin-bottom: 10px;
            }
            .subtitle {
                font-size: 1.2em;
                color: #666;
                margin-bottom: 40px;
            }
            .description {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 40px;
                text-align: left;
                line-height: 1.6;
            }
            .description h2 {
                color: #333;
                margin-bottom: 10px;
            }
            .description p {
                color: #666;
                margin-bottom: 10px;
            }
            .links {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 30px;
            }
            a {
                display: block;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
                transition: transform 0.2s;
            }
            a:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            .info {
                background: #e7f3ff;
                border-left: 4px solid #667eea;
                padding: 15px;
                border-radius: 4px;
                text-align: left;
                color: #333;
                font-size: 0.9em;
            }
            .info strong { color: #667eea; }
            @media (max-width: 600px) {
                .links { grid-template-columns: 1fr; }
                h1 { font-size: 2em; }
                .container { padding: 30px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ SentinelX</h1>
            <p class="subtitle">Financial Fraud Investigation Environment</p>
            
            <div class="description">
                <h2>What is SentinelX?</h2>
                <p>
                    SentinelX is a <strong>reinforcement learning environment</strong> where AI agents act as financial fraud analysts. 
                    The environment simulates three real-world fraud scenarios with increasing difficulty.
                </p>
                <p>
                    <strong>Key Feature:</strong> An adaptive adversary learns from the agent's detection patterns and escalates tactics across episodes.
                </p>
            </div>

            <div class="links">
                <a href="/web">🎮 Play Interactive UI</a>
                <a href="/docs">📚 API Documentation</a>
                <a href="/health">💚 Health Check</a>
                <a href="/openapi.json">⚙️ OpenAPI Spec</a>
            </div>

            <div class="info">
                <strong>📖 How it works:</strong><br>
                1. Call <code>/reset</code> to start an episode<br>
                2. Call <code>/step</code> to take actions (investigate or intervene)<br>
                3. Receive observations and rewards<br>
                4. Episode ends when you make a final decision<br><br>
                <strong>🎯 Three Tasks:</strong> Stolen Card (Easy) → Account Takeover (Medium) → Money Laundering (Hard)
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


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
async def reset(task_id: str = "stolen-card-easy", seed: Optional[int] = None, session_id: Optional[str] = None):
    """Reset the environment and start a new episode."""
    try:
        if seed is None:
            seed = 42
        
        session_id, env = get_session_env(session_id)
        logger.info(f"Reset called: session={session_id}, task_id={task_id}, seed={seed}")
        
        result = env.reset(task_id=task_id, seed=seed)
        logger.info(f"Reset successful: done={result.done}, has_transaction={bool(result.transaction)}")
        
        # Serialize observation
        obs_dict = result.model_dump() if hasattr(result, "model_dump") else result.__dict__
        
        return {
            "session_id": session_id,
            "observation": obs_dict,
            "done": False,
            "reward": None,
        }
    except Exception as e:
        logger.exception("Error in /reset")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# ============================================================================
# Step Endpoint
# ============================================================================

@app.post("/step")
async def step(action_data: dict, session_id: Optional[str] = None):
    """Execute one step in the environment."""
    try:
        if session_id is None:
            raise HTTPException(status_code=400, detail="session_id is required. Call /reset first to get a session_id.")
        
        session_id, env = get_session_env(session_id)
        
        # Parse action
        action = FraudAction(
            action_type=action_data.get("action_type", "monitor_only"),
            parameters=action_data.get("parameters", {}),
            reasoning=action_data.get("reasoning", ""),
        )
        
        # Step environment
        result = env.step(action)
        
        # Serialize observation
        obs_dict = result.model_dump() if hasattr(result, "model_dump") else result.__dict__
        
        return {
            "session_id": session_id,
            "observation": obs_dict,
            "reward": float(result.reward or 0.0),
            "done": bool(result.done),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /step")
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


# ============================================================================
# State Endpoint
# ============================================================================

@app.get("/state")
async def state(session_id: Optional[str] = None):
    """Get the current environment state."""
    try:
        if session_id is None:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        session_id, env = get_session_env(session_id)
        s = env.state
        state_dict = s.model_dump() if hasattr(s, "model_dump") else s.__dict__
        return {
            "session_id": session_id,
            "state": state_dict,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /state")
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}")


# ============================================================================
# Web UI
# ============================================================================

@app.get("/web")
async def web_ui():
    """Interactive web interface for playing the environment."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SentinelX — Fraud Investigation</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 12px; 
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { opacity: 0.9; font-size: 1.1em; }
            .content { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 30px; }
            .section { 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .section h2 { color: #333; margin-bottom: 15px; font-size: 1.3em; }
            .section label { display: block; margin-top: 10px; font-weight: 600; color: #555; }
            input, select, textarea { 
                width: 100%; 
                padding: 10px; 
                margin-top: 5px; 
                border: 1px solid #ddd; 
                border-radius: 4px;
                font-family: inherit;
            }
            input:focus, select:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            button { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
                margin-top: 10px;
                font-weight: 600;
                transition: transform 0.2s;
            }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
            button:active { transform: translateY(0); }
            .observation { 
                background: white; 
                padding: 15px; 
                border-radius: 4px; 
                max-height: 400px; 
                overflow-y: auto;
                border: 1px solid #ddd;
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 0.9em;
                white-space: pre-wrap;
                word-break: break-word;
            }
            .reward { 
                font-size: 1.2em; 
                font-weight: bold; 
                color: #28a745;
                margin-top: 10px;
            }
            .error { 
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }
            .success {
                color: #155724;
                background: #d4edda;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }
            @media (max-width: 768px) {
                .content { grid-template-columns: 1fr; }
                .header h1 { font-size: 1.8em; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ SentinelX</h1>
                <p>Financial Fraud Investigation Environment</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>Start New Episode</h2>
                    <label>Task:</label>
                    <select id="taskSelect">
                        <option value="stolen-card-easy">Stolen Card (Easy)</option>
                        <option value="account-takeover-medium">Account Takeover (Medium)</option>
                        <option value="money-laundering-hard">Money Laundering (Hard)</option>
                    </select>
                    <label>Seed (optional):</label>
                    <input type="number" id="seedInput" placeholder="Leave blank for random" />
                    <button onclick="resetEnv()">🔄 Reset Episode</button>
                    <div id="resetStatus"></div>
                </div>

                <div class="section">
                    <h2>Current Observation</h2>
                    <div id="observation" class="observation">No episode started yet. Click "Reset Episode" to begin.</div>
                </div>

                <div class="section">
                    <h2>Take Action</h2>
                    <label>Action Type:</label>
                    <select id="actionSelect">
                        <optgroup label="Investigation">
                            <option value="query_velocity">Query Velocity</option>
                            <option value="check_device_history">Check Device History</option>
                            <option value="lookup_ip_reputation">Lookup IP Reputation</option>
                            <option value="check_behavioral_biometrics">Check Behavioral Biometrics</option>
                            <option value="check_active_sessions">Check Active Sessions</option>
                            <option value="query_linked_accounts">Query Linked Accounts</option>
                            <option value="analyze_temporal_pattern">Analyze Temporal Pattern</option>
                            <option value="check_business_registration">Check Business Registration</option>
                        </optgroup>
                        <optgroup label="Intervention">
                            <option value="approve_transaction">Approve Transaction</option>
                            <option value="block_transaction">Block Transaction</option>
                            <option value="request_3ds">Request 3DS</option>
                            <option value="send_push_notification">Send Push Notification</option>
                            <option value="force_password_reset">Force Password Reset</option>
                            <option value="temporarily_freeze_account">Temporarily Freeze Account</option>
                        </optgroup>
                        <optgroup label="Regulatory">
                            <option value="file_sar">File SAR</option>
                            <option value="file_ctr">File CTR</option>
                            <option value="escalate_to_compliance">Escalate to Compliance</option>
                        </optgroup>
                        <optgroup label="Special">
                            <option value="monitor_only">Monitor Only</option>
                        </optgroup>
                    </select>
                    <label>Reasoning:</label>
                    <textarea id="reasoningInput" placeholder="Explain your action" rows="3"></textarea>
                    <button onclick="takeAction()">⚡ Execute Action</button>
                </div>

                <div class="section">
                    <h2>Result</h2>
                    <div id="result">No action taken yet.</div>
                </div>
            </div>
        </div>

        <script>
            let currentSessionId = null;

            async function resetEnv() {
                const task = document.getElementById('taskSelect').value;
                const seed = document.getElementById('seedInput').value;
                const statusDiv = document.getElementById('resetStatus');
                
                try {
                    statusDiv.innerHTML = '<div class="success">Loading...</div>';
                    const response = await fetch('/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            task_id: task, 
                            seed: seed ? parseInt(seed) : null 
                        })
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        statusDiv.innerHTML = '<div class="error">Error: ' + error.detail + '</div>';
                        return;
                    }
                    
                    const data = await response.json();
                    currentSessionId = data.session_id;
                    document.getElementById('observation').innerHTML = JSON.stringify(data.observation, null, 2);
                    statusDiv.innerHTML = '<div class="success">✓ Episode reset (Session: ' + currentSessionId.substring(0, 8) + '...)</div>';
                } catch (e) {
                    document.getElementById('resetStatus').innerHTML = '<div class="error">Error: ' + e.message + '</div>';
                }
            }

            async function takeAction() {
                if (!currentSessionId) {
                    document.getElementById('result').innerHTML = '<div class="error">Error: No active session. Click "Reset Episode" first.</div>';
                    return;
                }

                const actionType = document.getElementById('actionSelect').value;
                const reasoning = document.getElementById('reasoningInput').value;
                const resultDiv = document.getElementById('result');
                
                try {
                    resultDiv.innerHTML = '<div class="success">Executing...</div>';
                    
                    const url = new URL('/step', window.location.origin);
                    url.searchParams.append('session_id', currentSessionId);
                    
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            action_type: actionType, 
                            parameters: {}, 
                            reasoning: reasoning 
                        })
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        resultDiv.innerHTML = '<div class="error">Error: ' + error.detail + '</div>';
                        return;
                    }
                    
                    const data = await response.json();
                    document.getElementById('observation').innerHTML = JSON.stringify(data.observation, null, 2);
                    
                    const resultHtml = `
                        <div class="success">
                            <strong>Action:</strong> ${actionType}<br>
                            <strong>Reward:</strong> <span class="reward">${data.reward.toFixed(2)}</span><br>
                            <strong>Done:</strong> ${data.done ? '✓ Yes' : '✗ No'}
                        </div>
                    `;
                    resultDiv.innerHTML = resultHtml;
                } catch (e) {
                    resultDiv.innerHTML = '<div class="error">Error: ' + e.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
