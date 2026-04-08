"""
SentinelX Inference Script
===========================
Mandatory hackathon baseline script.

STDOUT FORMAT (strictly required):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Environment variables:
    API_BASE_URL   — LLM endpoint   (default: HF inference router)
    MODEL_NAME     — Model ID       (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       — API key
    SENTINELX_URL  — Running env URL (default: http://localhost:8000)
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from sentinelx.server.environment import SentinelXEnvironment
from sentinelx.models import FraudAction

# ---------------------------------------------------------------------------
# Configuration (Aligned with official baseline)
# ---------------------------------------------------------------------------

API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
ENV_URL       = os.getenv("SENTINELX_URL") or "http://localhost:7860"
BENCHMARK     = "sentinelx"
MAX_STEPS     = 15
TEMPERATURE   = 0.3
EXPLORATION_LIMIT = 5

TASKS = [
    {"task_id": "stolen-card-easy",       "seed": 42},
    {"task_id": "account-takeover-medium", "seed": 88},
    {"task_id": "money-laundering-hard",  "seed": 15},
]

# ---------------------------------------------------------------------------
# System Prompt (Advanced Steering)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an elite Fraud Security Analyst at SentinelX.
Your mission: Protect the financial ecosystem by investigating and resolving suspicious transactions.

RULES OF ENGAGEMENT:
1. MAX 5 INVESTIGATIONS: You must make a final decision (APPROVE or BLOCK) within 5 steps of starting.
2. EVIDENCE CHAIN: Use 'query_velocity' -> 'check_device_history' -> 'lookup_ip_reputation' to build a case.
3. DECISIVENESS: Once you see a high-risk signal (e.g. 5x velocity spike or spoofed device), BLOCK immediately.
4. JSON ONLY: Respond only with a raw JSON object.

EXAMPLE OF A SUCCESSFUL DECISION:
Step 1 reasoning: \"Amount is high; checking velocity.\" -> query_velocity
Step 2 reasoning: \"Velocity is normal but location is new; checking device.\" -> check_device_history
Step 3 reasoning: \"Device history shows multiple accounts on this phone; this is fraud.\" -> block_transaction

RESPONSE FORMAT:
{
  \"action_type\": \"<action_name>\",
  \"parameters\": {},
  \"reasoning\": \"<short analysis mapping evidence to risk>\"
}
""").strip()

# ---------------------------------------------------------------------------
# Official Logging Utilities
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Baseline requires 3 decimal places for score
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

client = OpenAI(api_key=API_KEY or "no-key", base_url=API_BASE_URL)

def call_llm(messages: List[Dict[str, str]]) -> str:
    """Call the LLM and return the raw content string."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return json.dumps({
            "action_type": "monitor_only",
            "parameters": {},
            "reasoning": f"LLM error: {exc}",
        })

def parse_action(raw: str) -> FraudAction:
    """Extract a FraudAction from LLM output, with safe fallback."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        atype = str(data.get("action_type", "monitor_only")).strip()
        reasons = str(data.get("reasoning", "")).strip()
        return FraudAction(
            action_type=atype,
            parameters=data.get("parameters", {}),
            reasoning=reasons,
        )
    except Exception:
        return FraudAction(
            action_type="monitor_only",
            parameters={},
            reasoning=f"Parse error — raw output: {raw[:120]}",
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def build_user_message(obs_dict: Dict[str, Any], force_terminal: bool = False) -> str:
    """Build a robust summary. If force_terminal is true, only show decision actions."""
    lines = ["### CURRENT OBSERVATION ###"]

    # Transaction info
    txn = obs_dict.get("transaction", {})
    amount = txn.get("amount", "?")
    lines.append(f"TRANSACTION: ID={txn.get('transaction_id', '?')}, Amount=${amount}, Merchant={txn.get('merchant', '?')}, Location={txn.get('location', '?')}")

    # Profile info
    profile = obs_dict.get("user_profile", {})
    lines.append(f"USER_PROFILE: Age={profile.get('account_age_days', '?')} days, Typical_Spend=${profile.get('typical_transaction_size', '?')}")

    # Revealed Data
    lines.append("\n### REVEALED EVIDENCE ###")
    found_any = False
    fields = ["velocity_data", "device_history", "ip_reputation", "network_connections", "behavioral_biometrics", "active_sessions", "temporal_pattern", "business_registration"]
    for key in fields:
        val = obs_dict.get(key)
        if val is not None:
            lines.append(f"- {key.upper()}: {json.dumps(val)}")
            found_any = True
    if not found_any:
        lines.append("- (No additional signals revealed yet.)")

    # Filter actions if we are forcing a decision
    available = obs_dict.get("available_actions", [])
    if force_terminal:
        terminal_only = ["approve_transaction", "block_transaction", "request_3ds", "temporarily_freeze_account", "file_sar", "file_ctr"]
        available = [a for a in available if a in terminal_only]
        lines.append("\n!!! IMPORTANT: Investigation tools are now CLOSED. You MUST pick a final decision below. !!!")

    lines.append(f"\nAVAILABLE_ACTIONS: {', '.join(available)}")
    lines.append(f"LAST_RESULT: {obs_dict.get('last_action_result', 'N/A')}")
    
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Core Async Loop
# ---------------------------------------------------------------------------

async def run_episode(task_id: str, seed: int) -> Dict[str, Any]:
    """Run one full episode and return result dict."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = SentinelXEnvironment()
        result = env.reset(task_id=task_id, seed=seed)
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            obs = result
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
            
            # Logic-Guided Steering Hints
            evidence_found = [f for f in ["velocity_data", "device_history", "ip_reputation", "network_connections", "behavioral_biometrics", "active_sessions"] if getattr(obs, f, None) is not None]
            if step > EXPLORATION_LIMIT:
                last_action = obs_dict.get("last_action_result", "")
                messages.append({"role": "system", "content": f"⚠️ TIMER ALERT: Step {step}/{MAX_STEPS}. Investigation phase CLOSED. You must choose a final decision action NOW (approve_transaction, block_transaction, request_3ds, etc). No more queries allowed."})
            elif step == EXPLORATION_LIMIT:
                messages.append({"role": "system", "content": f"ℹ️ PLANNING WINDOW: You have gathered initial signals. PREPARE your final decision for step {EXPLORATION_LIMIT + 1}."})
            elif evidence_found:
                messages.append({"role": "system", "content": f"ANALYSIS HINT: You found {', '.join(evidence_found)}. Factor this into your risk model."})

            # Build conversation turn
            user_msg = build_user_message(obs_dict, force_terminal=(step > EXPLORATION_LIMIT))
            messages.append({"role": "user", "content": user_msg})

            # Get LLM action
            raw = call_llm(messages)
            action = parse_action(raw)

            # Add assistant turn to history
            messages.append({"role": "assistant", "content": raw})

            # Step environment
            result = env.step(action)
            
            reward = float(result.reward or 0.0)
            # Force done=True if a terminal action (decision/regulation) is taken
            terminal_actions = {"approve_transaction", "block_transaction", "request_3ds", "send_push_notification", "force_password_reset", "temporarily_freeze_account", "file_sar", "file_ctr", "escalate_to_compliance"}
            is_terminal_action = action.action_type in terminal_actions
            done = bool(result.done) or is_terminal_action or (step == MAX_STEPS)
            
            rewards.append(reward)
            steps_taken = step
            
            action_str = f"{action.action_type}({json.dumps(action.parameters) if action.parameters else ''})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Scoring: sum rewards, clamp [0,1]
        total_raw = sum(rewards)
        score = round(max(0.0, min(1.0, total_raw)), 3)
        success = score > 0.3

    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(exc))

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "steps": steps_taken}

async def main() -> None:
    print(f"# SentinelX Baseline Inference (Async Mode)", flush=True)
    print(f"# Model : {MODEL_NAME}", flush=True)
    print(f"# Env   : {ENV_URL}", flush=True)
    print("", flush=True)

    start_time = time.time()
    all_results = []
    
    for task in TASKS:
        res = await run_episode(task["task_id"], task["seed"])
        all_results.append(res)
        print("", flush=True)

    elapsed = round(time.time() - start_time, 1)
    avg_score = round(sum(r["score"] for r in all_results) / len(all_results), 3)

    print(f"# === SUMMARY ===", flush=True)
    print(f"# Total time : {elapsed}s", flush=True)
    print(f"# Avg score  : {avg_score:.3f}", flush=True)
    for r in all_results:
        print(f"#   {r['task_id']:35s}  score={r['score']:.3f}  steps={r['steps']}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
