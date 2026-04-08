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

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from sentinelx.client import SentinelXEnv
from sentinelx.models import FraudAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LLM Configuration
# To use Ollama locally, set API_BASE_URL and MODEL_NAME in your .env file.
# Default for Ollama: API_BASE_URL=http://localhost:11434/v1, MODEL_NAME=qwen2.5:7b
API_KEY       = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "ollama"
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
ENV_URL       = os.getenv("SENTINELX_URL", "http://localhost:7860")
BENCHMARK     = "sentinelx"
MAX_STEPS     = 15
TEMPERATURE   = 0.3

TASKS = [
    {"task_id": "stolen-card-easy",       "seed": 42},
    {"task_id": "account-takeover-medium", "seed": 88},
    {"task_id": "money-laundering-hard",  "seed": 15},
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an elite Fraud Security Analyst at SentinelX.
Your mission: Protect the financial ecosystem by investigating and resolving suspicious transactions.

RULES OF ENGAGEMENT:
1. MAX 5 INVESTIGATIONS: You must make a final decision (APPROVE or BLOCK) within 5 steps of starting.
2. EVIDENCE CHAIN: Use 'query_velocity' -> 'check_device_history' -> 'ip_reputation' to build a case.
3. DECISIVENESS: Once you see a high-risk signal (e.g. 5x velocity spike or spoofed device), BLOCK immediately.
4. JSON ONLY: Respond only with a raw JSON object.

EXAMPLE OF A SUCCESSFUL DECISION:
Step 1 reasoning: "Amount is high; checking velocity." -> query_velocity
Step 2 reasoning: "Velocity is normal but location is new; checking device." -> check_device_history
Step 3 reasoning: "Device history shows multiple accounts on this phone; this is fraud." -> block_transaction

RESPONSE FORMAT:
{
  "action_type": "<action_name>",
  "parameters": {},
  "reasoning": "<short analysis mapping evidence to risk>"
}
""").strip()


# ---------------------------------------------------------------------------
# LLM client
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
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        return FraudAction(
            action_type=data.get("action_type", "monitor_only"),
            parameters=data.get("parameters", {}),
            reasoning=data.get("reasoning", ""),
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

def build_user_message(obs_dict: Dict[str, Any]) -> str:
    """Build a concise user-turn message from the current observation."""
    lines = []

    txn = obs_dict.get("transaction", {})
    lines.append("## Current Transaction")
    lines.append(f"  Amount   : ${txn.get('amount', '?')}")
    lines.append(f"  Merchant : {txn.get('merchant', '?')}")
    lines.append(f"  Location : {txn.get('location', '?')}")
    lines.append(f"  Time     : {txn.get('timestamp', '?')}")

    profile = obs_dict.get("user_profile", {})
    lines.append("\n## Account Profile")
    lines.append(f"  Account age    : {profile.get('account_age_days', '?')} days")
    lines.append(f"  Typical spend  : ${profile.get('typical_transaction_size', '?')}")
    lines.append(f"  Typical loc    : {profile.get('typical_location', '?')}")

    summary = obs_dict.get("evidence_summary", "")
    if summary:
        lines.append(f"\n## Evidence Summary\n{summary}")

    last = obs_dict.get("last_action_result", "")
    if last:
        lines.append(f"\n## Last Action Result\n{last}")

    available = obs_dict.get("available_actions", [])
    lines.append(f"\n## Available Actions\n{', '.join(available)}")
    lines.append(f"\nTime remaining: {obs_dict.get('time_remaining', '?')} ticks")

    return "\n".join(lines)


def run_episode(task_id: str, seed: int) -> Dict[str, Any]:
    """Run one full episode and return result dict."""
    rewards: List[float] = []
    step = 0
    success = False
    last_error: Optional[str] = None
    final_score = 0.0

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        with SentinelXEnv(base_url=ENV_URL).sync() as env:
            try:
                result = env.reset(task_id=task_id, seed=seed)
                messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

                for step in range(1, MAX_STEPS + 1):
                    obs = result.observation
                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

                    # Build conversation turn
                    user_msg = build_user_message(obs_dict)
                    messages.append({"role": "user", "content": user_msg})

                    # Inject pressure if investigative loop persists
                    if step >= 6:
                        messages.append({
                            "role": "system", 
                            "content": "CRITICAL: You have gathered enough evidence. You MUST now call 'approve_transaction' or 'block_transaction' to conclude the case. No further investigation is allowed."
                        })

                    # Get LLM action
                    raw = call_llm(messages)
                    action = parse_action(raw)

                    # Add assistant turn to history
                    messages.append({"role": "assistant", "content": raw})

                    # Step environment
                    result = env.step(action)
                    reward = float(result.reward or 0.0)
                    done = bool(result.done)
                    last_error = None

                    rewards.append(round(reward, 2))
                    action_str = f"{action.action_type}({json.dumps(action.parameters) if action.parameters else ''})"

                    print(
                        f"[STEP]  step={step} action={action_str} "
                        f"reward={reward:.2f} done={'true' if done else 'false'} "
                        f"error={'null' if not last_error else last_error}",
                        flush=True,
                    )
                    if action.reasoning:
                        print(f"        reasoning: {action.reasoning}", flush=True)

                    if done:
                        success = reward > 0
                        break

            except Exception as exc:
                last_error = f"Runtime error: {exc}"
                print(
                    f"[STEP]  step={step} action=error reward=0.00 done=true error={last_error}",
                    flush=True,
                )
    except Exception as exc:
        last_error = f"Connection error: {exc}"
        print(
            f"[STEP]  step=0 action=connection reward=0.00 done=true error={last_error}",
            flush=True,
        )
        print(f"FAILED to connect to environment at {ENV_URL}. Is the server running?", file=sys.stderr)

    # Compute final score: sum of positive rewards, capped to [0,1]
    if rewards:
        total = sum(rewards)
        final_score = round(max(0.0, min(1.0, total)), 2)
        success = final_score > 0.3

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={'true' if success else 'false'} "
        f"steps={step} score={final_score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps": step,
        "score": final_score,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"# SentinelX Baseline Inference", flush=True)
    print(f"# Model : {MODEL_NAME}", flush=True)
    print(f"# Env   : {ENV_URL}", flush=True)
    print(f"# Tasks : {[t['task_id'] for t in TASKS]}", flush=True)
    print("", flush=True)

    all_results = []
    start_time = time.time()

    for task in TASKS:
        result = run_episode(task["task_id"], task["seed"])
        all_results.append(result)
        print("", flush=True)

    elapsed = round(time.time() - start_time, 1)
    avg_score = round(sum(r["score"] for r in all_results) / len(all_results), 2)

    print(f"# === SUMMARY ===", flush=True)
    print(f"# Total time : {elapsed}s", flush=True)
    print(f"# Avg score  : {avg_score}", flush=True)
    for r in all_results:
        print(f"#   {r['task_id']:35s}  score={r['score']:.2f}  steps={r['steps']}", flush=True)
