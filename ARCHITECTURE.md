# 🛡️ SentinelX Architecture — How It Actually Works

## TL;DR

**SentinelX is NOT training a model.** It's a **deterministic simulation environment** that:
1. Simulates fraud scenarios (stolen cards, account takeovers, money laundering)
2. Accepts actions from an external agent (LLM, RL policy, human)
3. Returns observations and rewards based on the agent's decisions
4. Grades the agent's performance

The agent (e.g., `Qwen2.5-7B` in `inference.py`) is trained **outside** this environment using standard RL/LLM fine-tuning tools. SentinelX just provides the **evaluation harness**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    External Agent                           │
│              (LLM, RL Policy, or Human)                     │
│                                                              │
│  Example: Qwen2.5-7B via HF Inference Router               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTP POST /reset, /step
                     │ JSON: {"action_type": "...", "reasoning": "..."}
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Server (app.py)                    │
│                                                              │
│  - /reset   → Start new episode                            │
│  - /step    → Execute action                               │
│  - /state   → Get current state                            │
│  - /web     → Interactive UI                               │
│  - /health  → Health check                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            SentinelXEnvironment (environment.py)            │
│                                                              │
│  Core RL Logic:                                            │
│  - reset(task_id, seed) → Initial observation             │
│  - step(action) → (observation, reward, done, info)       │
│  - state → Current episode state                          │
│                                                              │
│  Manages:                                                  │
│  - Episode state (transaction, user profile, evidence)    │
│  - Reward computation (dense shaping)                     │
│  - Terminal conditions (time limit, final action)         │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    ┌────────┐  ┌──────────┐  ┌──────────┐
    │ Tasks  │  │ Adversary│  │ Graders  │
    │        │  │          │  │          │
    │ - SC   │  │Fraudster │  │ - grade_ │
    │ - ATO  │  │Agent     │  │   stolen │
    │ - ML   │  │          │  │ - grade_ │
    │        │  │Strategies│  │   ato    │
    └────────┘  │          │  │ - grade_ │
                │ - Velocity│  │   ml     │
                │ - Device  │  └──────────┘
                │ - Behavior│
                │ - Network │
                └──────────┘
```

---

## The Three Components

### 1. **Tasks** (`tasks/`)
Each task is a **scenario generator** that creates reproducible fraud situations:

```python
# tasks/stolen_card.py
def generate_episode(seed: int) -> Scenario:
    """Generate a stolen card fraud scenario."""
    # Returns a Scenario with:
    # - transaction: $2,499.99 at electronics store, 3 AM, new location
    # - user_profile: typical spend $50, home city is different
    # - fraud_type: "stolen_card"
    # - hidden_signals: velocity anomaly, device mismatch, IP risk
```

**No training happens here.** It's just data generation.

---

### 2. **Environment** (`sentinelx/server/environment.py`)
The **game engine** that:
- Manages episode state
- Dispatches actions to reveal evidence
- Computes rewards
- Checks terminal conditions

```python
class SentinelXEnvironment:
    def reset(task_id, seed):
        # Load scenario from task factory
        # Initialize hidden state
        # Return partial observation (only surface data visible)
        return FraudObservation(
            transaction={...},
            user_profile={...},
            velocity_data=None,  # Hidden until queried
            device_history=None,  # Hidden until queried
            ...
        )
    
    def step(action: FraudAction):
        # If action is investigation (e.g., "check_device_history"):
        #   - Reveal the hidden evidence
        #   - Reward: +0.05 if relevant, -0.02 if wasted
        # If action is intervention (e.g., "block_transaction"):
        #   - Check if fraud actually happened
        #   - Reward: +0.30 if correct, -0.40 if false positive
        # Return updated observation + reward + done flag
        return FraudObservation(...), reward, done, info
```

**No training happens here either.** It's just state management and reward computation.

---

### 3. **Adversary** (`sentinelx/adversary/`)
An **embedded opponent** that learns from the agent's detection patterns:

```python
class FraudsterAgent:
    def __init__(episode_history):
        # Analyze what detection methods the agent has used
        self.detection_methods_seen = ["velocity", "device"]
        
        # Escalate strategy to bypass those methods
        self.strategy = BehavioralMimicryStrategy()
    
    def generate_scenario(seed):
        # Generate fraud that avoids the agent's known detection methods
        # E.g., if agent always checks velocity, space out transactions
        return Scenario(...)
```

This creates **non-stationarity** — the environment gets harder as the agent learns.

---

## How an Episode Works

### Step 1: Reset
```
Agent calls: POST /reset
  {
    "task_id": "stolen-card-easy",
    "seed": 42
  }

Environment returns:
  {
    "observation": {
      "transaction": {"amount": 2499.99, "merchant": "BestBuy", ...},
      "user_profile": {"account_age": 180, "typical_spend": 50, ...},
      "velocity_data": null,  ← Hidden
      "device_history": null,  ← Hidden
      "available_actions": ["query_velocity", "check_device_history", ...],
      "time_remaining": 15
    },
    "reward": null,
    "done": false
  }
```

### Step 2: Agent Investigates
```
Agent calls: POST /step
  {
    "action_type": "check_device_history",
    "reasoning": "3 AM transaction from new device — suspicious"
  }

Environment:
  1. Checks if action is relevant to fraud type
  2. Reveals device_history from hidden state
  3. Computes reward: +0.05 (good investigation)
  4. Decrements time_remaining: 14
  5. Returns updated observation

Response:
  {
    "observation": {
      "transaction": {...},
      "user_profile": {...},
      "device_history": {
        "new_device": true,
        "first_use": "2 hours ago",
        "risk_score": 0.92
      },  ← Now revealed!
      "available_actions": [...],
      "time_remaining": 14
    },
    "reward": 0.05,
    "done": false
  }
```

### Step 3: Agent Decides
```
Agent calls: POST /step
  {
    "action_type": "block_transaction",
    "reasoning": "New device + 3 AM + location mismatch = fraud"
  }

Environment:
  1. Checks if fraud actually happened (ground truth)
  2. Fraud DID happen → Correct decision!
  3. Computes reward: +0.30 (correct block)
  4. Sets done=true (terminal action)
  5. Grades episode

Response:
  {
    "observation": {...},
    "reward": 0.30,
    "done": true,
    "info": {
      "score": 0.90,  ← Final grade [0, 1]
      "success": true,
      "steps": 2,
      "rewards": [0.05, 0.30]
    }
  }
```

---

## Reward Function (Dense Shaping)

The environment provides **per-step rewards**, not just end-of-episode:

| Event | Reward | Why |
|-------|--------|-----|
| Good investigation (relevant evidence) | +0.05 | Encourage thoroughness |
| Wasted investigation (irrelevant) | -0.02 | Discourage inefficiency |
| Correct block (fraud confirmed) | +0.30 | Primary objective |
| Correct approve (legitimate) | +0.30 | Avoid false positives |
| False positive (blocked legitimate) | -0.40 | Penalize customer harm |
| False negative (approved fraud) | -0.50 | Penalize fraud loss |
| Time pressure (per tick) | -0.01 | Encourage speed |
| SAR filed when required | +0.15 | Regulatory compliance |
| SAR missed | -0.20 | Regulatory failure |
| Over-relying on one signal | -0.10 | Teach robustness |

---

## Where Training Happens

**SentinelX does NOT train models.** Training happens **outside** in `inference.py`:

```python
# inference.py
from openai import OpenAI

client = OpenAI(api_key=HF_TOKEN, base_url="https://router.huggingface.co/v1")

for task in TASKS:
    obs = env.reset(task_id=task)
    
    for step in range(MAX_STEPS):
        # Call the LLM (Qwen2.5-7B)
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[...],  # Conversation history
            temperature=0.2,
        )
        
        # LLM generates an action
        action = parse_action(response.content)
        
        # Environment evaluates it
        obs, reward, done, info = env.step(action)
        
        if done:
            break
```

**The LLM is NOT being trained here.** It's just being **evaluated** on the environment.

If you wanted to **fine-tune** the LLM on SentinelX, you'd:
1. Run many episodes with different agents
2. Collect (observation, action, reward) tuples
3. Use those as training data for supervised fine-tuning or RL fine-tuning
4. But that happens **outside** SentinelX

---

## What SentinelX Provides

✅ **Deterministic simulation** of fraud scenarios  
✅ **Dense reward signals** for RL training  
✅ **Adaptive adversary** for non-stationary learning  
✅ **Grading system** to measure agent performance  
✅ **Web UI** for human play  
✅ **REST API** for agent integration  

❌ **NOT** a training framework  
❌ **NOT** a model  
❌ **NOT** fine-tuning code  

---

## For the Hackathon

The judges will:
1. Deploy your Space
2. Run `inference.py` with a baseline LLM (e.g., Qwen2.5-7B)
3. Measure the LLM's performance on all 3 tasks
4. Grade your environment on:
   - Real-world utility (30%)
   - Task quality (25%)
   - Environment design (20%)
   - Code quality (15%)
   - Creativity (10%)

Your job is to make a **good environment**, not train a model.

---

## Key Takeaway

```
┌──────────────────────────────────────────────────────────┐
│  SentinelX = Evaluation Harness                          │
│                                                          │
│  Like a gym for fraud detection:                        │
│  - You don't train in the gym                           │
│  - You test your fitness in the gym                     │
│  - The gym measures your performance                    │
│                                                          │
│  Similarly:                                             │
│  - You don't train LLMs in SentinelX                    │
│  - You test LLMs in SentinelX                           │
│  - SentinelX measures their performance                 │
└──────────────────────────────────────────────────────────┘
```

