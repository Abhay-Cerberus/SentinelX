# 🤖 Training an LLM for SentinelX

This guide explains how to train/fine-tune a Large Language Model (LLM) to play the SentinelX fraud investigation environment.

---

## Overview

**SentinelX is NOT a training framework.** It's an evaluation environment. To train an LLM, you need to:

1. **Collect training data** from the environment
2. **Fine-tune an LLM** using that data
3. **Evaluate the fine-tuned model** on SentinelX

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  1. Collect Data     →   2. Fine-tune LLM   →   3. Evaluate │
│     (SentinelX)            (External)            (SentinelX) │
│                                                              │
│  Run episodes with        Use collected          Test the    │
│  baseline agents          data to train          trained     │
│  → save (obs, act, rew)   a model                model       │
└─────────────────────────────────────────────────────────────┘
```

---

## Method 1: Supervised Fine-Tuning (SFT)

The simplest approach: train the LLM to imitate expert decisions.

### Step 1: Collect Expert Demonstrations

Create a script to collect expert trajectories:

```python
# collect_expert_data.py
import json
from sentinelx.client import SentinelXEnv
from sentinelx.models import FraudAction

def expert_policy(observation):
    """Simple rule-based expert policy."""
    txn = observation.transaction
    profile = observation.user_profile
    
    # Check for obvious fraud signals
    if observation.device_history and observation.device_history.get("is_new_device"):
        return FraudAction(
            action_type="block_transaction",
            reasoning="New device detected"
        )
    
    if observation.velocity_data and observation.velocity_data.get("past_1_hour_count", 0) > 5:
        return FraudAction(
            action_type="block_transaction",
            reasoning="High velocity detected"
        )
    
    # If no evidence yet, investigate
    if not observation.device_history:
        return FraudAction(
            action_type="check_device_history",
            reasoning="Check device first"
        )
    
    if not observation.velocity_data:
        return FraudAction(
            action_type="query_velocity",
            reasoning="Check velocity"
        )
    
    # Default: approve
    return FraudAction(
        action_type="approve_transaction",
        reasoning="No fraud signals detected"
    )

def collect_episode(env, task_id, seed):
    """Collect one episode of expert demonstrations."""
    trajectory = []
    
    result = env.reset(task_id=task_id, seed=seed)
    
    while not result.done:
        action = expert_policy(result.observation)
        
        # Save (observation, action) pair
        trajectory.append({
            "observation": result.observation.model_dump(),
            "action": action.model_dump(),
            "reward": result.reward,
        })
        
        result = env.step(action)
    
    # Add final result
    trajectory.append({
        "observation": result.observation.model_dump(),
        "action": None,
        "reward": result.reward,
        "done": True,
    })
    
    return trajectory

# Collect data
all_trajectories = []
with SentinelXEnv(base_url="http://localhost:8000").sync() as env:
    for task_id in ["stolen-card-easy", "account-takeover-medium", "money-laundering-hard"]:
        for seed in range(100):  # 100 episodes per task
            traj = collect_episode(env, task_id, seed)
            all_trajectories.append(traj)
            print(f"Collected {task_id} seed={seed}, reward={traj[-1]['reward']}")

# Save to JSONL
with open("expert_demonstrations.jsonl", "w") as f:
    for traj in all_trajectories:
        for step in traj:
            f.write(json.dumps(step) + "\n")

print(f"Collected {len(all_trajectories)} episodes")
```

### Step 2: Convert to Chat Format

Convert the demonstrations to a chat format for fine-tuning:

```python
# convert_to_chat_format.py
import json

def format_observation(obs):
    """Format observation as a user message."""
    txn = obs.get("transaction", {})
    profile = obs.get("user_profile", {})
    
    msg = f"""## Transaction
Amount: ${txn.get('amount', '?')}
Merchant: {txn.get('merchant', '?')}
Location: {txn.get('location', '?')}
Time: {txn.get('timestamp', '?')}

## User Profile
Account Age: {profile.get('account_age_days', '?')} days
Typical Spend: ${profile.get('typical_transaction_size', '?')}
Typical Location: {profile.get('typical_location', '?')}

## Available Actions
{', '.join(obs.get('available_actions', []))}

## Evidence Summary
{obs.get('evidence_summary', 'No evidence yet')}
"""
    return msg

def format_action(action):
    """Format action as assistant response."""
    return json.dumps({
        "action_type": action["action_type"],
        "parameters": action.get("parameters", {}),
        "reasoning": action.get("reasoning", ""),
    })

# Convert to chat format
chat_data = []
with open("expert_demonstrations.jsonl") as f:
    for line in f:
        step = json.loads(line)
        
        if step.get("action") is None:
            continue  # Skip terminal steps
        
        chat_data.append({
            "messages": [
                {"role": "system", "content": "You are a fraud analyst AI. Investigate transactions and make decisions."},
                {"role": "user", "content": format_observation(step["observation"])},
                {"role": "assistant", "content": format_action(step["action"])},
            ]
        })

# Save as JSONL
with open("training_data.jsonl", "w") as f:
    for item in chat_data:
        f.write(json.dumps(item) + "\n")

print(f"Created {len(chat_data)} training examples")
```

### Step 3: Fine-Tune with Hugging Face

Use the Hugging Face `trl` library:

```python
# finetune.py
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load training data
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# Fine-tune
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",  # or format as needed
    max_seq_length=512,
    tokenizer=tokenizer,
    args={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "learning_rate": 5e-5,
        "output_dir": "./sentinelx-model",
    },
)

trainer.train()

# Save model
trainer.save_model("./sentinelx-model")
tokenizer.save_pretrained("./sentinelx-model")
```

Or use the Hugging Face AutoTrain:

```bash
autotrain llm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-path training_data.jsonl \
  --output-dir ./sentinelx-model \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 5e-5
```

---

## Method 2: Reinforcement Learning from Human Feedback (RLHF)

More advanced: train the LLM to maximize reward using RL.

### Step 1: Create a Reward Model

Train a reward model from human preferences:

```python
# collect_preferences.py
import json
from sentinelx.client import SentinelXEnv
from sentinelx.models import FraudAction

def collect_preferences():
    """Collect human preferences between two actions."""
    preferences = []
    
    # Show humans pairs of actions and ask which is better
    # This is simplified - in practice, use a UI like Label Studio
    
    with SentinelXEnv(base_url="http://localhost:8000").sync() as env:
        for _ in range(1000):
            result = env.reset(task_id="stolen-card-easy")
            
            # Generate two candidate actions
            action1 = generate_candidate_action(result.observation)
            action2 = generate_candidate_action(result.observation)
            
            # Ask human which is better (simplified)
            # In practice, use a labeling interface
            preferred = input(f"Which action is better? (1/2)\n1: {action1}\n2: {action2}\n> ")
            
            preferences.append({
                "observation": result.observation.model_dump(),
                "action1": action1.model_dump(),
                "action2": action2.model_dump(),
                "preferred": int(preferred) - 1,  # 0 or 1
            })
    
    return preferences
```

### Step 2: Train Reward Model

```python
# train_reward_model.py
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    num_labels=1,  # Scalar reward output
)

# Train on preferences
# (Implementation depends on your preference data format)
```

### Step 3: PPO Training

Use Proximal Policy Optimization (PPO) with the reward model:

```python
# rlhf_training.py
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM

# Load policy model
policy = AutoModelForCausalLM.from_pretrained("./sentinelx-model")

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward-model")

# PPO config
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
)

# PPO trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=policy,
    tokenizer=tokenizer,
)

# Training loop
for epoch in range(100):
    # Generate trajectories
    # Compute rewards with reward model
    # Update policy with PPO
    pass
```

---

## Method 3: Direct Environment Interaction (Online RL)

Train directly on the environment using online RL.

### Using RL4LMs

```python
# online_rl_training.py
from rl4lm import RL4LMTrainer
from sentinelx.client import SentinelXEnv

# Create environment wrapper
class SentinelXWrapper:
    def __init__(self, base_url):
        self.env = SentinelXEnv(base_url=base_url)
        self.session_id = None
    
    def reset(self):
        result = self.env.reset()
        self.session_id = result.session_id
        return self._format_obs(result.observation)
    
    def step(self, action_text):
        action = self._parse_action(action_text)
        result = self.env.step(action, session_id=self.session_id)
        return self._format_obs(result.observation), result.reward, result.done
    
    def _format_obs(self, obs):
        # Format observation as text
        return f"Transaction: {obs.transaction}\n..."
    
    def _parse_action(self, text):
        # Parse LLM output to FraudAction
        import json
        data = json.loads(text)
        return FraudAction(**data)

# Train with RL4LMs
trainer = RL4LMTrainer(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    env=SentinelXWrapper("http://localhost:8000"),
    algorithm="ppo",
)

trainer.train(num_episodes=10000)
```

---

## Method 4: Prompt Engineering (No Training)

The simplest approach: use prompt engineering with an existing LLM.

### System Prompt

```python
SYSTEM_PROMPT = """You are SentinelX, an expert financial fraud analyst AI.

## Your Role
Investigate suspicious transactions and make decisions to prevent fraud while minimizing false positives.

## Workflow
1. Read the transaction and user profile carefully
2. Use investigation actions to gather evidence
3. Once you have enough evidence, take a decisive action
4. Always provide clear reasoning

## Available Actions

### Investigation (gathers evidence)
- query_velocity: Check transaction frequency patterns
- check_device_history: Get device fingerprint
- lookup_ip_reputation: Check IP risk score
- check_behavioral_biometrics: Analyze typing/mouse patterns
- check_active_sessions: See concurrent logins
- query_linked_accounts: Explore network connections
- analyze_temporal_pattern: Check timing coordination

### Intervention (ends episode)
- approve_transaction: Allow the transaction
- block_transaction: Deny the transaction
- request_3ds: Request 3D Secure verification
- send_push_notification: Alert the user
- force_password_reset: Require password change
- temporarily_freeze_account: Lock the account

### Regulatory
- file_sar: File Suspicious Activity Report
- file_ctr: File Currency Transaction Report
- escalate_to_compliance: Escalate to compliance team

## Response Format
You MUST respond with a single JSON object:
{
  "action_type": "<action_name>",
  "parameters": {},
  "reasoning": "<your reasoning>"
}

## Key Rules
- Do NOT approve unless confident it's legitimate
- For money laundering: check network depth-2 before filing SAR
- For account takeover: check behavioral biometrics first
- Respond ONLY with the JSON object
"""
```

### Few-Shot Examples

```python
FEW_SHOT_EXAMPLES = [
    {
        "user": "Transaction: $2,499.99 at ElectronicsStore, 3 AM, Miami FL\nUser: Typical spend $50, home city Austin TX",
        "assistant": '{"action_type": "check_device_history", "parameters": {}, "reasoning": "Large amount at 3 AM from new location - check device first"}'
    },
    {
        "user": "Device History: New device (never seen before), risk score 0.88",
        "assistant": '{"action_type": "query_velocity", "parameters": {}, "reasoning": "New device confirmed - check velocity pattern"}'
    },
    {
        "user": "Velocity: 8 transactions in last hour, typical is 3 per week",
        "assistant": '{"action_type": "block_transaction", "parameters": {}, "reasoning": "Clear fraud: new device + high velocity + location mismatch"}'
    },
]
```

---

## Evaluation

After training, evaluate your model:

```python
# evaluate_model.py
import json
from openai import OpenAI
from sentinelx.client import SentinelXEnv
from sentinelx.models import FraudAction

def evaluate_model(model_path, base_url):
    """Evaluate a fine-tuned model on SentinelX."""
    client = OpenAI(
        base_url=base_url,
        api_key="your-key",
    )
    
    results = []
    
    with SentinelXEnv(base_url="http://localhost:8000").sync() as env:
        for task_id in ["stolen-card-easy", "account-takeover-medium", "money-laundering-hard"]:
            for seed in range(10):  # 10 episodes per task
                result = env.reset(task_id=task_id, seed=seed)
                total_reward = 0
                
                while not result.done:
                    # Call your model
                    response = client.chat.completions.create(
                        model=model_path,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": format_obs(result.observation)},
                        ],
                    )
                    
                    # Parse action
                    action = parse_action(response.choices[0].message.content)
                    
                    # Step environment
                    result = env.step(action)
                    total_reward += result.reward or 0
                
                results.append({
                    "task_id": task_id,
                    "seed": seed,
                    "total_reward": total_reward,
                    "success": total_reward > 0.3,
                })
    
    # Compute metrics
    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    success_rate = sum(r["success"] for r in results) / len(results)
    
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Success Rate: {success_rate:.1%}")
    
    return results

# Evaluate
evaluate_model("./sentinelx-model", "https://router.huggingface.co/v1")
```

---

## Recommended Approach

For the hackathon, I recommend:

1. **Start with Prompt Engineering** (Method 4)
   - Fastest to implement
   - No training required
   - Use the existing `inference.py` with a good system prompt

2. **If you have time, try SFT** (Method 1)
   - Collect 100-500 expert demonstrations
   - Fine-tune a small model (7B parameters)
   - Evaluate on all 3 tasks

3. **Advanced: RLHF** (Method 2)
   - Only if you have significant compute and time
   - Requires preference data collection
   - Can achieve better performance

---

## Resources

- **Hugging Face TRL**: https://huggingface.co/docs/trl
- **RL4LMs**: https://github.com/allenai/rl4lm
- **OpenAI Fine-tuning**: https://platform.openai.com/docs/guides/fine-tuning
- **SentinelX Client**: `sentinelx/client.py`

---

## Quick Start: No Training Required

If you just want to run the baseline for the hackathon:

```bash
# 1. Set up environment
export HF_TOKEN=hf_your_token
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export SENTINELX_URL=https://abhay-maheshwari-sentinelx.hf.space

# 2. Run inference
python inference.py
```

The baseline uses prompt engineering with Qwen2.5-7B-Instruct. No training needed!