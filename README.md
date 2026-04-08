# SentinelX — Financial Fraud Investigation Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **OpenEnv Hackathon Round 1 Submission** — Solo: Abhay Maheshwari

SentinelX is a reinforcement learning environment where an AI agent acts as a financial fraud analyst. The environment simulates three real-world fraud types across escalating difficulty levels, with an **adaptive adversary** that learns from the agent's detection patterns across episodes — creating true non-stationarity that forces robust multi-signal reasoning.

---

## Environment Description

The agent investigates suspicious transactions by querying hidden evidence (velocity patterns, device fingerprints, IP reputation, behavioral biometrics, transaction networks) and then taking decisive actions (block, approve, freeze, file SAR, etc.) under time pressure.

**The killer feature:** The internal `FraudsterAgent` tracks which detection methods the investigating agent relies on and escalates its tactics accordingly — from simple velocity anomalies, to device spoofing, to full behavioral mimicry. Each episode the agent "teaches" the fraudster something new.

---

## Tasks

| Task | Difficulty | Expected Score | Description |
|------|-----------|---------------|-------------|
| `stolen-card-easy` | Easy | 0.85–0.95 | Stolen card with clear signals — location anomaly, 3 AM, new device, VPN IP |
| `account-takeover-medium` | Medium | 0.60–0.75 | Account takeover hidden behind normal transaction details; fraud lives in behavioral biometrics |
| `money-laundering-hard` | Hard | 0.35–0.55 | Multi-hop laundering network through shell companies; requires graph analysis + SAR filing |

---

## Action Space

```
Investigation:   query_velocity, check_device_history, lookup_ip_reputation,
                 check_behavioral_biometrics, check_active_sessions,
                 query_linked_accounts, analyze_temporal_pattern,
                 check_business_registration, request_kyc_documents

Intervention:    approve_transaction, block_transaction, request_3ds,
                 send_push_notification, force_password_reset,
                 temporarily_freeze_account

Regulatory:      file_sar, file_ctr, escalate_to_compliance

Special:         monitor_only, request_additional_info
```

All actions include a `reasoning` field for interpretability and logging.

---

## Observation Space

| Field | Visibility | Type |
|-------|-----------|------|
| `transaction` | Always | Dict (amount, merchant, location, timestamp) |
| `user_profile` | Always | Dict (account age, typical spend, typical location) |
| `velocity_data` | After `query_velocity` | Dict |
| `device_history` | After `check_device_history` | Dict |
| `ip_reputation` | After `lookup_ip_reputation` | Dict |
| `behavioral_biometrics` | After `check_behavioral_biometrics` | Dict |
| `active_sessions` | After `check_active_sessions` | List |
| `network_connections` | After `query_linked_accounts` | List |
| `temporal_pattern` | After `analyze_temporal_pattern` | Dict |
| `business_registration` | After `check_business_registration` | Dict |
| `available_actions` | Always | List[str] |
| `time_remaining` | Always | int (ticks) |
| `evidence_summary` | Always | str (natural language summary) |

---

## Reward Function

| Event | Reward |
|-------|--------|
| Relevant investigation action | +0.05 |
| Irrelevant / repeated action | −0.02 |
| Correct block (fraud) | +0.30 |
| Active fraud stopped | +0.10 bonus |
| Correct approve (legitimate) | +0.30 |
| False positive | −0.40 |
| False negative | −0.50 |
| Time pressure (per tick) | −0.01 |
| SAR filed when required | +0.15 |
| SAR missed | −0.20 |
| Single-signal overuse | −0.10 |

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-7B-Instruct` via HF Inference Router:

| Task | Score | Steps |
|------|-------|-------|
| `stolen-card-easy` | 0.87 | 3 |
| `account-takeover-medium` | 0.62 | 5 |
| `money-laundering-hard` | 0.40 | 10 |

> **Note:** You can swap the model via the `MODEL_NAME` env var. `7B` runs comfortably under 5 min per task on the HF router. Use `Qwen/Qwen2.5-14B-Instruct` for higher scores if time allows.

---

## Setup & Usage

### Local (Uvicorn)

```bash
git clone https://huggingface.co/spaces/Abhay-Cerberus/SentinelX
cd SentinelX
pip install -r requirements.txt
uvicorn sentinelx.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Local (Docker)

```bash
docker build -t sentinelx:latest .
docker run -d -p 8000:8000 sentinelx:latest
curl http://localhost:8000/health  # {"status": "healthy"}
```

### Baseline Inference

1. Create a `.env` file in the root directory:
```env
HF_TOKEN=hf_your_token
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
SENTINELX_URL=http://localhost:8000
```

2. Run the baseline evaluation:
```bash
python inference.py
```

### Python Client

```python
from sentinelx.client import SentinelXEnv
from sentinelx.models import FraudAction

with SentinelXEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="stolen-card-easy", seed=42)
    print(result.observation.evidence_summary)

    result = env.step(FraudAction(
        action_type="check_device_history",
        reasoning="3AM transaction from high-risk merchant — check device first"
    ))
    print(result.observation.device_history)
    print(f"Reward: {result.reward}")
```

---

## Project Structure

```
SentinelX/
├── inference.py              ← Hackathon baseline script
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← Container definition
├── pyproject.toml            ← Package metadata
│
├── sentinelx/
│   ├── models.py             ← FraudAction, FraudObservation, FraudInvestigationState
│   ├── client.py             ← SentinelXEnv (EnvClient subclass)
│   ├── adversary/
│   │   ├── fraudster.py      ← Adaptive FraudsterAgent
│   │   └── strategies.py    ← Pluggable fraud strategy implementations
│   └── server/
│       ├── environment.py    ← Core RL logic (reset/step/state)
│       ├── graders.py        ← Deterministic per-task scoring
│       └── app.py            ← FastAPI wiring
│
├── tasks/
│   ├── stolen_card.py        ← Task 1 episode factory
│   ├── account_takeover.py  ← Task 2 episode factory
│   └── money_laundering.py  ← Task 3 episode factory
│
├── data/
│   ├── transaction_profiles.json
│   ├── merchant_registry.json
│   ├── ip_reputation_db.json
│   └── shell_company_graph.json
│
└── tests/
    ├── test_models.py
    ├── test_environment.py
    ├── test_graders.py
    └── test_adversary.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/reset` | POST | Start new episode: `{"task_id": "...", "seed": 42}` |
| `/step` | POST | Execute action: `{"action_type": "...", "parameters": {}, "reasoning": "..."}` |
| `/state` | GET | Current episode state |
| `/ws` | WS | Persistent WebSocket session (used by Python client) |
| `/docs` | GET | Interactive OpenAPI documentation |
| `/web` | GET | Built-in web interface |

---

## License

MIT © 2024 Abhay Maheshwari
