"""Task 2 — Account Takeover (Medium)."""
from __future__ import annotations

from sentinelx.adversary.strategies import BehavioralMimicryStrategy, Scenario
from tasks import load_user_profile, TASK_USER_MAP

TASK_ID = "account-takeover-medium"
MAX_TICKS = 20

TASK_CONFIG = {
    "id": TASK_ID,
    "description": "Identify an account takeover hidden behind normal-looking transaction details.",
    "difficulty": "medium",
    "max_ticks": MAX_TICKS,
    "initial_adversary_strategy": "behavioral_mimicry",
    "expected_score_range": (0.60, 0.75),
}


def generate_episode(seed: int, adversary_strategy: str = "behavioral_mimicry") -> Scenario:
    user_profile = load_user_profile(TASK_USER_MAP[TASK_ID])
    strategy = BehavioralMimicryStrategy()
    return strategy.generate(TASK_ID, seed, user_profile)
