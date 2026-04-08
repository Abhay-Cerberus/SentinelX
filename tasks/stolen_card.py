"""Task 1 — Stolen Card Fraud (Easy)."""
from __future__ import annotations

from typing import Any, Dict

from sentinelx.adversary.strategies import HighVelocityStrategy, DeviceSpoofingStrategy, Scenario
from tasks import load_user_profile, TASK_USER_MAP

TASK_ID = "stolen-card-easy"
MAX_TICKS = 15

TASK_CONFIG = {
    "id": TASK_ID,
    "description": "Detect and block a stolen card transaction with clear anomaly signals.",
    "difficulty": "easy",
    "max_ticks": MAX_TICKS,
    "initial_adversary_strategy": "high_velocity",
    "expected_score_range": (0.85, 0.95),
}


def generate_episode(seed: int, adversary_strategy: str = "high_velocity") -> Scenario:
    """Return a reproducible stolen-card scenario."""
    user_profile = load_user_profile(TASK_USER_MAP[TASK_ID])

    strategy_cls = HighVelocityStrategy if adversary_strategy == "high_velocity" else DeviceSpoofingStrategy
    strategy = strategy_cls()
    return strategy.generate(TASK_ID, seed, user_profile)
