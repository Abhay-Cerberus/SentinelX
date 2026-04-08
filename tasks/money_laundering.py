"""Task 3 — Money Laundering Network (Hard)."""
from __future__ import annotations

from sentinelx.adversary.strategies import StructuredPaymentsStrategy, ShellNetworkStrategy, Scenario
from tasks import load_user_profile, TASK_USER_MAP

TASK_ID = "money-laundering-hard"
MAX_TICKS = 25

TASK_CONFIG = {
    "id": TASK_ID,
    "description": "Unravel a multi-hop money laundering network and file the required SAR.",
    "difficulty": "hard",
    "max_ticks": MAX_TICKS,
    "initial_adversary_strategy": "structured_payments",
    "expected_score_range": (0.35, 0.55),
}


def generate_episode(seed: int, adversary_strategy: str = "structured_payments") -> Scenario:
    user_profile = load_user_profile(TASK_USER_MAP[TASK_ID])
    if adversary_strategy == "shell_network":
        strategy = ShellNetworkStrategy()
    else:
        strategy = StructuredPaymentsStrategy()
    return strategy.generate(TASK_ID, seed, user_profile)
