"""Task factories and shared utilities."""
import json
import os
from typing import Any, Dict

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_user_profile(user_id: str) -> Dict[str, Any]:
    path = os.path.join(_DATA_DIR, "transaction_profiles.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for u in data["users"]:
        if u["user_id"] == user_id:
            return u
    raise KeyError(f"User {user_id!r} not found in transaction_profiles.json")


# Default user ids per task
TASK_USER_MAP = {
    "stolen-card-easy": "U001",
    "account-takeover-medium": "U002",
    "money-laundering-hard": "U003",
}
