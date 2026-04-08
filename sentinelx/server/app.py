"""FastAPI server for SentinelX — one meaningful line."""
import sys
import os

# Ensure the project root is on the path when run from this directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openenv.core.env_server import create_fastapi_app  # type: ignore
from sentinelx.server.environment import SentinelXEnvironment

# create_fastapi_app wires up all required endpoints:
#   /ws      — WebSocket persistent session (used by EnvClient)
#   /reset   — HTTP POST (stateless)
#   /step    — HTTP POST (stateless)
#   /state   — HTTP GET
#   /health  — HTTP GET
#   /web     — Interactive web UI
#   /docs    — OpenAPI docs
app = create_fastapi_app(SentinelXEnvironment)
