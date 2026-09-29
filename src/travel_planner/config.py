"""Central place for settings and API keys, loaded once from .env."""

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)  # .env values take precedence over shell env vars

# Live mode needs a real OpenAI key. Without one (a public, keyless deployment) the app runs on recorded
# runs only; a placeholder lets agents still be built and drawn, and every live button is disabled.
LIVE = bool(os.environ.get("OPENAI_API_KEY"))
if not LIVE:
    os.environ["OPENAI_API_KEY"] = "no-key-recorded-runs-only"

# Committed recordings (replays, Compare results, guide embeddings) that work with no keys
RECORDINGS_DIR = PROJECT_ROOT / "lab" / "recordings"

# Evals point this at evals/cache (committed), so CI replays recorded API and LLM responses
CACHE_DIR = Path(os.environ.get("TRAVEL_CACHE_DIR", PROJECT_ROOT / "cache"))


# Where the code is published, for "View on GitHub" links in the Agent Lab (optional)
REPO_URL = os.environ.get("TRAVEL_REPO_URL", "https://github.com/ashutoshkaushik/ai-travel-agent").rstrip("/")


def today() -> date:
    """Today's date, or TRAVEL_TODAY when set: evals pin it so prompts (and their cache keys) never change."""
    pinned = os.environ.get("TRAVEL_TODAY")
    return date.fromisoformat(pinned) if pinned else date.today()


def require(name: str) -> str:
    """Return an env var or fail loudly with a hint, instead of a confusing 401 later."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to {PROJECT_ROOT / '.env'} (see .env.example).")
    return value


def duffel_token() -> str:
    token = require("DUFFEL_ACCESS_TOKEN")
    if not token.startswith("duffel_test_"):
        raise RuntimeError("Only Duffel test-mode tokens (duffel_test_...) are allowed in this project.")
    return token
