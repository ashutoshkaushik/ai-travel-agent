"""A local usage ledger: one JSON line per external API call or LLM call.

Recorded automatically by http_cache.request (every provider API) and RunStats (every LLM call),
so nothing else has to remember to log. Read it with scripts/usage_report.py.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from travel_planner.config import PROJECT_ROOT

USAGE_LOG = Path(os.environ.get("TRAVEL_USAGE_LOG", PROJECT_ROOT / "usage.jsonl"))

PROVIDERS_BY_HOST = {
    "api.duffel.com": "duffel",
    "serpapi.com": "serpapi",
    "api.opentripmap.com": "opentripmap",
    "nominatim.openstreetmap.org": "nominatim",
    "api.open-meteo.com": "open-meteo",
    "archive-api.open-meteo.com": "open-meteo",
}


def provider_for(url: str) -> str:
    return PROVIDERS_BY_HOST.get(urlparse(url).hostname or "", "other")


def record(provider: str, **fields) -> None:
    """Append one usage event. Never raises: usage logging must not break the agent."""
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "provider": provider, **fields}
    try:
        with USAGE_LOG.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass


def read_events() -> list[dict]:
    if not USAGE_LOG.exists():
        return []
    return [json.loads(line) for line in USAGE_LOG.read_text().splitlines() if line.strip()]
