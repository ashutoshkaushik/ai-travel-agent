"""Every external HTTP call goes through request(): a disk cache plus real-world resilience.

- Cache: repeat runs are free (SerpApi has ~250 free searches/month), fast, and rate-limit proof.
  The key is method + URL + params + JSON body with secrets stripped, so keys never land on disk.
- Retries: timeouts, network errors, 429 and 5xx are retried with exponential backoff + jitter.
- Circuit breaker: after repeated failures a provider is "open" for a cooldown, and calls fail fast
  (or serve a stale cached copy) instead of hammering a service that is down.
- Never raises: on final failure it returns an error response the tools turn into
  {"status": "error", ...} data, like every other tool error.
- Learning hooks: inject_faults() simulates outages ("Break it" mode), RETRY_EVENTS lets the UI show
  each retry in the trace, and TRAVEL_FROZEN_CACHE=1 serves cached responses regardless of age so
  evals are deterministic.
"""

import gzip
import hashlib
import json
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

from travel_planner.config import CACHE_DIR
from travel_planner.usage import provider_for, record

SECRET_PARAMS = {"api_key", "apikey", "key", "token"}
DEFAULT_TTL_SECONDS = 24 * 60 * 60

MAX_ATTEMPTS = 3
BASE_DELAY_S = 0.5
MAX_DELAY_S = 8.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
BREAKER_THRESHOLD = 3  # consecutive failed requests (after retries) that open the breaker
BREAKER_COOLDOWN_S = 60.0

_lock = threading.Lock()
_breakers: dict[str, dict] = {}  # provider -> {"failures": int, "open_until": float}
_faults: dict[str, list[int]] = {}  # provider -> status codes to simulate on the next attempts
RETRY_EVENTS: deque = deque(maxlen=200)  # (provider, attempt, status, delay, simulated) for the live trace
_sleep = time.sleep  # patched in tests


@dataclass
class CachedResponse:
    status_code: int
    data: Any
    from_cache: bool


def frozen() -> bool:
    return os.environ.get("TRAVEL_FROZEN_CACHE") == "1"


def inject_faults(provider: str, count: int, status: int = 503) -> None:
    """Make the next `count` attempts to this provider fail with `status` (learning / tests)."""
    with _lock:
        _faults[provider] = [status] * count


def clear_faults() -> None:
    with _lock:
        _faults.clear()


def reset_breakers() -> None:
    with _lock:
        _breakers.clear()


def breaker_state() -> dict[str, dict]:
    with _lock:
        return {p: dict(b) for p, b in _breakers.items()}


def _take_fault(provider: str) -> int | None:
    with _lock:
        queue = _faults.get(provider)
        return queue.pop(0) if queue else None


def _breaker_open(provider: str) -> bool:
    with _lock:
        return _breakers.get(provider, {}).get("open_until", 0) > time.time()


def _breaker_result(provider: str, success: bool) -> None:
    with _lock:
        b = _breakers.setdefault(provider, {"failures": 0, "open_until": 0.0})
        if success:
            b["failures"], b["open_until"] = 0, 0.0
        else:
            b["failures"] += 1
            if b["failures"] >= BREAKER_THRESHOLD:
                b["open_until"] = time.time() + BREAKER_COOLDOWN_S


def _copy_from_seed(key: str, path) -> None:
    """Evals record into their own cache folder; TRAVEL_CACHE_SEED_DIR lets them reuse responses already
    cached by the app (saving SerpApi quota) instead of calling the API again."""
    seed = os.environ.get("TRAVEL_CACHE_SEED_DIR")
    if not seed:
        return
    from pathlib import Path

    for candidate in (Path(seed) / f"{key}.json.gz", Path(seed) / f"{key}.json"):
        if candidate.exists():
            raw = candidate.read_bytes()
            path.write_bytes(raw if candidate.suffix == ".gz" else gzip.compress(raw, mtime=0))
            return


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter: random in [0, min(max, base * 2^(attempt-1))]."""
    return random.uniform(0, min(MAX_DELAY_S, BASE_DELAY_S * 2 ** (attempt - 1)))


def _cache_key(method: str, url: str, params: dict | None, json_body: dict | None) -> str:
    safe_params = {k: v for k, v in (params or {}).items() if k.lower() not in SECRET_PARAMS}
    raw = json.dumps([method.upper(), url, safe_params, json_body], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def request(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    timeout: float = 30.0,
) -> CachedResponse:
    """Cached, retried, circuit-broken HTTP. Only 2xx responses are cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(method, url, params, json_body)
    path = CACHE_DIR / f"{key}.json.gz"  # gzip: a Duffel search is ~800 KB of JSON, ~60 KB compressed
    legacy = CACHE_DIR / f"{key}.json"
    if not path.exists() and legacy.exists():
        path = legacy
    if not path.exists():
        _copy_from_seed(key, path)
    provider = provider_for(url)
    engine = (params or {}).get("engine")  # which SerpApi engine, for the usage report

    def cached(stale_ok: bool) -> CachedResponse | None:
        if path.exists() and (stale_ok or frozen() or time.time() - path.stat().st_mtime < ttl_seconds):
            record(provider, cached=True, engine=engine)
            raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            return CachedResponse(200, json.loads(raw), from_cache=True)
        return None

    if _breaker_open(provider):
        # Fail fast while the provider is down; a stale copy beats no answer
        return cached(stale_ok=True) or CachedResponse(
            503, {"error": f"{provider} is temporarily unavailable (circuit breaker open, retry later)"}, from_cache=False)

    last_status, last_error = None, ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        simulated = _take_fault(provider)
        if simulated is None:
            hit = cached(stale_ok=False)
            if hit:
                return hit
            try:
                resp = httpx.request(method, url, params=params, json=json_body, headers=headers, timeout=timeout)
                last_status = resp.status_code
            except (httpx.TimeoutException, httpx.TransportError) as e:
                resp, last_status, last_error = None, None, f"{type(e).__name__}: {e}"
        else:
            resp, last_status, last_error = None, simulated, f"simulated {simulated}"
        record(provider, cached=False, status=last_status, engine=engine, attempt=attempt, simulated=simulated is not None)

        if resp is not None and resp.status_code not in RETRYABLE_STATUS:
            try:
                data = resp.json()
            except ValueError:
                data = {"raw_body": resp.text[:500]}
            if resp.is_success:
                (CACHE_DIR / f"{key}.json.gz").write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
            _breaker_result(provider, success=True)  # the service answered, even if with a 4xx
            return CachedResponse(resp.status_code, data, from_cache=False)

        if attempt < MAX_ATTEMPTS:
            delay = backoff_delay(attempt)
            RETRY_EVENTS.append((provider, attempt, last_status or last_error, round(delay, 2), simulated is not None))
            _sleep(delay)

    _breaker_result(provider, success=False)
    reason = f"HTTP {last_status}" if last_status else last_error
    return CachedResponse(last_status or 503, {"error": f"{provider} failed after {MAX_ATTEMPTS} attempts ({reason})"},
                          from_cache=False)
