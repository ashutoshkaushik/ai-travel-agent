"""Smoke-test Duffel test-mode access: flight search (should work) and Stays (may need approval).

Usage:
    1. Put DUFFEL_ACCESS_TOKEN=duffel_test_... in travel_planner/.env
    2. python3 scripts/check_duffel.py

Uses only the standard library so it runs before any project dependencies are installed.
"""

import json
import os
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

API = "https://api.duffel.com"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_token() -> str:
    token = os.environ.get("DUFFEL_ACCESS_TOKEN")
    if not token and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("DUFFEL_ACCESS_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not token:
        raise SystemExit(f"DUFFEL_ACCESS_TOKEN not set (checked env and {ENV_FILE})")
    if not token.startswith("duffel_test_"):
        raise SystemExit("Refusing to run: token is not a test-mode token (expected duffel_test_ prefix)")
    return token


def post(path: str, body: dict, token: str) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps({"data": body}).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Duffel-Version": "v2",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return e.code, {"raw_body": raw.decode(errors="replace")[:500]}


def check_flights(token: str) -> None:
    depart = (date.today() + timedelta(days=70)).isoformat()
    status, payload = post(
        "/air/offer_requests?return_offers=true&supplier_timeout=20000",
        {
            "slices": [{"origin": "SFO", "destination": "NRT", "departure_date": depart}],
            "passengers": [{"type": "adult"}],
            "cabin_class": "economy",
        },
        token,
    )
    print(f"\n[Flights] SFO -> NRT on {depart}: HTTP {status}")
    if status >= 400:
        print(json.dumps(payload.get("errors", payload), indent=2))
        return
    offers = payload["data"].get("offers", [])
    print(f"  {len(offers)} offers returned")
    for offer in offers[:3]:
        print(f"  - {offer['owner']['name']}: {offer['total_amount']} {offer['total_currency']}")


def check_stays(token: str) -> None:
    check_in = date.today() + timedelta(days=70)
    status, payload = post(
        "/stays/search",
        {
            "rooms": 1,
            "guests": [{"type": "adult"}],
            "check_in_date": check_in.isoformat(),
            "check_out_date": (check_in + timedelta(days=2)).isoformat(),
            # Duffel's test hotels only exist at these coordinates
            "location": {
                "radius": 5,
                "geographic_coordinates": {"latitude": -24.38, "longitude": -128.32},
            },
        },
        token,
    )
    print(f"\n[Stays] test-hotel search: HTTP {status}")
    if status >= 400:
        print(json.dumps(payload.get("errors", payload), indent=2))
        print("  -> Stays not enabled for this account (expected; it is per-request access)")
        return
    results = payload["data"].get("results", [])
    print(f"  {len(results)} test hotels returned")
    for r in results[:3]:
        print(f"  - {r['accommodation']['name']}")


if __name__ == "__main__":
    token = load_token()
    check_flights(token)
    check_stays(token)
