"""The booking ledger (SQLite) and the human-decision helpers for the approval middleware.

Bookings are simulated: a row in a local database with a confirmation id. The point is the
write-action discipline, not a real reservation:
- every write is gated by a human decision (HumanInTheLoopMiddleware),
- every write is idempotent (a unique key per item: booking twice returns the first booking),
- every rejection carries a reason, logged as feedback (future eval data).
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from travel_planner.config import PROJECT_ROOT

BOOKINGS_DB = Path(os.environ.get("TRAVEL_BOOKINGS_DB", PROJECT_ROOT / "bookings.db"))
FEEDBACK_LOG = Path(os.environ.get("TRAVEL_FEEDBACK_LOG", PROJECT_ROOT / "feedback.jsonl"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    confirmation    TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    kind            TEXT NOT NULL,
    summary         TEXT NOT NULL,
    amount_usd      REAL NOT NULL,
    details         TEXT NOT NULL,
    created_at      TEXT NOT NULL
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(BOOKINGS_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def create_booking(kind: str, idempotency_key: str, summary: str, amount_usd: float, details: dict) -> dict:
    """Insert a booking, or return the existing one if this exact item was already booked."""
    with _connect() as conn:
        existing = conn.execute("SELECT * FROM bookings WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
        if existing:
            return {"confirmation": existing["confirmation"], "already_booked": True}
        confirmation = f"{kind[:2].upper()}-{uuid.uuid4().hex[:6].upper()}"
        conn.execute(
            "INSERT INTO bookings VALUES (?, ?, ?, ?, ?, ?, ?)",
            (confirmation, idempotency_key, kind, summary, amount_usd, json.dumps(details),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        return {"confirmation": confirmation, "already_booked": False}


def list_bookings() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM bookings ORDER BY created_at")]


# ---------- human decisions for HumanInTheLoopMiddleware ----------

def record_feedback(action: dict, reason: str, request: dict | None = None) -> None:
    """The 'annotate' review pattern: every rejection is a labeled example of what the agent got wrong.
    With the trip request attached, evals/from_feedback.py can turn it into a regression case."""
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "decision": "reject",
             "action": action["name"], "args": action["args"], "reason": reason}
    if request:
        event["request"] = request
    with FEEDBACK_LOG.open("a") as f:
        f.write(json.dumps(event) + "\n")


def parse_decision(answer: str, action: dict) -> dict:
    """Turn a reviewer's reply into a middleware decision.

        "approve"                        -> approve as-is
        "reject: <reason>"               -> reject (reason required, logged as feedback)
        'edit: {"guests": 3}'            -> run the tool with these args changed

    Raises ValueError with a reviewer-readable message if the reply can't be used.
    """
    text = answer.strip()
    verb, _, rest = text.partition(":")
    verb, rest = verb.strip().lower(), rest.strip()

    if verb in ("approve", "yes", "y", "ok"):
        return {"type": "approve"}
    if verb == "reject":
        if not rest:
            raise ValueError("Please give a reason when rejecting, e.g. 'reject: too far from the center'.")
        record_feedback(action, rest)
        return {"type": "reject", "message": rest}
    if verb == "edit":
        try:
            changes = json.loads(rest)
        except json.JSONDecodeError:
            raise ValueError('Edits must be JSON, e.g. edit: {"guests": 3}') from None
        unknown = set(changes) - set(action["args"])
        if unknown:
            raise ValueError(f"Unknown field(s) {sorted(unknown)}. Editable: {sorted(action['args'])}")
        return {"type": "edit", "edited_action": {"name": action["name"], "args": {**action["args"], **changes}}}
    raise ValueError("Reply 'approve', 'reject: <reason>', or 'edit: {json of changed fields}'.")
