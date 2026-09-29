"""Keep test runs out of the real usage ledger, bookings database, and feedback log."""

import os
import tempfile

import pytest

os.environ["TRAVEL_USAGE_LOG"] = os.path.join(tempfile.gettempdir(), "travel_planner_test_usage.jsonl")


@pytest.fixture(autouse=True)
def isolated_bookings(tmp_path, monkeypatch):
    """Every test gets an empty bookings database and feedback log."""
    from travel_planner import bookings

    monkeypatch.setattr(bookings, "BOOKINGS_DB", tmp_path / "bookings.db")
    monkeypatch.setattr(bookings, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
