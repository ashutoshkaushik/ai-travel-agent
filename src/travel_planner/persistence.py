"""Durable state: conversations (checkpointer) and long-term memory (store), both in SQLite.

- Checkpointer: short-term/session memory. Every step of every thread, so a paused trip survives
  closing the app. Swapping InMemorySaver for SqliteSaver is the only change the graphs need.
- Store: long-term memory across threads, e.g. a traveler's profile.

Each gets its OWN connection: sharing one made their transactions collide
("cannot start a transaction within a transaction").
"""

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from travel_planner.config import PROJECT_ROOT

CHECKPOINTS_DB = Path(os.environ.get("TRAVEL_CHECKPOINTS_DB", PROJECT_ROOT / "checkpoints.db"))
MEMORY_DB = Path(os.environ.get("TRAVEL_MEMORY_DB", PROJECT_ROOT / "memory.db"))


def open_checkpointer(path: Path = CHECKPOINTS_DB) -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def open_store(path: Path = MEMORY_DB) -> SqliteStore:
    store = SqliteStore(sqlite3.connect(path, check_same_thread=False, isolation_level=None))
    store.setup()
    return store
