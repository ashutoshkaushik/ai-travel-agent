"""Shared helpers so every tool returns the same shape and never raises into the agent.

Every tool returns either:
    {"status": "ok", ...trimmed results...}
    {"status": "error", "reason": "...", "hint": "..."}

Why errors as data: if a tool raises, the agent loop crashes. If it returns an error dict,
the LLM reads it like any other observation and can decide what to do next
(fix its arguments, widen the search, or ask the traveler).
"""

import functools
from datetime import date

import httpx
from travel_planner.config import today


class ToolInputError(ValueError):
    """Bad arguments from the LLM. The message is written for the LLM to read and correct."""

    def __init__(self, reason: str, hint: str | None = None):
        super().__init__(reason)
        self.hint = hint


def ok(**data) -> dict:
    return {"status": "ok", **data}


def error(reason: str, hint: str | None = None) -> dict:
    result = {"status": "error", "reason": reason}
    if hint:
        result["hint"] = hint
    return result


def returns_errors_as_data(fn):
    """Convert exceptions inside a tool into error dicts the LLM can reason about."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolInputError as e:
            return error(str(e), e.hint)
        except httpx.TimeoutException:
            return error("The travel data provider timed out.", "Retry once; if it fails again, tell the traveler.")
        except httpx.HTTPError as e:
            return error(f"Network error talking to the provider: {e}", "Retry once; if it fails again, tell the traveler.")
        except (KeyError, IndexError, TypeError) as e:
            # The provider's response had an unexpected shape
            return error(f"Unexpected response from the provider ({type(e).__name__}: {e}).")

    return wrapper


def parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ToolInputError(f"{field}={value!r} is not a valid date.", "Use YYYY-MM-DD format.") from None


def parse_future_date(value: str, field: str) -> date:
    parsed = parse_date(value, field)
    if parsed <= today():
        raise ToolInputError(
            f"{field}={value} is not in the future (today is {today()}).",
            "Travel dates must be after today. Ask the traveler if the year is unclear.",
        )
    return parsed
