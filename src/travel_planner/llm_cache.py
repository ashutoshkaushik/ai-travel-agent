"""A disk cache for LLM responses, used by evals so reruns are deterministic and nearly free.

LangChain checks the global LLM cache before every chat-model call: same prompt + same model
settings -> the stored response, no API call. Combined with TRAVEL_FROZEN_CACHE=1 (cached tool
responses regardless of age), an eval run replays exactly. Only cache misses cost money, and
update() (called only on a miss) records that spend, so each run can report its real cost.
"""

import hashlib
import json
import os
from pathlib import Path

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads

from travel_planner.config import CACHE_DIR
from travel_planner.trace import PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT


class RecordingMissing(RuntimeError):
    """Offline (CI) run asked for an LLM response nobody recorded."""


VOLATILE = {"id", "usage_metadata", "response_metadata"}  # never sent to the model as content


def _strip_ids(node):
    """Drop what changes between identical conversations: LangGraph's random message ids, and usage
    metadata (a cache hit reports total_cost 0, a fresh call doesn't), so the same conversation hashes the same."""
    if isinstance(node, dict):
        kwargs = node.get("kwargs")
        if isinstance(kwargs, dict):
            node = {**node, "kwargs": {k: v for k, v in kwargs.items() if k not in VOLATILE}}
        return {k: _strip_ids(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_ids(v) for v in node]
    return node


def stable_key(prompt: str, llm_string: str) -> str:
    try:
        prompt = json.dumps(_strip_ids(json.loads(prompt)), sort_keys=True)
    except ValueError:
        pass
    return hashlib.sha256((llm_string + "|" + prompt).encode()).hexdigest()[:40]


class DiskLLMCache(BaseCache):
    def __init__(self, folder: Path = CACHE_DIR / "llm", offline: bool = False):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.offline = offline  # CI: a miss is an error ("record it locally"), never a paid call
        self.hits = 0
        self.misses = 0
        self.new_cost_usd = 0.0

    def _path(self, prompt: str, llm_string: str) -> Path:
        return self.folder / f"{stable_key(prompt, llm_string)}.json"

    def _debug(self, kind: str, prompt: str, llm_string: str) -> None:
        folder = os.environ.get("TRAVEL_LLM_CACHE_DEBUG")  # a folder: dump prompts to diff a replay miss
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            (Path(folder) / f"{kind}-{stable_key(prompt, llm_string)}.json").write_text(
                json.dumps({"llm_string": llm_string, "prompt": _strip_ids(json.loads(prompt))}, indent=1, sort_keys=True))

    def lookup(self, prompt: str, llm_string: str):
        path = self._path(prompt, llm_string)
        if not path.exists():
            self.misses += 1
            self._debug("miss", prompt, llm_string)
            if self.offline:
                raise RecordingMissing("No recorded LLM response for this prompt. Run the evals locally with an "
                                       "OpenAI key to record it, then commit evals/cache.")
            return None
        self._debug("hit", prompt, llm_string)
        self.hits += 1
        return loads(path.read_text())

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        self._debug("stored", prompt, llm_string)
        self._path(prompt, llm_string).write_text(dumps(return_val))
        for gen in return_val:
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
            self.new_cost_usd += (usage.get("input_tokens", 0) * PRICE_PER_M_INPUT
                                  + usage.get("output_tokens", 0) * PRICE_PER_M_OUTPUT) / 1_000_000

    def clear(self, **kwargs) -> None:
        for p in self.folder.glob("*.json"):
            p.unlink()

    def stats(self) -> dict:
        return {"llm_cache_hits": self.hits, "llm_cache_misses": self.misses, "new_cost_usd": round(self.new_cost_usd, 5)}


def roundtrip_ok() -> bool:
    """Sanity check used by tests: a stored generation loads back identically."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration

    gen = [ChatGeneration(message=AIMessage("hi", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}))]
    return json.loads(dumps(loads(dumps(gen)))) == json.loads(dumps(gen))
