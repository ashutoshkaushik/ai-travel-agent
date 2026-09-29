"""Run one eval case through the real assistant graph and record what happened.

The record is plain data (path, intent, pauses, tool calls, evidence, replies), so every scorer is a
small pure function over it. The graph is the production one: same prompts, tools and guardrails.
Determinism comes from the environment evals/run.py sets up (frozen API cache, LLM cache, pinned date).
"""

import json
import time
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from travel_planner.guardrails import unfence

PAUSE_NODE = {"review": "review", "escalation": "escalate", "question": "ask", "approval": "approve_booking"}
MAX_RESUMES = 4


class EvalRecorder(BaseCallbackHandler):
    """Counts LLM calls and captures every tool call with its (parsed) result."""

    def __init__(self):
        self.llm_calls = 0
        self.tool_calls: list[str] = []
        self.evidence: dict[str, list] = defaultdict(list)
        self._names: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.llm_calls += 1

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._names[run_id] = name
        self.tool_calls.append(name)

    def on_tool_end(self, output, *, run_id, **kwargs):
        name = self._names.get(run_id, "tool")
        content = getattr(output, "content", output)
        if isinstance(content, str):
            try:
                content = json.loads(unfence(content))
            except ValueError:
                return
        if isinstance(content, dict):
            self.evidence[name].append(content)


def pause_kind(payload: dict) -> str:
    if "action_requests" in payload:
        return "approval"
    return payload.get("kind") or "question"


def run_case(graph, case: dict) -> dict:
    config = {"configurable": {"thread_id": f"eval-{case['id']}-{time.time_ns()}", "user_id": "eval"}}
    recorder = EvalRecorder()
    run_config = {**config, "callbacks": [recorder]}
    if case.get("seed_state"):
        graph.update_state(config, case["seed_state"], as_node="save_profile")

    path, steps, pauses, intent, error = [], [], [], None, None  # path: flat node names (matched); steps: for reading
    answers = list(case.get("answers", []))
    started = time.perf_counter()
    try:
        for text in case["turns"]:
            agent_input = {"messages": [HumanMessage(text)]}
            for _ in range(MAX_RESUMES + 1):
                paused = None
                for namespace, chunk in graph.stream(agent_input, run_config, stream_mode="updates", subgraphs=True):
                    names = [ns.split(":")[0] for ns in namespace]
                    for node, update in chunk.items():
                        if node == "__interrupt__":
                            paused = update[0].value
                            continue
                        for name in names + [node]:
                            if not path or path[-1] != name:
                                path.append(name)
                        label = " › ".join(names + [node])
                        if not steps or steps[-1] != label:
                            steps.append(label)
                        if node == "classify_intent" and not namespace:
                            intent = (update or {}).get("intent")
                if paused is None:
                    break
                kind = pause_kind(paused)
                path.append(PAUSE_NODE[kind])
                steps.append(f"⏸ {PAUSE_NODE[kind]}")
                pauses.append({"kind": kind, **{k: paused.get(k) for k in ("question", "plan", "costs", "request")
                                                if paused.get(k) is not None}})
                if kind in ("question", "escalation") and answers:
                    agent_input = Command(resume={"answer": answers.pop(0)})
                else:
                    break  # review / booking approval: the eval stops at the human gate
    except Exception as e:  # a crashed case is a failed case, not a crashed run
        error = f"{type(e).__name__}: {e}"

    values = graph.get_state(config).values if error is None else {}
    replies = [m.content for m in values.get("messages", []) if isinstance(m, AIMessage) and m.content]
    return {
        "id": case["id"], "category": case["category"], "turns": case["turns"], "intent": intent, "path": path, "steps": steps,
        "pauses": pauses, "replies": replies, "questions": [p["question"] for p in pauses if p.get("question")],
        "tool_calls": recorder.tool_calls, "llm_calls": recorder.llm_calls, "evidence": dict(recorder.evidence),
        "trip_status": values.get("trip_status"), "error": error,
        "seconds": round(time.perf_counter() - started, 2),
    }


def build_graph():
    from travel_planner.agents.assistant import build_assistant

    return build_assistant(InMemorySaver(), InMemoryStore())
