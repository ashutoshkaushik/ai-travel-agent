# Travel Planner Agent — Phased Build Plan

A multi-agent travel planner built with LangChain + LangGraph, presented through Streamlit.
Built phase by phase so each phase teaches one agent concept and ends in something runnable.
Scope is deliberately constrained: every core agent concept is covered, nothing is built to perfection.

**One-liner (draft):** My agent helps a traveler plan a single-city trip in a Streamlit app,
replacing an evening of juggling flight, hotel, and sightseeing tabs and a budget spreadsheet.
It researches flights, hotels, and top sights, prices the trip, and swaps to a cheaper hotel
when over budget, on its own using ~6 tools; it hands off to the traveler when details are
missing, when the budget still can't be met after 2 tries, and before any booking. I'll know it
works when a traveler gets an in-budget, approved plan with sandbox bookings in under 5 minutes,
in 5 of 6 test scenarios.

---

## Architecture (router pattern)

```
                    ┌──────────────────────┐
  Traveler msg ───▶ │  Intent classifier   │  structured output
                    └──────────┬───────────┘
        ┌──────────────┬───────┴───────┬──────────────┐
        ▼              ▼               ▼              ▼
   Trip planner    Booking agent   Travel guide    Fallback
   (subgraph)      (HITL           agent (RAG)     (plain node)
                    middleware)

Trip planner subgraph:
   research_agent (create_agent) ─▶ budget_check (Python)
         ▲                                │
         └──── over budget: cheaper hotel, ≤2 tries
                                          ├── still over ─▶ interrupt: escalate to traveler
                                          ▼
                                  review_plan ─▶ interrupt: approve / modify
```

## Simplifying assumptions

| Assumption | Avoids |
|---|---|
| Destinations from a fixed list (~8 cities) pre-mapped to airports | Geocoding, IATA guessing, ambiguous places |
| Origin is a US airport code | Origin ambiguity |
| Round trip, economy, 1–4 adults, 2–14 nights, one city | Multi-city routing, cabin logic |
| One hotel per stay; rating ≥ 4.0, no dorms/hostels (filtered in code) | Bad hotel picks without LLM judgment |
| Activities = top sights only, ~2–3 per day | Interest ranking, geography optimization |
| Food = fixed $/person/day estimate per city | A restaurant data source |
| Bookings = local SQLite ledger (Duffel sandbox orders = stretch) | Passenger details, expiring offers, payments |
| Budget replanning only swaps to a cheaper hotel, max 2 tries | LLM critic, targeted rerouting |

## Stack

| Need | Provider |
|---|---|
| Flights | Duffel test mode (synthetic prices) |
| Hotels | SerpApi Google Hotels (real, read-only) |
| Top sights | SerpApi Google top sights (popularity + ticket prices) |
| Weather | Open-Meteo (no key) |
| Travel guide RAG | Local Markdown docs + in-memory vector store + OpenAI embeddings |
| LLM | `gpt-4o-mini` via `init_chat_model` (Nebius optional for the classifier) |
| Persistence | `InMemorySaver` → `SqliteSaver` |
| Bookings | SQLite ledger |
| UI | Streamlit |

---

## Phase map

| # | Phase | Concept | Status |
|---|---|---|---|
| 0 | Setup + API smoke tests, disk cache | — | ✅ |
| 1 | Tools in isolation | `@tool`, docstring-as-prompt, errors as data | ✅ |
| 2 | Hand-wired ReAct agent | State, reducer, nodes, edges, loop | ✅ |
| 3 | Ask-the-traveler interrupt | `interrupt()`, checkpointer, `thread_id`, `Command(resume)` | ✅ |
| 4 | Trip request + assumptions | `with_structured_output`, Pydantic validation | ✅ |
| 5 | Research agent with `create_agent` | `create_agent`, structured `response_format` | ✅ |
| 6 | Trip planner subgraph | Custom state, conditional edges, loop guard, escalation + review interrupts | ✅ |
| 7 | Booking agent + approval middleware | `HumanInTheLoopMiddleware` approve/edit/reject, idempotency | ✅ |
| 8 | Travel guide agent | Agentic RAG, retriever as a tool | ✅ |
| 9 | Router + parent graph + saved trips | Router pattern, subgraphs, checkpointer propagation, `SqliteSaver` | ✅ |
| 10 | Streamlit app | Graph driven from a UI, interrupts as buttons | ✅ |
| 11 | Robustness, guardrails + evaluation | Retries, fallbacks, run caps, guardrails, 3-level evals, CI | ✅ |
| 12 | Ship | README, framework table, demo video, write-up | |

Stretch (only if time allows): LLM critic, parallel specialist agents, itinerary map,
`.ics` calendar export, real Duffel sandbox orders, full staged walkthrough tab.

---

## Done: Phases 0–3

- **0** — `uv` project (Python 3.12), `.env` + `.env.example`, `config.py` (refuses non-test Duffel tokens),
  `http_cache.py` (secrets stripped from keys, only 2xx cached), API check scripts.
- **1** — `search_flights`, `search_hotels`, `search_top_sights`, `search_places_by_interest`, `geocode`,
  `get_weather`. Trimmed outputs (Duffel 806 KB → ~1.5 KB), `{"status": "ok" | "error"}` contract,
  validation before API calls. `scripts/inspect_tools.py` shows what the LLM sees.
- **2** — `build_react_agent()`: reason/action nodes, conditional + back edge, `recursion_limit`,
  streamed trace with token cost (~$0.001/run).
- **3** — `ask_traveler` tool calling `interrupt()`, `InMemorySaver`, resume with
  `Command(resume={"answer": ...})`, follow-ups on the same thread. Tests prove: interrupts are not
  swallowed by error handling, threads are isolated, a node re-runs from the top on resume.

---

## Phase 4 — Trip request + assumptions

- `SUPPORTED_CITIES` dict: city → airport code, food $/person/day
- `TripRequest` (Pydantic): origin, city (`Literal` of supported cities), depart/return dates, adults,
  budget_usd; validators enforce the assumptions (2–14 nights, 1–4 adults, future dates)
- `parse_trip_request` node: `with_structured_output(TripRequest)` on the conversation; missing or
  invalid fields → `ask_traveler` interrupt with a specific question
- `search_hotels` default: `min_rating=4.0`, exclude dorm/hostel names

**Done when:** "Tokyo in December for 2, $4k" → asks for exact dates + origin → valid `TripRequest`.

## Phase 5 — Research agent with `create_agent`

- Rebuild the Phase 3 agent with `create_agent(model, tools, system_prompt, response_format=TripPlan)`
- `TripPlan`: chosen flight, chosen hotel, 2–3 sights/day, cost breakdown
- Compare with the hand-wired version: same graph underneath (`get_graph()`)

**Done when:** a `TripRequest` in → a validated `TripPlan` out, with every price from a tool.

## Phase 6 — Trip planner subgraph

- `PlannerState`: request, plan, budget_report, attempts
- `budget_check` node (pure Python): flight + hotel + sights + food estimate vs budget
- Over budget → back to research with "find a hotel under $X/night"; `attempts` guard (max 2)
- Still over → `interrupt()`: "raise budget / fewer nights / accept anyway?"
- `review_plan` node → `interrupt()`: approve or modify (modify → back to research)

**Done when:** a $2,500 Tokyo budget triggers a hotel swap, and a $1,000 budget escalates.

## Phase 7 — Booking agent + approval middleware

- `book_flight`, `book_hotel` tools writing to a SQLite ledger, returning a confirmation id
- `create_agent(..., middleware=[HumanInTheLoopMiddleware(interrupt_on={...approve/edit/reject})])`
- Idempotency key (thread + item) so a resumed node never double-books
- Reject requires a one-line reason → appended to `feedback.jsonl` (annotate pattern; future eval data)

**Done when:** approve → confirmation id; edit → changed args used; reject → nothing written.

## Phase 8 — Travel guide agent (RAG)

- ~8 short Markdown guides in `guides/` (visa, etiquette, transit, neighborhoods for a few cities)
- Chunk by heading (`MarkdownHeaderTextSplitter`), heading path + `city` as metadata
- `InMemoryVectorStore` + `OpenAIEmbeddings`; `search_travel_guide` tool with a `city` metadata filter
- Confidence threshold: best score below cutoff → "not in my guides" + offer escalation
- `create_agent` that answers only from retrieved text, cites chunk ids, refuses when not found
- Optional: hybrid BM25 + dense via `EnsembleRetriever`

**Done when:** "Do I need a visa for Japan?" answers from the guide; unknown topics say so.

## Phase 9 — Router + parent graph + saved trips

- Intent classifier (`plan_trip` / `book` / `travel_question` / `other`) with `Literal` structured output
- Parent `StateGraph`: classifier → planner subgraph / booking agent / guide agent / fallback
- One checkpointer at the parent (propagates to sub-agents; interrupts bubble up)
- Swap `InMemorySaver` → `SqliteSaver`: a trip survives restarting the app
- Long-term memory: traveler profile (home airport, hotel prefs) in a LangGraph `Store` —
  read on entry, written back after an approved trip

**Done when:** one thread goes plan → approve → book → visa question, across a restart.

## Phase 10 — Streamlit app

- **Hero concept diagram (landing page, one image):** the progression
  plain LLM → RAG → tool calling → single agent → multi-agent system (this app), with this travel
  planner highlighted as the last step. Each stage: what it adds, what it can't do (the reason for
  the next stage). So a viewer grasps the whole
  app's intent at a glance before using it.

- Graph + checkpointer in `st.cache_resource`; `thread_id` in `st.session_state`
- Chat + live trace panel (node, tool, args, result)
- `__interrupt__` → question box / Approve-Edit-Reject buttons → `Command(resume=...)`
- Approval card shows what a reviewer needs: task, recommended action, reasoning, evidence (tool results), cost
- Plan cards with budget bar; "How it's built" tab: graph image + one paragraph per concept

## Phase 11 — Robustness, guardrails + evaluation ✅

- **Robustness:** timeouts, retries with exponential backoff + jitter on 429/5xx, a circuit breaker per
  provider (stale cache while open), errors as data (never raises); primary model → fallback model after
  2 failures (which model answered is in the trace); per-run token/dollar cap and per-session budget
  that stop cleanly with a message. "Break it" mode simulates each failure in the app.
- **Guardrails (Agent Lab 10):** tool results fenced as untrusted data, instruction-like fields stripped;
  input guard before the classifier (abuse, injection, off-topic, booking outside the approved plan);
  output guard (PII the user never gave, prices with no evidence).
- **Evals (Agent Lab 9):** 26 cases in `evals/dataset.jsonl` across planner, budget, intake, guide (RAG),
  booking, guardrails (3 injection cases) and router. Scored at three levels: final result (code),
  steps (code), quality (LLM judge, fixed rubric, pass rule in code). Recorded API + LLM responses
  and a pinned date make reruns deterministic and free; `--offline` never calls a paid API.
  `evals/from_feedback.py` turns booking rejections into cases. CI: `.github/workflows/evals.yml`.
- **Replays:** the four example runs are recorded (`scripts/record_replays.py`) for a keyless,
  cost-free "Replay a recorded run" mode and the Start page's "Watch it work".

## Phase 12 — Ship

- README with architecture diagram, setup, screenshots; handout framework table
- Demo video (≤ 5 min); write-up: prompts used while vibe-coding, iterations, learnings
- Optional: Nebius for the intent classifier; Streamlit Community Cloud with bring-your-own-key

---

## Concept coverage

Legend: ✅ built · 📅 planned · ➕ small addition · 📝 write-up only

**Additions agreed in review** (folded into existing phases):
1. ➕ Long-term memory — traveler profile (home airport, hotel prefs) in a LangGraph `Store`:
   retrieve on entry, write back after an approved trip (Phase 9). Covers working / session / long-term memory.
2. ➕ Confidence threshold in the RAG agent — low retrieval score → "not in my guides" + offer escalation (Phase 8).
3. ➕ Annotate on reject — capture a one-line reason when a booking is rejected; append to a feedback log (Phase 7).
4. ➕ RAG upgrades — heading-based chunking, `city` metadata filter, cited chunk ids, refusal clause (Phase 8);
   8–10 question golden set with recall@k (Phase 11).

| Concept | Status | Where |
|---|---|---|
| Agent loop + stop conditions | ✅ | Phase 2–3, `recursion_limit` |
| Deterministic + agentic mix | 📅 | Phase 6 (budget check in code, research by agent) |
| Decision tree: prompt → workflow → RAG → tool → agent | 📝 | Each component uses the lightest option |
| Autonomy level 3 ("act, ask on risk") | 📝 | Reads autonomous, writes approval-gated |
| Tools: read vs write, small tool sets | ✅📅 | 6 read tools, 2 write tools (7), per-agent subsets (5) |
| Memory: working / session / long-term | ✅✅➕ | messages / checkpointer / profile store (9) |
| Plan → execute → replan | 📅 | Phase 6 budget replan |
| Continue / retry / stop / escalate | 📅 | Phases 6, 11 |
| HITL: approve / edit / reject / retry / escalate / annotate | 📅➕ | Phases 6, 7 |
| Confidence thresholds | ➕ | Phase 8 |
| Review queue contents (task, action, reasoning, evidence, confidence) | 📅 | Streamlit approval card (10) |
| Router + subagents; single → multi when limits are named | ✅📅 | Phases 2–3 showed limits; router in 9 |
| Supervisor, planner-executor, reflection, handoffs, skills | 📝 | Explained; budget check = code-based reflection |
| 5 failure modes (loops, hallucinated calls, token waste, bad decisions, wrong tools) | ✅📅 | recursion limit, Literal + arg validation, trimming + cache, code constraints, tool subsets |
| ADLC, MINT ("earn each layer") | 📝 | The phase order is the MINT ladder |
| MCP, A2A, Deep Agents | 📝 | Out of scope; explained in the write-up |
| Cost & latency: small models, caching, step caps, parallel calls, streaming | ✅📅 | gpt-4o-mini, disk cache, caps, Streamlit streaming |
| RAG: embeddings, vector store, chunking, metadata filter, citations, refusal | 📅➕ | Phase 8 |
| Hybrid BM25 + rerank | 📝 | Optional hybrid; rerank skipped (tiny corpus) |
| RAG evals (golden set, recall@k) | ➕ | Phase 11 |
| Context engineering (trim, order, compress) | ✅ | Tool-output trimming |

---

## Guardrails (the "never do" list)

- Never book without an explicit approve decision
- Never use a non-test Duffel token (code refuses non `duffel_test_` tokens)
- Never exceed the budget without asking the traveler
- Never retry the budget loop more than 2 times without escalating
- Never send raw API payloads to the LLM (trim first — cost and accuracy)
