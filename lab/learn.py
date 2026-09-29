"""'Learn from this step' content for each Agent Lab page.

Every insight comes from building and running this project (terminal runs, tests, real failures),
so it stays true to this app. Questions render as expanders so a reader can think first.
"""

import streamlit as st

LEARN: dict[str, dict] = {
    "tools": {
        "goals": ["What the LLM actually receives for a tool: a name, a description and a JSON schema", "Why tools trim, validate and cache before an LLM sees anything", "How errors come back as data the agent can recover from"],
        "insights": [
            "**A tool is a function plus a description the LLM reads.** The docstring becomes the argument "
            "descriptions sent to the model on every call; `Literal` types become an enum it can't go outside.",
            "**Trim before the LLM sees it.** One Duffel flight search returned **806 KB** (about 200,000 tokens). "
            "The trimmed result is about 1.5 KB. Without trimming, one search would cost more than a whole trip plan.",
            "**Return errors as data, never raise.** `{\"status\": \"error\", \"reason\": …, \"hint\": …}` lets the agent "
            "correct itself (\"Tokyo\" → \"NRT\") instead of crashing the loop.",
            "**Don't trust a provider's filter blindly.** SerpApi's `max_price` returned zero hotels for Tokyo, so the "
            "tool filters prices itself. Bonus: one cached search then serves every budget the planner tries.",
            "**Google's top-sights panel is inconsistent:** \"Tokyo\" has one but \"Tokyo, Japan\" doesn't; \"Rome\" "
            "doesn't but \"Rome, Italy\" does. The tool tries both, and both are cached.",
        ],
        "qa": [
            ("Why does every tool call cost tokens even when the agent doesn't use that tool?",
             "All bound tool schemas are sent with every LLM call (about 1,300 tokens for six tools here). That's one "
             "reason each specialist agent later gets only the tools it needs.", "Duffel returned 806 KB"),
            ("Why cache API responses on disk?",
             "Agents call the same searches repeatedly while you develop and replan. The cache made repeat runs free, "
             "kept SerpApi under its 250-searches/month free plan, and makes a live demo immune to rate limits."),
        ],
    },
    "loop": {
        "goals": ["The ReAct loop: reason, act, observe, repeat", "How a conditional back edge lets the model take as many steps as it needs", "Why tool errors must not crash the loop"],
        "insights": [
            "**The LLM never executes anything.** It emits a structured request; `action_node` (plain Python) runs the "
            "tool and returns the result as a `ToolMessage`.",
            "**The back edge is the whole idea.** `action → reason` lets the model take as many steps as the task needs. "
            "For a Tokyo trip it called flights, hotels and sights **in one turn** (parallel tool calls), then answered.",
            "**Dependent steps need extra laps.** \"Find the hotel, then sights near it\" can't be one turn: the second "
            "call needs the first call's result.",
            "**Reducers decide how updates merge.** `add_messages` appends; without it every node would overwrite the "
            "conversation.",
            "**Cost per run: about $0.001** with gpt-4o-mini, visible in every trace.",
        ],
        "qa": [
            ("The agent asked good clarifying questions for \"Plan me a trip to Japan\", then stopped. What was lost?",
             "Everything: without a checkpointer the run ended at END and nothing was saved. Answering would start a "
             "brand-new run that never saw \"Japan\". That's what interrupts + checkpointers fix."),
            ("What caps a runaway loop?",
             "`recursion_limit` in the run config: a hard ceiling on node executions, so a model that never stops "
             "can't burn the budget.", "Unclear reply looped"),
        ],
    },
    "hitl": {
        "goals": ["How interrupt() pauses a graph and Command(resume=...) continues it", "Why a checkpointer is required to pause at all", "Why a human loop needs a stop condition"],
        "insights": [
            "**Four human checkpoints, four mechanisms.** Clarify (interrupt inside a node, code decides when), "
            "review (interrupt + options), escalate (interrupt with choices), booking (HumanInTheLoopMiddleware).",
            "**The autonomy level is 3: act, ask on risk.** Reads (searches) are autonomous; writes (bookings) and "
            "anything over budget need a human.",
            "**A paused node re-runs from the top on resume.** A test proves a tool called before the interrupt runs "
            "twice, which is why every write is idempotent.",
            "**`interrupt()` is an exception.** A catch-all `except Exception` silently swallowed the pause until it "
            "was re-raised explicitly. A test guards it.",
            "**Human loops need stop conditions too.** An unclear reply once looped 4,063 times (no LLM cost, pure "
            "code). Now two unclear replies cancel.",
        ],
        "qa": [
            ("Why did the runner stop resuming with an empty answer?",
             "Never invent a human's reply. With no answer available, the run stays paused: its state is checkpointed "
             "and can be answered later, even after a restart.", "Runner resumed with an empty answer"),
            ("Where should checkpoints go?",
             "Before external or irreversible actions (booking), before customer-facing output (the plan), and when "
             "confidence is low (the budget can't be met, the guides don't cover it)."),
        ],
    },
    "structured": {
        "goals": ["Turning free text into a validated Pydantic object", "Letting code, not the LLM, decide what is missing", "Asking one clarifying question instead of guessing"],
        "insights": [
            "**The LLM extracts, the code validates.** `TripRequestDraft` lets the model say \"unknown\" (null); "
            "`TripRequest` can only exist if every rule holds.",
            "**It left dates empty for \"in December\"** instead of guessing, exactly as the field description asked.",
            "**All assumptions live in code:** 8 cities, 24 US airports, 2-14 nights, 1-4 adults. Kyoto has no airport, "
            "so it maps to Osaka Kansai (KIX).",
            "**The clarifying question is written by code, not the LLM,** so it lists exactly what the validator found.",
        ],
        "qa": [
            ("Why two models instead of one?",
             "If the extraction model enforced the rules, the LLM would have to produce valid values or fail, which "
             "pushes it to guess. The permissive draft plus strict validation separates \"what was said\" from \"what "
             "is allowed\"."),
            ("Why did the token counter need a callback?",
             "Structured-output calls never become messages in state, so counting messages missed them. A LangChain "
             "callback sees every model call."),
        ],
    },
    "planner": {
        "goals": ["A workflow graph: research, budget check, replan, escalate, review", "Why costs are added up in code, never by the model", "How evidence checks catch invented prices and sights"],
        "insights": [
            "**Seven of eight nodes are plain code.** Only research is an agent; budget math, retries and when to ask "
            "you are all code: \"LLM decides, code guarantees\".",
            "**The replan is computed, not guessed.** Over by $X across N nights → the hotel must be at most "
            "(current rate − X/N) per night. The agent gets a checkable number.",
            "**Constraints must add up, never replace.** Three real bugs were one lesson: a retry dropped the hotel "
            "cap; a budget replan dropped \"rated 4.5 or higher\"; a replan added four sights and ate the savings.",
            "**Code carries over what shouldn't change.** Replans zeroed sight prices twice and once paired Iberia's "
            "price with Duffel's offer id (the ids differed by one character). Now code keeps the flight and sights.",
            "**Every choice is checked against the evidence.** An invented sight (\"Free Day\") and a mismatched price "
            "were both caught before a human ever saw them.",
        ],
        "qa": [
            ("Why escalate immediately when the flight alone exceeds the budget?",
             "No cheaper hotel can fix that, so retrying would waste calls. Code computes whether a hotel cap is even "
             "possible before replanning."),
            ("Why did one agent choose a campsite?",
             "\"Camping Village Fabulous\" was under the price cap and rated 4.1. The quality filter only knew about "
             "hostels and capsules; \"camping\" was added.", "A campsite chosen"),
        ],
    },
    "approval": {
        "goals": ["HumanInTheLoopMiddleware: approve, edit or reject before a tool runs", "Idempotent bookings, so a retry never books twice", "Why every rejection is saved as feedback"],
        "insights": [
            "**The gate is configuration, not code paths.** `HumanInTheLoopMiddleware(interrupt_on=…)` pauses before "
            "each write tool, so no code path can forget it.",
            "**Approve, edit, or reject, per booking.** Edits must name real fields; rejections must give a reason.",
            "**Every rejection is a labeled example.** Reasons go to `feedback.jsonl` (the \"annotate\" pattern) and "
            "become evaluation data later.",
            "**Idempotency beats trust.** Booking the same offer again returned the original confirmation "
            "(`already_booked: true`): one flight in the ledger, not two.",
        ],
        "qa": [
            ("Why pause before the tool instead of confirming afterwards?",
             "After is too late for a write: the booking already happened. The middleware pauses between the model's "
             "request and the tool's execution."),
            ("What does the reviewer need to see?",
             "The action, the money, and what it means (\"Sandbox booking, no real charge\"). The description function "
             "builds that card from the tool call's arguments."),
        ],
    },
    "rag": {
        "goals": ["Agentic RAG: the agent decides when to search the guides", "Confidence bands: answer, answer carefully, or say it isn't covered", "Citations that code verifies were actually retrieved"],
        "insights": [
            "**40 chunks from 8 guides, split by heading,** with \"Tokyo > Visa and entry\" prepended so each vector "
            "carries its context.",
            "**Filter by city before ranking.** Paris text never competes with a Tokyo question.",
            "**One threshold can't separate covered from uncovered.** \"Can I chew gum?\" scored 0.35 and is covered; "
            "a Shinkansen fare scored 0.40 and isn't. So: proceed ≥ 0.45, confirm 0.30-0.45, stop < 0.30.",
            "**Fix the documents, not the model.** \"Where should I stay\" missed until the headings said "
            "\"Neighborhoods and where to stay\".",
            "**Citations are verified in code,** and the entry-rules note is added by code (the model skipped it).",
        ],
        "qa": [
            ("Why not just let the LLM answer travel questions from memory?",
             "Entry rules change and the model can't cite anything. The guides are a controlled source, and the agent "
             "refuses when they don't cover the question."),
            ("What happens in the confirm band?",
             "The passage is returned with a note: answer only if it directly answers the question. That's how the gum "
             "question was answered and the Shinkansen fare refused.", "One threshold couldn't separate"),
        ],
    },
    "router": {
        "goals": ["The router pattern: one cheap classifier, specialist sub-agents", "Preconditions in code (no booking without an approved plan)", "Working, session and long-term memory in one graph"],
        "insights": [
            "**One cheap classifier call routes; specialists do the work.** Four intents: plan_trip, book, "
            "travel_question, other.",
            "**Preconditions are code.** \"Book it\" with no approved plan can't reach the booking tools, whatever the "
            "classifier says.",
            "**One checkpointer at the top.** Sub-agents inherit it, so a pause deep inside the planner survives a "
            "restart and resumes in a new process.",
            "**All three memory types:** working (this turn), session (the checkpointed thread), long-term (the "
            "traveler profile, read on entry and written only after approval).",
            "**A returning traveler didn't repeat themselves:** \"Now plan London…\" got SFO and 2 adults from the "
            "saved profile.",
        ],
        "qa": [
            ("Why did the checkpointer and the store need separate SQLite connections?",
             "Sharing one made their transactions collide (\"cannot start a transaction within a transaction\"). A "
             "small experiment caught it before the real build.", "Checkpointer and store shared"),
            ("What's a known limitation of the classifier?",
             "It sees only the latest message: \"what's the capital of France?\" went to the guide (which politely "
             "refused), and \"go there\" meant the saved trip's city. Passing recent context would fix both."),
        ],
    },
    "evals": {
        "goals": ["Scoring an agent at three levels: result, steps, quality", "Record once, replay forever: deterministic, free reruns", "Turning human feedback into regression cases"],
        "insights": [
            "**Score three things separately.** The final result (right answer?), the steps (right way?), and the "
            "quality (would a traveler be happy?). A plan can be correct and still take a path you never want.",
            "**Code checks first, the LLM judge last.** Budget, prices, sights and citations are facts: code checks "
            "them for free. The judge only scores what code can't, with a fixed rubric.",
            "**Record once, replay forever.** API and LLM responses are cached by the exact prompt, and today's date "
            "is pinned. A rerun costs \\$0 and gives the same answer, so a failure means the code changed.",
            "**Replay has to be exact.** Two things broke it: random message ids, and usage metadata that differs "
            "between a fresh call and a cache hit. Both are stripped from the cache key.",
            "**Every human \"no\" is a test.** A rejected booking becomes a case that replays the trip and asserts "
            "the plan avoids what was rejected.",
        ],
        "qa": [
            ("Why doesn't the judge decide pass or fail?",
             "In testing it gave identical scores (5/3/4/5) a pass on one case and a fail on another. Now it only "
             "scores, and code applies the pass rule: every score at least 3, average at least 4.", "The judge passed and failed"),
            ("What does CI catch without any API keys?",
             "Any change that alters a prompt, a tool call or the graph path: the replay misses (a clear failure) "
             "or a code check fails. To accept an intended change, re-record locally and commit evals/cache.", "Tool tests passed locally only"),
            ("Why does a planner case stop at the review pause?",
             "Everything worth checking (the plan, its prices, the budget) is in the review payload, and stopping "
             "at the human gate keeps each case cheap and free of side effects."),
        ],
    },
    "guardrails": {
        "goals": ["Treating tool output as untrusted input (prompt injection)", "Input and output guards that run in code, before and after the LLM", "Hard limits on tokens and spend per run and per session"],
        "insights": [
            "**Tool output is untrusted input.** A hotel description is written by someone else. The agent fences it "
            "as data, and code strips instruction-like text before the model reads it.",
            "**The prompt rule and the code rule back each other up.** The prompt says tool content is data; the "
            "code makes sure the worst of it never arrives.",
            "**Block early, in code.** The input guard runs before the classifier, so a jailbreak costs zero tokens "
            "and never reaches an agent with tools.",
            "**Check the reply, not just the request.** The output guard removes personal data the user never gave "
            "and any price no search result or calculation backs.",
            "**Limits are guardrails too.** A per-run token cap and a per-session budget stop runaway cost cleanly, "
            "with a message instead of a crash.",
        ],
        "qa": [
            ("If the model ignored the fence and picked the \\$5,340 hotel, what would happen?",
             "The planner's budget check is code, so the plan would fail verification and replan or escalate. The "
             "guardrail lowers the odds; the code check guarantees the outcome.", "A hotel description said"),
            ("Why is 'book' on the instruction-like list for tool fields but allowed in user messages?",
             "A traveler saying \"book it\" is the normal booking flow. A hotel description saying \"book\" is "
             "never legitimate: data doesn't give orders."),
            ("What does a regex guard miss?",
             "Paraphrases and other languages. That's why it is one layer of several: the fence, the prompt rule, "
             "the code budget check and human approval before any booking."),
        ],
    },
}


def learn_section(key: str) -> None:
    content = LEARN[key]
    st.markdown("### Learn from this step")
    for insight in content["insights"]:
        st.markdown(f"- {insight}")
    st.markdown("<div class='section-label'>Check your understanding</div>", unsafe_allow_html=True)
    from lab.overview import lesson_link

    for question, answer, *lesson in content["qa"]:
        with st.expander(question):
            st.markdown(answer)
            link = lesson_link(lesson[0]) if lesson else None
            if link:
                st.markdown(f"[See it in What broke →]({link})")
