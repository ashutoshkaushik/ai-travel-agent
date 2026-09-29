"""Evals: a fixed dataset of cases, scored at three levels (final result, steps, quality).

    uv run python -m evals.run                      # all cases, with the LLM judge
    uv run python -m evals.run --category planner   # one category
    uv run python -m evals.run --offline --no-judge # CI: recorded responses only, code checks only
"""
