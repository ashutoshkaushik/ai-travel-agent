"""Print exactly what the LLM receives for each tool: name, description, and argument schema.

This is the "docstring is the prompt" idea made visible. Everything printed here is sent to
the model on every call; nothing else about the Python function is.

    uv run python scripts/inspect_tools.py
"""

import json

from langchain_core.utils.function_calling import convert_to_openai_tool

from travel_planner.tools import ALL_TOOLS

total_chars = 0
for t in ALL_TOOLS:
    schema = convert_to_openai_tool(t)
    text = json.dumps(schema, indent=2)
    total_chars += len(json.dumps(schema))
    print(f"\n{'=' * 80}\n{t.name}\n{'=' * 80}\n{text}")

print(f"\nAll {len(ALL_TOOLS)} tool schemas: ~{total_chars // 4} tokens sent with every LLM call.")
