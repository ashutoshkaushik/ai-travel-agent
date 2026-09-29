"""The ask-a-human tool.

The interrupt is exposed AS A TOOL rather than hard-coded at a fixed point in the graph,
so the LLM decides when it is stuck. We don't have to predict every situation that needs
clarification: missing dates, ambiguous cities, conflicting constraints.
"""

from langchain.tools import tool
from langgraph.types import interrupt


@tool(parse_docstring=True)
def ask_traveler(question: str) -> str:
    """Ask the traveler a question and wait for their answer.

    Use this whenever information you need is missing or ambiguous (departure city, travel dates,
    number of travelers, budget, or which of several places they mean). Ask instead of guessing.
    Combine everything you need into one clear question.

    Args:
        question: The question to show the traveler.
    """
    # Execution STOPS on this line. The graph's state is saved by the checkpointer, and the
    # caller receives {"question": ...} under the "__interrupt__" key.
    # When the caller resumes with Command(resume={"answer": "..."}), this node re-runs from
    # the top, interrupt() returns that dict instead of pausing, and the tool returns the answer.
    reply = interrupt({"question": question})
    return reply["answer"]  # contract with the resume side: it must send a dict with "answer"
