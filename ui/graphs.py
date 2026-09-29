"""Draw the real, compiled LangGraph graphs as Graphviz diagrams, coloured by who acts at each node.

The diagrams are generated from the code (graph.get_graph()), so they can't drift from what runs.
"""

from ui.theme import tokens

# Which actor runs each node: the LLM (agent), plain code, or a human (nodes that interrupt).
ACTORS = {
    "agent": {"classify_intent", "extract", "research", "model", "reason_node", "guide", "booking", "intake", "planner"},
    "human": {"ask", "review", "escalate"},
}


def actor_of(node: str) -> str:
    for actor, nodes in ACTORS.items():
        if node in nodes:
            return actor
    return "code"


def _hex(color: str) -> str:
    """'rgba(42,120,214,.10)' -> '#2a78d61a' (Graphviz understands #RRGGBBAA, not CSS rgba)."""
    if not color.startswith("rgba"):
        return color
    r, g, b, a = [x.strip() for x in color[5:-1].split(",")]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}{round(float(a) * 255):02x}"


def graph_dot(compiled, direction: str = "TB") -> str:
    t = {k: _hex(v) for k, v in tokens().items()}
    g = compiled.get_graph()
    fill = {"agent": t["agent-soft"], "code": t["code-soft"], "human": t["human-soft"]}
    edge_color = t["muted"]
    lines = [f'digraph {{ rankdir={direction}; bgcolor="transparent"; nodesep=0.35; ranksep=0.4;',
             f'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=1.6, color="{edge_color}"];',
             f'edge [color="{edge_color}", fontname="Helvetica", fontsize=9, arrowsize=0.7];']
    for node_id in g.nodes:
        if node_id in ("__start__", "__end__"):
            label = "START" if node_id == "__start__" else "END"
            lines.append(f'"{node_id}" [label="{label}", shape=oval, style="filled", fillcolor="{t["line"]}", fontcolor="{edge_color}"];')
            continue
        actor = actor_of(node_id)
        lines.append(f'"{node_id}" [label="{node_id}", fillcolor="{fill[actor]}", color="{t[actor]}", fontcolor="{t[actor]}"];')
    for e in g.edges:
        style = ' [style=dashed]' if e.conditional else ""
        lines.append(f'"{e.source}" -> "{e.target}"{style};')
    lines.append("}")
    return "\n".join(lines)


def legend_html() -> str:
    return ("<span class='badge agent'>agent: the LLM decides</span>"
            "<span class='badge code'>code: guarantees</span>"
            "<span class='badge human'>human: you decide</span>"
            " <span class='muted' style='font-size:.8rem'>dashed = conditional edge</span>")
