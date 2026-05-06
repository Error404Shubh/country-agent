"""
LangGraph graph definition for the Country Information Agent.

Flow:
                    ┌──────────────────┐
     user query ──► │  parse_intent    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  _route_intent   │  (conditional edge)
                    └──┬──────────┬───┘
                  ok   │          │  error
          ┌────────────┘          └────────────┐
          │                                    │
  ┌───────▼──────────┐               ┌─────────▼────────┐
  │   fetch_data     │               │    synthesize     │
  └───────┬──────────┘               │  (error message)  │
          │                          └──────────┬────────┘
  ┌───────▼──────────┐                          │
  │  _route_fetch    │  (conditional edge)       │
  └──┬──────────┬───┘                           │
 ok  │          │ error                          │
 ┌───┘          └───────────────────────┐        │
 │                                      │        │
 ▼                                      ▼        │
synthesize ◄────────────────────────────         │
     │                                           │
     └──────────────────►  END ◄─────────────────┘

Design notes:
- All error states are funnelled into `synthesize`, which checks `state.error`
  and emits a user-friendly message. This keeps the graph linear and avoids
  a proliferating error-node tree.
- The graph is compiled once at module load and reused across requests (thread-safe).
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import parse_intent, fetch_data, synthesize

# ── Conditional edge functions ────────────────────────────────────────────────


def _route_intent(state: AgentState) -> str:
    """After parse_intent: go to fetch_data or short-circuit to synthesize."""
    return "synthesize" if state.get("failed_at") == "intent" else "fetch_data"


def _route_fetch(state: AgentState) -> str:
    """After fetch_data: go to synthesize in all cases (error handled inside)."""
    return "synthesize"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("parse_intent", parse_intent)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("synthesize", synthesize)

    # Edges
    graph.set_entry_point("parse_intent")
    graph.add_conditional_edges("parse_intent", _route_intent)
    graph.add_conditional_edges("fetch_data", _route_fetch)
    graph.add_edge("synthesize", END)

    return graph


# ── Compiled singleton — reused across all requests ───────────────────────────

_compiled_graph = build_graph().compile()


async def run_agent(query: str) -> dict:
    """
    Public entry point. Runs the graph and returns the final state.
    Callers should read `state["answer"]` for the response.
    """
    initial_state: AgentState = {
        "query": query,
        "country_name": None,
        "requested_fields": None,
        "raw_data": None,
        "answer": None,
        "error": None,
        "failed_at": None,
    }
    final_state = await _compiled_graph.ainvoke(initial_state)
    return final_state
