"""
AgentState is the single source of truth that flows through every node in the graph.
Each node reads from and writes to this shared state — no side channels.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                              # Raw user question, never mutated

    # ── Intent parsing output ─────────────────────────────────────────────────
    country_name: Optional[str]             # Normalised country name, e.g. "Germany"
    requested_fields: Optional[list[str]]   # e.g. ["population", "capital"]

    # ── API fetch output ──────────────────────────────────────────────────────
    raw_data: Optional[dict[str, Any]]      # Subset of REST Countries payload

    # ── Final answer ──────────────────────────────────────────────────────────
    answer: Optional[str]                   # Natural-language response to user
    error: Optional[str]                    # Human-readable error, if any

    # ── Routing helpers ───────────────────────────────────────────────────────
    # Tracks which node produced an error so the synthesiser can tailor the message
    failed_at: Optional[str]               # "intent" | "fetch" | None
