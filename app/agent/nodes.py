from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.state import AgentState
from app.agent.tools import fetch_country, CountryFetchError, FIELD_EXTRACTORS

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
MODEL           = "moonshotai/kimi-k2.6"

_LLM = ChatOpenAI(
    model=MODEL,
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    temperature=0,
    max_tokens=512,
)

_SYNTH_LLM = ChatOpenAI(
    model=MODEL,
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    temperature=0.2,
    max_tokens=1024,
)

KNOWN_FIELDS = list(FIELD_EXTRACTORS.keys())

# ── Intent parsing via JSON prompt (works on any OpenAI-compatible model) ──────

_INTENT_SYSTEM = f"""You are an intent parser for a country information API.
Given a user question, respond with ONLY a valid JSON object — no markdown, no explanation.

Schema:
{{
  "country_name": "<country in English, properly capitalised>",
  "requested_fields": ["<field1>", "<field2>"]
}}

Valid fields: {", ".join(KNOWN_FIELDS)}

Rules:
- requested_fields must be a non-empty array of valid fields.
- If the question is general (e.g. "tell me about X"), use: ["population", "capital", "currency", "region", "languages", "area"]
- If a specific attribute is asked, include only that: "What currency?" → ["currency"]
- Never include fields outside the valid list.
- Respond with raw JSON only."""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, stripping any markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


async def parse_intent(state: AgentState) -> dict[str, Any]:
    logger.info("parse_intent: query=%r", state["query"])
    try:
        response = await _LLM.ainvoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=state["query"]),
        ])
        parsed = _parse_json_response(response.content)

        country = parsed.get("country_name", "").strip()
        fields = [f for f in parsed.get("requested_fields", []) if f in FIELD_EXTRACTORS]

        if not country:
            raise ValueError("No country extracted")
        if not fields:
            fields = ["population", "capital", "currency", "region"]

        logger.info("parse_intent: country=%r fields=%r", country, fields)
        return {
            "country_name": country,
            "requested_fields": fields,
            "error": None,
            "failed_at": None,
        }

    except Exception as exc:
        logger.warning("parse_intent failed: %s", exc)
        return {
            "country_name": None,
            "requested_fields": None,
            "error": (
                "I couldn't understand your question. "
                "Try: 'What is the population of France?'"
            ),
            "failed_at": "intent",
        }


async def fetch_data(state: AgentState) -> dict[str, Any]:
    logger.info("fetch_data: country=%r", state["country_name"])
    try:
        data = await fetch_country(
            country_name=state["country_name"],
            requested_fields=state["requested_fields"] or [],
        )
        return {"raw_data": data, "error": None, "failed_at": None}
    except CountryFetchError as exc:
        logger.warning("fetch_data: %s", exc)
        return {"raw_data": None, "error": str(exc), "failed_at": "fetch"}
    except Exception as exc:
        logger.error("fetch_data: unexpected error: %s", exc, exc_info=True)
        return {
            "raw_data": None,
            "error": "Unexpected error fetching country data.",
            "failed_at": "fetch",
        }


_SYNTH_SYSTEM = """You are a concise country information assistant.
Answer using ONLY the JSON data provided. Never fabricate facts.
Format numbers with commas. Be direct and factual."""


async def synthesize(state: AgentState) -> dict[str, Any]:
    if state.get("error"):
        return {"answer": state["error"]}

    data_blob = json.dumps(state["raw_data"], indent=2, ensure_ascii=False)
    prompt = f"Question: {state['query']}\n\nData:\n{data_blob}\n\nAnswer concisely using only the data above."

    try:
        response = await _SYNTH_LLM.ainvoke([
            SystemMessage(content=_SYNTH_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return {"answer": response.content}
    except Exception as exc:
        logger.error("synthesize failed: %s", exc, exc_info=True)
        return {"answer": "Error generating answer. Please try again."}