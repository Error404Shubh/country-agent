"""
Tests for the Country Information Agent.

Run with:  pytest tests/ -v

These tests use httpx mocking so they don't hit the real API
— safe to run in CI without network access.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# ── Unit tests: tools ──────────────────────────────────────────────────────────

class TestFieldExtractors:
    def test_extract_currencies(self):
        from app.agent.tools import _extract_currencies
        data = {
            "currencies": {
                "EUR": {"name": "Euro", "symbol": "€"},
            }
        }
        result = _extract_currencies(data)
        assert result == ["Euro (€)"]

    def test_extract_currencies_multiple(self):
        from app.agent.tools import _extract_currencies
        data = {
            "currencies": {
                "USD": {"name": "United States dollar", "symbol": "$"},
                "EUR": {"name": "Euro", "symbol": "€"},
            }
        }
        result = _extract_currencies(data)
        assert len(result) == 2

    def test_extract_currencies_empty(self):
        from app.agent.tools import _extract_currencies
        assert _extract_currencies({}) == []

    def test_extract_calling_code_single_suffix(self):
        from app.agent.tools import _extract_calling_code
        data = {"idd": {"root": "+4", "suffixes": ["9"]}}
        assert _extract_calling_code(data) == "+49"

    def test_extract_calling_code_no_root(self):
        from app.agent.tools import _extract_calling_code
        assert _extract_calling_code({}) is None


class TestBestMatch:
    def test_exact_common_name_match(self):
        from app.agent.tools import _best_match
        results = [
            {"name": {"common": "Guinea", "official": "Republic of Guinea"}},
            {"name": {"common": "Guinea-Bissau", "official": "Republic of Guinea-Bissau"}},
        ]
        match = _best_match(results, "Guinea")
        assert match["name"]["common"] == "Guinea"

    def test_falls_back_to_first(self):
        from app.agent.tools import _best_match
        results = [
            {"name": {"common": "New Zealand", "official": "New Zealand"}},
        ]
        match = _best_match(results, "nz")
        assert match["name"]["common"] == "New Zealand"


# ── Integration tests: graph nodes ────────────────────────────────────────────

@pytest.mark.asyncio
class TestParseIntent:
    async def test_happy_path(self):
        from app.agent.nodes import parse_intent
        from app.agent.state import AgentState

        mock_result = MagicMock()
        mock_result.country_name = "Germany"
        mock_result.requested_fields = ["population"]

        with patch("app.agent.nodes._intent_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_result)
            state: AgentState = {
                "query": "What is the population of Germany?",
                "country_name": None, "requested_fields": None,
                "raw_data": None, "answer": None, "error": None, "failed_at": None,
            }
            result = await parse_intent(state)

        assert result["country_name"] == "Germany"
        assert "population" in result["requested_fields"]
        assert result["error"] is None
        assert result["failed_at"] is None

    async def test_llm_failure_sets_error(self):
        from app.agent.nodes import parse_intent
        from app.agent.state import AgentState

        with patch("app.agent.nodes._intent_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
            state: AgentState = {
                "query": "???",
                "country_name": None, "requested_fields": None,
                "raw_data": None, "answer": None, "error": None, "failed_at": None,
            }
            result = await parse_intent(state)

        assert result["failed_at"] == "intent"
        assert result["error"] is not None
        assert result["country_name"] is None


@pytest.mark.asyncio
class TestFetchData:
    async def test_happy_path(self):
        from app.agent.nodes import fetch_data
        from app.agent.state import AgentState

        mock_data = {"population": 83200000, "common_name": "Germany"}

        with patch("app.agent.nodes.fetch_country", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_data
            state: AgentState = {
                "query": "population of Germany",
                "country_name": "Germany",
                "requested_fields": ["population"],
                "raw_data": None, "answer": None, "error": None, "failed_at": None,
            }
            result = await fetch_data(state)

        assert result["raw_data"] == mock_data
        assert result["error"] is None

    async def test_404_sets_error(self):
        from app.agent.nodes import fetch_data
        from app.agent.tools import CountryFetchError
        from app.agent.state import AgentState

        with patch("app.agent.nodes.fetch_country", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = CountryFetchError("Country 'Xyzzy' not found.", status_code=404)
            state: AgentState = {
                "query": "capital of Xyzzy",
                "country_name": "Xyzzy",
                "requested_fields": ["capital"],
                "raw_data": None, "answer": None, "error": None, "failed_at": None,
            }
            result = await fetch_data(state)

        assert result["failed_at"] == "fetch"
        assert "not found" in result["error"].lower()
        assert result["raw_data"] is None


@pytest.mark.asyncio
class TestSynthesize:
    async def test_returns_error_message_on_error_state(self):
        from app.agent.nodes import synthesize
        from app.agent.state import AgentState

        state: AgentState = {
            "query": "What is the capital of Neverland?",
            "country_name": "Neverland",
            "requested_fields": ["capital"],
            "raw_data": None,
            "answer": None,
            "error": "Country 'Neverland' not found.",
            "failed_at": "fetch",
        }
        result = await synthesize(state)
        assert result["answer"] == "Country 'Neverland' not found."

    async def test_calls_llm_on_success(self):
        from app.agent.nodes import synthesize
        from app.agent.state import AgentState

        mock_response = MagicMock()
        mock_response.content = "Germany has a population of 83,200,000."

        with patch("app.agent.nodes._SYNTHESIS_LLM") as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            state: AgentState = {
                "query": "What is the population of Germany?",
                "country_name": "Germany",
                "requested_fields": ["population"],
                "raw_data": {"population": 83200000, "common_name": "Germany"},
                "answer": None,
                "error": None,
                "failed_at": None,
            }
            result = await synthesize(state)

        assert "83,200,000" in result["answer"] or "Germany" in result["answer"]


# ── Graph routing tests ────────────────────────────────────────────────────────

class TestRouting:
    def test_route_intent_ok(self):
        from app.agent.graph import _route_intent
        state = {"failed_at": None, "error": None}
        assert _route_intent(state) == "fetch_data"

    def test_route_intent_error(self):
        from app.agent.graph import _route_intent
        state = {"failed_at": "intent", "error": "bad input"}
        assert _route_intent(state) == "synthesize"

    def test_route_fetch_always_synthesize(self):
        from app.agent.graph import _route_fetch
        assert _route_fetch({}) == "synthesize"
        assert _route_fetch({"failed_at": "fetch"}) == "synthesize"


# ── Schema validation ──────────────────────────────────────────────────────────

class TestSchemas:
    def test_query_request_strips_whitespace(self):
        from app.schemas.models import QueryRequest
        req = QueryRequest(question="  What is the capital of France?  ")
        assert req.question == "What is the capital of France?"

    def test_query_request_rejects_too_short(self):
        from app.schemas.models import QueryRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            QueryRequest(question="Hi")
