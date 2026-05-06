"""
API schemas — strict Pydantic models for request validation and response shaping.
Keeping these separate from agent state ensures the public API contract
can evolve independently of internal graph state.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="A natural-language question about a country.",
        examples=[
            "What is the population of Germany?",
            "What currency does Japan use?",
            "Tell me about Brazil.",
        ],
    )

    @field_validator("question")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class QueryResponse(BaseModel):
    answer: str = Field(description="Natural-language answer grounded in REST Countries data.")
    country: Optional[str] = Field(None, description="Resolved country name, if identified.")
    fields_retrieved: Optional[list[str]] = Field(
        None, description="Data fields that were fetched from the API."
    )
    raw_data: Optional[dict[str, Any]] = Field(
        None, description="Raw extracted data (useful for debugging / client-side rendering)."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
