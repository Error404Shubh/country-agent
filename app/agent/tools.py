"""
Country data tool — thin wrapper around the REST Countries v3.1 API.

Design decisions:
- httpx.AsyncClient with a shared session (connection pooling, keep-alive)
- Conservative timeout (5 s connect, 10 s read) to fail fast in production
- Returns only the fields the agent asked for — smaller payload, less LLM noise
- Structured errors so the calling node can route appropriately
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Field extraction helpers ───────────────────────────────────────────────────
# Maps our canonical field names → extraction functions over the raw API dict

FIELD_EXTRACTORS: dict[str, Any] = {
    "population": lambda d: d.get("population"),
    "capital": lambda d: (d.get("capital") or [None])[0],
    "currency": lambda d: _extract_currencies(d),
    "languages": lambda d: list((d.get("languages") or {}).values()),
    "area": lambda d: d.get("area"),
    "region": lambda d: d.get("region"),
    "subregion": lambda d: d.get("subregion"),
    "timezones": lambda d: d.get("timezones"),
    "flag": lambda d: d.get("flag"),           # emoji flag
    "flag_url": lambda d: (d.get("flags") or {}).get("png"),
    "calling_code": lambda d: _extract_calling_code(d),
    "borders": lambda d: d.get("borders"),
    "continent": lambda d: (d.get("continents") or [None])[0],
    "official_name": lambda d: (d.get("name") or {}).get("official"),
    "common_name": lambda d: (d.get("name") or {}).get("common"),
    "independence": lambda d: d.get("independent"),
    "un_member": lambda d: d.get("unMember"),
    "landlocked": lambda d: d.get("landlocked"),
    "google_maps": lambda d: (d.get("maps") or {}).get("googleMaps"),
}

# Fields we always fetch regardless of what the user asked — needed for a coherent answer
_BASELINE_FIELDS = {"common_name", "official_name", "flag"}


def _extract_currencies(data: dict) -> list[str]:
    currencies = data.get("currencies") or {}
    return [
        f"{info.get('name', code)} ({info.get('symbol', '')})"
        for code, info in currencies.items()
    ]


def _extract_calling_code(data: dict) -> str | None:
    idd = data.get("idd") or {}
    root = idd.get("root", "")
    suffixes = idd.get("suffixes") or []
    if not root:
        return None
    if len(suffixes) == 1:
        return f"{root}{suffixes[0]}"
    return root  # multiple suffixes → return root only


# ── Shared async HTTP client ───────────────────────────────────────────────────

_CLIENT: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient(
            base_url="https://restcountries.com/v3.1",
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=2.0),
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
    return _CLIENT


async def close_http_client() -> None:
    global _CLIENT
    if _CLIENT and not _CLIENT.is_closed:
        await _CLIENT.aclose()


# ── Public interface ───────────────────────────────────────────────────────────

class CountryFetchError(Exception):
    """Raised when the REST Countries API cannot fulfil the request."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


async def fetch_country(
    country_name: str,
    requested_fields: list[str],
) -> dict[str, Any]:
    """
    Fetch country data from the REST Countries API and return only the
    fields the agent needs. Raises CountryFetchError on any failure.
    """
    client = get_http_client()
    url = f"/name/{httpx.URL(country_name)}"

    try:
        logger.info("Fetching country data", extra={"country": country_name, "fields": requested_fields})
        response = await client.get(url, params={"fullText": "false"})
    except httpx.TimeoutException as exc:
        raise CountryFetchError(f"Request to REST Countries API timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise CountryFetchError(f"Network error reaching REST Countries API: {exc}") from exc

    if response.status_code == 404:
        raise CountryFetchError(
            f"Country '{country_name}' not found. Please check the spelling.",
            status_code=404,
        )
    if response.status_code != 200:
        raise CountryFetchError(
            f"REST Countries API returned HTTP {response.status_code}.",
            status_code=response.status_code,
        )

    results: list[dict] = response.json()
    if not results:
        raise CountryFetchError(f"No data returned for '{country_name}'.")

    # Pick the best match: prefer exact common-name match, fall back to first result
    raw = _best_match(results, country_name)

    # Extract only the fields we need (plus baselines)
    fields_to_extract = set(requested_fields) | _BASELINE_FIELDS
    extracted: dict[str, Any] = {}
    for field in fields_to_extract:
        extractor = FIELD_EXTRACTORS.get(field)
        if extractor:
            value = extractor(raw)
            if value is not None and value != [] and value != "":
                extracted[field] = value
        else:
            logger.debug("Unknown field requested: %s", field)

    logger.info("Country data fetched successfully", extra={"country": country_name, "fields_returned": list(extracted.keys())})
    return extracted


def _best_match(results: list[dict], query: str) -> dict:
    """
    Pick the most relevant result when the API returns multiple matches.
    Prefers an exact common-name or official-name match (case-insensitive).
    """
    q = query.lower().strip()
    for r in results:
        name = r.get("name") or {}
        if name.get("common", "").lower() == q or name.get("official", "").lower() == q:
            return r
    return results[0]
