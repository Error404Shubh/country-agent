"""
FastAPI application — production-ready wrapper around the LangGraph agent.

Endpoints:
  POST /api/query    — main agent endpoint
  GET  /api/health   — liveness check (load balancer / k8s probe)
  GET  /             — serves the demo UI (static/index.html)

Production features baked in:
  - Request-scoped correlation IDs (X-Request-ID header)
  - Structured JSON logging
  - Global exception handler (never leaks stack traces to clients)
  - CORS configured for local dev (tighten origins in production)
  - Graceful shutdown via lifespan context (closes HTTP client cleanly)
  - /metrics stub comment — plug in prometheus-fastapi-instrumentator in prod
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import run_agent
from app.agent.tools import close_http_client
from app.schemas.models import QueryRequest, QueryResponse, HealthResponse
from app.utils.logger import configure_logging

# ── Bootstrap ──────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JSON_LOGS = os.getenv("JSON_LOGS", "false").lower() == "true"
configure_logging(level=LOG_LEVEL, json_logs=JSON_LOGS)

logger = logging.getLogger(__name__)

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Country Agent starting up (version=%s)", APP_VERSION)
    yield
    logger.info("Country Agent shutting down — closing HTTP client")
    await close_http_client()


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Country Information Agent",
    description=(
        "A LangGraph-powered AI agent that answers natural-language questions "
        "about countries using the REST Countries v3.1 API."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for demo; restrict to your domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)

# Serve static files (demo UI)
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.monotonic()

    response = await call_next(request)

    elapsed_ms = round((time.monotonic() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

    logger.info(
        "%s %s %s %dms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        extra={"request_id": request_id},
    )
    return response


# ── Global error handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    index = os.path.join(static_dir, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return JSONResponse({"message": "Country Information Agent API", "docs": "/docs"})


@app.get("/api/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness probe — always returns 200 if the process is alive."""
    return HealthResponse(status="ok", version=APP_VERSION)


@app.post("/api/query", response_model=QueryResponse, tags=["agent"])
async def query(body: QueryRequest, request: Request):
    """
    Run the Country Information Agent against a natural-language question.

    The agent will:
    1. Parse the intent (country + requested fields)
    2. Fetch data from the REST Countries API
    3. Synthesise a grounded natural-language answer
    """
    request_id = request.headers.get("X-Request-ID", "")
    logger.info("New query", extra={"request_id": request_id, "question": body.question})

    try:
        state = await run_agent(body.question)
    except Exception as exc:
        logger.error("Agent run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Agent encountered an internal error.")

    return QueryResponse(
        answer=state.get("answer") or "I could not generate an answer. Please try again.",
        country=state.get("country_name"),
        fields_retrieved=state.get("requested_fields"),
        raw_data=state.get("raw_data"),
    )
