# -*- coding: utf-8 -*-
"""美妆电商客服 Agent 的 FastAPI 与网页演示入口。"""
from __future__ import annotations

import asyncio
import os
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_observability import (
    METRICS,
    conversation_id_var,
    finish_agent_trace,
    log_event,
    request_id_var,
    reset_agent_trace,
    start_agent_trace,
)
from agent_pipeline import close_agent_resources, handle_user_question
from agent_safety import redact_payload
from configs import AGENT_RUNTIME_config, ES_search_config, LLM_deepseek_config
from handoff_store import get_handoff_store


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"


class ChatMessage(BaseModel):
    role: Literal["buyer", "seller", "user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=3, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    messages: list[ChatMessage] = Field(default_factory=list, max_length=30)
    user_id: str | None = Field(default=None, max_length=100)
    knowledge_top_k: int = Field(default=3, ge=1, le=8)
    tool_context: dict[str, Any] | None = None

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str) -> str:
        if not all(char.isalnum() or char in "-_" for char in value):
            raise ValueError("conversation_id 只能包含字母、数字、短横线和下划线")
        return value


class HandoffRequest(BaseModel):
    conversation_id: str = Field(min_length=3, max_length=100)
    reason: str = Field(default="user_requested", max_length=100)
    summary: str = Field(min_length=1, max_length=1000)
    priority: Literal["normal", "high", "urgent"] = "normal"
    context: dict[str, Any] = Field(default_factory=dict)


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = max(1, limit)
        self.window_seconds = max(1, window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                return False, retry_after
            events.append(now)
            if not events:
                self._events.pop(key, None)
            return True, 0


RATE_LIMITER = SlidingWindowRateLimiter(
    int(AGENT_RUNTIME_config["rate_limit_requests"]),
    int(AGENT_RUNTIME_config["rate_limit_window_seconds"]),
)
AGENT_CAPACITY = asyncio.BoundedSemaphore(
    max(1, int(AGENT_RUNTIME_config["max_concurrent_requests"]))
)


async def _es_health() -> dict[str, Any]:
    url = str(ES_search_config["url"]).rstrip("/")

    def check() -> dict[str, Any]:
        request = urllib.request.Request(f"{url}/_cluster/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return {"available": response.status == 200, "url": url}
        except (urllib.error.URLError, TimeoutError, OSError):
            return {"available": False, "url": url}

    return await asyncio.to_thread(check)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log_event(
        "application_started",
        max_concurrency=AGENT_RUNTIME_config["max_concurrent_requests"],
        rate_limit=AGENT_RUNTIME_config["rate_limit_requests"],
    )
    yield
    await close_agent_resources()
    log_event("application_stopped")


app = FastAPI(
    title="美妆电商客服 Agent",
    description="带 RAG、业务工具、安全防护与人工转接的面试演示 API。",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def request_trace_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request_token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log_event(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response
    finally:
        request_id_var.reset(request_token)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "deepseek_configured": bool(LLM_deepseek_config.get("api_key")),
        "elasticsearch": await _es_health(),
        "limits": {
            "max_concurrency": AGENT_RUNTIME_config["max_concurrent_requests"],
            "requests_per_window": AGENT_RUNTIME_config["rate_limit_requests"],
            "window_seconds": AGENT_RUNTIME_config["rate_limit_window_seconds"],
        },
    }


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    return METRICS.snapshot()


@app.get("/api/handoffs")
async def handoffs(limit: int = 20) -> dict[str, Any]:
    return {"items": get_handoff_store().list_recent(limit=max(1, min(limit, 100)))}


@app.post("/api/handoff")
async def create_handoff(payload: HandoffRequest) -> dict[str, Any]:
    ticket = get_handoff_store().create(
        conversation_id=payload.conversation_id,
        reason=payload.reason,
        summary=payload.summary,
        priority=payload.priority,
        context=payload.context,
    )
    METRICS.increment("handoffs")
    log_event("handoff_created", ticket_id=ticket["ticket_id"], reason=payload.reason)
    return ticket


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = await RATE_LIMITER.check(client_key)
    if not allowed:
        METRICS.increment("rate_limited")
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试。",
            headers={"Retry-After": str(retry_after)},
        )

    capacity_wait = min(5.0, float(AGENT_RUNTIME_config["request_timeout_seconds"]))
    try:
        await asyncio.wait_for(AGENT_CAPACITY.acquire(), timeout=capacity_wait)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="Agent 正在处理较多请求，请稍后重试。") from exc

    conversation_token = conversation_id_var.set(payload.conversation_id)
    trace_token = start_agent_trace(request_id_var.get(), payload.conversation_id)
    started = time.perf_counter()
    result: dict[str, Any] | None = None
    route = "error"
    try:
        history = {
            "conversation_id": payload.conversation_id,
            "user_id": payload.user_id,
            "messages": [message.model_dump() for message in payload.messages],
        }
        result = await asyncio.wait_for(
            handle_user_question(
                history_dialogue=history,
                question=payload.question,
                knowledge_top_k=payload.knowledge_top_k,
                tool_context=payload.tool_context,
            ),
            timeout=float(AGENT_RUNTIME_config["request_timeout_seconds"]),
        )
        route = str(result.get("route") or "unknown")
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if route == "security_block":
            METRICS.increment("security_blocks")
        if result.get("handoff_required"):
            METRICS.increment("handoffs")
        METRICS.record_request(route=route, latency_ms=latency_ms)
        result["trace"] = finish_agent_trace(route=route)
        result["trace"].update(
            {
                "request_id": request_id_var.get(),
                "latency_ms": latency_ms,
                "intent_count": len(result.get("intents") or []),
                "retrieval_count": len(result.get("knowledge_docs") or []),
                "pii_redacted": result.get("pii_redacted") or [],
            }
        )
        log_event(
            "agent_request_completed",
            route=route,
            latency_ms=latency_ms,
            intents=result.get("intents"),
            citations=result.get("citations"),
            handoff_required=bool(result.get("handoff_required")),
        )
        return redact_payload(result)
    except asyncio.TimeoutError as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        METRICS.record_request(route="timeout", latency_ms=latency_ms, error=True)
        log_event("agent_request_timeout", latency_ms=latency_ms)
        raise HTTPException(status_code=504, detail="Agent 处理超时，请稍后重试或转人工。") from exc
    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        METRICS.record_request(route=route, latency_ms=latency_ms, error=True)
        log_event("agent_request_failed", error=type(exc).__name__, message=str(exc))
        raise HTTPException(status_code=503, detail="Agent 暂时不可用，请检查模型与知识库服务。") from exc
    finally:
        reset_agent_trace(trace_token)
        conversation_id_var.reset(conversation_token)
        AGENT_CAPACITY.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host=os.getenv("WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("WEB_PORT", "8000")),
        reload=False,
    )
