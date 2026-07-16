# -*- coding: utf-8 -*-
"""结构化日志、请求追踪和轻量运行指标。"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from collections import Counter
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import Any

from agent_safety import redact_payload
from configs import AGENT_RUNTIME_config


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="-")
agent_trace_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_trace", default=None
)


def start_agent_trace(trace_id: str, conversation_id: str):
    """Start a context-local trace and return the ContextVar reset token."""
    return agent_trace_var.set(
        {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "started_perf": time.perf_counter(),
            "stages": [],
        }
    )


def record_stage(
    name: str,
    latency_ms: float,
    *,
    status: str = "success",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
    **fields: Any,
) -> None:
    trace = agent_trace_var.get()
    if trace is None:
        return
    stage = {
        "name": name,
        "latency_ms": round(float(latency_ms), 2),
        "status": status,
    }
    if prompt_tokens or completion_tokens:
        stage.update(
            {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
                "cache_hit_tokens": int(cache_hit_tokens),
                "cache_miss_tokens": int(cache_miss_tokens),
            }
        )
    if fields:
        stage.update(redact_payload(fields))
    trace["stages"].append(stage)


def finish_agent_trace(*, route: str = "unknown") -> dict[str, Any]:
    trace = agent_trace_var.get()
    if trace is None:
        return {}
    stages = list(trace.get("stages") or [])
    prompt_tokens = sum(int(stage.get("prompt_tokens") or 0) for stage in stages)
    completion_tokens = sum(int(stage.get("completion_tokens") or 0) for stage in stages)
    cache_hit_tokens = sum(int(stage.get("cache_hit_tokens") or 0) for stage in stages)
    cache_miss_tokens = sum(int(stage.get("cache_miss_tokens") or 0) for stage in stages)
    return {
        "trace_id": trace.get("trace_id"),
        "conversation_id": trace.get("conversation_id"),
        "route": route,
        "total_latency_ms": round(
            (time.perf_counter() - float(trace["started_perf"])) * 1000, 2
        ),
        "stages": stages,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "cache_hit": cache_hit_tokens,
            "cache_miss": cache_miss_tokens,
        },
    }


def reset_agent_trace(token) -> None:
    agent_trace_var.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "request_id": request_id_var.get(),
            "conversation_id": conversation_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = redact_payload(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("beauty_agent")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = JsonFormatter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_path = Path(AGENT_RUNTIME_config["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rotating = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)
    return logger


LOGGER = configure_logging()


def log_event(event: str, **fields: Any) -> None:
    LOGGER.info(event, extra={"event": event, "fields": fields})


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self.started_at = time.time()
        self.requests = 0
        self.errors = 0
        self.rate_limited = 0
        self.security_blocks = 0
        self.handoffs = 0
        self.total_latency_ms = 0.0
        self.routes: Counter[str] = Counter()

    def record_request(self, *, route: str, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.requests += 1
            self.errors += int(error)
            self.total_latency_ms += latency_ms
            self.routes[route or "unknown"] += 1

    def increment(self, name: str) -> None:
        with self._lock:
            if name == "rate_limited":
                self.rate_limited += 1
            elif name == "security_blocks":
                self.security_blocks += 1
            elif name == "handoffs":
                self.handoffs += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average = self.total_latency_ms / self.requests if self.requests else 0.0
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests": self.requests,
                "errors": self.errors,
                "rate_limited": self.rate_limited,
                "security_blocks": self.security_blocks,
                "handoffs": self.handoffs,
                "average_latency_ms": round(average, 1),
                "routes": dict(self.routes),
            }


METRICS = MetricsRegistry()
