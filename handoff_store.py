# -*- coding: utf-8 -*-
"""持久化人工转接工单，供 Agent 与 Web 演示共同使用。"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from agent_safety import redact_payload
from configs import AGENT_RUNTIME_config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HandoffStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or AGENT_RUNTIME_config["handoff_db_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS handoffs (
                    ticket_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_handoffs_status ON handoffs(status, created_at)"
            )

    def create(
        self,
        *,
        conversation_id: str,
        reason: str,
        summary: str,
        priority: str = "normal",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ticket_id = f"HO-{uuid.uuid4().hex[:10].upper()}"
        now = _now()
        safe_summary = str(redact_payload(summary))[:1000]
        safe_context = redact_payload(context or {})
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO handoffs(
                    ticket_id, conversation_id, reason, priority, status,
                    summary, context_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    conversation_id,
                    reason,
                    priority,
                    safe_summary,
                    json.dumps(safe_context, ensure_ascii=False, default=str),
                    now,
                    now,
                ),
            )
        return self.get(ticket_id)

    def get(self, ticket_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM handoffs WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Handoff ticket not found: {ticket_id}")
        return self._from_row(row)

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM handoffs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "ticket_id": row["ticket_id"],
            "conversation_id": row["conversation_id"],
            "reason": row["reason"],
            "priority": row["priority"],
            "status": row["status"],
            "summary": row["summary"],
            "context": json.loads(row["context_json"] or "{}"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


_DEFAULT_STORE: HandoffStore | None = None


def get_handoff_store() -> HandoffStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = HandoffStore()
    return _DEFAULT_STORE
