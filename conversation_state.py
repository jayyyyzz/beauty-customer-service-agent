# -*- coding: utf-8 -*-
"""客服 Agent 的持久化会话、槽位和任务状态存储。"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "runtime" / "conversation_state.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class ConversationStore:
    """使用 SQLite 保存会话历史、共享槽位和可恢复任务。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'active',
                    slots_json TEXT NOT NULL DEFAULT '{}',
                    active_task_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    intent_json TEXT NOT NULL,
                    route TEXT NOT NULL,
                    status TEXT NOT NULL,
                    required_slots_json TEXT NOT NULL DEFAULT '[]',
                    slots_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                CREATE INDEX IF NOT EXISTS idx_tasks_conversation_status
                    ON tasks(conversation_id, status, sequence_no);
                """
            )

    def ensure_conversation(self, conversation_id: str) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(
                    conversation_id, status, slots_json, created_at, updated_at
                ) VALUES (?, 'active', '{}', ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (conversation_id, now, now),
            )

    def seed_history(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        """仅在新会话中导入调用方已有历史，避免每轮重复写入。"""
        self.ensure_conversation(conversation_id)
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            if count:
                return
            now = _now()
            rows = [
                (
                    conversation_id,
                    str(message.get("role") or "buyer"),
                    str(message.get("content") or ""),
                    json.dumps(message.get("metadata") or {}, ensure_ascii=False),
                    now,
                )
                for message in messages
                if str(message.get("content") or "").strip()
            ]
            conn.executemany(
                """
                INSERT INTO messages(
                    conversation_id, role, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        self.ensure_conversation(conversation_id)
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages(
                    conversation_id, role, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            return int(cursor.lastrowid)

    def get_messages(self, conversation_id: str, limit: int = 30) -> list[dict[str, Any]]:
        self.ensure_conversation(conversation_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "metadata": _loads(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def get_history_dialogue(self, conversation_id: str, limit: int = 30) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "messages": self.get_messages(conversation_id, limit=limit),
        }

    def get_slots(self, conversation_id: str) -> dict[str, Any]:
        self.ensure_conversation(conversation_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT slots_json FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return _loads(row["slots_json"] if row else None, {})

    def update_slots(self, conversation_id: str, slots: dict[str, Any]) -> dict[str, Any]:
        current = self.get_slots(conversation_id)
        current.update({key: value for key, value in slots.items() if value not in (None, "")})
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET slots_json = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (json.dumps(current, ensure_ascii=False), now, conversation_id),
            )
        return current

    def create_task(
        self,
        conversation_id: str,
        question: str,
        intent: dict[str, Any],
        route: str,
        *,
        status: str = "in_progress",
        required_slots: list[str] | None = None,
        slots: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_conversation(conversation_id)
        task_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            sequence_no = conn.execute(
                """
                SELECT COALESCE(MAX(sequence_no), 0) + 1
                FROM tasks WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO tasks(
                    task_id, conversation_id, sequence_no, question,
                    intent_json, route, status, required_slots_json,
                    slots_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    conversation_id,
                    sequence_no,
                    question,
                    json.dumps(intent, ensure_ascii=False),
                    route,
                    status,
                    json.dumps(required_slots or [], ensure_ascii=False),
                    json.dumps(slots or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE conversations
                SET active_task_id = ?, status = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    task_id,
                    "waiting_user" if status == "waiting_user" else "active",
                    now,
                    conversation_id,
                ),
            )
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return self._task_from_row(row)

    def list_tasks(self, conversation_id: str) -> list[dict[str, Any]]:
        self.ensure_conversation(conversation_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE conversation_id = ?
                ORDER BY sequence_no
                """,
                (conversation_id,),
            ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def get_waiting_tasks(self, conversation_id: str) -> list[dict[str, Any]]:
        self.ensure_conversation(conversation_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE conversation_id = ? AND status = 'waiting_user'
                ORDER BY sequence_no
                """,
                (conversation_id,),
            ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        slots: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task = self.get_task(task_id)
        merged_slots = dict(task["slots"])
        if slots:
            merged_slots.update(slots)
        next_status = status or task["status"]
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, slots_json = ?, result_json = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    next_status,
                    json.dumps(merged_slots, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False, default=str)
                    if result is not None
                    else (
                        json.dumps(task["result"], ensure_ascii=False, default=str)
                        if task["result"] is not None
                        else None
                    ),
                    now,
                    task_id,
                ),
            )
            waiting_count = conn.execute(
                """
                SELECT COUNT(*) FROM tasks
                WHERE conversation_id = ? AND status = 'waiting_user'
                """,
                (task["conversation_id"],),
            ).fetchone()[0]
            conn.execute(
                """
                UPDATE conversations
                SET status = ?, active_task_id = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    "waiting_user" if waiting_count else "active",
                    task_id if next_status in {"waiting_user", "in_progress"} else None,
                    now,
                    task["conversation_id"],
                ),
            )
        return self.get_task(task_id)

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "conversation_id": row["conversation_id"],
            "sequence_no": row["sequence_no"],
            "question": row["question"],
            "intent": _loads(row["intent_json"], {}),
            "route": row["route"],
            "status": row["status"],
            "required_slots": _loads(row["required_slots_json"], []),
            "slots": _loads(row["slots_json"], {}),
            "result": _loads(row["result_json"], None),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
