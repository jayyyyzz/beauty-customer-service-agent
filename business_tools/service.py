# -*- coding: utf-8 -*-
"""带权限、确认、幂等、重试和审计的本地电商业务工具服务。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ToolAction(str, Enum):
    QUERY_ORDER = "query_order"
    URGE_SHIPMENT = "urge_shipment"
    REQUEST_REFUND = "request_refund"
    CANCEL_ORDER = "cancel_order"
    UPDATE_ADDRESS = "update_address"
    REQUEST_INVOICE = "request_invoice"


MUTATING_ACTIONS = {
    ToolAction.URGE_SHIPMENT,
    ToolAction.REQUEST_REFUND,
    ToolAction.CANCEL_ORDER,
    ToolAction.UPDATE_ADDRESS,
    ToolAction.REQUEST_INVOICE,
}

SHIPPED_FULFILLMENT_STATES = {"shipped", "delivered"}


@dataclass(frozen=True)
class ActorContext:
    actor_id: str
    role: str
    user_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ActorContext":
        data = value or {}
        user_id = str(data.get("user_id") or "").strip() or None
        actor_id = str(data.get("actor_id") or user_id or "anonymous").strip()
        role = str(data.get("role") or "customer").strip().lower()
        return cls(actor_id=actor_id, role=role, user_id=user_id)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mask_phone(phone: str) -> str:
    digits = str(phone)
    if len(digits) < 7:
        return "***"
    return f"{digits[:3]}****{digits[-4:]}"


def _mask_email(email: str) -> str:
    name, sep, domain = str(email).partition("@")
    if not sep:
        return "***"
    return f"{name[:1]}***@{domain}"


class BusinessToolService:
    """SQLite-backed mock commerce service with production-style safeguards."""

    def __init__(
        self,
        order_csv: str | Path,
        db_path: str | Path,
        *,
        confirmation_ttl_seconds: int = 600,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.05,
    ) -> None:
        self.order_csv = Path(order_csv)
        self.db_path = Path(db_path)
        self.confirmation_ttl_seconds = confirmation_ttl_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialize(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    product_id TEXT,
                    product_name TEXT,
                    order_status TEXT,
                    payment_status TEXT,
                    fulfillment_status TEXT,
                    tracking_number TEXT,
                    carrier TEXT,
                    estimated_delivery_time TEXT,
                    is_cancelable INTEGER NOT NULL,
                    is_refundable INTEGER NOT NULL,
                    created_at TEXT,
                    updated_at TEXT,
                    mock_note TEXT,
                    extra_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS tool_operations (
                    operation_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    arguments_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confirmation_token TEXT UNIQUE,
                    confirmation_expires_at TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(order_id) REFERENCES orders(order_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tool_operations_order
                    ON tool_operations(order_id, created_at);
                """
            )
            count = int(conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0])
            if count == 0:
                self._import_orders(conn)

    def _import_orders(self, conn: sqlite3.Connection) -> None:
        if not self.order_csv.exists():
            raise FileNotFoundError(f"订单初始化文件不存在: {self.order_csv}")
        with self.order_csv.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        conn.executemany(
            """
            INSERT INTO orders (
                order_id, user_id, product_id, product_name, order_status,
                payment_status, fulfillment_status, tracking_number, carrier,
                estimated_delivery_time, is_cancelable, is_refundable,
                created_at, updated_at, mock_note, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
            """,
            [
                (
                    row.get("order_id", "").upper(), row.get("user_id", ""),
                    row.get("product_id", ""), row.get("product_name", ""),
                    row.get("order_status", ""), row.get("payment_status", ""),
                    row.get("fulfillment_status", ""), row.get("tracking_number", ""),
                    row.get("carrier", ""), row.get("estimated_delivery_time", ""),
                    int(_as_bool(row.get("is_cancelable"))),
                    int(_as_bool(row.get("is_refundable"))),
                    row.get("created_at", ""), row.get("updated_at", ""),
                    row.get("mock_note", ""),
                )
                for row in rows
            ],
        )

    def _retry(self, func: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                return func()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay_seconds * (2 ** (attempt - 1)))
        raise RuntimeError("unreachable")

    def execute(
        self,
        action: str | ToolAction,
        order_id: str | None,
        *,
        actor: ActorContext,
        arguments: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        confirmation_token: str | None = None,
    ) -> dict[str, Any]:
        try:
            parsed_action = ToolAction(action)
        except ValueError:
            return self._error("unsupported_action", f"不支持的业务操作: {action}")

        normalized_order_id = str(order_id or "").strip().upper()
        if not re.fullmatch(r"MOCK\d{12,}", normalized_order_id):
            return self._error("need_user_info", "请提供有效订单号，例如 MOCK202606260003。")
        if actor.role not in {"customer", "agent", "admin"}:
            return self._error("permission_denied", f"未知角色: {actor.role}")

        args = arguments or {}
        if not isinstance(args, dict):
            return self._error("invalid_arguments", "工具参数必须是 JSON 对象。")

        return self._retry(
            lambda: self._execute_once(
                parsed_action,
                normalized_order_id,
                actor,
                args,
                idempotency_key,
                confirmation_token,
            )
        )

    def _execute_once(
        self,
        action: ToolAction,
        order_id: str,
        actor: ActorContext,
        arguments: dict[str, Any],
        idempotency_key: str | None,
        confirmation_token: str | None,
    ) -> dict[str, Any]:
        with closing(self._connect()) as conn, conn:
            order = self._get_order(conn, order_id)
            if not order:
                return self._error("not_found", f"没有查到订单 {order_id}。")
            permission_error = self._authorize(actor, order)
            if permission_error:
                return permission_error

            if action is ToolAction.QUERY_ORDER:
                result = {
                    "status": "succeeded",
                    "tool_name": action.value,
                    "message": "订单查询成功。",
                    "order": self._public_order(order),
                }
                self._audit_query(conn, action, order_id, actor, arguments, result)
                return result

            args_json = _json(arguments)
            args_hash = hashlib.sha256(args_json.encode("utf-8")).hexdigest()

            if confirmation_token:
                if not idempotency_key:
                    return self._error("invalid_request", "确认执行时必须同时提供幂等键。")
                return self._confirm(
                    conn, action, order, actor, arguments, args_json, args_hash,
                    idempotency_key, confirmation_token,
                )

            validation_error = self._validate_action(action, order, arguments)
            if validation_error:
                return validation_error

            return self._prepare(
                conn, action, order, actor, arguments, args_json, args_hash,
                idempotency_key or f"op_{uuid.uuid4().hex}",
            )

    def _get_order(self, conn: sqlite3.Connection, order_id: str) -> dict[str, Any] | None:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return dict(row) if row else None

    def _authorize(self, actor: ActorContext, order: dict[str, Any]) -> dict[str, Any] | None:
        if actor.role in {"agent", "admin"}:
            return None
        if not actor.user_id:
            return self._error("authentication_required", "执行订单操作前需要登录用户身份。")
        if actor.user_id != order["user_id"]:
            return self._error("permission_denied", "当前用户无权访问该订单。")
        return None

    def _validate_action(
        self, action: ToolAction, order: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        cancelled = order["order_status"] == "cancelled"
        if action is ToolAction.URGE_SHIPMENT:
            if (
                cancelled
                or order["payment_status"] != "paid"
                or order["fulfillment_status"] in SHIPPED_FULFILLMENT_STATES
                or order["fulfillment_status"] == "shipping_exception"
                or order["order_status"] == "exception"
            ):
                return self._error(
                    "business_rule_rejected",
                    "仅已支付且尚未发货、无物流异常的订单可以催发货。物流异常请转异常查询或人工处理。",
                )
        elif action is ToolAction.CANCEL_ORDER:
            if cancelled:
                return self._error("business_rule_rejected", "订单已经取消，无需重复操作。")
            if (
                not bool(order["is_cancelable"])
                or order["fulfillment_status"] in SHIPPED_FULFILLMENT_STATES
            ):
                return self._error("business_rule_rejected", "当前订单状态不允许取消。")
        elif action is ToolAction.REQUEST_REFUND:
            reason = str(arguments.get("reason") or "").strip()
            if len(reason) < 2:
                return self._error("invalid_arguments", "申请退款需要提供退款原因。")
            if not bool(order["is_refundable"]) or cancelled or order["order_status"] == "refund_requested":
                return self._error("business_rule_rejected", "当前订单状态不支持发起退款。")
        elif action is ToolAction.UPDATE_ADDRESS:
            if (
                cancelled
                or order["fulfillment_status"] in SHIPPED_FULFILLMENT_STATES
                or not bool(order["is_cancelable"])
            ):
                return self._error("business_rule_rejected", "订单已发货或当前状态不允许修改地址。")
            address = arguments.get("new_address")
            required = ("recipient", "phone", "province", "city", "detail")
            if not isinstance(address, dict) or any(not str(address.get(k) or "").strip() for k in required):
                return self._error("invalid_arguments", "新地址必须包含收件人、电话、省、市和详细地址。")
            if not re.fullmatch(r"[0-9+\- ]{7,20}", str(address["phone"])):
                return self._error("invalid_arguments", "联系电话格式不正确。")
        elif action is ToolAction.REQUEST_INVOICE:
            invoice_type = str(arguments.get("invoice_type") or "")
            title = str(arguments.get("title") or "").strip()
            email = str(arguments.get("email") or "").strip()
            if order["payment_status"] != "paid" or cancelled:
                return self._error("business_rule_rejected", "仅已支付且未取消的订单可以申请发票。")
            if invoice_type not in {"personal", "company"} or not title:
                return self._error("invalid_arguments", "请提供发票类型和发票抬头。")
            if invoice_type == "company" and not str(arguments.get("tax_id") or "").strip():
                return self._error("invalid_arguments", "企业发票必须提供纳税人识别号。")
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                return self._error("invalid_arguments", "电子邮箱格式不正确。")
        return None

    def _prepare(
        self,
        conn: sqlite3.Connection,
        action: ToolAction,
        order: dict[str, Any],
        actor: ActorContext,
        arguments: dict[str, Any],
        args_json: str,
        args_hash: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        existing = conn.execute(
            "SELECT * FROM tool_operations WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if existing:
            return self._existing_operation(dict(existing), action, order["order_id"], actor, args_hash)

        operation_id = f"biz_{uuid.uuid4().hex}"
        token = f"confirm_{uuid.uuid4().hex}"
        now = _utc_now()
        expires_at = now + timedelta(seconds=self.confirmation_ttl_seconds)
        conn.execute(
            """
            INSERT INTO tool_operations (
                operation_id, idempotency_key, action, order_id, actor_id, actor_role,
                arguments_json, arguments_hash, status, confirmation_token,
                confirmation_expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_confirmation', ?, ?, ?, ?)
            """,
            (
                operation_id, idempotency_key, action.value, order["order_id"], actor.actor_id,
                actor.role, args_json, args_hash, token, expires_at.isoformat(timespec="seconds"),
                now.isoformat(timespec="seconds"), now.isoformat(timespec="seconds"),
            ),
        )
        return {
            "status": "confirmation_required",
            "tool_name": action.value,
            "operation_id": operation_id,
            "idempotency_key": idempotency_key,
            "confirmation_token": token,
            "confirmation_expires_at": expires_at.isoformat(timespec="seconds"),
            "confirmation_summary": self._confirmation_summary(action, order, arguments),
            "message": "该操作会修改订单，请用户明确确认后再执行。",
        }

    def _existing_operation(
        self,
        operation: dict[str, Any],
        action: ToolAction,
        order_id: str,
        actor: ActorContext,
        args_hash: str,
    ) -> dict[str, Any]:
        if (
            operation["action"] != action.value
            or operation["order_id"] != order_id
            or operation["actor_id"] != actor.actor_id
            or operation["arguments_hash"] != args_hash
        ):
            return self._error("idempotency_conflict", "该幂等键已被另一个不同请求使用。")
        if operation["status"] == "succeeded" and operation.get("result_json"):
            result = json.loads(operation["result_json"])
            result["idempotent_replay"] = True
            return result
        if operation["status"] == "pending_confirmation":
            return {
                "status": "confirmation_required",
                "tool_name": operation["action"],
                "operation_id": operation["operation_id"],
                "idempotency_key": operation["idempotency_key"],
                "confirmation_token": operation["confirmation_token"],
                "confirmation_expires_at": operation["confirmation_expires_at"],
                "message": "相同请求正在等待确认。",
            }
        return self._error("operation_unavailable", f"该操作当前状态为 {operation['status']}。")

    def _confirm(
        self,
        conn: sqlite3.Connection,
        action: ToolAction,
        order: dict[str, Any],
        actor: ActorContext,
        arguments: dict[str, Any],
        args_json: str,
        args_hash: str,
        idempotency_key: str,
        token: str,
    ) -> dict[str, Any]:
        conn.execute("BEGIN IMMEDIATE")
        operation_row = conn.execute(
            "SELECT * FROM tool_operations WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if not operation_row:
            conn.rollback()
            return self._error("confirmation_not_found", "没有找到待确认操作，请重新发起。")
        operation = dict(operation_row)
        mismatch = (
            operation["action"] != action.value
            or operation["order_id"] != order["order_id"]
            or operation["actor_id"] != actor.actor_id
            or operation["arguments_hash"] != args_hash
        )
        if mismatch:
            conn.rollback()
            return self._error("confirmation_mismatch", "确认信息与原始请求不一致。")
        if operation["status"] == "succeeded" and operation.get("result_json"):
            conn.rollback()
            result = json.loads(operation["result_json"])
            result["idempotent_replay"] = True
            return result
        if operation["status"] != "pending_confirmation" or operation["confirmation_token"] != token:
            conn.rollback()
            return self._error("invalid_confirmation", "确认令牌无效。")
        expires_at = datetime.fromisoformat(operation["confirmation_expires_at"])
        if _utc_now() > expires_at:
            conn.execute(
                "UPDATE tool_operations SET status = 'expired', updated_at = ? WHERE operation_id = ?",
                (_iso_now(), operation["operation_id"]),
            )
            conn.commit()
            return self._error("confirmation_expired", "确认已过期，请重新发起操作。")

        current_order = self._get_order(conn, order["order_id"])
        validation_error = self._validate_action(action, current_order or order, arguments)
        if validation_error:
            conn.rollback()
            return validation_error

        updated_order, details = self._apply_action(action, current_order or order, arguments)
        conn.execute(
            """
            UPDATE orders SET
                order_status = ?, payment_status = ?, fulfillment_status = ?,
                is_cancelable = ?, is_refundable = ?, updated_at = ?, extra_json = ?
            WHERE order_id = ?
            """,
            (
                updated_order["order_status"], updated_order["payment_status"],
                updated_order["fulfillment_status"], int(bool(updated_order["is_cancelable"])),
                int(bool(updated_order["is_refundable"])), updated_order["updated_at"],
                updated_order["extra_json"], updated_order["order_id"],
            ),
        )
        result = {
            "status": "succeeded",
            "tool_name": action.value,
            "operation_id": operation["operation_id"],
            "idempotency_key": idempotency_key,
            "order_id": order["order_id"],
            "message": self._success_message(action),
            "result": details,
        }
        conn.execute(
            """
            UPDATE tool_operations SET status = 'succeeded', result_json = ?, updated_at = ?
            WHERE operation_id = ?
            """,
            (_json(result), _iso_now(), operation["operation_id"]),
        )
        conn.commit()
        return result

    def _apply_action(
        self, action: ToolAction, order: dict[str, Any], arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        updated = dict(order)
        extra = json.loads(updated.get("extra_json") or "{}")
        now = _iso_now()
        details: dict[str, Any]

        if action is ToolAction.URGE_SHIPMENT:
            ticket_id = f"URGE{uuid.uuid4().hex[:12].upper()}"
            extra["shipment_urge"] = {"ticket_id": ticket_id, "status": "queued", "created_at": now}
            details = {"ticket_id": ticket_id, "status": "queued"}
        elif action is ToolAction.REQUEST_REFUND:
            request_id = f"REF{uuid.uuid4().hex[:12].upper()}"
            updated["order_status"] = "refund_requested"
            extra["refund"] = {
                "request_id": request_id, "status": "pending_review",
                "reason": str(arguments["reason"]).strip(), "created_at": now,
            }
            details = {"refund_request_id": request_id, "status": "pending_review"}
        elif action is ToolAction.CANCEL_ORDER:
            updated["order_status"] = "cancelled"
            updated["fulfillment_status"] = "not_shipped"
            if updated["payment_status"] == "paid":
                updated["payment_status"] = "refund_pending"
            updated["is_cancelable"] = 0
            updated["is_refundable"] = 0
            extra["cancellation"] = {
                "status": "accepted", "reason": str(arguments.get("reason") or "").strip(),
                "created_at": now,
            }
            details = {"order_status": "cancelled", "payment_status": updated["payment_status"]}
        elif action is ToolAction.UPDATE_ADDRESS:
            address = dict(arguments["new_address"])
            extra["shipping_address"] = {**address, "updated_at": now}
            details = {
                "recipient": address["recipient"],
                "phone": _mask_phone(address["phone"]),
                "region": f"{address['province']} {address['city']}",
                "status": "updated",
            }
        elif action is ToolAction.REQUEST_INVOICE:
            invoice_id = f"INV{uuid.uuid4().hex[:12].upper()}"
            invoice = {
                "invoice_id": invoice_id,
                "invoice_type": arguments["invoice_type"],
                "title": str(arguments["title"]).strip(),
                "tax_id": str(arguments.get("tax_id") or "").strip(),
                "email": str(arguments["email"]).strip(),
                "status": "pending_issue",
                "created_at": now,
            }
            extra["invoice"] = invoice
            details = {
                "invoice_id": invoice_id, "status": "pending_issue",
                "title": invoice["title"], "email": _mask_email(invoice["email"]),
            }
        else:
            raise ValueError(f"Unsupported mutation: {action.value}")

        updated["updated_at"] = now
        updated["extra_json"] = _json(extra)
        return updated, details

    def _audit_query(
        self,
        conn: sqlite3.Connection,
        action: ToolAction,
        order_id: str,
        actor: ActorContext,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        now = _iso_now()
        args_json = _json(arguments)
        conn.execute(
            """
            INSERT INTO tool_operations (
                operation_id, idempotency_key, action, order_id, actor_id, actor_role,
                arguments_json, arguments_hash, status, result_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'succeeded', ?, ?, ?)
            """,
            (
                f"biz_{uuid.uuid4().hex}", f"query_{uuid.uuid4().hex}", action.value,
                order_id, actor.actor_id, actor.role, args_json,
                hashlib.sha256(args_json.encode("utf-8")).hexdigest(), _json(result), now, now,
            ),
        )

    def _public_order(self, order: dict[str, Any]) -> dict[str, Any]:
        return {
            key: order.get(key)
            for key in (
                "order_id", "product_name", "order_status", "payment_status",
                "fulfillment_status", "tracking_number", "carrier",
                "estimated_delivery_time", "is_cancelable", "is_refundable", "updated_at",
            )
        }

    def _confirmation_summary(
        self, action: ToolAction, order: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "action": action.value,
            "order_id": order["order_id"],
            "product_name": order["product_name"],
        }
        if action is ToolAction.UPDATE_ADDRESS:
            address = arguments["new_address"]
            summary["new_address"] = {
                "recipient": address["recipient"],
                "phone": _mask_phone(address["phone"]),
                "region": f"{address['province']} {address['city']}",
                "detail": address["detail"],
            }
        elif action is ToolAction.REQUEST_INVOICE:
            summary["invoice"] = {
                "invoice_type": arguments["invoice_type"],
                "title": arguments["title"],
                "email": _mask_email(arguments["email"]),
            }
        elif arguments.get("reason"):
            summary["reason"] = str(arguments["reason"])
        return summary

    @staticmethod
    def _success_message(action: ToolAction) -> str:
        return {
            ToolAction.URGE_SHIPMENT: "催发货工单已创建。",
            ToolAction.REQUEST_REFUND: "退款申请已提交审核。",
            ToolAction.CANCEL_ORDER: "订单已取消。",
            ToolAction.UPDATE_ADDRESS: "收货地址已更新。",
            ToolAction.REQUEST_INVOICE: "电子发票申请已提交。",
        }[action]

    @staticmethod
    def _error(status: str, message: str) -> dict[str, Any]:
        return {"status": status, "message": message}
