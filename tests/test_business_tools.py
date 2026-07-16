# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from business_tools import (
    ActorContext,
    BusinessToolService,
    extract_business_arguments,
    infer_business_action,
    missing_business_arguments,
)
from conversation_state import ConversationStore


ROOT = Path(__file__).resolve().parents[1]
ORDER_CSV = ROOT / "data" / "processed" / "order_mock_data.csv"


class BusinessToolServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "business_tools.db"
        self.service = BusinessToolService(ORDER_CSV, self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def actor(user_id: str) -> ActorContext:
        return ActorContext(actor_id=user_id, role="customer", user_id=user_id)

    def prepare_and_confirm(
        self,
        action: str,
        order_id: str,
        user_id: str,
        arguments: dict,
        key: str,
    ) -> tuple[dict, dict]:
        actor = self.actor(user_id)
        prepared = self.service.execute(
            action,
            order_id,
            actor=actor,
            arguments=arguments,
            idempotency_key=key,
        )
        self.assertEqual("confirmation_required", prepared["status"])
        confirmed = self.service.execute(
            action,
            order_id,
            actor=actor,
            arguments=arguments,
            idempotency_key=key,
            confirmation_token=prepared["confirmation_token"],
        )
        self.assertEqual("succeeded", confirmed["status"])
        return prepared, confirmed

    def test_query_requires_order_owner(self) -> None:
        allowed = self.service.execute(
            "query_order", "MOCK202606260003", actor=self.actor("mock_user_004")
        )
        denied = self.service.execute(
            "query_order", "MOCK202606260003", actor=self.actor("mock_user_999")
        )
        self.assertEqual("succeeded", allowed["status"])
        self.assertEqual("permission_denied", denied["status"])

    def test_cancel_requires_confirmation_and_is_idempotent(self) -> None:
        arguments = {"reason": "用户不再需要"}
        prepared, confirmed = self.prepare_and_confirm(
            "cancel_order",
            "MOCK202606260002",
            "mock_user_003",
            arguments,
            "cancel-0002",
        )
        replayed = self.service.execute(
            "cancel_order",
            "MOCK202606260002",
            actor=self.actor("mock_user_003"),
            arguments=arguments,
            idempotency_key="cancel-0002",
            confirmation_token=prepared["confirmation_token"],
        )
        self.assertEqual("cancelled", confirmed["result"]["order_status"])
        self.assertTrue(replayed["idempotent_replay"])
        with closing(sqlite3.connect(self.db_path)) as conn:
            succeeded = conn.execute(
                "SELECT COUNT(*) FROM tool_operations WHERE idempotency_key=? AND status='succeeded'",
                ("cancel-0002",),
            ).fetchone()[0]
        self.assertEqual(1, succeeded)

    def test_refund_creates_review_request(self) -> None:
        _, confirmed = self.prepare_and_confirm(
            "request_refund",
            "MOCK202606260003",
            "mock_user_004",
            {"reason": "使用后出现过敏"},
            "refund-0003",
        )
        self.assertEqual("pending_review", confirmed["result"]["status"])

    def test_update_address_masks_phone_in_result(self) -> None:
        address = {
            "recipient": "张三",
            "phone": "13800138000",
            "province": "上海市",
            "city": "上海市",
            "detail": "浦东新区测试路88号",
        }
        _, confirmed = self.prepare_and_confirm(
            "update_address",
            "MOCK202606260002",
            "mock_user_003",
            {"new_address": address},
            "address-0002",
        )
        self.assertEqual("138****8000", confirmed["result"]["phone"])

    def test_invoice_validates_company_tax_id(self) -> None:
        invalid = self.service.execute(
            "request_invoice",
            "MOCK202606260003",
            actor=self.actor("mock_user_004"),
            arguments={
                "invoice_type": "company",
                "title": "测试公司",
                "email": "finance@example.com",
            },
            idempotency_key="invoice-invalid",
        )
        self.assertEqual("invalid_arguments", invalid["status"])

        _, confirmed = self.prepare_and_confirm(
            "request_invoice",
            "MOCK202606260003",
            "mock_user_004",
            {
                "invoice_type": "company",
                "title": "测试公司",
                "tax_id": "91310000TEST12345X",
                "email": "finance@example.com",
            },
            "invoice-0003",
        )
        self.assertEqual("pending_issue", confirmed["result"]["status"])
        self.assertEqual("f***@example.com", confirmed["result"]["email"])

    def test_urge_shipment_creates_ticket(self) -> None:
        _, confirmed = self.prepare_and_confirm(
            "urge_shipment",
            "MOCK202606260002",
            "mock_user_003",
            {},
            "urge-0002",
        )
        self.assertEqual("queued", confirmed["result"]["status"])
        self.assertTrue(confirmed["result"]["ticket_id"].startswith("URGE"))

    def test_shipping_exception_cannot_use_normal_urge_tool(self) -> None:
        rejected = self.service.execute(
            "urge_shipment",
            "MOCK202606260008",
            actor=self.actor("mock_user_001"),
            arguments={"reason": "物流一直没有更新"},
            idempotency_key="urge-exception",
        )
        self.assertEqual("business_rule_rejected", rejected["status"])

    def test_structured_planner_extracts_refund_address_and_invoice_arguments(self) -> None:
        refund_question = "订单 MOCK202606260019 使用后过敏，帮我申请退款"
        refund_action = infer_business_action(refund_question, {"intent_level1": "after_sale"})
        refund_args = extract_business_arguments(refund_question, refund_action)
        self.assertEqual("request_refund", refund_action)
        self.assertEqual([], missing_business_arguments(refund_action, refund_args))

        address_question = (
            "订单 MOCK202606260026 改到上海市浦东新区测试路88号，"
            "收件人张三，电话13800138000"
        )
        address_action = infer_business_action(address_question, {"intent_level1": "after_sale"})
        address_args = extract_business_arguments(address_question, address_action)
        self.assertEqual("update_address", address_action)
        self.assertEqual([], missing_business_arguments(address_action, address_args))

        invoice_question = (
            "订单 MOCK202606260027 开企业电子发票，抬头测试公司，"
            "税号91310000TEST12345X，发到finance@example.com"
        )
        invoice_action = infer_business_action(invoice_question, {"intent_level1": "invoice"})
        invoice_args = extract_business_arguments(invoice_question, invoice_action)
        self.assertEqual("request_invoice", invoice_action)
        self.assertEqual([], missing_business_arguments(invoice_action, invoice_args))

    def test_current_turn_order_has_priority_over_history(self) -> None:
        import agent_pipeline

        history = {
            "messages": [{"role": "buyer", "content": "之前查的是 MOCK202606260003"}]
        }
        current = "改查订单 MOCK202606260004 到哪里了"
        self.assertEqual("MOCK202606260004", agent_pipeline.extract_order_id(current, history))
        slots = agent_pipeline.extract_slots(current, history, {"order_id": "MOCK202606260003"})
        self.assertEqual("MOCK202606260004", slots["order_id"])

    def test_tool_failure_maps_to_non_completed_task_status(self) -> None:
        import agent_pipeline

        self.assertEqual("waiting_user", agent_pipeline.business_status_to_task_status("invalid_arguments"))
        self.assertEqual("blocked", agent_pipeline.business_status_to_task_status("permission_denied"))
        self.assertEqual("rejected", agent_pipeline.business_status_to_task_status("business_rule_rejected"))

    def test_idempotency_key_cannot_be_reused_for_different_arguments(self) -> None:
        actor = self.actor("mock_user_004")
        first = self.service.execute(
            "request_refund",
            "MOCK202606260003",
            actor=actor,
            arguments={"reason": "过敏"},
            idempotency_key="same-key",
        )
        second = self.service.execute(
            "request_refund",
            "MOCK202606260003",
            actor=actor,
            arguments={"reason": "不喜欢"},
            idempotency_key="same-key",
        )
        self.assertEqual("confirmation_required", first["status"])
        self.assertEqual("idempotency_conflict", second["status"])

    def test_agent_wrapper_returns_reusable_confirmation_context(self) -> None:
        import agent_pipeline

        previous_service = agent_pipeline._BUSINESS_SERVICE
        agent_pipeline._BUSINESS_SERVICE = self.service
        try:
            history = {"user_id": "mock_user_003", "messages": []}
            intent = {"intent_level1": "after_sale"}
            first = agent_pipeline.call_business_api(
                "取消订单 MOCK202606260002",
                intent,
                history,
                actor_context={
                    "actor_id": "mock_user_003",
                    "role": "customer",
                    "user_id": "mock_user_003",
                },
                tool_context={
                    "action": "cancel_order",
                    "order_id": "MOCK202606260002",
                    "arguments": {"reason": "用户不再需要"},
                    "idempotency_key": "agent-cancel-0002",
                },
            )
            self.assertEqual("confirmation_required", first["result"]["status"])
            self.assertIn("next_tool_context", first)

            second = agent_pipeline.call_business_api(
                "确认取消",
                intent,
                history,
                actor_context={
                    "actor_id": "mock_user_003",
                    "role": "customer",
                    "user_id": "mock_user_003",
                },
                tool_context=first["next_tool_context"],
            )
            self.assertEqual("succeeded", second["result"]["status"])
        finally:
            agent_pipeline._BUSINESS_SERVICE = previous_service


class AgentBusinessConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_user_question_completes_two_phase_action(self) -> None:
        import agent_pipeline

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            service = BusinessToolService(ORDER_CSV, temp / "business.db")
            state_store = ConversationStore(temp / "conversation.db")
            previous_service = agent_pipeline._BUSINESS_SERVICE
            agent_pipeline._BUSINESS_SERVICE = service
            intent = {
                "intent_level1": "after_sale",
                "intent_level2": "after_sale.cancel",
                "intent_level3": "after_sale.cancel.order",
                "intent_logic": "用户要求取消订单",
                "intent_confidence": 0.99,
                "keywords": ["取消订单"],
            }
            history = {
                "conversation_id": "conv-business-confirm",
                "user_id": "mock_user_003",
                "messages": [],
            }
            try:
                with (
                    patch("agent_pipeline.recognize_intents", new=AsyncMock(return_value=[intent])) as recognizer,
                    patch("agent_pipeline.search_knowledge", return_value=[]),
                    patch("agent_pipeline.generate_answer", new=AsyncMock(return_value="请确认操作")),
                ):
                    first = await agent_pipeline.handle_user_question(
                        history,
                        "取消订单 MOCK202606260002",
                        state_store=state_store,
                        tool_context={
                            "action": "cancel_order",
                            "order_id": "MOCK202606260002",
                            "arguments": {"reason": "用户不再需要"},
                            "idempotency_key": "handle-cancel-0002",
                        },
                    )
                    self.assertEqual(
                        "confirmation_required", first["api_data"]["result"]["status"]
                    )
                    self.assertEqual("waiting_confirmation", first["tasks"][0]["status"])

                    second = await agent_pipeline.handle_user_question(
                        history,
                        "确认取消",
                        state_store=state_store,
                        tool_context=first["api_data"]["next_tool_context"],
                    )
                self.assertEqual("succeeded", second["api_data"]["result"]["status"])
                self.assertEqual("business_api_confirmation", second["route"])
                self.assertEqual(1, recognizer.await_count)
            finally:
                agent_pipeline._BUSINESS_SERVICE = previous_service


if __name__ == "__main__":
    unittest.main()
