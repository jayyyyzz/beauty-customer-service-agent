# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_pipeline import generate_answer, handle_user_question
from conversation_state import ConversationStore


def make_intent(name: str, confidence: float = 0.95) -> dict:
    return {
        "intent_level1": name,
        "intent_level2": f"{name}.test",
        "intent_level3": f"{name}.test.case",
        "intent_logic": "测试意图",
        "intent_confidence": confidence,
        "keywords": ["测试"],
    }


def fake_doc(document_id: str = "doc-1") -> dict:
    return {
        "citation_id": "S1",
        "document_id": document_id,
        "document_type": "faq",
        "title": "测试知识",
        "topic": "测试",
        "text": "测试知识内容",
        "source_name": "测试知识库",
        "source_url": "",
    }


class ConversationStoreTests(unittest.TestCase):
    def test_messages_slots_and_tasks_persist_across_store_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.db"
            store = ConversationStore(db_path)
            store.seed_history(
                "conv-persist",
                [{"role": "buyer", "content": "我是敏感肌"}],
            )
            store.update_slots("conv-persist", {"order_id": "MOCK202606260003"})
            task = store.create_task(
                "conv-persist",
                "查物流",
                make_intent("logistics"),
                "business_api",
                status="waiting_user",
                required_slots=["order_id"],
            )

            reopened = ConversationStore(db_path)
            self.assertEqual(reopened.get_messages("conv-persist")[0]["content"], "我是敏感肌")
            self.assertEqual(reopened.get_slots("conv-persist")["order_id"], "MOCK202606260003")
            self.assertEqual(reopened.get_task(task["task_id"])["status"], "waiting_user")


class AgentStateMachineTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_task_resumes_after_slot_is_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp) / "state.db")
            history = {
                "conversation_id": "conv-resume",
                "user_id": "mock_user_004",
                "messages": [],
            }
            recognizer = AsyncMock(
                side_effect=[
                    [make_intent("logistics")],
                    [make_intent("routine")],
                ]
            )

            with (
                patch("agent_pipeline.recognize_intents", recognizer),
                patch("agent_pipeline.search_knowledge", return_value=[fake_doc()]),
                patch("agent_pipeline.generate_answer", new=AsyncMock(return_value="处理完成")),
            ):
                first = await handle_user_question(
                    history, "帮我查一下快递到哪里了", state_store=store
                )
                self.assertEqual(first["route"], "waiting_slots")
                self.assertEqual(len(first["pending_tasks"]), 1)
                self.assertIn("order_id", first["pending_tasks"][0]["missing_slots"])

                second = await handle_user_question(
                    history, "那先告诉我护肤顺序", state_store=store
                )
                self.assertEqual(second["route"], "knowledge_base")
                self.assertEqual(len(second["pending_tasks"]), 1)
                statuses = [task["status"] for task in store.list_tasks("conv-resume")]
                self.assertEqual(statuses, ["waiting_user", "completed"])

                third = await handle_user_question(
                    history, "订单号是 MOCK202606260003", state_store=store
                )

            self.assertTrue(third["resumed"])
            self.assertEqual(third["route"], "resume_tasks")
            self.assertEqual(third["pending_tasks"], [])
            self.assertEqual(third["slots"]["order_id"], "MOCK202606260003")
            self.assertEqual(third["api_data"]["result"]["status"], "succeeded")
            self.assertEqual(recognizer.await_count, 2)
            self.assertEqual(
                [task["status"] for task in store.list_tasks("conv-resume")],
                ["completed", "completed"],
            )

    async def test_multiple_intents_create_and_complete_independent_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp) / "state.db")
            history = {
                "conversation_id": "conv-multi",
                "user_id": "mock_user_004",
                "messages": [],
            }
            intents = [make_intent("routine"), make_intent("logistics")]

            with (
                patch("agent_pipeline.recognize_intents", new=AsyncMock(return_value=intents)),
                patch(
                    "agent_pipeline.search_knowledge",
                    side_effect=[[fake_doc("routine-doc")], [fake_doc("shipping-doc")]],
                ),
                patch("agent_pipeline.generate_answer", new=AsyncMock(return_value="两个问题都已处理")),
            ):
                result = await handle_user_question(
                    history,
                    "护肤顺序是什么？另外查一下订单 MOCK202606260003 的物流",
                    state_store=store,
                )

            self.assertEqual(result["route"], "multi_intent")
            self.assertEqual(len(result["intents"]), 2)
            self.assertEqual(len(result["tasks"]), 2)
            self.assertTrue(all(task["status"] == "completed" for task in result["tasks"]))
            self.assertEqual(result["api_data"]["result"]["status"], "succeeded")
            # 纯物流任务直接使用业务工具，不再额外执行知识库检索。
            self.assertEqual(len(result["knowledge_docs"]), 1)

    async def test_generate_answer_receives_full_recent_history(self):
        history = {
            "conversation_id": "conv-history",
            "messages": [
                {"role": "buyer", "content": "我最近容易泛红"},
                {"role": "seller", "content": "建议先精简护肤"},
                {"role": "buyer", "content": "那这个怎么用"},
            ],
        }
        completion = AsyncMock(return_value="测试回复")
        with patch("agent_pipeline._chat_completion", new=completion):
            answer = await generate_answer(
                "那这个怎么用",
                make_intent("usage"),
                route="knowledge_base",
                history_dialogue=history,
            )

        self.assertEqual(answer, "测试回复")
        messages = completion.await_args.args[0]
        prompt = messages[1]["content"]
        self.assertIn("我最近容易泛红", prompt)
        self.assertIn("建议先精简护肤", prompt)
        self.assertIn("那这个怎么用", prompt)


if __name__ == "__main__":
    unittest.main()
