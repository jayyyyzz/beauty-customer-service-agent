import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import handoff_store
from agent_pipeline import handle_user_question
from handoff_policy import assess_handoff_policy
from handoff_store import HandoffStore


class HandoffPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_user_request_and_account_security_are_deterministic(self):
        requested = assess_handoff_policy("别再自动回复，马上接人工")
        security = assess_handoff_policy("这笔异常支付不是我付的，账号可能被盗")
        self.assertTrue(requested.should_handoff)
        self.assertEqual("user_requested", requested.reason)
        self.assertTrue(security.should_handoff)
        self.assertEqual("account_security", security.reason)
        self.assertEqual("urgent", security.priority)

    def test_repeated_failure_uses_history(self):
        decision = assess_handoff_policy(
            "别再让我重复了，直接升级处理",
            {
                "messages": [
                    {"role": "buyer", "content": "订单号已经给过两次"},
                    {"role": "seller", "content": "系统仍然查询失败"},
                ]
            },
        )
        self.assertTrue(decision.should_handoff)
        self.assertEqual("repeated_failure", decision.reason)

    async def test_explicit_handoff_bypasses_llm_and_preserves_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff_store._DEFAULT_STORE = HandoffStore(Path(tmp) / "handoffs.db")
            with patch("agent_pipeline.recognize_intents", new=AsyncMock(side_effect=AssertionError("LLM should not run"))):
                result = await handle_user_question(
                    history_dialogue={
                        "conversation_id": "handoff-policy-test",
                        "messages": [{"role": "buyer", "content": "我已经解释过问题"}],
                    },
                    question="给我转人工客服",
                )
        self.assertEqual("human_handoff", result["route"])
        self.assertEqual("user_requested", result["handoff"]["reason"])
        self.assertTrue(result["handoff"]["context"]["history"])


if __name__ == "__main__":
    unittest.main()
