import json
import unittest
from unittest.mock import AsyncMock, patch

import agent_pipeline
from intent_taxonomy import INTENT_LEVEL1_VALUES, INTENT_LEVEL3_PATHS


class IntentTaxonomyTests(unittest.IsolatedAsyncioTestCase):
    def test_taxonomy_has_22_level1_intents_and_valid_paths(self):
        self.assertEqual(len(INTENT_LEVEL1_VALUES), 22)
        self.assertIn("price.discount.coupon", INTENT_LEVEL3_PATHS)
        self.assertIn("other.unclear", INTENT_LEVEL3_PATHS)

    def test_normalize_derives_parent_levels_from_valid_level3(self):
        result = agent_pipeline._normalize_intent({
            "intent_level1": "wrong",
            "intent_level2": "wrong.path",
            "intent_level3": "usage.method.frequency",
            "intent_confidence": 0.91,
            "keywords": ["频率"],
            "needs_clarification": False,
        })
        self.assertEqual(result["intent_level1"], "usage")
        self.assertEqual(result["intent_level2"], "usage.method")
        self.assertEqual(result["intent_level3"], "usage.method.frequency")

    def test_clear_business_intent_does_not_use_intent_level_clarification(self):
        result = agent_pipeline._normalize_intent({
            "intent_level1": "comparison",
            "intent_level2": "comparison.product",
            "intent_level3": "comparison.product.suitability",
            "intent_confidence": 0.85,
            "keywords": ["两款", "肤质"],
            "needs_clarification": True,
            "missing_information": ["商品名称"],
        })
        self.assertFalse(result["needs_clarification"])

    def test_prompt_contains_exact_paths_and_boundary_rules(self):
        messages = agent_pipeline.build_intent_recognition_messages(
            {"conversation_id": "prompt-test", "messages": []},
            "订单怎么还没发？",
        )
        system = messages[0]["content"]
        self.assertIn("price.discount.coupon", system)
        self.assertIn("urge_shipment.urgent.deadline", system)
        self.assertIn("中性询问多久发货属于 logistics", system)
        self.assertIn("needs_clarification", system)

    async def test_vague_request_is_clarified_without_model_call(self):
        with patch("agent_pipeline._chat_completion", new=AsyncMock()) as completion:
            intents, raw, used_fallback = await agent_pipeline.recognize_intents_with_trace(
                {"conversation_id": "clarify-test", "messages": []},
                "帮我处理一下。",
            )
        completion.assert_not_awaited()
        self.assertFalse(used_fallback)
        self.assertIn("clarification_heuristic", raw)
        self.assertEqual(intents[0]["intent_level3"], "other.unclear")
        self.assertTrue(intents[0]["needs_clarification"])

    async def test_invalid_path_triggers_restricted_repair(self):
        first = json.dumps({
            "intents": [{
                "intent_level1": "price",
                "intent_level2": "discount",
                "intent_level3": "coupon",
                "intent_logic": "咨询优惠券",
                "intent_confidence": 0.9,
                "keywords": ["优惠券"],
            }]
        }, ensure_ascii=False)
        repaired = json.dumps({
            "intent_level1": "price",
            "intent_level2": "price.discount",
            "intent_level3": "price.discount.coupon",
            "intent_logic": "咨询优惠券",
            "intent_confidence": 0.9,
            "keywords": ["优惠券"],
            "needs_clarification": False,
            "missing_information": [],
            "clarification_question": "",
        }, ensure_ascii=False)
        completion = AsyncMock(side_effect=[first, repaired])
        with patch("agent_pipeline._chat_completion", new=completion):
            intents, _, used_fallback = await agent_pipeline.recognize_intents_with_trace(
                {"conversation_id": "repair-test", "messages": []},
                "有优惠券吗？",
            )
        self.assertFalse(used_fallback)
        self.assertEqual(completion.await_count, 2)
        self.assertEqual(intents[0]["intent_level3"], "price.discount.coupon")


if __name__ == "__main__":
    unittest.main()
