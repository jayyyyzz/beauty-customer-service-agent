# -*- coding: utf-8 -*-

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import agent_pipeline


class BusinessAnswerTemplateTests(unittest.TestCase):
    def test_confirmation_required_never_claims_success(self):
        answer = agent_pipeline.render_business_answer({
            "order_id": "MOCK202606260003",
            "result": {
                "status": "confirmation_required",
                "summary": "拟为订单 MOCK202606260003 提交退款申请。",
            },
        })
        self.assertIn("请确认是否继续执行", answer)
        self.assertNotIn("退款成功", answer)

    def test_found_order_only_renders_returned_fields(self):
        answer = agent_pipeline.render_business_answer({
            "order_id": "MOCK202606260008",
            "result": {
                "status": "found",
                "order": {
                    "order_id": "MOCK202606260008",
                    "fulfillment_status": "shipping_exception",
                    "carrier": "ZTO Express",
                    "tracking_number": "ZTO6690455566",
                },
            },
        })
        self.assertIn("物流异常", answer)
        self.assertIn("ZTO6690455566", answer)
        self.assertNotIn("延迟", answer)
        self.assertNotIn("优先处理", answer)

    def test_tool_only_answer_bypasses_llm_and_has_no_citation(self):
        completion = AsyncMock(return_value="不应调用")
        with patch.object(agent_pipeline, "_chat_completion", new=completion):
            answer = asyncio.run(agent_pipeline.generate_answer(
                "取消成功了吗？",
                {"intent_level1": "after_sale"},
                route="business_api_confirmation",
                api_data={
                    "order_id": "MOCK202606260002",
                    "result": {"status": "succeeded", "message": "订单已取消。"},
                },
            ))
        completion.assert_not_awaited()
        self.assertEqual(answer, "订单已取消。")
        self.assertNotIn("[S", answer)


class EvidenceGuardrailTests(unittest.TestCase):
    def test_invalid_citation_is_removed(self):
        completion = AsyncMock(return_value="建议每次使用1泵。[S1] 不要自行换算。[S9]")
        docs = [{
            "citation_id": "S1",
            "title": "用量",
            "document_type": "faq",
            "source_name": "测试",
            "source_url": "",
            "text": "每次使用1泵。",
        }]
        with patch.object(agent_pipeline, "_chat_completion", new=completion):
            answer = asyncio.run(agent_pipeline.generate_answer(
                "一次用多少？",
                {"intent_level1": "usage"},
                route="knowledge_base",
                knowledge_docs=docs,
            ))
        self.assertIn("[S1]", answer)
        self.assertNotIn("[S9]", answer)

    def test_realtime_stock_without_tool_abstains(self):
        completion = AsyncMock(return_value="库存还有100件")
        with patch.object(agent_pipeline, "_chat_completion", new=completion):
            answer = asyncio.run(agent_pipeline.generate_answer(
                "这款今天还有多少库存，最低成交价是多少？",
                {"intent_level1": "price"},
                route="knowledge_base",
                knowledge_docs=[{"citation_id": "S1", "text": "普通商品介绍"}],
            ))
        completion.assert_not_awaited()
        self.assertIn("没有实时库存", answer)
        self.assertNotIn("100件", answer)

    def test_refund_status_without_tool_does_not_claim_query(self):
        completion = AsyncMock(return_value="没有查到相关记录")
        with patch.object(agent_pipeline, "_chat_completion", new=completion):
            answer = asyncio.run(agent_pipeline.generate_answer(
                "我的退款到哪一步了？",
                {"intent_level1": "after_sale"},
                route="hybrid",
                knowledge_docs=[{"citation_id": "S1", "text": "无退款工具结果"}],
            ))
        completion.assert_not_awaited()
        self.assertIn("请提供订单号", answer)
        self.assertIn("不能确认", answer)
        self.assertNotIn("没有查到", answer)

    def test_unopened_shelf_life_without_evidence_abstains(self):
        completion = AsyncMock(return_value="未开封通常三年")
        with patch.object(agent_pipeline, "_chat_completion", new=completion):
            answer = asyncio.run(agent_pipeline.generate_answer(
                "化妆品没开封一般能放多久？",
                {"intent_level1": "authenticity_shelf_life"},
                route="knowledge_base",
                knowledge_docs=[{"citation_id": "S1", "text": "开封后建议6个月内用完"}],
            ))
        completion.assert_not_awaited()
        self.assertIn("没有提供", answer)
        self.assertIn("产品包装", answer)
        self.assertNotIn("三年", answer)

    def test_unreleased_formula_uses_knowledge_gap_guard(self):
        completion = AsyncMock(return_value="内部配方如下")
        with patch.object(agent_pipeline, "_chat_completion", new=completion):
            answer = asyncio.run(agent_pipeline.generate_answer(
                "下季度还没发布的新品完整配方能发我吗？",
                {"intent_level1": "ingredient"},
                route="knowledge_base",
                knowledge_docs=[],
            ))
        completion.assert_not_awaited()
        self.assertIn("未发布", answer)
        self.assertIn("不能", answer)


class ContextualRetrievalTests(unittest.TestCase):
    def test_referential_question_uses_recent_user_context(self):
        query = agent_pipeline.build_contextual_retrieval_question(
            "那它白天也能用吗？",
            {
                "messages": [
                    {"role": "buyer", "content": "我说的是视黄醇精华"},
                    {"role": "seller", "content": "建议低频使用"},
                ]
            },
        )
        self.assertIn("视黄醇精华", query)
        self.assertIn("白天也能用吗", query)

        queries = agent_pipeline.build_retrieval_queries(
            "那它白天也能用吗？",
            {"intent_level1": "usage"},
            {
                "messages": [
                    {"role": "buyer", "content": "我说的是视黄醇精华"},
                ]
            },
        )
        self.assertEqual(len(queries), 1)
        self.assertIn("视黄醇精华", queries[0])

    def test_contextual_query_normalizes_history_terms(self):
        query = agent_pipeline.build_contextual_retrieval_question(
            "那能同一晚叠吗？",
            {
                "messages": [
                    {"role": "buyer", "content": "我晚上正在用A醇和果酸"},
                ]
            },
        )
        self.assertIn("视黄醇", query)
        self.assertIn("一起使用", query)

    def test_multi_query_results_are_fused_and_renumbered(self):
        side_effect = [
            [
                {"document_id": "doc-a", "citation_id": "S1", "text": "A"},
                {"document_id": "doc-b", "citation_id": "S2", "text": "B"},
            ],
            [
                {"document_id": "doc-b", "citation_id": "S1", "text": "B"},
                {"document_id": "doc-c", "citation_id": "S2", "text": "C"},
            ],
        ]
        with (
            patch.object(
                agent_pipeline,
                "build_retrieval_queries",
                return_value=["query-1", "query-2"],
            ),
            patch.object(agent_pipeline, "search_knowledge", side_effect=side_effect),
        ):
            docs = agent_pipeline.search_knowledge_multi(
                "问题", {"intent_level1": "usage"}, k=3
            )
        self.assertEqual(docs[0]["document_id"], "doc-b")
        self.assertEqual([doc["citation_id"] for doc in docs], ["S1", "S2", "S3"])

    def test_reranker_error_uses_lexical_safety_fallback(self):
        docs = [
            {
                "document_id": "generic",
                "title": "敏感肌产品",
                "text": "某款果酸身体乳适合敏感肌。",
            },
            {
                "document_id": "safety",
                "title": "新手使用提示",
                "text": "新手先局部试用建立耐受，持续刺痛泛红时停用。",
            },
            {
                "document_id": "noise",
                "title": "其他",
                "text": "普通保湿信息。",
            },
            {
                "document_id": "extra",
                "title": "其他2",
                "text": "普通商品信息。",
            },
        ]
        with patch.object(
            agent_pipeline,
            "_chat_completion",
            new=AsyncMock(side_effect=RuntimeError("reranker unavailable")),
        ), patch.dict(agent_pipeline.RERANK_config, {"enabled": True}):
            ranked = asyncio.run(agent_pipeline.rerank_knowledge_docs(
                "敏感肌第一次用酸应该注意什么？", docs, top_k=3
            ))
        self.assertEqual(ranked[0]["document_id"], "safety")
        self.assertEqual(ranked[0]["rerank_status"], "fallback_lexical_error")


if __name__ == "__main__":
    unittest.main()
