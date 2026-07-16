# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "implementation"))

from optimized_retrieval import (
    build_contextual_query,
    build_optimized_filters,
    is_likely_knowledge_gap,
    rerank_and_deduplicate,
)


class OptimizedRetrievalTests(unittest.TestCase):
    def test_contextual_query_includes_history_and_normalizes_colloquial_text(self):
        case = {
            "intent": "compatibility",
            "keywords": ["A醇", "果酸"],
            "history": [{"role": "buyer", "content": "我晚上正在用A醇"}],
            "question": "那能同一晚叠吗？",
        }
        query = build_contextual_query(case)
        self.assertIn("视黄醇", query)
        self.assertIn("果酸", query)
        self.assertIn("一起使用", query)

    def test_english_product_query_is_not_blocked_by_language_or_brand_case(self):
        case = {
            "intent": "product_info",
            "question": "What is the price of Almay Smart Shade Butter Kiss Lipstick?",
        }
        filters = build_optimized_filters(case)
        self.assertIn("product", filters["document_type"])
        self.assertNotIn("language", filters)
        self.assertNotIn("brand", filters)

    def test_high_confidence_knowledge_gaps_are_detected(self):
        questions = [
            "能告诉我仓库此刻准确还剩多少件吗？",
            "下季度还没发布的新品完整成分百分比是多少？",
            "你直接诊断一下我是不是激素脸",
        ]
        for question in questions:
            self.assertTrue(is_likely_knowledge_gap({"question": question, "history": []}))
        self.assertFalse(is_likely_knowledge_gap({"question": "烟酰胺浓度是多少？", "history": []}))

    def test_content_hash_and_near_duplicate_are_removed(self):
        case = {"intent": "after_sale", "question": "过敏可以退货吗", "keywords": ["过敏", "退货"]}
        hits = [
            {"_score": 10.0, "_source": {"document_id": "a", "content_hash": "same", "document_type": "faq", "title": "过敏退货", "content": "过敏后可以申请退货"}},
            {"_score": 9.0, "_source": {"document_id": "b", "content_hash": "same", "document_type": "faq", "title": "过敏退货", "content": "过敏后可以申请退货"}},
            {"_score": 8.0, "_source": {"document_id": "c", "content_hash": "other", "document_type": "policy", "title": "售后规则", "content": "请保留凭证并联系售后"}},
        ]
        output = rerank_and_deduplicate(hits, case, final_k=3)
        self.assertEqual(len(output), 2)
        self.assertEqual(len({item["_source"]["content_hash"] for item in output}), 2)


if __name__ == "__main__":
    unittest.main()
