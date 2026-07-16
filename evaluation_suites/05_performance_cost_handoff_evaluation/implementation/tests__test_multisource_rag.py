# -*- coding: utf-8 -*-

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "es_store"))

import agent_pipeline
import es_ingest
import es_search
from knowledge_sources import KnowledgeDocument, load_documents


class KnowledgeSourceTests(unittest.TestCase):
    def test_all_sources_use_unique_ids_and_expected_types(self):
        documents = load_documents(["conversation", "product", "faq", "policy", "shipping"])
        document_ids = [document.document_id for document in documents]
        self.assertEqual(len(document_ids), len(set(document_ids)))
        self.assertEqual(
            {document.document_type for document in documents},
            {"conversation", "product", "faq", "policy", "shipping"},
        )
        self.assertTrue(all(len(document.content_hash) == 64 for document in documents))

    def test_query_applies_filters_to_bm25_and_knn(self):
        filters = {"document_type": ["policy", "faq"], "brand": ["ColourPop"]}
        bm25 = es_search.build_query("bm25", "退货政策", None, filters=filters)
        self.assertEqual(len(bm25["query"]["bool"]["filter"]), 2)

        knn = es_search.build_query("knn", "退货政策", [0.0] * 512, filters=filters)
        self.assertEqual(len(knn["knn"]["filter"]["bool"]["filter"]), 2)

    def test_intent_selects_relevant_document_types(self):
        filters = agent_pipeline.build_metadata_filters({"intent_level1": "after_sale"})
        self.assertEqual(filters["document_type"], ["policy", "faq", "conversation"])

    def test_brand_is_inferred_from_question(self):
        filters = agent_pipeline.build_metadata_filters(
            {"intent_level1": "after_sale", "keywords": ["退货"]},
            "ColourPop 的退货政策是什么",
        )
        self.assertEqual(filters["brand"], ["ColourPop"])

    def test_es_search_uses_reusable_http_client(self):
        response = MagicMock()
        response.content = b'{"hits":{"hits":[]}}'
        client = MagicMock()
        client.post.return_value = response

        with patch.object(es_search, "_get_http_client", return_value=client) as get_client:
            hits = es_search.es_search(
                "http://127.0.0.1:9200",
                "knowledge",
                {"query": {"match_all": {}}},
            )

        self.assertEqual(hits, [])
        get_client.assert_called_once_with(False)
        client.post.assert_called_once()
        response.raise_for_status.assert_called_once_with()

    def test_http_client_is_cached_and_ignores_environment_proxy(self):
        es_search._get_http_client.cache_clear()

    def test_ingest_request_uses_reusable_http_client(self):
        response = MagicMock()
        response.status_code = 200
        response.content = b'{"acknowledged":true}'
        client = MagicMock()
        client.request.return_value = response

        with patch.object(es_ingest, "_get_http_client", return_value=client) as get_client:
            status, payload = es_ingest.request(
                "GET",
                "http://127.0.0.1:9200",
            )

        self.assertEqual(status, 200)
        self.assertTrue(payload["acknowledged"])
        get_client.assert_called_once_with(False)
        client.request.assert_called_once()

    def test_incremental_ingest_skips_reembedding_for_metadata_only_change(self):
        document = KnowledgeDocument(
            document_id="faq:1",
            document_type="faq",
            source_record_id="1",
            title="使用方法",
            content="问题：怎么用？\n答案：洁面后使用。",
            question_text="怎么用？",
            answer_text="洁面后使用。",
            content_hash="new-hash",
        )
        existing = {
            document.document_id: {
                "content_hash": "old-hash",
                "embedding_version": es_ingest.EMBEDDING_VERSION,
                "embedding_input_hash": es_ingest._embedding_input_hash(document),
            }
        }

        reembed, metadata_only = es_ingest._plan_incremental_updates(
            [document], existing
        )

        self.assertEqual(reembed, [])
        self.assertEqual(metadata_only, [document])

    def test_incremental_ingest_reembeds_when_retrieval_text_changes(self):
        document = KnowledgeDocument(
            document_id="faq:1",
            document_type="faq",
            source_record_id="1",
            title="使用方法",
            content="问题：什么时候用？\n答案：洁面后使用。",
            question_text="什么时候用？",
            answer_text="洁面后使用。",
            content_hash="new-hash",
        )
        existing = {
            document.document_id: {
                "content_hash": "old-hash",
                "embedding_version": es_ingest.EMBEDDING_VERSION,
                "embedding_input_hash": "different-input-hash",
            }
        }

        reembed, metadata_only = es_ingest._plan_incremental_updates(
            [document], existing
        )

        self.assertEqual(reembed, [document])
        self.assertEqual(metadata_only, [])
        client = MagicMock()

        with patch.object(es_search.httpx, "Client", return_value=client) as client_factory:
            first = es_search._get_http_client(False)
            second = es_search._get_http_client(False)

        self.assertIs(first, second)
        client_factory.assert_called_once_with(
            timeout=60.0,
            verify=True,
            trust_env=False,
        )
        es_search._get_http_client.cache_clear()


class AgentCitationTests(unittest.TestCase):
    def test_search_results_are_deduplicated_and_numbered(self):
        hits = [
            {
                "_score": 1.0,
                "_source": {
                    "document_id": "policy:1",
                    "document_type": "policy",
                    "title": "退货政策",
                    "content": "签收后十四天内处理。",
                    "content_hash": "same",
                    "source_name": "官方政策页",
                    "source_url": "https://example.com/policy",
                },
            },
            {
                "_score": 0.9,
                "_source": {
                    "document_id": "policy:duplicate",
                    "document_type": "policy",
                    "title": "退货政策副本",
                    "content": "签收后十四天内处理。",
                    "content_hash": "same",
                },
            },
            {
                "_score": 0.8,
                "_source": {
                    "document_id": "faq:1",
                    "document_type": "faq",
                    "title": "怎么申请退货",
                    "content": "请从订单页提交申请。",
                    "content_hash": "other",
                },
            },
        ]
        with patch.dict(agent_pipeline.ES_search_config, {"mode": "bm25"}), patch.object(
            es_search, "es_search", return_value=hits
        ):
            documents = agent_pipeline.search_knowledge(
                "怎么退货", {"intent_level1": "after_sale", "keywords": ["退货"]}, k=3
            )
        self.assertEqual([doc["citation_id"] for doc in documents], ["S1", "S2"])
        self.assertEqual([doc["document_id"] for doc in documents], ["policy:1", "faq:1"])

    def test_answer_prompt_contains_traceable_sources(self):
        captured = {}

        async def fake_completion(messages, **kwargs):
            captured["messages"] = messages
            return "可以申请退货。[S1]"

        documents = [{
            "citation_id": "S1",
            "title": "退货政策",
            "document_type": "policy",
            "source_name": "官方政策页",
            "source_url": "https://example.com/policy",
            "text": "签收后十四天内处理。",
        }]
        with patch.object(agent_pipeline, "_chat_completion", side_effect=fake_completion):
            answer = asyncio.run(
                agent_pipeline.generate_answer(
                    "怎么退货",
                    {"intent_level1": "after_sale"},
                    route="hybrid",
                    knowledge_docs=documents,
                )
            )
        self.assertIn("[S1]", captured["messages"][1]["content"])
        self.assertIn("https://example.com/policy", captured["messages"][1]["content"])
        self.assertEqual(answer, "可以申请退货。[S1]")


if __name__ == "__main__":
    unittest.main()
