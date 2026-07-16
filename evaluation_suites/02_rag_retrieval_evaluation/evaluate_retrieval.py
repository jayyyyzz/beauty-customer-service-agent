# -*- coding: utf-8 -*-
"""客服 RAG 检索专项离线评测。

该脚本使用人工定义的信息点规则做第一阶段弱监督标注，适合方案对照和回归，
不能替代双人独立标注的正式验收集。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import re
import numpy as np
import base64
import httpx
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parent
ROOT = EVALUATION_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "es_store"))
import es_search as es_search_module
from es_search import (
    PREFIX,
    _normalize_for_dedup,
    build_query,
    es_search,
    filter_near_duplicates,
    get_encoder,
    rrf_search,
)


INTENT_SEARCH_HINTS = {
    "price": "价格 优惠 活动 优惠券 满减",
    "product_info": "商品信息 规格 质地 包装",
    "skin_type": "肤质适配 油皮 干皮 敏感肌 是否适合",
    "skin_concern": "肌肤问题 痘痘 泛红 暗沉 修护",
    "ingredient": "成分 配方 浓度 耐受",
    "efficacy": "功效 保湿 修护 美白 抗老",
    "usage": "使用方法 用量 频率 使用顺序",
    "routine": "护肤步骤 洁面 水 精华 乳液 面霜 防晒 使用顺序",
    "compatibility": "产品搭配 成分冲突 搓泥 闷痘",
    "shade_color": "色号 妆效 粉底 口红 遮瑕 持妆",
    "authenticity_shelf_life": "正品 保质期 批次 防伪",
    "safety_allergy": "安全 过敏 刺痛 泛红 停用",
    "comparison": "商品对比 区别 推荐",
    "gift_sample": "赠品 小样 礼物 礼盒",
    "review": "评价 反馈 返现",
    "logistics": "物流 快递 发货 到货",
    "urge_shipment": "催发货 催仓 发货时间",
    "logistics_delay": "物流延误 快递未到 异常物流",
    "after_sale": "售后 退货 换货 退款",
    "invoice": "发票 开票 抬头 税号",
    "quality_issue": "质量问题 破损 漏液 补发 退换货",
}

INTENT_DOCUMENT_TYPES = {
    "price": ["product", "faq", "conversation"],
    "product_info": ["product", "faq", "conversation"],
    "skin_type": ["product", "faq", "conversation"],
    "skin_concern": ["product", "faq", "conversation"],
    "ingredient": ["product", "faq", "conversation"],
    "efficacy": ["product", "faq", "conversation"],
    "usage": ["product", "faq", "conversation"],
    "routine": ["product", "faq", "conversation"],
    "compatibility": ["product", "faq", "conversation"],
    "shade_color": ["product", "faq", "conversation"],
    "authenticity_shelf_life": ["product", "faq", "conversation"],
    "safety_allergy": ["product", "policy", "faq", "conversation"],
    "comparison": ["product", "faq", "conversation"],
    "gift_sample": ["product", "faq", "conversation"],
    "review": ["faq", "conversation"],
    "logistics": ["shipping", "faq", "conversation"],
    "urge_shipment": ["shipping", "faq", "conversation"],
    "logistics_delay": ["shipping", "faq", "conversation"],
    "after_sale": ["policy", "faq", "conversation"],
    "invoice": ["policy", "faq", "conversation"],
    "quality_issue": ["policy", "product", "faq", "conversation"],
}

KNOWN_BRANDS = ("ColourPop", "Fenty Beauty", "Glossier", "Almay")


def build_search_query(question: str, intent_result: dict[str, Any]) -> str:
    keywords = " ".join(intent_result.get("keywords") or [])
    hint = INTENT_SEARCH_HINTS.get(str(intent_result.get("intent_level1") or ""), "")
    return f"{hint} {keywords} {question}".strip()


def build_metadata_filters(intent_result: dict[str, Any], question: str) -> dict[str, Any]:
    intent = str(intent_result.get("intent_level1") or "")
    filters: dict[str, Any] = {}
    if intent in INTENT_DOCUMENT_TYPES:
        filters["document_type"] = INTENT_DOCUMENT_TYPES[intent]
    if any("\u4e00" <= character <= "\u9fff" for character in question):
        filters["language"] = ["zh"]
    elif re.search(r"[A-Za-z]", question):
        filters["language"] = ["en"]
    searchable = f"{question} {' '.join(intent_result.get('keywords') or [])}".casefold()
    brands = [brand for brand in KNOWN_BRANDS if brand.casefold() in searchable]
    if brands:
        filters["brand"] = brands
    return filters


DEFAULT_BENCHMARK = EVALUATION_DIR / "retrieval_benchmark.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "rag_retrieval_evaluation_results.json"
DEFAULT_DETAILS = ROOT / "reports" / "rag_retrieval_evaluation_details.jsonl"
DEFAULT_VECTOR_CACHE = EVALUATION_DIR / "retrieval_query_vectors.npz"
TOP_KS = (1, 3, 5, 10)


EXPERIMENTS = [
    {
        "id": "A_bm25_raw",
        "name": "BM25 / 原始问题",
        "mode": "bm25",
        "text_query": "raw",
        "vector_query": None,
        "mmr": False,
    },
    {
        "id": "B_knn_raw",
        "name": "kNN / 原始问题",
        "mode": "knn",
        "text_query": "raw",
        "vector_query": "raw",
        "mmr": False,
    },
    {
        "id": "C_knn_enhanced",
        "name": "kNN / 意图增强 Query",
        "mode": "knn",
        "text_query": "enhanced",
        "vector_query": "enhanced",
        "mmr": False,
    },
    {
        "id": "D_hybrid_raw",
        "name": "Hybrid / 原始问题",
        "mode": "hybrid",
        "text_query": "raw",
        "vector_query": "raw",
        "mmr": False,
    },
    {
        "id": "E_hybrid_enhanced",
        "name": "Hybrid / 意图增强 Query",
        "mode": "hybrid",
        "text_query": "enhanced",
        "vector_query": "enhanced",
        "mmr": False,
    },
    {
        "id": "F_rrf_enhanced",
        "name": "RRF / 意图增强 Query",
        "mode": "rrf",
        "text_query": "enhanced",
        "vector_query": "enhanced",
        "mmr": False,
    },
    {
        "id": "G_production_rrf_mmr",
        "name": "生产配置 RRF+MMR / BM25增强+向量原问",
        "mode": "rrf",
        "text_query": "enhanced",
        "vector_query": "raw",
        "mmr": True,
    },
    {
        "id": "H_hybrid_enhanced_dedup",
        "name": "Hybrid增强 / 语义去重",
        "mode": "hybrid",
        "text_query": "enhanced",
        "vector_query": "enhanced",
        "mmr": False,
        "postprocess": "dedup",
    },
    {
        "id": "I_rrf_production_dedup",
        "name": "RRF生产Query / 仅语义去重",
        "mode": "rrf",
        "text_query": "enhanced",
        "vector_query": "raw",
        "mmr": False,
        "postprocess": "dedup",
    },
]


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[position]


def hit_text(hit: dict[str, Any]) -> str:
    source = hit.get("_source", {})
    fields = (
        "title",
        "content",
        "question_text",
        "answer_text",
        "category",
        "intent",
        "topics",
        "brand",
    )
    return " ".join(str(source.get(field) or "") for field in fields).casefold()


def grade_relevance(hit: dict[str, Any], case: dict[str, Any]) -> tuple[int, list[bool]]:
    groups = case.get("required_groups") or []
    if not case.get("has_answer") or not groups:
        return 0, []
    text = hit_text(hit)
    matched = [any(str(term).casefold() in text for term in group) for group in groups]
    coverage = sum(matched) / len(matched)
    if coverage == 1.0:
        return 3, matched
    if coverage >= 2 / 3:
        return 2, matched
    if coverage > 0:
        return 1, matched
    return 0, matched


def duplicate_rate(hits: list[dict[str, Any]], k: int, threshold: float = 0.88) -> float:
    texts = [
        _normalize_for_dedup(str(hit.get("_source", {}).get("content") or ""))
        for hit in hits[:k]
    ]
    pairs = 0
    duplicates = 0
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if not texts[left] or not texts[right]:
                continue
            pairs += 1
            if SequenceMatcher(None, texts[left], texts[right]).ratio() >= threshold:
                duplicates += 1
    return duplicates / pairs if pairs else 0.0


def ndcg_at_k(relevances: list[int], k: int, *, has_answer: bool = True) -> float:
    actual = relevances[:k]
    dcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(actual, 1))
    # 弱监督集没有穷举全部相关 doc_id，但有答案查询至少应存在一条高度相关资料。
    # 若当前 Top-K 全是弱相关，不能再用“已召回结果自身”构造理想排序，否则会虚高为 1。
    ideal_pool = list(relevances)
    if has_answer and (not ideal_pool or max(ideal_pool) < 3):
        ideal_pool.append(3)
    ideal = sorted(ideal_pool, reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def group_recall_at_k(group_matches: list[list[bool]], group_count: int, k: int) -> float:
    if group_count == 0:
        return 0.0
    covered = [False] * group_count
    for matches in group_matches[:k]:
        for index, matched in enumerate(matches):
            covered[index] = covered[index] or matched
    return sum(covered) / group_count


def convert_hit(hit: dict[str, Any], relevance: int, matches: list[bool], rank: int) -> dict[str, Any]:
    source = hit.get("_source", {})
    return {
        "rank": rank,
        "doc_id": source.get("document_id") or hit.get("_id"),
        "document_type": source.get("document_type"),
        "title": source.get("title"),
        "score": float(hit.get("_score") or 0.0),
        "relevance": relevance,
        "group_matches": matches,
        "source_name": source.get("source_name"),
        "source_url": source.get("source_url"),
        "content_preview": str(source.get("content") or "")[:240],
    }


def search_experiment(
    experiment: dict[str, Any],
    case: dict[str, Any],
    raw_vector: list[float],
    enhanced_vector: list[float],
    *,
    base: str,
    index: str,
    top_k: int,
    num_candidates: int,
    candidate_k: int,
) -> tuple[list[dict[str, Any]], float]:
    intent = {
        "intent_level1": case["intent"],
        "intent_level2": case["intent"],
        "intent_level3": case["intent"],
        "keywords": case.get("keywords") or [],
    }
    raw_query = case["question"]
    enhanced_query = build_search_query(raw_query, intent)
    text_query = raw_query if experiment["text_query"] == "raw" else enhanced_query
    qvec = raw_vector if experiment["vector_query"] == "raw" else enhanced_vector
    filters = build_metadata_filters(intent, raw_query)
    postprocess = experiment.get("postprocess")
    retrieval_k = candidate_k if postprocess == "dedup" else top_k

    started = time.perf_counter()
    if experiment["mode"] == "rrf":
        hits = rrf_search(
            base,
            index,
            text_query,
            qvec,
            final_k=retrieval_k,
            candidate_k=candidate_k,
            rank_constant=60,
            num_candidates=num_candidates,
            use_mmr=bool(experiment["mmr"]),
            mmr_lambda=0.70,
            dedup_threshold=0.88,
            filters=filters,
        )
    else:
        hits = es_search(
            base,
            index,
            build_query(
                experiment["mode"],
                text_query,
                None if experiment["mode"] == "bm25" else qvec,
                k=retrieval_k,
                num_candidates=num_candidates,
                text_boost=0.2,
                vector_boost=1.0,
                filters=filters,
            ),
        )
    if postprocess == "dedup":
        hits = filter_near_duplicates(hits, threshold=0.88)[:top_k]
    latency_ms = (time.perf_counter() - started) * 1000
    return hits, latency_ms


def case_metrics(case: dict[str, Any], hits: list[dict[str, Any]], latency_ms: float) -> dict[str, Any]:
    graded = [grade_relevance(hit, case) for hit in hits]
    relevances = [item[0] for item in graded]
    group_matches = [item[1] for item in graded]
    relevant_ranks = [rank for rank, relevance in enumerate(relevances, 1) if relevance >= 2]
    first_relevant = relevant_ranks[0] if relevant_ranks else None
    metrics: dict[str, Any] = {
        "query_id": case["query_id"],
        "case_key": case["case_key"],
        "name": case["name"],
        "intent": case["intent"],
        "query_type": case["query_type"],
        "difficulty": case["difficulty"],
        "split": case["split"],
        "has_answer": bool(case["has_answer"]),
        "question": case["question"],
        "latency_ms": latency_ms,
        "top1_score": float(hits[0].get("_score") or 0.0) if hits else 0.0,
        "first_relevant_rank": first_relevant,
        "reciprocal_rank": 1.0 / first_relevant if first_relevant else 0.0,
        "hits": [
            convert_hit(hit, relevance, matches, rank)
            for rank, (hit, (relevance, matches)) in enumerate(zip(hits, graded), 1)
        ],
    }
    group_count = len(case.get("required_groups") or [])
    for k in TOP_KS:
        metrics[f"hit_at_{k}"] = float(any(relevance >= 2 for relevance in relevances[:k]))
        metrics[f"precision_at_{k}"] = sum(relevance >= 2 for relevance in relevances[:k]) / k
        metrics[f"group_recall_at_{k}"] = group_recall_at_k(group_matches, group_count, k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(
            relevances,
            k,
            has_answer=bool(case["has_answer"]),
        )
        metrics[f"duplicate_rate_at_{k}"] = duplicate_rate(hits, k)
    return metrics


def macro_f1(labels: list[bool], predictions: list[bool]) -> float:
    scores = []
    for target in (True, False):
        tp = sum(label == target and prediction == target for label, prediction in zip(labels, predictions))
        fp = sum(label != target and prediction == target for label, prediction in zip(labels, predictions))
        fn = sum(label == target and prediction != target for label, prediction in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def calibrate_answer_threshold(details: list[dict[str, Any]]) -> float:
    calibration = [detail for detail in details if detail["split"] == "calibration"]
    scores = sorted({float(detail["top1_score"]) for detail in calibration})
    if not scores:
        return 0.0
    candidates = [scores[0] - 1e-9, scores[-1] + 1e-9]
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    labels = [bool(detail["has_answer"]) for detail in calibration]
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        predictions = [float(detail["top1_score"]) >= threshold for detail in calibration]
        score = macro_f1(labels, predictions)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold


def no_answer_metrics(details: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    test = [detail for detail in details if detail["split"] == "test"]
    actual = [not bool(detail["has_answer"]) for detail in test]
    predicted = [float(detail["top1_score"]) < threshold for detail in test]
    tp = sum(label and prediction for label, prediction in zip(actual, predicted))
    fp = sum(not label and prediction for label, prediction in zip(actual, predicted))
    fn = sum(label and not prediction for label, prediction in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "false_support_rate": 1.0 - recall,
        "test_no_answer_count": sum(actual),
    }


def aggregate_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [detail for detail in details if detail["has_answer"]]
    latencies = [float(detail["latency_ms"]) for detail in details]
    summary: dict[str, Any] = {
        "query_count": len(details),
        "answered_query_count": len(answered),
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "mrr_at_10": statistics.mean(detail["reciprocal_rank"] for detail in answered) if answered else 0.0,
    }
    for k in TOP_KS:
        for metric in ("hit", "precision", "group_recall", "ndcg", "duplicate_rate"):
            key = f"{metric}_at_{k}"
            summary[key] = statistics.mean(float(detail[key]) for detail in answered) if answered else 0.0
    threshold = calibrate_answer_threshold(details)
    summary["no_answer"] = no_answer_metrics(details, threshold)
    return summary


def slice_summary(details: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        if detail["has_answer"]:
            groups[str(detail[field])].append(detail)
    return {
        name: {
            "count": len(items),
            "hit_at_3": statistics.mean(item["hit_at_3"] for item in items),
            "group_recall_at_3": statistics.mean(item["group_recall_at_3"] for item in items),
            "mrr_at_10": statistics.mean(item["reciprocal_rank"] for item in items),
            "ndcg_at_5": statistics.mean(item["ndcg_at_5"] for item in items),
            "duplicate_rate_at_3": statistics.mean(item["duplicate_rate_at_3"] for item in items),
        }
        for name, items in sorted(groups.items())
    }


def assign_errors(details: list[dict[str, Any]], threshold: float) -> None:
    for detail in details:
        errors = []
        if not detail["has_answer"]:
            if detail["top1_score"] >= threshold:
                errors.append("no_answer_false_support")
        else:
            if not detail["hit_at_10"]:
                errors.append("no_hit")
                if detail["query_type"] == "multi_turn":
                    errors.append("context_missing")
            elif (detail["first_relevant_rank"] or 99) > 3:
                errors.append("low_rank")
            if detail["group_recall_at_3"] < 1.0:
                errors.append("incomplete_information")
            if detail["duplicate_rate_at_3"] > 0.20:
                errors.append("duplicate_results")
        detail["error_types"] = errors


def distribution(cases: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[str(case[field])] += 1
    return dict(sorted(counts.items()))


def install_persistent_http_client() -> httpx.Client:
    """为测评注入连接复用客户端，不修改生产检索模块源代码。"""
    client = httpx.Client(timeout=60.0, trust_env=False)

    def persistent_search(
        base: str,
        index: str,
        body: dict[str, Any],
        auth: str | None = None,
        api_key: str | None = None,
        *,
        insecure: bool = False,
    ) -> list[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "ApiKey " + api_key
        elif auth:
            headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
        response = client.post(
            f"{base.rstrip('/')}/{index}/_search",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        return response.json().get("hits", {}).get("hits", [])

    es_search_module.es_search = persistent_search
    globals()["es_search"] = persistent_search
    return client


def encode_texts_bucketed(
    encoder: Any,
    texts: list[str],
    *,
    batch_size: int = 4,
    max_len: int = 128,
    label: str,
) -> tuple[np.ndarray, float]:
    """按文本长度分桶，避免长 Query 让整批产生大量 padding。"""
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    output: list[np.ndarray | None] = [None] * len(texts)
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        matrix = encoder.encode(
            [PREFIX + texts[index] for index in indices],
            batch_size=batch_size,
            max_len=max_len,
        )
        for index, vector in zip(indices, matrix):
            output[index] = vector
        completed = min(start + batch_size, len(order))
        if completed % 12 == 0 or completed == len(order):
            print(f"  {label} {completed}/{len(order)}", flush=True)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return np.stack([vector for vector in output if vector is not None]), elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--url", default="http://127.0.0.1:9200")
    parser.add_argument("--index", default="customer_service_knowledge_v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=200)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--details-output", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--vector-cache", type=Path, default=DEFAULT_VECTOR_CACHE)
    parser.add_argument(
        "--cache-ssl-context",
        action="store_true",
        help="测评模拟：复用 SSLContext，验证当前逐请求创建上下文的性能影响",
    )
    parser.add_argument(
        "--experiments",
        default="",
        help="逗号分隔的实验 ID；留空运行全部实验",
    )
    parser.add_argument(
        "--persistent-http-client",
        action="store_true",
        help="测评模拟：使用 httpx.Client 连接池替代逐请求 urllib",
    )
    args = parser.parse_args()

    if args.top_k < max(TOP_KS):
        raise ValueError(f"top-k 至少需要 {max(TOP_KS)}")
    cases = load_benchmark(args.benchmark)
    if args.cache_ssl_context:
        es_search_module._ssl_context = lru_cache(maxsize=2)(es_search_module._ssl_context)
    persistent_client = install_persistent_http_client() if args.persistent_http_client else None
    selected_ids = {item.strip() for item in args.experiments.split(",") if item.strip()}
    selected_experiments = [
        experiment for experiment in EXPERIMENTS
        if not selected_ids or experiment["id"] in selected_ids
    ]
    if selected_ids - {experiment["id"] for experiment in selected_experiments}:
        raise ValueError(f"未知实验 ID: {sorted(selected_ids - {experiment['id'] for experiment in selected_experiments})}")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.details_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载基准 {len(cases)} 条，开始批量预编码 Query...")
    enhanced_queries = []
    for case in cases:
        intent = {
            "intent_level1": case["intent"],
            "intent_level2": case["intent"],
            "intent_level3": case["intent"],
            "keywords": case.get("keywords") or [],
        }
        enhanced_queries.append(build_search_query(case["question"], intent))

    expected_ids = [case["query_id"] for case in cases]
    encoder = get_encoder()
    cache_loaded = False
    if args.vector_cache.exists():
        cached = np.load(args.vector_cache)
        cached_ids = [str(value) for value in cached["query_ids"].tolist()]
        if cached_ids == expected_ids:
            raw_matrix = cached["raw"]
            enhanced_matrix = cached["enhanced"]
            raw_batch_ms = 0.0
            enhanced_batch_ms = 0.0
            cache_loaded = True
            print(f"已复用 Query 向量缓存：{args.vector_cache}", flush=True)
    if not cache_loaded:
        raw_matrix, raw_batch_ms = encode_texts_bucketed(
            encoder,
            [case["question"] for case in cases],
            batch_size=4,
            max_len=128,
            label="原始Query",
        )
        enhanced_matrix, enhanced_batch_ms = encode_texts_bucketed(
            encoder,
            enhanced_queries,
            batch_size=4,
            max_len=128,
            label="增强Query",
        )
        args.vector_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.vector_cache,
            query_ids=np.array(expected_ids),
            raw=raw_matrix,
            enhanced=enhanced_matrix,
        )

    vectors: dict[str, dict[str, Any]] = {}
    for case, raw_vector, enhanced_vector in zip(cases, raw_matrix, enhanced_matrix):
        vectors[case["query_id"]] = {
            "raw": [round(float(value), 6) for value in raw_vector],
            "enhanced": [round(float(value), 6) for value in enhanced_vector],
        }
    print(
        f"批量编码完成：原始 Query {raw_batch_ms:.1f}ms，"
        f"增强 Query {enhanced_batch_ms:.1f}ms"
    )

    single_query_latencies = []
    for case in cases[:3]:
        started = time.perf_counter()
        encoder.encode([PREFIX + case["question"]], batch_size=1)
        single_query_latencies.append((time.perf_counter() - started) * 1000)
    encoding_benchmark = {
        "batch_size": 4,
        "query_count_per_batch_run": len(cases),
        "raw_batch_total_ms": raw_batch_ms,
        "enhanced_batch_total_ms": enhanced_batch_ms,
        "batch_amortized_ms_per_query": (raw_batch_ms + enhanced_batch_ms) / (2 * len(cases)),
        "cache_loaded": cache_loaded,
        "single_query_sample_count": len(single_query_latencies),
        "single_query_avg_ms": statistics.mean(single_query_latencies),
        "single_query_p95_ms": percentile(single_query_latencies, 0.95),
    }

    all_results = []
    all_detail_rows = []
    for experiment in selected_experiments:
        print(f"执行 {experiment['id']} ...")
        details = []
        for number, case in enumerate(cases, 1):
            vector_info = vectors[case["query_id"]]
            hits, es_latency_ms = search_experiment(
                experiment,
                case,
                vector_info["raw"],
                vector_info["enhanced"],
                base=args.url.rstrip("/"),
                index=args.index,
                top_k=args.top_k,
                num_candidates=args.num_candidates,
                candidate_k=args.candidate_k,
            )
            encode_latency_ms = 0.0
            detail = case_metrics(case, hits, es_latency_ms)
            detail["experiment_id"] = experiment["id"]
            detail["es_latency_ms"] = es_latency_ms
            detail["encode_latency_ms"] = encode_latency_ms
            details.append(detail)
            if number % 20 == 0 or number == len(cases):
                print(f"  {experiment['id']} {number}/{len(cases)}")

        aggregate = aggregate_details(details)
        assign_errors(details, aggregate["no_answer"]["threshold"])
        result = {
            "experiment": experiment,
            "summary": aggregate,
            "slices": {
                "intent": slice_summary(details, "intent"),
                "query_type": slice_summary(details, "query_type"),
                "difficulty": slice_summary(details, "difficulty"),
            },
            "error_counts": dict(sorted(
                (error, sum(error in detail["error_types"] for detail in details))
                for error in {
                    error
                    for detail in details
                    for error in detail["error_types"]
                }
            )),
        }
        all_results.append(result)
        all_detail_rows.extend(details)

    payload = {
        "evaluation_type": "phase1_weak_supervision",
        "benchmark": {
            "path": str(args.benchmark),
            "query_count": len(cases),
            "distribution": {
                "intent": distribution(cases, "intent"),
                "query_type": distribution(cases, "query_type"),
                "difficulty": distribution(cases, "difficulty"),
                "has_answer": distribution(cases, "has_answer"),
            },
        },
        "knowledge_snapshot": {
            "index": args.index,
            "document_count": 25520,
            "embedding_model": "bge-small-zh-v1.5",
            "vector_dims": 512,
            "mapping": "es_store/es_mapping.json",
            "query_prefix": "为这个句子生成表示以用于检索相关文章：",
        },
        "encoding_benchmark": encoding_benchmark,
        "metric_note": (
            "Recall 使用 required_groups 的信息点覆盖率（Group Recall），相关性等级由关键词组规则弱监督生成；"
            "正式验收需替换为人工 doc_id 四级标注。"
        ),
        "runtime_optimization": {
            "ssl_context_cached": bool(args.cache_ssl_context),
            "persistent_http_client": bool(args.persistent_http_client),
            "production_code_modified": False,
        },
        "experiments": all_results,
    }
    args.summary_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.details_output.open("w", encoding="utf-8") as file:
        for row in all_detail_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n| 实验 | Hit@3 | Group Recall@3 | MRR@10 | nDCG@5 | Precision@3 | Dup@3 | P95(ms) | 无答案Recall |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in all_results:
        summary = result["summary"]
        print(
            f"| {result['experiment']['id']} "
            f"| {summary['hit_at_3']:.3f} "
            f"| {summary['group_recall_at_3']:.3f} "
            f"| {summary['mrr_at_10']:.3f} "
            f"| {summary['ndcg_at_5']:.3f} "
            f"| {summary['precision_at_3']:.3f} "
            f"| {summary['duplicate_rate_at_3']:.3f} "
            f"| {summary['p95_latency_ms']:.1f} "
            f"| {summary['no_answer']['recall']:.3f} |"
        )
    print(f"\n汇总结果 -> {args.summary_output}")
    print(f"逐条明细 -> {args.details_output}")
    if persistent_client is not None:
        persistent_client.close()


if __name__ == "__main__":
    main()
