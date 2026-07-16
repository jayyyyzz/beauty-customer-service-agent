# -*- coding: utf-8 -*-
"""统一知识库的 BM25、向量和混合检索，支持元数据过滤。"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vector_store"))
from bge_numpy import BgeEncoder


MODEL = str(HERE.parent / "models" / "bge-small-zh-v1.5")
PREFIX = "为这个句子生成表示以用于检索相关文章："
FILTER_FIELDS = {"document_type", "brand", "category", "intent", "language", "need_human"}
SOURCE_FIELDS = [
    "document_id", "document_type", "source_record_id", "title", "content",
    "question_text", "answer_text", "source_name", "source_url", "brand",
    "category", "intent", "topics", "language", "need_human", "updated_at",
    "content_hash",
]

DEDUP_FILLERS = (
    "买家: 你好",
    "买家: 在吗",
    "买家: 亲，在不在",
    "买家: 客服客服",
    "买家: hello有人吗",
    "买家: ？？？",
    "买家: 回个话啊",
)


@lru_cache(maxsize=1)
def get_encoder() -> BgeEncoder:
    """进程内复用模型，避免每个问题都重新加载 BGE。"""
    return BgeEncoder(MODEL)


@lru_cache(maxsize=512)
def _encode_query_cached(query: str) -> tuple[float, ...]:
    """缓存高频查询向量；模型本身由 get_encoder 进程内单例复用。"""
    vector = get_encoder().encode([PREFIX + query], batch_size=1)[0]
    return tuple(round(float(value), 6) for value in vector)


def encode_query(query: str) -> list[float]:
    return list(_encode_query_cached(query.strip()))


@lru_cache(maxsize=2)
def _get_http_client(insecure: bool) -> httpx.Client:
    """复用连接池并忽略系统代理，避免本机 ES 请求的固定握手开销。"""
    return httpx.Client(
        timeout=60.0,
        verify=not insecure,
        trust_env=False,
    )


def _parse_response(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return {"error": "empty response"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "non-json response", "body": text[:1000]}


def es_search(
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
    try:
        response = _get_http_client(insecure).post(
            f"{base}/{index}/_search",
            json=body,
            headers=headers,
        )
        result = _parse_response(response.content)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        result = _parse_response(exc.response.content)
        raise RuntimeError(
            f"ES search failed: HTTP {exc.response.status_code} {result}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"ES search failed: {exc}") from exc
    return result.get("hits", {}).get("hits", [])


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[str, list[dict[str, Any]], float]],
    *,
    final_k: int = 5,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """按排名融合多路召回结果，避免直接比较不同召回器的原始分数。"""
    merged: dict[str, dict[str, Any]] = {}

    for channel, hits, weight in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            source = hit.get("_source", {})
            document_id = source.get("document_id") or hit.get("_id")
            if not document_id:
                continue

            entry = merged.setdefault(
                document_id,
                {
                    "hit": hit,
                    "rrf_score": 0.0,
                    "ranks": {},
                    "raw_scores": {},
                },
            )
            entry["rrf_score"] += weight / (rank_constant + rank)
            entry["ranks"][channel] = rank
            entry["raw_scores"][channel] = float(hit.get("_score") or 0.0)

    ordered = sorted(
        merged.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    results = []
    for entry in ordered[:final_k]:
        hit = dict(entry["hit"])
        hit["_score"] = entry["rrf_score"]
        hit["_rrf"] = {
            "ranks": entry["ranks"],
            "raw_scores": entry["raw_scores"],
        }
        results.append(hit)
    return results


def _normalize_for_dedup(text: str) -> str:
    """去除寒暄、空白和标点，保留真正影响业务含义的文字。"""
    normalized = text.lower()
    for filler in DEDUP_FILLERS:
        normalized = normalized.replace(filler.lower(), "")
    return re.sub(r"[^\u4e00-\u9fffa-z0-9]+", "", normalized)


def filter_near_duplicates(
    hits: list[dict[str, Any]],
    *,
    threshold: float = 0.88,
) -> list[dict[str, Any]]:
    """按当前排名保留首条，过滤仅有寒暄、标点等差异的近重复结果。"""
    kept: list[dict[str, Any]] = []
    normalized_texts: list[str] = []

    for hit in hits:
        content = str(hit.get("_source", {}).get("content") or "")
        normalized = _normalize_for_dedup(content)
        if not normalized:
            continue
        if any(
            SequenceMatcher(None, normalized, previous).ratio() >= threshold
            for previous in normalized_texts
        ):
            continue
        kept.append(hit)
        normalized_texts.append(normalized)
    return kept


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def maximal_marginal_relevance(
    hits: list[dict[str, Any]],
    query_vector: list[float],
    *,
    final_k: int = 5,
    lambda_mult: float = 0.70,
) -> list[dict[str, Any]]:
    """在查询相关性和候选之间的差异性之间取得平衡。"""
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("MMR lambda 必须在 0 到 1 之间")

    remaining = []
    for hit in hits:
        vector = hit.get("_source", {}).get("content_vector") or []
        if vector:
            remaining.append((hit, vector))

    selected: list[dict[str, Any]] = []
    selected_vectors: list[list[float]] = []

    while remaining and len(selected) < final_k:
        best_index = 0
        best_score = float("-inf")
        best_relevance = 0.0
        best_redundancy = 0.0

        for index, (hit, vector) in enumerate(remaining):
            relevance = _cosine_similarity(query_vector, vector)
            redundancy = max(
                (_cosine_similarity(vector, chosen) for chosen in selected_vectors),
                default=0.0,
            )
            mmr_score = (
                lambda_mult * relevance
                - (1.0 - lambda_mult) * redundancy
            )
            if mmr_score > best_score:
                best_index = index
                best_score = mmr_score
                best_relevance = relevance
                best_redundancy = redundancy

        hit, vector = remaining.pop(best_index)
        chosen = dict(hit)
        chosen["_mmr"] = {
            "score": best_score,
            "relevance": best_relevance,
            "redundancy": best_redundancy,
        }
        selected.append(chosen)
        selected_vectors.append(vector)

    return selected


def _strip_source_vectors(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MMR 完成后移除向量，避免把大字段传给上层 Agent。"""
    cleaned = []
    for hit in hits:
        item = dict(hit)
        source = dict(item.get("_source", {}))
        source.pop("content_vector", None)
        item["_source"] = source
        cleaned.append(item)
    return cleaned


def rrf_search(
    base: str,
    index: str,
    query: str,
    qvec: list[float],
    *,
    final_k: int = 5,
    candidate_k: int = 30,
    rank_constant: int = 60,
    num_candidates: int = 200,
    use_mmr: bool = False,
    mmr_lambda: float = 0.70,
    dedup_threshold: float = 0.88,
    filters: dict[str, Any] | None = None,
    auth: str | None = None,
    api_key: str | None = None,
    insecure: bool = False,
) -> list[dict[str, Any]]:
    """分别执行 BM25 和 kNN，再使用 RRF 合并排名。"""
    bm25_body = build_query(
        "bm25",
        query,
        None,
        k=candidate_k,
        filters=filters,
        include_vector=use_mmr,
    )
    knn_body = build_query(
        "knn",
        query,
        qvec,
        k=candidate_k,
        num_candidates=num_candidates,
        filters=filters,
        include_vector=use_mmr,
    )

    # 两路召回互不依赖，并行执行可将等待时间从两次请求之和降到较慢一路的耗时。
    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(
            es_search,
            base,
            index,
            bm25_body,
            auth,
            api_key,
            insecure=insecure,
        )
        knn_future = executor.submit(
            es_search,
            base,
            index,
            knn_body,
            auth,
            api_key,
            insecure=insecure,
        )
        bm25_hits = bm25_future.result()
        knn_hits = knn_future.result()
    fused = reciprocal_rank_fusion(
        [
            ("bm25", bm25_hits, 1.0),
            ("knn", knn_hits, 1.0),
        ],
        final_k=max(candidate_k * 2, final_k),
        rank_constant=rank_constant,
    )
    if not use_mmr:
        return fused[:final_k]

    deduplicated = filter_near_duplicates(
        fused,
        threshold=dedup_threshold,
    )
    selected = maximal_marginal_relevance(
        deduplicated,
        qvec,
        final_k=final_k,
        lambda_mult=mmr_lambda,
    )
    return _strip_source_vectors(selected)


def _filter_clauses(filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    clauses = []
    for field, raw_value in (filters or {}).items():
        if field not in FILTER_FIELDS or raw_value in (None, "", []):
            continue
        values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
        values = list(values)
        if field == "need_human" and len(values) == 1:
            clauses.append({"term": {field: bool(values[0])}})
        else:
            clauses.append({"terms": {field: values}})
    return clauses


def build_query(
    mode: str,
    query: str,
    qvec: list[float] | None,
    k: int = 3,
    num_candidates: int = 100,
    text_boost: float = 0.2,
    vector_boost: float = 1.0,
    vector_field: str = "content_vector",
    filters: dict[str, Any] | None = None,
    include_vector: bool = False,
) -> dict[str, Any]:
    clauses = _filter_clauses(filters)
    text_query: dict[str, Any] = {
        "multi_match": {
            "query": query,
            "fields": ["title^3", "question_text^2", "content", "answer_text", "topics^2"],
            "boost": text_boost,
        }
    }
    if clauses:
        text_query = {"bool": {"must": [text_query], "filter": clauses}}

    source_fields = SOURCE_FIELDS + (["content_vector"] if include_vector else [])
    source = {"_source": source_fields, "size": k}
    if mode == "bm25":
        return {**source, "query": text_query}
    if qvec is None:
        raise ValueError(f"{mode} 模式必须提供查询向量")

    knn: dict[str, Any] = {
        "field": vector_field,
        "query_vector": qvec,
        "k": k,
        "num_candidates": max(num_candidates, k),
    }
    if clauses:
        knn["filter"] = {"bool": {"filter": clauses}}

    if mode == "knn":
        return {**source, "knn": knn}
    if mode != "hybrid":
        raise ValueError(f"不支持的检索模式: {mode}")
    return {
        **source,
        "query": text_query,
        "knn": {**knn, "boost": vector_boost},
    }


def _csv_values(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument(
        "--mode",
        choices=["knn", "bm25", "hybrid", "rrf", "rrf_mmr"],
        default="rrf_mmr",
    )
    parser.add_argument("--url", default=os.getenv("ES_URL", "http://127.0.0.1:9200"))
    parser.add_argument("--api-key", default=os.getenv("ES_API_KEY", ""))
    parser.add_argument("--user", default=os.getenv("ES_USER", ""))
    parser.add_argument("--password", default=os.getenv("ES_PASSWORD", ""))
    parser.add_argument("--index", default=os.getenv("ES_INDEX", "customer_service_knowledge_v1"))
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--document-type", help="逗号分隔: conversation,product,faq,policy,shipping")
    parser.add_argument("--brand")
    parser.add_argument("--category")
    parser.add_argument("--intent")
    parser.add_argument("--language")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=200)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
        help="每路召回参与融合的候选数量",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF 排名常数",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.70,
        help="MMR相关性权重，越大越偏向相关性",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.88,
        help="近重复阈值，越低去重越激进",
    )
    parser.add_argument("--text-boost", type=float, default=0.2)
    parser.add_argument("--vector-boost", type=float, default=1.0)
    args = parser.parse_args()

    query = " ".join(args.query)
    filters = {
        "document_type": _csv_values(args.document_type),
        "brand": _csv_values(args.brand),
        "category": _csv_values(args.category),
        "intent": _csv_values(args.intent),
        "language": _csv_values(args.language),
    }
    qvec = None if args.mode == "bm25" else encode_query(query)
    auth = f"{args.user}:{args.password}" if args.user else None
    if args.mode in {"rrf", "rrf_mmr"}:
        hits = rrf_search(
            args.url.rstrip("/"),
            args.index,
            query,
            qvec,
            final_k=args.k,
            candidate_k=args.candidate_k,
            rank_constant=args.rrf_k,
            num_candidates=args.num_candidates,
            use_mmr=args.mode == "rrf_mmr",
            mmr_lambda=args.mmr_lambda,
            dedup_threshold=args.dedup_threshold,
            filters=filters,
            auth=auth,
            api_key=args.api_key or None,
            insecure=args.insecure,
        )
    else:
        hits = es_search(
            args.url.rstrip("/"),
            args.index,
            build_query(
                args.mode,
                query,
                qvec,
                k=args.k,
                num_candidates=args.num_candidates,
                text_boost=args.text_boost,
                vector_boost=args.vector_boost,
                filters=filters,
            ),
            auth,
            args.api_key or None,
            insecure=args.insecure,
        )
    print(f"查询: {query} [mode={args.mode}, filters={filters}]")
    for position, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        print(
            f"  [{position}] score={hit.get('_score', 0):.4f} "
            f"type={source.get('document_type')} title={source.get('title')}"
        )
        rrf_info = hit.get("_rrf")
        if rrf_info:
            print(
                f"      RRF排名: {rrf_info['ranks']} "
                f"原始分数: {rrf_info['raw_scores']}"
            )
        mmr_info = hit.get("_mmr")
        if mmr_info:
            print(
                "      MMR: "
                f"score={mmr_info['score']:.4f} "
                f"relevance={mmr_info['relevance']:.4f} "
                f"redundancy={mmr_info['redundancy']:.4f}"
            )
        print(f"      来源: {source.get('source_name') or '内部知识库'} {source.get('source_url') or ''}")
        print(f"      {source.get('content', '')[:160].replace(chr(10), ' | ')}")


if __name__ == "__main__":
    main()
