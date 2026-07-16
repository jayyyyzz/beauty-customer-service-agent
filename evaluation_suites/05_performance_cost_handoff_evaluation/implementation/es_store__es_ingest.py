# -*- coding: utf-8 -*-
"""多源知识库增量入库：对话、商品、FAQ、政策和物流规则。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vector_store"))

from bge_numpy import BgeEncoder
from knowledge_sources import SUPPORTED_SOURCES, KnowledgeDocument, load_documents


MODEL = HERE.parent / "models" / "bge-small-zh-v1.5"
MAPPING_FILE = HERE / "es_mapping.json"
DOCUMENT_PREFIX = "为这个句子生成表示以用于检索相关文章："
EMBEDDING_VERSION = "bge-small-zh-v1.5-question-focused-v2"


@lru_cache(maxsize=2)
def _get_http_client(insecure: bool) -> httpx.Client:
    """复用 Elasticsearch 入库连接，避免批次间重复建立网络栈。"""
    return httpx.Client(
        timeout=180.0,
        verify=not insecure,
        trust_env=False,
    )


def _parse_response(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "non-json response", "body": text[:1000]}


def request(
    method: str,
    url: str,
    body: str | bytes | None = None,
    *,
    ndjson: bool = False,
    auth: str | None = None,
    api_key: str | None = None,
    insecure: bool = False,
) -> tuple[int, dict]:
    headers = {"Content-Type": "application/x-ndjson" if ndjson else "application/json"}
    if api_key:
        headers["Authorization"] = "ApiKey " + api_key
    elif auth:
        headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
    data = body if isinstance(body, bytes) else body.encode("utf-8") if body else None
    try:
        response = _get_http_client(insecure).request(
            method,
            url,
            content=data,
            headers=headers,
        )
        return response.status_code, _parse_response(response.content)
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _retrieval_text_from_source(source: dict) -> str:
    document_type = str(source.get("document_type") or "")
    question_text = str(source.get("question_text") or "")
    topics = source.get("topics") or []
    if document_type in {"faq", "conversation"} and question_text:
        parts = [source.get("title"), question_text, " ".join(topics)]
    else:
        parts = [source.get("title"), source.get("content")]
    return "\n".join(str(part) for part in parts if part)


def _retrieval_text(document: KnowledgeDocument) -> str:
    return _retrieval_text_from_source(document.to_source())


def _embedding_input_hash(document: KnowledgeDocument) -> str:
    text = DOCUMENT_PREFIX + _retrieval_text(document)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_embedding_input_hash(source: dict) -> str:
    stored = str(source.get("embedding_input_hash") or "")
    if stored:
        return stored
    text = DOCUMENT_PREFIX + _retrieval_text_from_source(source)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector(values: Iterable[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def _parse_sources(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(SUPPORTED_SOURCES)
    sources = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(sources) - set(SUPPORTED_SOURCES))
    if invalid:
        raise ValueError(f"不支持的数据源: {', '.join(invalid)}")
    return sources


def _existing_states(
    base: str,
    index: str,
    document_ids: list[str],
    *,
    auth: str | None,
    api_key: str | None,
    insecure: bool,
) -> dict[str, dict[str, str]]:
    states: dict[str, dict[str, str]] = {}
    for batch in _batches(document_ids, 1000):
        # Elasticsearch 8.x 的 _mget 顶层请求只接受 docs/ids；
        # 这里读取完整 _source 后仅取 content_hash，保持各小版本兼容。
        body = json.dumps({"ids": batch})
        code, response = request(
            "POST",
            f"{base}/{index}/_mget?_source_includes="
            "content_hash,embedding_version,embedding_input_hash,"
            "document_type,title,question_text,topics,content",
            body,
            auth=auth,
            api_key=api_key,
            insecure=insecure,
        )
        if code != 200:
            raise RuntimeError(f"读取现有文档哈希失败: HTTP {code} {response}")
        for item in response.get("docs", []):
            if item.get("found"):
                source = item.get("_source", {})
                states[item["_id"]] = {
                    "content_hash": str(source.get("content_hash") or ""),
                    "embedding_version": str(source.get("embedding_version") or ""),
                    "embedding_input_hash": _source_embedding_input_hash(source),
                }
    return states


def _plan_incremental_updates(
    documents: list[KnowledgeDocument],
    existing: dict[str, dict[str, str]],
) -> tuple[list[KnowledgeDocument], list[KnowledgeDocument]]:
    """区分需要重新编码的文档和只需刷新元数据的文档。"""
    reembed: list[KnowledgeDocument] = []
    metadata_only: list[KnowledgeDocument] = []
    for document in documents:
        state = existing.get(document.document_id)
        input_hash = _embedding_input_hash(document)
        if (
            not state
            or state.get("embedding_version") != EMBEDDING_VERSION
            or state.get("embedding_input_hash") != input_hash
        ):
            reembed.append(document)
        elif state.get("content_hash") != document.content_hash:
            metadata_only.append(document)
    return reembed, metadata_only


def _bulk_index(
    base: str,
    index: str,
    documents: list[KnowledgeDocument],
    *,
    encoder: BgeEncoder,
    encode_batch_size: int,
    auth: str | None,
    api_key: str | None,
    insecure: bool,
) -> int:
    inputs = [DOCUMENT_PREFIX + _retrieval_text(document) for document in documents]
    vectors = encoder.encode(inputs, batch_size=encode_batch_size)
    lines = []
    for document, vector in zip(documents, vectors):
        source = document.to_source()
        source["embedding_version"] = EMBEDDING_VERSION
        source["embedding_input_hash"] = _embedding_input_hash(document)
        source["content_vector"] = _vector(vector)
        lines.append(json.dumps({"index": {"_index": index, "_id": document.document_id}}, ensure_ascii=False))
        lines.append(json.dumps(source, ensure_ascii=False, separators=(",", ":")))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    code, response = request(
        "POST",
        f"{base}/_bulk",
        body,
        ndjson=True,
        auth=auth,
        api_key=api_key,
        insecure=insecure,
    )
    if code != 200 or response.get("errors"):
        failures = [
            item for item in response.get("items", [])
            if item.get("index", {}).get("error")
        ][:3]
        raise RuntimeError(f"批量入库失败: HTTP {code}; 示例={failures or response}")
    return len(response.get("items", []))


def _bulk_update_metadata(
    base: str,
    index: str,
    documents: list[KnowledgeDocument],
    *,
    auth: str | None,
    api_key: str | None,
    insecure: bool,
) -> int:
    """更新非向量字段并保留已有向量，避免无意义地重新编码。"""
    lines = []
    for document in documents:
        source = document.to_source()
        source["embedding_version"] = EMBEDDING_VERSION
        source["embedding_input_hash"] = _embedding_input_hash(document)
        lines.append(json.dumps({"update": {"_index": index, "_id": document.document_id}}, ensure_ascii=False))
        lines.append(json.dumps({"doc": source}, ensure_ascii=False, separators=(",", ":")))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    code, response = request(
        "POST",
        f"{base}/_bulk",
        body,
        ndjson=True,
        auth=auth,
        api_key=api_key,
        insecure=insecure,
    )
    if code != 200 or response.get("errors"):
        failures = [
            item for item in response.get("items", [])
            if item.get("update", {}).get("error")
        ][:3]
        raise RuntimeError(f"批量更新元数据失败: HTTP {code}; 示例={failures or response}")
    return len(response.get("items", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="将多源客服知识增量写入 Elasticsearch")
    parser.add_argument("--url", default=os.getenv("ES_URL", "http://127.0.0.1:9200"))
    parser.add_argument("--api-key", default=os.getenv("ES_API_KEY", ""))
    parser.add_argument("--user", default=os.getenv("ES_USER", ""))
    parser.add_argument("--password", default=os.getenv("ES_PASSWORD", ""))
    parser.add_argument("--index", default=os.getenv("ES_INDEX", "customer_service_knowledge_v1"))
    parser.add_argument("--sources", default="all", help="all 或逗号分隔: conversation,product,faq,policy,shipping")
    parser.add_argument("--recreate", action="store_true", help="删除旧索引并全量重建")
    parser.add_argument("--dry-run", action="store_true", help="只读取和校验数据，不连接 ES")
    parser.add_argument("--insecure", action="store_true", help="仅限本地自签名 HTTPS，关闭证书校验")
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--bulk-size", type=int, default=128)
    args = parser.parse_args()

    sources = _parse_sources(args.sources)
    documents = load_documents(sources)
    counts = Counter(document.document_type for document in documents)
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        duplicates = [doc_id for doc_id, count in Counter(ids).items() if count > 1][:10]
        raise RuntimeError(f"发现重复 document_id: {duplicates}")

    print(f"已加载 {len(documents)} 条知识文档: {dict(counts)}")
    if args.dry_run:
        print("dry-run 完成：字段转换、文档 ID 和数据源校验通过，未连接 Elasticsearch。")
        return

    base = args.url.rstrip("/")
    auth = f"{args.user}:{args.password}" if args.user else None
    api_key = args.api_key or None
    code, info = request("GET", base, auth=auth, api_key=api_key, insecure=args.insecure)
    if code != 200:
        raise RuntimeError(f"无法连接 Elasticsearch: HTTP {code} {info}")
    print(f"Elasticsearch {info.get('version', {}).get('number', 'unknown')} 连接成功")

    index_url = f"{base}/{args.index}"
    if args.recreate:
        request("DELETE", index_url, auth=auth, api_key=api_key, insecure=args.insecure)
    code, _ = request("GET", index_url, auth=auth, api_key=api_key, insecure=args.insecure)
    if code != 200:
        mapping = MAPPING_FILE.read_text(encoding="utf-8")
        code, response = request(
            "PUT", index_url, mapping, auth=auth, api_key=api_key, insecure=args.insecure
        )
        if code not in {200, 201}:
            raise RuntimeError(f"创建索引失败: HTTP {code} {response}")
        print(f"索引 {args.index} 创建成功")

    existing = _existing_states(
        base,
        args.index,
        ids,
        auth=auth,
        api_key=api_key,
        insecure=args.insecure,
    )
    reembed, metadata_only = _plan_incremental_updates(documents, existing)
    unchanged = len(documents) - len(reembed) - len(metadata_only)
    print(
        "增量比较完成: "
        f"需重新编码 {len(reembed)} 条，仅更新元数据 {len(metadata_only)} 条，"
        f"未变化 {unchanged} 条"
    )

    if reembed:
        print("加载 BGE 编码器，只为新增/变更文档生成向量...")
        encoder = BgeEncoder(str(MODEL))
        indexed = 0
        for number, batch in enumerate(_batches(reembed, args.bulk_size), start=1):
            indexed += _bulk_index(
                base,
                args.index,
                batch,
                encoder=encoder,
                encode_batch_size=args.encode_batch_size,
                auth=auth,
                api_key=api_key,
                insecure=args.insecure,
            )
            print(f"  编码批次 {number}: 已写入 {indexed}/{len(reembed)}")

    if metadata_only:
        updated = 0
        for number, batch in enumerate(_batches(metadata_only, max(args.bulk_size, 512)), start=1):
            updated += _bulk_update_metadata(
                base,
                args.index,
                batch,
                auth=auth,
                api_key=api_key,
                insecure=args.insecure,
            )
            print(f"  元数据批次 {number}: 已更新 {updated}/{len(metadata_only)}")

    request("POST", f"{index_url}/_refresh", auth=auth, api_key=api_key, insecure=args.insecure)
    _, count_response = request("GET", f"{index_url}/_count", auth=auth, api_key=api_key, insecure=args.insecure)
    print(f"入库完成，索引文档总数: {count_response.get('count', 'unknown')}")
    print("提示：源文件发生删除时，请使用 --recreate 全量同步，避免保留已删除的旧文档。")


if __name__ == "__main__":
    main()
