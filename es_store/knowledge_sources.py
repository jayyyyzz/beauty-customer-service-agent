# -*- coding: utf-8 -*-
"""把不同来源的数据转换为统一的 Elasticsearch 知识文档。"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNK_FILE = ROOT / "chunk" / "qa_topic_chunks.jsonl"

SUPPORTED_SOURCES = (
    "conversation",
    "product",
    "faq",
    "canonical",
    "policy",
    "shipping",
)


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _join_fields(pairs: Iterable[tuple[str, object]]) -> str:
    lines = []
    for label, value in pairs:
        text = _clean(value)
        if text:
            lines.append(f"{label}：{text}")
    return "\n".join(lines)


def _as_bool(value: object) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "y"}


def split_dialogue_qa(text: str) -> tuple[str, str]:
    questions, answers = [], []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("买家:"):
            questions.append(line.removeprefix("买家:").strip())
        elif line.startswith("商家:"):
            answers.append(line.removeprefix("商家:").strip())
    return "\n".join(questions), "\n".join(answers)


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    document_type: str
    source_record_id: str
    title: str
    content: str
    question_text: str = ""
    answer_text: str = ""
    source_name: str = ""
    source_url: str = ""
    brand: str = ""
    category: str = ""
    intent: str = ""
    topics: tuple[str, ...] = ()
    language: str = "zh"
    need_human: bool = False
    updated_at: str = ""
    content_hash: str = ""

    def to_source(self) -> dict:
        data = asdict(self)
        data["topics"] = list(self.topics)
        if not data.get("updated_at"):
            data.pop("updated_at", None)
        return data


def _build_document(**kwargs: object) -> KnowledgeDocument:
    payload = {
        key: value
        for key, value in kwargs.items()
        if key not in {"content_hash"}
    }
    digest_input = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    content_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return KnowledgeDocument(**kwargs, content_hash=content_hash)


def iter_conversation_documents() -> Iterator[KnowledgeDocument]:
    """读取对话知识，并按完整问答内容进行精确去重。"""
    seen_qa_keys: set[tuple[str, str]] = set()

    with CHUNK_FILE.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            record_id = _clean(row["chunk_id"])
            content = row.get("core_text") or row.get("text") or ""
            question, answer = split_dialogue_qa(content)

            # 只删除问题和答案都完全相同的记录。
            # 不按问题单独去重，避免相同问题对应不同业务答案时被误删。
            dedup_key = (_clean(question), _clean(answer))
            if question and answer:
                if dedup_key in seen_qa_keys:
                    continue
                seen_qa_keys.add(dedup_key)

            main_topic = _clean(row.get("main_topic"))
            yield _build_document(
                document_id=f"conversation:{record_id}",
                document_type="conversation",
                source_record_id=record_id,
                title=f"{main_topic or '通用'}客服问答",
                content=content,
                question_text=question,
                answer_text=answer,
                source_name="美妆电商客服合成对话数据集",
                category=main_topic,
                topics=tuple(_clean(x) for x in row.get("topics", []) if _clean(x)),
                language="zh",
            )


def _read_csv(name: str) -> Iterator[dict[str, str]]:
    path = PROCESSED_DIR / name
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def iter_product_documents() -> Iterator[KnowledgeDocument]:
    for row in _read_csv("product_knowledge.csv"):
        record_id = _clean(row.get("product_id"))
        name = _clean(row.get("name"))
        brand = _clean(row.get("brand"))
        category = _clean(row.get("sub_category") or row.get("category"))
        content = _join_fields([
            ("商品", name),
            ("品牌", brand),
            ("类别", category),
            ("价格", f"{_clean(row.get('price'))} {_clean(row.get('currency'))}"),
            ("规格", row.get("specification")),
            ("色号", row.get("colors")),
            ("卖点", row.get("selling_points")),
            ("详情", row.get("detail")),
            ("功效", row.get("efficacy")),
            ("适用人群", row.get("applicable_people")),
            ("不适用人群", row.get("not_suitable_people")),
            ("使用方法", row.get("usage_method")),
            ("注意事项", row.get("cautions")),
            ("成分", row.get("ingredients")),
        ])
        language_sample = " ".join(
            _clean(row.get(field))
            for field in (
                "name", "brand", "category", "sub_category", "selling_points",
                "detail", "efficacy", "applicable_people", "not_suitable_people",
                "usage_method", "cautions", "ingredients",
            )
            if _clean(row.get(field))
        )
        topics = tuple(x for x in (_clean(row.get("category")), category, _clean(row.get("efficacy"))) if x)
        yield _build_document(
            document_id=f"product:{record_id}",
            document_type="product",
            source_record_id=record_id,
            title=" ".join(x for x in (brand, name) if x) or record_id,
            content=content,
            question_text=" ".join(x for x in (brand, name, category) if x),
            answer_text=content,
            source_name=_clean(row.get("source")),
            source_url=_clean(row.get("product_url") or row.get("source_url")),
            brand=brand,
            category=category,
            topics=topics,
            language=(
                "zh"
                if any("\u4e00" <= ch <= "\u9fff" for ch in language_sample)
                else "en"
            ),
            updated_at=_clean(row.get("retrieved_at")),
        )


def iter_faq_documents() -> Iterator[KnowledgeDocument]:
    for row in _read_csv("faq_knowledge.csv"):
        record_id = _clean(row.get("faq_id"))
        question = _clean(row.get("question"))
        answer = _clean(row.get("answer"))
        category = _clean(row.get("category"))
        intent = _clean(row.get("intent"))
        variant_hash = hashlib.sha256(
            f"{question}\n{answer}\n{intent}\n{category}".encode("utf-8")
        ).hexdigest()[:12]
        content = _join_fields([("问题", question), ("答案", answer)])
        # 不使用带“问题/答案”中文标签的 content 判断，否则英文 FAQ 会被误标为中文。
        language_sample = f"{question}\n{answer}"
        language = (
            "zh"
            if any("\u4e00" <= ch <= "\u9fff" for ch in language_sample)
            else "en"
        )
        yield _build_document(
            document_id=f"faq:{record_id}:{variant_hash}",
            document_type="faq",
            source_record_id=record_id,
            title=question[:160] or record_id,
            content=content,
            question_text=question,
            answer_text=answer,
            source_name=_clean(row.get("source")),
            source_url=_clean(row.get("source_url")),
            category=category,
            intent=intent,
            topics=tuple(x for x in (category, intent) if x),
            language=language,
            need_human=_as_bool(row.get("need_human")),
            updated_at=_clean(row.get("retrieved_at")),
        )


def iter_canonical_faq_documents() -> Iterator[KnowledgeDocument]:
    """读取人工确认的内部标准 FAQ，优先承载通用业务规则。"""
    for row in _read_csv("canonical_faq_knowledge.csv"):
        record_id = _clean(row.get("faq_id"))
        question = _clean(row.get("question"))
        answer = _clean(row.get("answer"))
        category = _clean(row.get("category"))
        intent = _clean(row.get("intent"))
        content = _join_fields([("问题", question), ("答案", answer)])
        yield _build_document(
            document_id=f"canonical:{record_id}",
            document_type="faq",
            source_record_id=record_id,
            title=question[:160] or record_id,
            content=content,
            question_text=question,
            answer_text=answer,
            source_name=_clean(row.get("source")) or "内部标准FAQ",
            source_url=_clean(row.get("source_url")),
            category=category,
            intent=intent,
            topics=tuple(x for x in (category, intent) if x),
            language="zh",
            need_human=_as_bool(row.get("need_human")),
            updated_at=_clean(row.get("retrieved_at")),
        )


def iter_policy_documents() -> Iterator[KnowledgeDocument]:
    for row in _read_csv("policy_knowledge.csv"):
        record_id = _clean(row.get("policy_id"))
        brand = _clean(row.get("brand"))
        policy_type = _clean(row.get("policy_type"))
        content = _join_fields([
            ("品牌", brand),
            ("适用场景", row.get("applicable_scenario")),
            ("退货规则", row.get("return_rule")),
            ("换货规则", row.get("exchange_rule")),
            ("退款规则", row.get("refund_rule")),
            ("运费规则", row.get("freight_rule")),
            ("保修规则", row.get("warranty_rule")),
            ("特殊限制", row.get("special_limits")),
            ("标准答复", row.get("standard_answer_template")),
        ])
        yield _build_document(
            document_id=f"policy:{record_id}",
            document_type="policy",
            source_record_id=record_id,
            title=f"{brand} {policy_type}政策".strip(),
            content=content,
            answer_text=_clean(row.get("standard_answer_template")) or content,
            source_name=_clean(row.get("source")),
            source_url=_clean(row.get("source_url")),
            brand=brand,
            category=policy_type,
            intent="after_sale",
            topics=("售后政策", policy_type),
            language="zh",
            need_human=True,
            updated_at=_clean(row.get("retrieved_at")),
        )


def iter_shipping_documents() -> Iterator[KnowledgeDocument]:
    for row in _read_csv("shipping_rules.csv"):
        record_id = _clean(row.get("shipping_rule_id"))
        brand = _clean(row.get("brand"))
        content = _join_fields([
            ("品牌", brand),
            ("发货时间", row.get("dispatch_time")),
            ("包邮门槛", row.get("free_shipping_threshold")),
            ("偏远地区规则", row.get("remote_region_rule")),
            ("承运商", row.get("carriers")),
            ("预计送达", row.get("estimated_delivery_time")),
            ("异常物流处理", row.get("abnormal_shipping_handling")),
            ("标准答复", row.get("standard_answer_template")),
        ])
        yield _build_document(
            document_id=f"shipping:{record_id}",
            document_type="shipping",
            source_record_id=record_id,
            title=f"{brand}物流规则".strip(),
            content=content,
            answer_text=_clean(row.get("standard_answer_template")) or content,
            source_name=_clean(row.get("source")),
            source_url=_clean(row.get("source_url")),
            brand=brand,
            category="shipping",
            intent="logistics",
            topics=("物流", "发货", "配送"),
            language="zh",
            need_human=True,
            updated_at=_clean(row.get("retrieved_at")),
        )


LOADERS = {
    "conversation": iter_conversation_documents,
    "product": iter_product_documents,
    "faq": iter_faq_documents,
    "canonical": iter_canonical_faq_documents,
    "policy": iter_policy_documents,
    "shipping": iter_shipping_documents,
}


def load_documents(sources: Iterable[str]) -> list[KnowledgeDocument]:
    documents_by_id: dict[str, KnowledgeDocument] = {}
    for source in sources:
        if source not in LOADERS:
            raise ValueError(f"不支持的数据源: {source}")
        for document in LOADERS[source]():
            existing = documents_by_id.get(document.document_id)
            if existing and existing.content_hash != document.content_hash:
                raise ValueError(f"同一 document_id 对应不同内容: {document.document_id}")
            documents_by_id[document.document_id] = document
    return list(documents_by_id.values())
