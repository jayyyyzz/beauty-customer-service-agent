# -*- coding: utf-8 -*-
"""RAG 检索专项优化候选实现。

保持现有 ES Mapping 不变，集中处理四类问题：
1. 口语、错别字和缩写归一化；
2. 多轮历史信息补全；
3. 英文商品查询被 language / brand 严格过滤误伤；
4. 候选结果的近重复、来源优先级和关键词覆盖排序。
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any


INTENT_HINTS = {
    "price": "价格 优惠 活动 优惠券 满减",
    "product_info": "商品信息 名称 品牌 规格 价格 色号",
    "skin_type": "肤质适配 油皮 干皮 敏感肌 闷痘 是否适合",
    "skin_concern": "肌肤问题 痘痘 泛红 暗沉 屏障修护",
    "ingredient": "成分 配方 浓度 含量 耐受",
    "efficacy": "功效 保湿 修护 美白 抗老",
    "usage": "使用方法 用量 频率",
    "routine": "护肤流程 使用顺序 洁面 水 精华 乳液 面霜 防晒",
    "compatibility": "产品搭配 成分冲突 叠加使用",
    "shade_color": "色号 妆效 粉底 口红 显白 遮瑕 持妆",
    "authenticity_shelf_life": "正品 保质期 批次 防伪 开封保存",
    "safety_allergy": "安全 过敏 刺痛 泛红 停用",
    "comparison": "商品对比 区别 推荐",
    "gift_sample": "赠品 小样 礼物 礼盒",
    "review": "评价 晒图 好评 反馈 返现 奖励",
    "logistics": "物流 快递 发货 到货",
    "urge_shipment": "催发货 催仓 发货时间",
    "logistics_delay": "物流延误 快递停滞 未更新 异常物流 丢件",
    "after_sale": "售后 退货 换货 退款 拆封",
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
    "safety_allergy": ["policy", "faq", "conversation", "product"],
    "comparison": ["product", "faq", "conversation"],
    "gift_sample": ["product", "faq", "conversation"],
    "review": ["faq", "conversation"],
    "logistics": ["shipping", "faq", "conversation"],
    "urge_shipment": ["shipping", "faq", "conversation"],
    "logistics_delay": ["shipping", "faq", "conversation"],
    "after_sale": ["policy", "faq", "conversation"],
    "invoice": ["policy", "faq", "conversation"],
    "quality_issue": ["policy", "faq", "conversation", "product"],
}

QUERY_REWRITES = (
    ("烟酰安", "烟酰胺"),
    ("a醇", "视黄醇 A醇"),
    ("A醇", "视黄醇 A醇"),
    ("红又痒", "过敏 泛红 瘙痒"),
    ("爆红", "过敏 泛红"),
    ("烂脸", "过敏 刺痛 泛红"),
    ("拆了", "拆封"),
    ("开盖", "开封"),
    ("大油田", "油皮 油性皮肤"),
    ("糊一脸", "油腻 闷痘"),
    ("咋整", "怎么办"),
    ("啥时候", "多久"),
    ("能开票不", "可以开发票吗"),
    ("给不给换", "能否换货"),
    ("同一晚叠", "一起使用 搭配"),
    ("晒图好评", "评价 晒图 奖励"),
)

NO_ANSWER_PATTERNS = (
    re.compile(r"临床|双盲|受试者|人体功效试验", re.I),
    re.compile(r"碳足迹|碳排放|二氧化碳", re.I),
    re.compile(r"实时库存|此刻.*库存|现在.*还剩|仓库.*(?:几件|多少件|还剩)", re.I),
    re.compile(r"原始报告|原始数据|逐人数据|完整检测报告", re.I),
    re.compile(r"胎儿|直接诊断|确诊|是不是.*皮炎|是不是.*激素脸", re.I),
    re.compile(r"UVA[- ]?PF.*(精确|实测|数值)", re.I),
    re.compile(r"私人行程|家庭住址|私人住址", re.I),
    re.compile(r"未发布|还没发布|尚未发布|下一季度新品|下季度.*新品|保密配方|内部配方", re.I),
)


def normalize_query(text: str) -> str:
    normalized = str(text or "").strip()
    for source, target in QUERY_REWRITES:
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized)


def build_contextual_query(case: dict[str, Any]) -> str:
    history = normalize_query(" ".join(str(item.get("content") or "") for item in case.get("history") or []))
    keywords = normalize_query(" ".join(str(item) for item in case.get("keywords") or []))
    question = normalize_query(str(case.get("question") or ""))
    intent = str(case.get("intent") or "")
    hint = INTENT_HINTS.get(intent, "")
    return re.sub(r"\s+", " ", f"{hint} {keywords} {history} {question}").strip()


def build_optimized_filters(case: dict[str, Any]) -> dict[str, Any]:
    """保留业务类型过滤，但避免英文商品被语言和品牌大小写误伤。"""
    intent = str(case.get("intent") or "")
    question = str(case.get("question") or "")
    filters: dict[str, Any] = {}
    if intent in INTENT_DOCUMENT_TYPES:
        filters["document_type"] = INTENT_DOCUMENT_TYPES[intent]

    has_chinese = any("\u4e00" <= char <= "\u9fff" for char in question)
    # 商品表的 language=zh 代表统一结构化说明语言，标题和商品名仍可能是英文。
    # 因此 product_info 不做 language 硬过滤；其他意图继续限制中文，避免英文 FAQ 污染。
    if intent != "product_info" and has_chinese:
        filters["language"] = ["zh"]

    policy_brands = [brand for brand in ("ColourPop", "Fenty Beauty", "Glossier") if brand.casefold() in question.casefold()]
    if policy_brands and intent in {"after_sale", "quality_issue", "logistics", "logistics_delay"}:
        filters["brand"] = policy_brands
    return filters


def is_likely_knowledge_gap(case: dict[str, Any]) -> bool:
    text = " ".join(
        [str(case.get("question") or "")]
        + [str(item.get("content") or "") for item in case.get("history") or []]
    )
    return any(pattern.search(text) for pattern in NO_ANSWER_PATTERNS)


def _searchable_text(hit: dict[str, Any]) -> str:
    source = hit.get("_source", {})
    return " ".join(
        str(source.get(field) or "")
        for field in ("title", "content", "question_text", "answer_text", "topics", "brand")
    ).casefold()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffa-z0-9]+", "", text.casefold())


def _source_priority(intent: str, document_type: str) -> float:
    priorities = {
        "after_sale": {"policy": 1.0, "faq": 0.95, "conversation": 0.65},
        "quality_issue": {"policy": 1.0, "faq": 0.95, "conversation": 0.75, "product": 0.40},
        "invoice": {"faq": 1.0, "policy": 0.85, "conversation": 0.70},
        "logistics": {"shipping": 1.0, "faq": 0.90, "conversation": 0.70},
        "logistics_delay": {"shipping": 1.0, "faq": 0.90, "conversation": 0.75},
        "product_info": {"product": 1.0, "faq": 0.80, "conversation": 0.60},
        "review": {"faq": 1.0, "conversation": 0.75},
    }
    return priorities.get(intent, {"faq": 0.90, "product": 0.85, "policy": 0.90, "shipping": 0.90, "conversation": 0.70}).get(document_type, 0.50)


def rerank_and_deduplicate(
    hits: list[dict[str, Any]],
    case: dict[str, Any],
    *,
    final_k: int = 10,
    dedup_threshold: float = 0.88,
    profile: str = "balanced",
) -> list[dict[str, Any]]:
    if not hits:
        return []

    weights = {
        "balanced": (0.55, 0.25, 0.10, 0.10),
        "lexical": (0.45, 0.35, 0.10, 0.10),
        "source": (0.50, 0.20, 0.10, 0.20),
    }[profile]
    score_weight, keyword_weight, title_weight, source_weight = weights

    raw_scores = [float(hit.get("_score") or 0.0) for hit in hits]
    low, high = min(raw_scores), max(raw_scores)
    keywords = [normalize_query(str(value)).casefold() for value in case.get("keywords") or [] if str(value).strip()]
    query = normalize_query(str(case.get("question") or "")).casefold()
    intent = str(case.get("intent") or "")

    rescored: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        text = _searchable_text(hit)
        title = str(source.get("title") or "").casefold()
        es_score = float(hit.get("_score") or 0.0)
        normalized_es = (es_score - low) / (high - low) if high > low else 1.0 / rank
        keyword_coverage = sum(keyword in text for keyword in keywords) / len(keywords) if keywords else 0.0
        title_similarity = SequenceMatcher(None, _normalize_text(query), _normalize_text(title)).ratio() if title else 0.0
        source_score = _source_priority(intent, str(source.get("document_type") or ""))
        score = (
            score_weight * normalized_es
            + keyword_weight * keyword_coverage
            + title_weight * title_similarity
            + source_weight * source_score
        )
        copied = dict(hit)
        copied["_optimized_score"] = score
        copied["_optimization"] = {
            "original_rank": rank,
            "normalized_es": normalized_es,
            "keyword_coverage": keyword_coverage,
            "title_similarity": title_similarity,
            "source_priority": source_score,
        }
        rescored.append(copied)

    rescored.sort(key=lambda item: float(item["_optimized_score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    normalized_kept: list[str] = []
    seen_hashes: set[str] = set()
    seen_sessions: dict[str, int] = {}
    for hit in rescored:
        source = hit.get("_source", {})
        content_hash = str(source.get("content_hash") or "")
        if content_hash and content_hash in seen_hashes:
            continue
        normalized = _normalize_text(str(source.get("content") or ""))
        if normalized and any(SequenceMatcher(None, normalized, previous).ratio() >= dedup_threshold for previous in normalized_kept):
            continue
        # 对话数据最多保留同一 session 的一条，减少 overlap 和模板化内容占满 Top-K。
        session_id = str(source.get("session_id") or "")
        if source.get("document_type") == "conversation" and session_id:
            if seen_sessions.get(session_id, 0) >= 1:
                continue
            seen_sessions[session_id] = seen_sessions.get(session_id, 0) + 1
        if content_hash:
            seen_hashes.add(content_hash)
        if normalized:
            normalized_kept.append(normalized)
        copied = dict(hit)
        copied["_score"] = float(hit["_optimized_score"])
        kept.append(copied)
        if len(kept) >= final_k:
            break
    return kept
