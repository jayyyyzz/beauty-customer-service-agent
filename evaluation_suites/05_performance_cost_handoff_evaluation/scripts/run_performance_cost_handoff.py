# -*- coding: utf-8 -*-
"""Live performance, token/cost and human-handoff benchmark for the Agent.

The benchmark calls the real DeepSeek and Elasticsearch services when configured.
It writes raw, reproducible JSON; the Markdown assessment is generated separately
from that artifact so measured values are never mixed with estimates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "agent_pipeline.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_pipeline
import handoff_store
from agent_observability import (
    finish_agent_trace,
    reset_agent_trace,
    start_agent_trace,
)
from conversation_state import ConversationStore
from handoff_store import HandoffStore


SPECIAL_DIR = ROOT / "evaluation_suites" / "05_performance_cost_handoff_evaluation"
REPORT_DIR = SPECIAL_DIR / "optimized"
RAW_REPORT = REPORT_DIR / "raw_results.json"

# Official DeepSeek API pricing observed on 2026-07-16 for deepseek-chat,
# which the official page maps to deepseek-v4-flash non-thinking mode.
# Unit: USD per 1M tokens.
PRICE = {
    "source": "https://api-docs.deepseek.com/quick_start/pricing",
    "observed_at": "2026-07-16",
    "input_cache_hit_per_million_usd": 0.0028,
    "input_cache_miss_per_million_usd": 0.14,
    "output_per_million_usd": 0.28,
}


PERF_CASES: list[dict[str, Any]] = [
    {
        "case_id": "perf_knowledge",
        "history": {"messages": [{"role": "buyer", "content": "我是干皮，偶尔泛红"}]},
        "question": "日常护肤流程应该怎么安排？",
    },
    {
        "case_id": "perf_business",
        "history": {"messages": []},
        "question": "订单 MOCK202606260003 的物流到哪里了？",
    },
    {
        "case_id": "perf_hybrid",
        "history": {
            "messages": [
                {"role": "buyer", "content": "订单是 MOCK202606260003"},
                {"role": "seller", "content": "请问商品有什么问题？"},
            ]
        },
        "question": "收到的精华漏液了，应该怎么处理？",
    },
    {
        "case_id": "perf_clarify",
        "history": {"messages": []},
        "question": "这个怎么办？",
    },
    {
        "case_id": "perf_fallback",
        "history": {"messages": []},
        "question": "给我讲一个关于星空的笑话。",
    },
    {
        "case_id": "perf_handoff",
        "history": {"messages": []},
        "question": "用了精华后嘴唇肿而且呼吸困难。",
    },
]


HANDOFF_CASES: list[dict[str, Any]] = [
    # User explicitly asks for a human.
    {"case_id": "ho_user_01", "question": "给我转人工客服。", "should_handoff": True, "reason": "user_requested", "priority": "high", "facts": ["转人工"]},
    {"case_id": "ho_user_02", "question": "我不想和机器人说了，我要找真人。", "should_handoff": True, "reason": "user_requested", "priority": "high", "facts": ["真人"]},
    {"case_id": "ho_user_03", "question": "别再自动回复，马上接人工。", "should_handoff": True, "reason": "user_requested", "priority": "high", "facts": ["人工"]},
    # Urgent and caution skincare safety.
    {"case_id": "ho_urgent_01", "question": "用了面霜后呼吸困难，喉咙也肿了。", "should_handoff": True, "reason": "urgent_skincare_safety", "priority": "urgent", "facts": ["呼吸困难", "喉咙肿"]},
    {"case_id": "ho_urgent_02", "question": "敷完面膜后全身荨麻疹，快喘不过气了。", "should_handoff": True, "reason": "urgent_skincare_safety", "priority": "urgent", "facts": ["全身荨麻疹", "喘不过气"]},
    {"case_id": "ho_urgent_03", "question": "精华进眼后眼睛肿，还有大面积水疱。", "should_handoff": True, "reason": "urgent_skincare_safety", "priority": "urgent", "facts": ["眼睛肿", "大面积水疱"]},
    {"case_id": "ho_reaction_01", "question": "用了以后脸很红、刺痛，想申请售后。", "should_handoff": True, "reason": "skincare_reaction", "priority": "high", "facts": ["刺痛", "售后"]},
    {"case_id": "ho_reaction_02", "question": "新精华用了两次就过敏发痒，能退吗？", "should_handoff": True, "reason": "skincare_reaction", "priority": "high", "facts": ["过敏", "退"]},
    # Complaint, fraud and repeated unresolved cases.
    {"case_id": "ho_complaint_01", "question": "订单 MOCK202606260003 处理三次还没结果，我要投诉到消协。", "should_handoff": True, "reason": "complaint_escalation", "priority": "high", "facts": ["MOCK202606260003", "消协"]},
    {"case_id": "ho_fraud_01", "question": "订单 MOCK202606260003 出现异常支付，我怀疑账号被盗。", "should_handoff": True, "reason": "account_security", "priority": "urgent", "facts": ["MOCK202606260003", "账号被盗"]},
    {"case_id": "ho_repeat_01", "history": [{"role": "buyer", "content": "订单号我已经给过两次了"}, {"role": "seller", "content": "系统还是查询失败"}], "question": "别再让我重复了，直接升级处理。", "should_handoff": True, "reason": "repeated_failure", "priority": "high", "facts": ["两次", "查询失败"]},
    {"case_id": "ho_dispute_01", "question": "这笔退款金额有两千多，平台和商家说法冲突。", "should_handoff": True, "reason": "high_value_dispute", "priority": "high", "facts": ["两千多", "退款"]},
    # Eligible for automation or one clarification; must not hand off immediately.
    {"case_id": "auto_01", "question": "油皮适合用面霜吗？", "should_handoff": False},
    {"case_id": "auto_02", "question": "烟酰胺精华早晚都能用吗？", "should_handoff": False},
    {"case_id": "auto_03", "question": "订单 MOCK202606260003 到哪里了？", "should_handoff": False},
    {"case_id": "auto_04", "question": "发票应该怎么申请？", "should_handoff": False},
    {"case_id": "auto_05", "question": "这款粉底黄皮选哪个色号？", "should_handoff": False},
    {"case_id": "auto_06", "question": "商品保质期一般怎么看？", "should_handoff": False},
    {"case_id": "auto_07", "question": "有没有满减优惠？", "should_handoff": False},
    {"case_id": "auto_08", "question": "玻尿酸和烟酰胺可以一起用吗？", "should_handoff": False},
    {"case_id": "clarify_01", "question": "这个怎么弄？", "should_handoff": False},
    {"case_id": "clarify_02", "question": "不太对，怎么办？", "should_handoff": False},
    {"case_id": "clarify_03", "question": "我想处理一下。", "should_handoff": False},
    {"case_id": "clarify_04", "question": "还有别的办法吗？", "should_handoff": False},
]


def percentile(values: list[float], q: float, *, digits: int = 2) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], digits)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return round(ordered[lo], digits)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return round(value, digits)


def distribution(values: list[float], *, digits: int = 2) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "p50": percentile(values, 0.50, digits=digits),
        "p90": percentile(values, 0.90, digits=digits),
        "p95": percentile(values, 0.95, digits=digits),
        "p99": percentile(values, 0.99, digits=digits),
        "max": round(max(values), digits),
        "mean": round(statistics.fmean(values), digits),
    }


def calculate_cost(tokens: dict[str, Any]) -> dict[str, float]:
    prompt = int(tokens.get("prompt") or 0)
    completion = int(tokens.get("completion") or 0)
    cache_hit = int(tokens.get("cache_hit") or 0)
    cache_miss = int(tokens.get("cache_miss") or 0)
    if cache_hit + cache_miss == 0:
        cache_miss = prompt
    elif cache_hit + cache_miss < prompt:
        cache_miss += prompt - cache_hit - cache_miss
    cost = (
        cache_hit * PRICE["input_cache_hit_per_million_usd"]
        + cache_miss * PRICE["input_cache_miss_per_million_usd"]
        + completion * PRICE["output_per_million_usd"]
    ) / 1_000_000
    return {
        "usd": round(cost, 8),
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
    }


async def execute_case(case: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    conversation_id = f"{prefix}_{case['case_id']}_{uuid.uuid4().hex[:8]}"
    raw_history = case.get("history") or case.get("history_messages") or []
    if isinstance(raw_history, dict):
        messages = list(raw_history.get("messages") or [])
    else:
        messages = list(raw_history)
    history = {
        "conversation_id": conversation_id,
        "messages": messages,
    }
    trace_id = f"trace_{uuid.uuid4().hex}"
    trace_token = start_agent_trace(trace_id, conversation_id)
    started = time.perf_counter()
    try:
        result = await agent_pipeline.handle_user_question(
            history_dialogue=history,
            question=case["question"],
            knowledge_top_k=int(case.get("knowledge_top_k") or 3),
        )
        success = True
        error = None
    except Exception as exc:
        result = {}
        success = False
        error = f"{type(exc).__name__}: {exc}"
    trace = finish_agent_trace(route=str(result.get("route") or "error"))
    reset_agent_trace(trace_token)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    cost = calculate_cost(trace.get("tokens") or {})
    handoff = result.get("handoff") or {}
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "success": success,
        "error": error,
        "route": result.get("route"),
        "intent": (result.get("intent") or {}).get("intent_level1"),
        "intent_confidence": (result.get("intent") or {}).get("intent_confidence"),
        "elapsed_ms": elapsed_ms,
        "trace": trace,
        "cost": cost,
        "handoff_required": bool(result.get("handoff_required")),
        "handoff": handoff,
        "answer": str(result.get("answer") or ""),
        "knowledge_topics": [
            doc.get("topic") for doc in (result.get("knowledge_docs") or [])
        ],
        "knowledge_document_ids": [
            doc.get("document_id") for doc in (result.get("knowledge_docs") or [])
        ],
        "knowledge_document_types": [
            doc.get("document_type") for doc in (result.get("knowledge_docs") or [])
        ],
    }


async def run_batch(cases: list[dict[str, Any]], *, concurrency: int, prefix: str) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await execute_case(case, prefix=prefix)

    started = time.perf_counter()
    results = await asyncio.gather(*(one(case) for case in cases))
    return results, round(time.perf_counter() - started, 3)


def aggregate_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]
    by_route: dict[str, list[float]] = {}
    by_stage: dict[str, list[float]] = {}
    costs: list[float] = []
    tokens: list[float] = []
    for row in successful:
        by_route.setdefault(str(row.get("route") or "unknown"), []).append(float(row["elapsed_ms"]))
        costs.append(float(row["cost"]["usd"]))
        tokens.append(float((row["trace"].get("tokens") or {}).get("total") or 0))
        for stage in row["trace"].get("stages") or []:
            by_stage.setdefault(str(stage.get("name") or "unknown"), []).append(float(stage.get("latency_ms") or 0))
    return {
        "requests": len(rows),
        "successes": len(successful),
        "success_rate": round(len(successful) / len(rows), 4) if rows else 0,
        "e2e": distribution([float(row["elapsed_ms"]) for row in successful]),
        "by_route": {route: distribution(values) for route, values in sorted(by_route.items())},
        "by_stage": {stage: distribution(values) for stage, values in sorted(by_stage.items())},
        "tokens_per_request": distribution(tokens),
        "cost_usd_per_request": distribution(costs, digits=8),
        "total_tokens": int(sum(tokens)),
        "total_cost_usd": round(sum(costs), 8),
    }


def handoff_metrics(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {case["case_id"]: case for case in cases}
    tp = fp = fn = tn = 0
    critical_total = critical_missed = 0
    summary_scores: list[float] = []
    context_scores: list[float] = []
    details = []
    for row in rows:
        case = expected[row["case_id"]]
        should = bool(case["should_handoff"])
        did = bool(row["handoff_required"] and row.get("handoff"))
        if should and did:
            tp += 1
        elif not should and did:
            fp += 1
        elif should and not did:
            fn += 1
        else:
            tn += 1
        if case.get("priority") == "urgent":
            critical_total += 1
            critical_missed += int(not did)

        ticket = row.get("handoff") or {}
        combined = json.dumps(
            {"summary": ticket.get("summary"), "context": ticket.get("context")},
            ensure_ascii=False,
        )
        if should and did:
            required = ["question", "reason"]
            if "safety" in str(case.get("reason")) or case.get("priority") == "urgent":
                required.append("safety")
            if case.get("history"):
                required.append("history")
            checks = {
                "question": bool(ticket.get("summary")),
                "reason": bool(ticket.get("reason")),
                "safety": bool((ticket.get("context") or {}).get("safety")),
                "history": bool((ticket.get("context") or {}).get("history")),
            }
            summary_scores.append(sum(int(checks[name]) for name in required) / len(required))
            facts = list(case.get("facts") or [])
            if facts:
                context_scores.append(sum(int(fact in combined) for fact in facts) / len(facts))
        details.append(
            {
                "case_id": case["case_id"],
                "should_handoff": should,
                "did_handoff": did,
                "expected_reason": case.get("reason"),
                "actual_reason": ticket.get("reason"),
                "priority": case.get("priority"),
                "route": row.get("route"),
                "success": row.get("success"),
            }
        )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        "sample_count": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "missed_handoff_rate": round(fn / (tp + fn), 4) if tp + fn else 0,
        "over_handoff_rate": round(fp / (fp + tn), 4) if fp + tn else 0,
        "critical_missed_handoff_rate": round(critical_missed / critical_total, 4) if critical_total else 0,
        "handoff_success_rate": round(sum(int(bool(row.get("handoff", {}).get("ticket_id"))) for row in rows if row["handoff_required"]) / max(1, sum(int(row["handoff_required"]) for row in rows)), 4),
        "summary_completeness": round(statistics.fmean(summary_scores), 4) if summary_scores else None,
        "context_preservation_rate": round(statistics.fmean(context_scores), 4) if context_scores else None,
        "repeat_information_rate": None,
        "details": details,
    }


async def run_fault_tests() -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    async def execute(name: str, question: str) -> None:
        case = {"case_id": name, "question": question, "history": []}
        result = await execute_case(case, prefix="fault")
        safe = bool(result.get("answer")) and not any(
            phrase in result.get("answer", "") for phrase in ("已经成功退款", "操作已完成")
        )
        tests.append({"name": name, "safe_degradation": bool(result["success"] and safe), **result})

    original_intents = agent_pipeline.recognize_intents
    original_search = agent_pipeline.search_knowledge
    original_tool = agent_pipeline.call_business_api
    try:
        async def usage_intent(*_args, **_kwargs):
            return [{"intent_level1": "usage", "intent_level2": "usage.method", "intent_level3": "usage.method.order", "intent_logic": "fault test", "intent_confidence": 0.99, "keywords": ["用法"]}]

        async def logistics_intent(*_args, **_kwargs):
            return [{"intent_level1": "logistics", "intent_level2": "logistics.query", "intent_level3": "logistics.query.status", "intent_logic": "fault test", "intent_confidence": 0.99, "keywords": ["物流"]}]

        agent_pipeline.recognize_intents = usage_intent
        agent_pipeline.search_knowledge = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected ES outage"))
        await execute("es_unavailable", "护肤流程是什么？")

        agent_pipeline.recognize_intents = logistics_intent
        agent_pipeline.call_business_api = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected tool outage"))
        await execute("business_tool_unavailable", "订单 MOCK202606260003 到哪里了？")

        async def llm_timeout(*_args, **_kwargs):
            raise TimeoutError("injected LLM timeout")

        agent_pipeline.recognize_intents = llm_timeout
        await execute("llm_timeout", "这款产品怎么用？")

        await execute("urgent_path_without_llm", "用了以后呼吸困难而且喉咙肿。")
    finally:
        agent_pipeline.recognize_intents = original_intents
        agent_pipeline.search_knowledge = original_search
        agent_pipeline.call_business_api = original_tool
    return tests


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="agent_perf_") as tmp:
        tmp_dir = Path(tmp)
        agent_pipeline._DEFAULT_STATE_STORE = ConversationStore(tmp_dir / "state.db")
        handoff_store._DEFAULT_STORE = HandoffStore(tmp_dir / "handoffs.db")

        # First request in a fresh process is the cold-start sample.
        cold_rows, cold_wall = await run_batch(PERF_CASES[:1], concurrency=1, prefix="cold")
        warm_rows, warm_wall = await run_batch(PERF_CASES, concurrency=1, prefix="warm")
        concurrent_cases = PERF_CASES * 2
        concurrent_rows, concurrent_wall = await run_batch(
            concurrent_cases,
            concurrency=max(1, args.concurrency),
            prefix="concurrent",
        )
        handoff_rows, handoff_wall = await run_batch(
            HANDOFF_CASES,
            concurrency=max(1, args.concurrency),
            prefix="handoff",
        )
        fault_rows = await run_fault_tests()

    all_perf = cold_rows + warm_rows + concurrent_rows
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version,
            "model": agent_pipeline.LLM_deepseek_config.get("model"),
            "es_url": agent_pipeline.ES_search_config.get("url"),
            "es_index": agent_pipeline.ES_search_config.get("index"),
            "configured_concurrency": args.concurrency,
            "ttft_measurable": False,
            "ttft_note": "Current Agent uses non-streaming ChatCompletions; only full-call latency is measurable.",
            "pricing": PRICE,
        },
        "performance": {
            "aggregate": aggregate_performance(all_perf),
            "cold_start": {"wall_seconds": cold_wall, "rows": cold_rows},
            "warm_sequential": {"wall_seconds": warm_wall, "rows": warm_rows},
            "concurrent": {
                "level": args.concurrency,
                "wall_seconds": concurrent_wall,
                "throughput_rps": round(len(concurrent_rows) / concurrent_wall, 4) if concurrent_wall else None,
                "rows": concurrent_rows,
            },
        },
        "handoff": {
            "wall_seconds": handoff_wall,
            "metrics": handoff_metrics(HANDOFF_CASES, handoff_rows),
            "rows": handoff_rows,
        },
        "fault_injection": {
            "graceful_degradation_rate": round(sum(int(row["safe_degradation"]) for row in fault_rows) / len(fault_rows), 4),
            "rows": fault_rows,
        },
    }
    RAW_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(RAW_REPORT),
        "performance": report["performance"]["aggregate"],
        "handoff": report["handoff"]["metrics"],
        "fault_injection_rate": report["fault_injection"]["graceful_degradation_rate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
