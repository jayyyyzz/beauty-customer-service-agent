# -*- coding: utf-8 -*-
"""Run repeatable end-to-end acceptance cases against the real Agent pipeline."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_pipeline import handle_user_question


REPORT_PATH = ROOT / "reports" / "e2e_acceptance_latest.json"

CASES: list[dict[str, Any]] = [
    {
        "name": "knowledge_routine_short_query",
        "history": {
            "conversation_id": "acceptance_routine_001",
            "messages": [
                {"role": "buyer", "content": "我最近脸有点干，还容易泛红"},
                {"role": "seller", "content": "可以看看修护精华，主打舒缓保湿"},
            ],
        },
        "question": "护肤流程有吗？",
        "expected_route": "knowledge_base",
        "expected_topics": {"用法用量"},
        "require_knowledge": True,
        "require_order": False,
    },
    {
        "name": "knowledge_skin_type",
        "history": {
            "conversation_id": "acceptance_knowledge_001",
            "messages": [
                {"role": "buyer", "content": "我是油皮，平时比较容易闷痘"},
            ],
        },
        "question": "油皮适合用面霜吗？",
        "expected_route": "knowledge_base",
        "expected_topics": {"肤质匹配"},
        "require_knowledge": True,
        "require_order": False,
    },
    {
        "name": "business_logistics",
        "history": {
            "conversation_id": "acceptance_business_001",
            "user_id": "mock_user_004",
            "messages": [],
        },
        "question": "订单 MOCK202606260003 的快递到哪里了？",
        "expected_route": "business_api",
        "expected_topics": set(),
        "require_knowledge": False,
        "require_order": True,
    },
    {
        "name": "hybrid_quality_issue",
        "history": {
            "conversation_id": "acceptance_hybrid_001",
            "user_id": "mock_user_004",
            "messages": [
                {"role": "buyer", "content": "我的订单号是 MOCK202606260003"},
                {"role": "seller", "content": "好的，请问商品遇到了什么问题？"},
            ],
        },
        "question": "收到的精华漏液了，应该怎么处理？",
        "expected_route": "hybrid",
        "expected_topics": {"破损补发", "过敏售后", "退换货"},
        "require_knowledge": True,
        "require_order": True,
    },
]


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    route = result.get("route")
    intent = result.get("intent") or {}
    docs = result.get("knowledge_docs") or []
    api_data = result.get("api_data") or {}
    api_result = api_data.get("result") or {}
    answer = (result.get("answer") or "").strip()

    if route != case["expected_route"]:
        failures.append(f"route expected {case['expected_route']}, got {route}")

    confidence = float(intent.get("intent_confidence") or 0)
    if confidence < 0.65:
        failures.append(f"intent confidence is too low: {confidence}")

    if case["require_knowledge"] and not docs:
        failures.append("knowledge retrieval returned no documents")

    topics = {str(doc.get("topic") or "") for doc in docs}
    expected_topics = case["expected_topics"]
    if expected_topics and not topics.intersection(expected_topics):
        failures.append(
            "retrieval topic mismatch: expected one of "
            f"{sorted(expected_topics)}, got {sorted(topics)}"
        )

    if case["require_order"] and api_result.get("status") not in {"found", "succeeded"}:
        failures.append(
            f"order lookup expected found/succeeded, got {api_result.get('status')}"
        )

    if len(answer) < 10:
        failures.append("final answer is empty or too short")

    summary = {
        "route": route,
        "intent_level1": intent.get("intent_level1"),
        "intent_confidence": confidence,
        "knowledge_topics": [doc.get("topic") for doc in docs],
        "knowledge_chunk_ids": [doc.get("chunk_id") for doc in docs],
        "order_status": api_result.get("status"),
        "answer": answer,
    }
    return failures, summary


async def run() -> int:
    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cases": [],
    }

    print("Customer Service Agent end-to-end acceptance")
    print("=" * 52)

    for case in CASES:
        print(f"\n[RUN] {case['name']}: {case['question']}")
        started = time.perf_counter()
        try:
            result = await handle_user_question(
                history_dialogue=case["history"],
                question=case["question"],
            )
            elapsed = round(time.perf_counter() - started, 3)
            failures, summary = evaluate_case(case, result)
            passed = not failures
            report["cases"].append(
                {
                    "name": case["name"],
                    "question": case["question"],
                    "expected_route": case["expected_route"],
                    "expected_topics": sorted(case["expected_topics"]),
                    "passed": passed,
                    "elapsed_seconds": elapsed,
                    "failures": failures,
                    **summary,
                }
            )
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] route={summary['route']} elapsed={elapsed}s")
            print(f"       intent={summary['intent_level1']} confidence={summary['intent_confidence']}")
            print(f"       topics={summary['knowledge_topics']} order={summary['order_status']}")
            for failure in failures:
                print(f"       - {failure}")
        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 3)
            report["cases"].append(
                {
                    "name": case["name"],
                    "question": case["question"],
                    "passed": False,
                    "elapsed_seconds": elapsed,
                    "failures": [f"unhandled exception: {type(exc).__name__}: {exc}"],
                }
            )
            print(f"[FAIL] unhandled exception after {elapsed}s: {type(exc).__name__}: {exc}")

    passed_count = sum(1 for case in report["cases"] if case["passed"])
    total_count = len(report["cases"])
    report["summary"] = {
        "passed": passed_count,
        "failed": total_count - passed_count,
        "total": total_count,
        "all_passed": passed_count == total_count,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 52)
    print(f"Result: {passed_count}/{total_count} passed")
    print(f"Report: {REPORT_PATH}")
    return 0 if report["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
