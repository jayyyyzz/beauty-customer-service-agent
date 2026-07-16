# -*- coding: utf-8 -*-
"""Small live A/B latency and token experiment for retrieval/rerank settings."""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "agent_pipeline.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_pipeline
import handoff_store
from conversation_state import ConversationStore
from handoff_store import HandoffStore
from run_performance_cost_handoff import execute_case


OUTPUT = (
    ROOT
    / "evaluation_suites"
    / "05_performance_cost_handoff_evaluation"
    / "experiments"
    / "optimization_experiments.json"
)

CASES = [
    {"case_id": "opt_routine", "question": "日常护肤流程应该怎么安排？"},
    {"case_id": "opt_skin", "question": "油皮适合用面霜吗？"},
    {"case_id": "opt_auth", "question": "商品保质期和批次应该怎么看？"},
]

VARIANTS = [
    {"id": "A_baseline", "mode": "rrf_mmr", "rerank": True},
    {"id": "B_disable_llm_rerank", "mode": "rrf_mmr", "rerank": False},
    {"id": "C_bm25_no_rerank", "mode": "bm25", "rerank": False},
]


async def main() -> None:
    original_mode = agent_pipeline.ES_search_config.get("mode")
    original_rerank = agent_pipeline.RERANK_config.get("enabled")
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="agent_opt_") as tmp:
            agent_pipeline._DEFAULT_STATE_STORE = ConversationStore(Path(tmp) / "state.db")
            handoff_store._DEFAULT_STORE = HandoffStore(Path(tmp) / "handoffs.db")
            for variant in VARIANTS:
                agent_pipeline.ES_search_config["mode"] = variant["mode"]
                agent_pipeline.RERANK_config["enabled"] = variant["rerank"]
                rows = []
                for repeat in range(2):
                    for case in CASES:
                        rows.append(
                            await execute_case(
                                case,
                                prefix=f"{variant['id']}_{repeat}",
                            )
                        )
                latencies = [float(row["elapsed_ms"]) for row in rows if row["success"]]
                token_values = [
                    int((row["trace"].get("tokens") or {}).get("total") or 0)
                    for row in rows if row["success"]
                ]
                costs = [float(row["cost"]["usd"]) for row in rows if row["success"]]
                retrieval_counts = [len(row.get("knowledge_document_ids") or []) for row in rows]
                results.append(
                    {
                        **variant,
                        "requests": len(rows),
                        "success_rate": sum(int(row["success"]) for row in rows) / len(rows),
                        "mean_latency_ms": round(statistics.fmean(latencies), 2),
                        "median_latency_ms": round(statistics.median(latencies), 2),
                        "mean_tokens": round(statistics.fmean(token_values), 2),
                        "mean_cost_usd": round(statistics.fmean(costs), 8),
                        "retrieval_nonempty_rate": round(
                            sum(int(count > 0) for count in retrieval_counts) / len(rows), 4
                        ),
                        "rows": rows,
                    }
                )
    finally:
        agent_pipeline.ES_search_config["mode"] = original_mode
        agent_pipeline.RERANK_config["enabled"] = original_rerank

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "Small live latency/cost comparison; retrieval quality requires the full retrieval benchmark before deployment.",
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
