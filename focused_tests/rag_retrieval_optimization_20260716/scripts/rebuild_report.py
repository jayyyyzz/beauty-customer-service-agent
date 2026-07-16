# -*- coding: utf-8 -*-
"""不重复执行 ES 检索，基于逐条结果重建阈值、错误标签和最终报告。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import run_rag_optimization as runner


def main() -> None:
    cases = runner.load_cases()
    case_map = {case["query_id"]: case for case in cases}
    detail_path = PACKAGE_ROOT / "results" / "retrieval_details.jsonl"
    with detail_path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["experiment_id"]].append(row)

    baseline_ids = [
        "A_bm25_raw",
        "B_knn_raw",
        "C_hybrid_enhanced",
        "D_hybrid_enhanced_dedup",
        "P_production_rrf_mmr",
    ]
    baseline_metrics = [
        runner.compact_metrics(experiment_id, groups[experiment_id], case_map, use_gap_rule=False)
        for experiment_id in baseline_ids
    ]
    optimized_details = groups["O_hybrid_contextual_rerank"]
    optimized_metric = runner.compact_metrics(
        "O_hybrid_contextual_rerank", optimized_details, case_map, use_gap_rule=True
    )

    runner.write_json(PACKAGE_ROOT / "results" / "baseline_metrics.json", baseline_metrics)
    runner.write_json(PACKAGE_ROOT / "results" / "optimized_metrics.json", optimized_metric)
    comparison = {
        "baseline_best": max(baseline_metrics, key=lambda item: runner.objective(item["held_out_test"])),
        "optimized": optimized_metric,
        "best_params": json.loads((PACKAGE_ROOT / "run_manifest.json").read_text(encoding="utf-8"))["best_params"],
    }
    runner.write_json(PACKAGE_ROOT / "results" / "optimization_comparison.json", comparison)
    runner.write_jsonl(PACKAGE_ROOT / "results" / "retrieval_details.jsonl", rows)
    runner.write_jsonl(
        PACKAGE_ROOT / "results" / "optimized_failure_cases.jsonl",
        [item for item in optimized_details if item.get("error_types")],
    )

    manifest = json.loads((PACKAGE_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    sweep = json.loads((PACKAGE_ROOT / "results" / "parameter_sweep.json").read_text(encoding="utf-8"))
    runner.generate_report(
        manifest["snapshot"],
        cases,
        baseline_metrics,
        optimized_metric,
        manifest["best_params"],
        sweep,
        manifest["encoding"],
        optimized_details,
    )
    print("已基于现有检索结果重建优化指标和报告。")


if __name__ == "__main__":
    main()
