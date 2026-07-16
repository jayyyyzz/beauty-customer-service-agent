# -*- coding: utf-8 -*-
"""使用已选参数快速重跑优化方案，不重复基线和参数扫描。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import run_rag_optimization as runner


def main() -> None:
    cases = runner.load_cases()
    case_map = {case["query_id"]: case for case in cases}
    index_by_id = {case["query_id"]: index for index, case in enumerate(cases)}
    cache = np.load(PACKAGE_ROOT / "results" / "retrieval_query_vectors_v2.npz")
    vectors = cache["optimized"]
    manifest = json.loads((PACKAGE_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    params = manifest["best_params"]

    client = runner.ElasticsearchClient()
    runner.es_module.es_search = client.search
    runner.eval_module.es_search = client.search
    details = runner.evaluate_cases(
        cases,
        lambda case: runner.search_optimized(
            client,
            case,
            vectors[index_by_id[case["query_id"]]],
            **params,
        ),
        "O_hybrid_contextual_rerank",
    )
    metric = runner.compact_metrics(
        "O_hybrid_contextual_rerank", details, case_map, use_gap_rule=True
    )

    detail_path = PACKAGE_ROOT / "results" / "retrieval_details.jsonl"
    previous = [json.loads(line) for line in detail_path.open(encoding="utf-8") if line.strip()]
    combined = [row for row in previous if row["experiment_id"] != "O_hybrid_contextual_rerank"] + details
    runner.write_jsonl(detail_path, combined)
    runner.write_json(PACKAGE_ROOT / "results" / "optimized_metrics.json", metric)
    runner.write_jsonl(
        PACKAGE_ROOT / "results" / "optimized_failure_cases.jsonl",
        [item for item in details if item.get("error_types")],
    )
    client.close()
    print("优化方案已按当前实现重新执行。")


if __name__ == "__main__":
    main()
