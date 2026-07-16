# -*- coding: utf-8 -*-
"""基于已保存的模型回答与Judge结果，修正规则指标并重建回答生成报告。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from run_answer_evaluation import (
    DEFAULT_DATASET,
    ROOT,
    aggregate,
    citation_rule_metrics,
    derive_claim_metrics,
    forbidden_matches,
    group_metrics,
    pass_fail,
    point_coverage,
    searchable_text,
    sha256_file,
    write_report,
)
from configs import ES_search_config, LLM_deepseek_config, RERANK_config


TEST_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TEST_DIR / "official_results"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def refresh(row: dict) -> dict:
    answer_coverage, answer_details = point_coverage(
        row.get("required_points") or [], row.get("answer") or ""
    )
    evidence_payload = row.get("knowledge_docs") or row.get("api_data")
    evidence_coverage, evidence_details = point_coverage(
        row.get("required_points") or [], searchable_text(evidence_payload)
    )
    citations = citation_rule_metrics(
        row.get("answer") or "",
        row.get("knowledge_docs") or [],
        bool(row.get("requires_citation")),
    )
    forbidden = forbidden_matches(
        row.get("answer") or "", row.get("forbidden_claims") or []
    )
    row["claim_metrics"] = derive_claim_metrics(row.get("judge", {}).get("claims") or [])
    row["rule_metrics"] = {
        "answer_required_point_coverage": answer_coverage,
        "answer_point_details": answer_details,
        "evidence_required_point_coverage": evidence_coverage,
        "evidence_point_details": evidence_details,
        "forbidden_matches": forbidden,
        "citations": citations,
    }
    errors = set(row.get("judge", {}).get("error_types") or [])
    if row.get("generation_error"):
        errors.add("generation_error")
    if row.get("judge_error"):
        errors.add("judge_error")
    if forbidden:
        errors.add("forbidden_claim")
    if answer_coverage < 1.0:
        errors.add("missing_required_point")
    if (
        row.get("requires_citation")
        and row.get("judge", {}).get("retrieval_context_sufficient")
        and not citations["has_required_citation"]
    ):
        errors.add("citation_missing")
    if citations["invalid"]:
        errors.add("citation_mismatch")
    retrieval_attribution_applicable = (
        row.get("evaluation_track") == "B_end_to_end_rag"
        and bool(row.get("has_answer"))
        and row.get("sample_type")
        not in {"safety", "prompt_injection", "information_insufficient", "no_answer"}
    )
    if retrieval_attribution_applicable:
        if evidence_coverage < 1.0:
            errors.add("retrieval_incomplete")
        elif answer_coverage < 1.0:
            errors.add("generation_omission")
    row["error_types"] = sorted(errors)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    details_path = args.output_dir / "answer_eval_predictions.jsonl"
    rows = [refresh(row) for row in load_jsonl(details_path)]
    manifest_path = TEST_DIR / "evaluation_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    run_key = "optimized" if "优化" in args.output_dir.name else "baseline"
    run_manifest = manifest.get(run_key, {})
    overall = aggregate(rows)
    payload = {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "recalculated_from_saved_results": True,
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "runner_sha256": sha256_file(TEST_DIR / "run_answer_evaluation.py"),
            "agent_pipeline_sha256": sha256_file(ROOT / "agent_pipeline.py"),
            "model": run_manifest.get("answer_model", LLM_deepseek_config.get("model")),
            "judge_model": run_manifest.get("judge_model", LLM_deepseek_config.get("model")),
            "answer_generation_note": run_manifest.get("answer_generation_note", ""),
            "base_url": LLM_deepseek_config.get("base_url"),
            "answer_temperature": LLM_deepseek_config.get("answer_temperature", 0.0),
            "judge_temperature": 0,
            "es_index": ES_search_config.get("index"),
            "es_mode": ES_search_config.get("mode"),
            "rerank_enabled": bool(RERANK_config.get("enabled")),
        },
        "distribution": {
            "track": dict(Counter(row["evaluation_track"] for row in rows)),
            "risk_level": dict(Counter(row["risk_level"] for row in rows)),
            "sample_type": dict(Counter(row["sample_type"] for row in rows)),
        },
        "overall": overall,
        "by_track": group_metrics(rows, "evaluation_track"),
        "by_risk_level": group_metrics(rows, "risk_level"),
        "by_sample_type": group_metrics(rows, "sample_type"),
        "acceptance": pass_fail(overall),
    }
    (args.output_dir / "answer_eval_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with details_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (args.output_dir / "p0_p1_review_queue.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            if row.get("judge", {}).get("critical_error") or any(
                claim.get("severity") in {"P0", "P1"}
                and claim.get("verdict") in {"contradicted", "not_enough_info"}
                for claim in row.get("judge", {}).get("claims", [])
            ):
                file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    write_report(args.output_dir, payload, rows)
    print(json.dumps({
        "faithfulness": overall["faithfulness"],
        "weighted_faithfulness": overall["weighted_faithfulness"],
        "claim_hallucination_rate": overall["claim_hallucination_rate"],
        "answer_hallucination_rate": overall["answer_hallucination_rate"],
        "critical_hallucination_rate": overall["critical_hallucination_rate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
