# -*- coding: utf-8 -*-
"""运行当前部署版多意图识别器的专项离线测评。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for intent_dir in (ROOT / "意图识别", ROOT / "intention_prompt"):
    if intent_dir.exists() and str(intent_dir) not in sys.path:
        sys.path.insert(0, str(intent_dir))

from agent_pipeline import (  # noqa: E402
    ALLOWED_INTENTS,
    _json_dumps,
    close_agent_resources,
    recognize_intents_with_trace,
)
from configs import AGENT_RUNTIME_config, LLM_deepseek_config  # noqa: E402


DEFAULT_DATASET = ROOT / "evaluation" / "intent_benchmark_v1.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "intent_evaluation_v1"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployed_messages(history_dialogue: dict[str, Any], question: str) -> list[dict[str, str]]:
    taxonomy = ", ".join(sorted(ALLOWED_INTENTS))
    return [
        {
            "role": "system",
            "content": (
                "你是美妆电商客服的多意图识别器。请结合历史对话，拆分用户当前输入中的"
                "所有独立诉求；不要把同一诉求的修饰语拆成多个意图。"
                f"intent_level1 只能取以下值：{taxonomy}。"
                "严格输出 JSON 对象，格式为 {\"intents\": [...]}。每个元素必须包含 "
                "intent_level1、intent_level2、intent_level3、intent_logic、"
                "intent_confidence、keywords。不要输出 Markdown 或解释文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"历史对话：\n{_json_dumps(history_dialogue)}\n\n"
                f"当前输入：\n{question}"
            ),
        },
    ]


async def recognize_deployed_with_raw(
    history_dialogue: dict[str, Any],
    question: str,
) -> tuple[list[dict[str, Any]], str, bool]:
    """调用当前生产识别链路，并保留模型原始输出。"""
    return await recognize_intents_with_trace(history_dialogue, question)


def path_prefix_valid(intent: dict[str, Any]) -> bool:
    level1 = str(intent.get("intent_level1") or "")
    level2 = str(intent.get("intent_level2") or "")
    level3 = str(intent.get("intent_level3") or "")
    if not level1 or not level2 or not level3:
        return False
    if level1 == "other":
        return level2.startswith("other") and level3.startswith("other")
    return level2.startswith(level1 + ".") and level3.startswith(level2 + ".")


async def evaluate_one(case: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    history_dialogue = {
        "conversation_id": f"intent_eval_{case['case_id']}",
        "messages": case.get("history") or [],
    }
    error = None
    raw_output = ""
    intents: list[dict[str, Any]] = []
    used_fallback = False
    latency_ms = 0.0

    async with semaphore:
        started = time.perf_counter()
        for attempt in range(1, 4):
            try:
                intents, raw_output, used_fallback = await recognize_deployed_with_raw(
                    history_dialogue, case["question"]
                )
                break
            except Exception as exc:  # API/JSON 错误均保留并重试
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    await asyncio.sleep(1.5 * attempt)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

    primary = intents[0] if intents else {}
    gold_l1 = case["gold_intent_level1"]
    gold_l2 = case["gold_intent_level2"]
    gold_l3 = case["gold_intent_level3"]
    pred_l1 = str(primary.get("intent_level1") or "__invalid__")
    pred_l2 = str(primary.get("intent_level2") or "__invalid__")
    pred_l3 = str(primary.get("intent_level3") or "__invalid__")
    confidence = float(primary.get("intent_confidence") or 0.0)
    l1_ok = pred_l1 == gold_l1
    l2_ok = pred_l2 == gold_l2
    l3_ok = pred_l3 == gold_l3
    path_score = 0.0
    if l1_ok:
        path_score += 0.4
        if l2_ok:
            path_score += 0.3
            if l3_ok:
                path_score += 0.3

    gold_intents = case.get("gold_intents") or []
    gold_l1_set = sorted({item["level1"] for item in gold_intents})
    gold_l3_set = sorted({item["level3"] for item in gold_intents})
    pred_l1_set = sorted({item.get("intent_level1") for item in intents if item.get("intent_level1")})
    pred_l3_set = sorted({item.get("intent_level3") for item in intents if item.get("intent_level3")})
    json_output_valid = bool(intents) and error is None
    paths_valid = bool(intents) and all(path_prefix_valid(item) for item in intents)
    valid_output = json_output_valid and paths_valid
    minimum_confidence = min(
        [float(item.get("intent_confidence") or 0.0) for item in intents],
        default=0.0,
    )
    explicit_needs_clarification = any(
        bool(item.get("needs_clarification")) for item in intents
    )

    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "history": case.get("history") or [],
        "scenario": case.get("scenario"),
        "difficulty": case.get("difficulty"),
        "has_noise": bool(case.get("has_noise")),
        "emotion": case.get("emotion"),
        "is_multi_intent": bool(case.get("is_multi_intent")),
        "should_clarify": bool(case.get("should_clarify")),
        "explicit_needs_clarification": explicit_needs_clarification,
        "gold": {
            "level1": gold_l1,
            "level2": gold_l2,
            "level3": gold_l3,
            "intents": gold_intents,
        },
        "prediction": {
            "level1": pred_l1,
            "level2": pred_l2,
            "level3": pred_l3,
            "confidence": confidence,
            "minimum_confidence": minimum_confidence,
            "intents": intents,
        },
        "level1_correct": l1_ok,
        "level2_correct": l2_ok,
        "level3_correct": l3_ok,
        "full_path_correct": l1_ok and l2_ok and l3_ok,
        "path_score": round(path_score, 2),
        "gold_level1_set": gold_l1_set,
        "gold_level3_set": gold_l3_set,
        "pred_level1_set": pred_l1_set,
        "pred_level3_set": pred_l3_set,
        "valid_output": valid_output,
        "json_output_valid": json_output_valid,
        "path_prefix_valid": paths_valid,
        "used_fallback": used_fallback,
        "latency_ms": latency_ms,
        "raw_output": raw_output,
        "error": error,
    }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean_or_zero(values: Iterable[float | bool]) -> float:
    rows = list(values)
    return statistics.mean(rows) if rows else 0.0


def classification_metrics(
    results: list[dict[str, Any]],
    gold_key: str,
    pred_key: str,
) -> dict[str, Any]:
    labels = sorted({str(item["gold"][gold_key]) for item in results})
    per_class: dict[str, dict[str, float | int]] = {}
    total_correct = 0
    total = len(results)
    sum_tp = sum_fp = sum_fn = 0
    for label in labels:
        tp = sum(1 for item in results if item["gold"][gold_key] == label and item["prediction"][pred_key] == label)
        fp = sum(1 for item in results if item["gold"][gold_key] != label and item["prediction"][pred_key] == label)
        fn = sum(1 for item in results if item["gold"][gold_key] == label and item["prediction"][pred_key] != label)
        support = sum(1 for item in results if item["gold"][gold_key] == label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        total_correct += tp
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
    macro_f1 = statistics.mean([float(row["f1"]) for row in per_class.values()]) if per_class else 0.0
    weighted_f1 = safe_div(
        sum(float(row["f1"]) * int(row["support"]) for row in per_class.values()),
        total,
    )
    # 本评测每个层级均按单标签主意图计算，Micro-F1 与 Accuracy 等价。
    micro_f1 = safe_div(total_correct, total)
    return {
        "accuracy": safe_div(total_correct, total),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def set_prf(gold: set[str], predicted: set[str]) -> tuple[float, float, float]:
    tp = len(gold & predicted)
    precision = safe_div(tp, len(predicted))
    recall = safe_div(tp, len(gold))
    return precision, recall, safe_div(2 * precision * recall, precision + recall)


def multi_intent_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [item for item in results if item["is_multi_intent"]]
    if not subset:
        return {"cases": 0}
    level1_rows = [set_prf(set(x["gold_level1_set"]), set(x["pred_level1_set"])) for x in subset]
    level3_rows = [set_prf(set(x["gold_level3_set"]), set(x["pred_level3_set"])) for x in subset]
    return {
        "cases": len(subset),
        "level1_exact_set_accuracy": statistics.mean(set(x["gold_level1_set"]) == set(x["pred_level1_set"]) for x in subset),
        "level1_precision": statistics.mean(x[0] for x in level1_rows),
        "level1_recall": statistics.mean(x[1] for x in level1_rows),
        "level1_f1": statistics.mean(x[2] for x in level1_rows),
        "level3_exact_set_accuracy": statistics.mean(set(x["gold_level3_set"]) == set(x["pred_level3_set"]) for x in subset),
        "level3_precision": statistics.mean(x[0] for x in level3_rows),
        "level3_recall": statistics.mean(x[1] for x in level3_rows),
        "level3_f1": statistics.mean(x[2] for x in level3_rows),
    }


def clarification_metrics(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for item in results:
        actual = item["should_clarify"]
        predicted = bool(item.get("explicit_needs_clarification")) or (
            float(item["prediction"]["minimum_confidence"]) < threshold
        )
        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "error_release_rate": safe_div(fn, tp + fn),
        "over_clarify_rate": safe_div(fp, fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [
        row for row in rows
        if row["precision"] >= 0.85
        and row["error_release_rate"] <= 0.05
        and row["over_clarify_rate"] <= 0.10
    ]
    if qualified:
        selected = dict(max(qualified, key=lambda row: (row["f1"], row["recall"])))
        selected["meets_targets"] = True
        return selected
    selected = dict(min(
        rows,
        key=lambda row: (
            3 * row["fn"] + row["fp"],
            -row["f1"],
        ),
    ))
    selected["meets_targets"] = False
    return selected


def calibration_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.000001)]
    rows = []
    ece = 0.0
    for lower, upper in bins:
        members = [
            item for item in results
            if lower <= float(item["prediction"]["confidence"]) < upper
        ]
        if not members:
            rows.append({"range": f"{lower:.1f}-{min(upper, 1.0):.1f}", "count": 0, "avg_confidence": 0.0, "accuracy": 0.0})
            continue
        avg_conf = statistics.mean(float(item["prediction"]["confidence"]) for item in members)
        accuracy = statistics.mean(bool(item["level1_correct"]) for item in members)
        ece += len(members) / len(results) * abs(avg_conf - accuracy)
        rows.append({
            "range": f"{lower:.1f}-{min(upper, 1.0):.1f}",
            "count": len(members),
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
        })
    brier = statistics.mean(
        (float(item["prediction"]["confidence"]) - float(item["level1_correct"])) ** 2
        for item in results
    )
    high_conf_errors = [
        item for item in results
        if float(item["prediction"]["confidence"]) >= 0.9 and not item["level1_correct"]
    ]
    high_conf_total = sum(1 for item in results if float(item["prediction"]["confidence"]) >= 0.9)
    return {
        "ece": ece,
        "brier_score": brier,
        "bins": rows,
        "high_confidence_errors": len(high_conf_errors),
        "high_confidence_total": high_conf_total,
        "high_confidence_error_rate": safe_div(len(high_conf_errors), high_conf_total),
    }


def slice_metrics(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[str(item.get(field))].append(item)
    output: dict[str, Any] = {}
    for name, members in sorted(groups.items()):
        output[name] = {
            "count": len(members),
            "level1_accuracy": statistics.mean(bool(x["level1_correct"]) for x in members),
            "level2_accuracy": statistics.mean(bool(x["level2_correct"]) for x in members),
            "level3_accuracy": statistics.mean(bool(x["level3_correct"]) for x in members),
            "full_path_accuracy": statistics.mean(bool(x["full_path_correct"]) for x in members),
            "mean_path_score": statistics.mean(float(x["path_score"]) for x in members),
        }
    return output


def top_confusions(results: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    counter = Counter(
        (item["gold"]["level1"], item["prediction"]["level1"])
        for item in results
        if not item["level1_correct"]
    )
    return [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in counter.most_common(limit)
    ]


def infer_error_type(item: dict[str, Any], threshold: float) -> str | None:
    predicted_clarify = bool(item.get("explicit_needs_clarification")) or (
        float(item["prediction"]["minimum_confidence"]) < threshold
    )
    if not item.get("json_output_valid"):
        return "output_format"
    if item["should_clarify"] and not predicted_clarify:
        return "clarify_missed"
    if not item["should_clarify"] and predicted_clarify:
        return "over_clarify"
    if item["is_multi_intent"] and set(item["gold_level3_set"]) != set(item["pred_level3_set"]):
        return "multi_intent"
    if not item["level1_correct"] and float(item["prediction"]["confidence"]) >= 0.9:
        return "over_confident"
    if item["level1_correct"] and predicted_clarify:
        return "under_confident"
    if not item["level1_correct"] and item["scenario"] == "context":
        return "context_missing"
    if not item["level1_correct"] and item["scenario"] == "noise":
        return "noise_robustness"
    if not item["level1_correct"] and item["scenario"] == "confusable":
        return "taxonomy_boundary"
    if not item["level1_correct"]:
        return "classification_error"
    if not item.get("path_prefix_valid") or not item["full_path_correct"]:
        return "hierarchy_path"
    return None


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_metrics(results: list[dict[str, Any]], dataset_path: Path) -> dict[str, Any]:
    l1 = classification_metrics(results, "level1", "level1")
    l2 = classification_metrics(results, "level2", "level2")
    l3 = classification_metrics(results, "level3", "level3")
    clarification_rows = [clarification_metrics(results, threshold) for threshold in THRESHOLDS]
    selected = choose_threshold(clarification_rows)
    for item in results:
        item["error_type"] = infer_error_type(item, float(selected["threshold"]))

    prompt_path = ROOT / "agent_pipeline.py"
    return {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "deployed_prompt_source": str(prompt_path),
            "deployed_prompt_source_sha256": sha256_file(prompt_path),
            "model": LLM_deepseek_config.get("model"),
            "base_url": LLM_deepseek_config.get("base_url"),
            "temperature": LLM_deepseek_config.get("intent_temperature", 0.0),
            "current_clarify_threshold": AGENT_RUNTIME_config.get("clarify_confidence_threshold"),
            "cases": len(results),
        },
        "dataset_distribution": {
            "level1": dict(Counter(item["gold"]["level1"] for item in results)),
            "scenario": dict(Counter(item["scenario"] for item in results)),
            "difficulty": dict(Counter(item["difficulty"] for item in results)),
            "has_history": sum(bool(item["history"]) for item in results),
            "should_clarify": sum(bool(item["should_clarify"]) for item in results),
            "multi_intent": sum(bool(item["is_multi_intent"]) for item in results),
        },
        "classification": {"level1": l1, "level2": l2, "level3": l3},
        "full_path_accuracy": statistics.mean(bool(item["full_path_correct"]) for item in results),
        "mean_hierarchical_path_score": statistics.mean(float(item["path_score"]) for item in results),
        "multi_intent": multi_intent_metrics(results),
        "clarification": {
            "thresholds": clarification_rows,
            "selected": selected,
        },
        "calibration": calibration_metrics(results),
        "slices": {
            "scenario": slice_metrics(results, "scenario"),
            "difficulty": slice_metrics(results, "difficulty"),
            "has_noise": slice_metrics(results, "has_noise"),
            "emotion": slice_metrics(results, "emotion"),
        },
        "context_accuracy": mean_or_zero(
            bool(item["level1_correct"]) for item in results if item["scenario"] == "context"
        ),
        "confusable_accuracy": mean_or_zero(
            bool(item["level1_correct"]) for item in results if item["scenario"] == "confusable"
        ),
        "output": {
            "valid_rate": statistics.mean(bool(item["valid_output"]) for item in results),
            "json_valid_rate": statistics.mean(bool(item.get("json_output_valid")) for item in results),
            "path_prefix_valid_rate": statistics.mean(bool(item.get("path_prefix_valid")) for item in results),
            "fallback_count": sum(bool(item["used_fallback"]) for item in results),
            "exception_count": sum(bool(item["error"]) for item in results),
        },
        "latency_ms": {
            "mean": statistics.mean(float(item["latency_ms"]) for item in results),
            "median": statistics.median(float(item["latency_ms"]) for item in results),
            "p95": sorted(float(item["latency_ms"]) for item in results)[max(0, math.ceil(len(results) * 0.95) - 1)],
            "max": max(float(item["latency_ms"]) for item in results),
        },
        "top_level1_confusions": top_confusions(results),
        "error_types": dict(Counter(item.get("error_type") or "correct" for item in results)),
    }


def write_confusion_csv(results: list[dict[str, Any]], path: Path) -> None:
    labels = sorted(ALLOWED_INTENTS | {"__invalid__"})
    matrix = Counter((item["gold"]["level1"], item["prediction"]["level1"]) for item in results)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["gold\\predicted", *labels])
        for gold in labels:
            writer.writerow([gold, *[matrix[(gold, predicted)] for predicted in labels]])


def report_markdown(metrics: dict[str, Any], results: list[dict[str, Any]]) -> str:
    meta = metrics["metadata"]
    cls = metrics["classification"]
    selected = metrics["clarification"]["selected"]
    calibration = metrics["calibration"]
    lines = [
        "# 美妆电商客服 Agent——意图识别专项测评报告",
        "",
        f"> 测评时间：{meta['generated_at']}  ",
        f"> 模型：{meta['model']}  ",
        f"> 测试集：intent_benchmark_v1，共 {meta['cases']} 条  ",
        "> 数据性质：人工定义标签模板生成的冻结合成基准，不等同于真实线上流量人工标注集。",
        "",
        "## 1. 执行摘要",
        "",
        f"- Level-1 Accuracy：**{percent(cls['level1']['accuracy'])}**，Macro-F1：**{percent(cls['level1']['macro_f1'])}**。",
        f"- Level-2 Accuracy：**{percent(cls['level2']['accuracy'])}**；Level-3 Accuracy：**{percent(cls['level3']['accuracy'])}**。",
        f"- 三级完整路径准确率：**{percent(metrics['full_path_accuracy'])}**；平均层级路径得分：**{metrics['mean_hierarchical_path_score']:.3f}**。",
        f"- 多轮上下文 Level-1 Accuracy：**{percent(metrics['context_accuracy'])}**；易混淆样本：**{percent(metrics['confusable_accuracy'])}**。",
        f"- JSON 可解析率：**{percent(metrics['output']['json_valid_rate'])}**；合法层级前缀率：**{percent(metrics['output']['path_prefix_valid_rate'])}**；P95 模型识别耗时：**{metrics['latency_ms']['p95']:.0f} ms**。",
        f"- 置信度校准 ECE：**{calibration['ece']:.3f}**，Brier Score：**{calibration['brier_score']:.3f}**。",
        "",
        "## 2. 测试集规模与分布",
        "",
        f"共 {meta['cases']} 条：基础 {metrics['dataset_distribution']['scenario'].get('base', 0)}、口语噪声 {metrics['dataset_distribution']['scenario'].get('noise', 0)}、多轮上下文 {metrics['dataset_distribution']['scenario'].get('context', 0)}、易混淆 {metrics['dataset_distribution']['scenario'].get('confusable', 0)}、澄清 {metrics['dataset_distribution']['scenario'].get('clarify', 0)}、多意图 {metrics['dataset_distribution']['scenario'].get('multi_intent', 0)}。",
        "",
        "每个一级意图均有基础覆盖；测试集哈希已写入 metrics JSON，用于后续回归复现。",
        "",
        "## 3. 核心分类指标",
        "",
        "| 层级 | Accuracy | Macro-F1 | Micro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for level in ("level1", "level2", "level3"):
        row = cls[level]
        lines.append(
            f"| {level.title()} | {percent(row['accuracy'])} | {percent(row['macro_f1'])} | {percent(row['micro_f1'])} | {percent(row['weighted_f1'])} |"
        )

    lines += [
        "",
        "## 4. 一级意图分类表现",
        "",
        "| 意图 | Precision | Recall | F1 | 样本数 |",
        "|---|---:|---:|---:|---:|",
    ]
    per_class = cls["level1"]["per_class"]
    for label, row in sorted(per_class.items(), key=lambda pair: (pair[1]["f1"], pair[0])):
        lines.append(
            f"| {label} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | {row['support']} |"
        )

    lines += [
        "",
        "## 5. 场景切片",
        "",
        "| 场景 | 样本数 | L1 Accuracy | L2 Accuracy | L3 Accuracy | 完整路径 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics["slices"]["scenario"].items():
        lines.append(
            f"| {name} | {row['count']} | {percent(row['level1_accuracy'])} | {percent(row['level2_accuracy'])} | {percent(row['level3_accuracy'])} | {percent(row['full_path_accuracy'])} |"
        )

    multi = metrics["multi_intent"]
    lines += [
        "",
        "### 多意图识别",
        "",
        f"- 样本数：{multi.get('cases', 0)}",
        f"- Level-1 意图集合 F1：{percent(multi.get('level1_f1', 0.0))}，完全匹配率：{percent(multi.get('level1_exact_set_accuracy', 0.0))}",
        f"- Level-3 意图集合 F1：{percent(multi.get('level3_f1', 0.0))}，完全匹配率：{percent(multi.get('level3_exact_set_accuracy', 0.0))}",
        "",
        "## 6. 澄清策略与阈值",
        "",
        "| 阈值 | Precision | Recall | 错误放行率 | 过度澄清率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["clarification"]["thresholds"]:
        marker_label = "建议" if selected.get("meets_targets") else "诊断最优"
        marker = f" **←{marker_label}**" if row["threshold"] == selected["threshold"] else ""
        lines.append(
            f"| {row['threshold']:.2f}{marker} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['error_release_rate'])} | {percent(row['over_clarify_rate'])} |"
        )
    lines += [
        "",
        (
            f"当前配置阈值为 {meta['current_clarify_threshold']}；**没有任何候选阈值达到方案目标**。"
            f"{selected['threshold']:.2f} 仅是当前候选中加权损失相对较小的诊断值，不能直接作为上线建议。"
            if not selected.get("meets_targets")
            else f"当前配置阈值为 {meta['current_clarify_threshold']}；按本基准风险约束，候选阈值为 **{selected['threshold']:.2f}**，仍需在真实人工标注集复测后再修改生产配置。"
        ),
        "",
        "## 7. 置信度校准",
        "",
        "| 置信度区间 | 样本数 | 平均置信度 | 实际 L1 Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in calibration["bins"]:
        lines.append(
            f"| {row['range']} | {row['count']} | {percent(row['avg_confidence'])} | {percent(row['accuracy'])} |"
        )
    lines += [
        "",
        f"高置信度（≥0.9）错误 {calibration['high_confidence_errors']} 条 / {calibration['high_confidence_total']} 条，高置信度错误率 {percent(calibration['high_confidence_error_rate'])}。",
        "",
        "## 8. 主要混淆关系",
        "",
        "| 真实意图 | 错误预测 | 数量 |",
        "|---|---|---:|",
    ]
    for row in metrics["top_level1_confusions"]:
        lines.append(f"| {row['gold']} | {row['predicted']} | {row['count']} |")

    high_conf_errors = sorted(
        [item for item in results if not item["level1_correct"]],
        key=lambda item: float(item["prediction"]["confidence"]),
        reverse=True,
    )[:15]
    lines += [
        "",
        "## 9. 高风险错误样本",
        "",
        "| Case | 问题 | Gold | Prediction | 置信度 | 错误类型 |",
        "|---|---|---|---|---:|---|",
    ]
    for item in high_conf_errors:
        question = str(item["question"]).replace("|", "\\|")
        lines.append(
            f"| {item['case_id']} | {question} | {item['gold']['level1']} | {item['prediction']['level1']} | {item['prediction']['confidence']:.2f} | {item.get('error_type') or ''} |"
        )

    lines += [
        "",
        "## 10. 验收判断",
        "",
    ]
    targets = [
        ("Level-1 Accuracy", cls["level1"]["accuracy"], 0.88, 0.92),
        ("Level-1 Macro-F1", cls["level1"]["macro_f1"], 0.85, 0.90),
        ("Level-2 Accuracy", cls["level2"]["accuracy"], 0.82, 0.88),
        ("Level-3 Accuracy", cls["level3"]["accuracy"], 0.75, 0.82),
        ("多轮上下文准确率", metrics["context_accuracy"], 0.80, 0.88),
        ("易混淆准确率", metrics["confusable_accuracy"], 0.75, 0.85),
    ]
    lines += ["| 指标 | 实际 | 合格线 | 较好目标 | 结论 |", "|---|---:|---:|---:|---|"]
    for name, actual, pass_line, good_line in targets:
        conclusion = "较好" if actual >= good_line else "合格" if actual >= pass_line else "未达标"
        lines.append(f"| {name} | {percent(actual)} | {percent(pass_line)} | {percent(good_line)} | {conclusion} |")

    lines += [
        "",
        "## 11. 限制说明",
        "",
        "- 本次为冻结的合成专项基准，表达覆盖较广，但不能替代真实客服流量的双人复核标注集。",
        "- 当前意图识别已显式设置 temperature=0；供应商服务和模型版本更新仍可能造成小幅波动。",
        "- 多意图样本数量较少，适合作为冒烟与方向判断，不适合作为线上多意图占比估计。",
        "- 澄清判断按当前 Agent 的“最低意图置信度低于阈值”逻辑复算，尚未评估槽位缺失触发的追问。",
        "",
    ]
    return "\n".join(lines)


def optimization_markdown(metrics: dict[str, Any]) -> str:
    cls = metrics["classification"]
    selected = metrics["clarification"]["selected"]
    confusions = metrics["top_level1_confusions"][:8]
    weak_intents = sorted(
        cls["level1"]["per_class"].items(),
        key=lambda pair: pair[1]["f1"],
    )[:8]
    if metrics["output"].get("path_prefix_valid_rate", 0) >= 0.99 and cls["level3"]["accuracy"] >= 0.80:
        lines = [
            "# 意图识别优化后续建议",
            "",
            "## 已完成的关键优化",
            "",
            "- 多意图识别复用统一三级 taxonomy，输出路径可机器校验。",
            "- 意图识别固定 temperature=0，并记录 Prompt 与测试集哈希。",
            "- 引入显式 `other.unclear` 澄清语义，和商品名、订单号等槽位补全解耦。",
            "- 非法路径只在对应一级意图子树内定向修复。",
            "- 增加物流/催发货、质量/售后、功效/肌肤问题、用法/流程等边界规则。",
            "",
            "## 下一阶段重点",
            "",
            "1. **继续优化剩余混淆对。** 优先为 `routine/usage`、`comparison/skin_type`、`product_info/comparison` 增加成对反例，并明确“单产品用法”与“整套流程”、“两款对比”与“单款适配”的判断条件。",
            "2. **降低高置信度错误。** 对边界意图增加置信度上限或轻量校准器；高风险意图可用小规模人工标注集训练 isotonic/Platt 校准。",
            "3. **控制 Prompt 成本与延迟。** 当前完整 taxonomy 提升了准确率，但增加输入长度；可改为一级意图候选召回后，仅注入 Top-2 一级子树做第二阶段分类。",
            "4. **建立真实盲测集。** 从脱敏客服日志抽取 300–500 条，由双人复核；冻结一份不参与 Prompt 调优的最终验收集。",
            "5. **增加线上反馈闭环。** 记录人工改判、最终工具、澄清结果和任务完成状态，按周更新混淆矩阵。",
            "",
            "## 当前最弱意图",
            "",
            "| 意图 | Precision | Recall | F1 | 样本数 |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, row in weak_intents:
            lines.append(
                f"| {label} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | {row['support']} |"
            )
        lines += [
            "",
            "## 剩余主要混淆",
            "",
            "| Gold | Prediction | 数量 |",
            "|---|---|---:|",
        ]
        for row in confusions:
            lines.append(f"| {row['gold']} | {row['predicted']} | {row['count']} |")
        lines += [
            "",
            "## 后续回归门禁",
            "",
            "- Level-1 Accuracy 与 Macro-F1 不低于本版本。",
            "- Level-2 Accuracy ≥ 90%，Level-3 Accuracy ≥ 88%。",
            "- 合法路径率和 JSON 可解析率保持 100%。",
            "- 多轮上下文、易混淆样本 Accuracy ≥ 90%。",
            "- 高置信度错误率逐步降低到 2% 以下。",
            "- 任何 Prompt/taxonomy 变更必须使用冻结集和真实盲测集双重回归。",
            "",
        ]
        return "\n".join(lines)

    lines = [
        "# 意图识别优化建议",
        "",
        "## P0：先解决影响评测可信度和线上决策的问题",
        "",
        "1. **统一单意图与多意图 Prompt 的三级分类标准。** 当前线上使用的多意图 Prompt 只枚举一级标签，却要求模型自行生成二、三级路径；应复用完整 taxonomy 或以结构化枚举动态注入允许路径，避免一级正确但层级路径漂移。",
        "2. **显式固定推理参数。** 在评测和生产意图识别中设置 temperature=0，并记录模型版本、Prompt 哈希和请求参数；当前使用供应商默认 temperature，会引入回归波动。",
        "3. **将“是否澄清”从自报置信度中解耦。** 增加 `needs_clarification`、`missing_information` 和 `clarification_question` 字段，按意图歧义与必要槽位判断；模型置信度只作为辅助信号。",
        (
            f"4. **不要仅靠调高阈值修复澄清策略。** 0.50–0.80 均未达到验收目标；{selected['threshold']:.2f} 只是诊断上损失相对较小。应先增加显式澄清判断字段，再重新扫描阈值。"
            if not selected.get("meets_targets")
            else f"4. **暂以 {selected['threshold']:.2f} 作为候选澄清阈值做回归验证。** 不直接修改线上配置，先用真实人工标注样本复测。"
        ),
        "",
        "## P1：提升分类边界和三级路径稳定性",
        "",
        "1. **采用两阶段分层分类。** 第一步识别一级意图及多意图集合；第二步只向模型提供对应一级意图的二、三级子树。这样可以缩小候选空间并提升 Level-2/3 准确率。",
        "2. **补充边界判定规则和反例。** 在 Prompt 中明确：中性询问发货时间属于 `logistics`，带催促/截止时间属于 `urge_shipment`；已发出后停滞属于 `logistics_delay`；商品本体异常优先 `quality_issue`，纯退换流程优先 `after_sale`。",
        "3. **为主诉与附属诉求制定排序规则。** 多意图结果应按风险优先级、是否需要工具和用户表达顺序排序，并对同一一级意图下的两个独立三级诉求保留两条结果。",
        "4. **增加结构校验与受限重试。** 使用完整合法路径集合校验 level2/level3；发现非法路径时仅携带对应子树进行一次修复，而不是直接接受前缀看似正确的自造标签。",
        "",
        "## P2：建立真实数据闭环",
        "",
        "1. 从真实或脱敏客服日志抽取 300–500 条独立回归集，由两名标注者复核，记录分歧与最终裁决。",
        "2. 将本次错误样本加入开发集，但保留一份从未参与 Prompt 调优的盲测集，防止针对测试集过拟合。",
        "3. 在线记录意图、置信度、是否澄清、最终工具、人工改判和任务结果，按周更新混淆矩阵与校准曲线。",
        "4. 对高风险意图（过敏、质量投诉、退款、物流异常）单独设置 Recall 目标和人工转接策略，不只看整体 Accuracy。",
        "",
        "## 本次最弱意图",
        "",
        "| 意图 | Precision | Recall | F1 | 样本数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in weak_intents:
        lines.append(
            f"| {label} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | {row['support']} |"
        )
    lines += [
        "",
        "## 优先处理的混淆对",
        "",
        "| Gold | Prediction | 数量 |",
        "|---|---|---:|",
    ]
    for row in confusions:
        lines.append(f"| {row['gold']} | {row['predicted']} | {row['count']} |")
    lines += [
        "",
        "## 建议回归门禁",
        "",
        "- Level-1 Accuracy ≥ 92%，Macro-F1 ≥ 90%。",
        "- Level-2 Accuracy ≥ 88%，Level-3 Accuracy ≥ 82%。",
        "- 多轮上下文与易混淆样本 Accuracy 均 ≥ 85%。",
        "- 高置信度（≥0.9）一级意图错误率 ≤ 2%。",
        "- 输出 JSON 与合法路径校验通过率 = 100%。",
        "- Prompt、模型或 taxonomy 变更必须运行同一冻结集，并保存与上个版本的差异报告。",
        "",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "intent_eval_predictions.jsonl"
    metrics_path = output_dir / "intent_eval_metrics.json"
    report_path = output_dir / "意图识别专项测评报告.md"
    suggestions_path = output_dir / "意图识别优化建议.md"
    confusion_path = output_dir / "level1_confusion_matrix.csv"

    cases = load_jsonl(dataset_path)
    if not cases:
        raise RuntimeError(f"测试集为空或不存在：{dataset_path}")
    if args.limit:
        cases = cases[: args.limit]

    existing = {item["case_id"]: item for item in load_jsonl(predictions_path)} if args.resume else {}
    pending = [case for case in cases if case["case_id"] not in existing]
    print(f"dataset={len(cases)} existing={len(existing)} pending={len(pending)}")
    print(f"model={LLM_deepseek_config.get('model')} concurrency={args.concurrency}")

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(evaluate_one(case, semaphore)) for case in pending]
    completed = 0
    if tasks:
        with predictions_path.open("a" if args.resume else "w", encoding="utf-8") as file:
            for future in asyncio.as_completed(tasks):
                item = await future
                existing[item["case_id"]] = item
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
                file.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}")

    results = [existing[case["case_id"]] for case in cases]
    metrics = build_metrics(results, dataset_path)

    # 用选定阈值重算错误类型后，按 case_id 重写，保证最终文件有序且信息完整。
    with predictions_path.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_markdown(metrics, results), encoding="utf-8")
    suggestions_path.write_text(optimization_markdown(metrics), encoding="utf-8")
    write_confusion_csv(results, confusion_path)

    print(json.dumps({
        "level1_accuracy": metrics["classification"]["level1"]["accuracy"],
        "level1_macro_f1": metrics["classification"]["level1"]["macro_f1"],
        "level2_accuracy": metrics["classification"]["level2"]["accuracy"],
        "level3_accuracy": metrics["classification"]["level3"]["accuracy"],
        "full_path_accuracy": metrics["full_path_accuracy"],
        "selected_threshold": metrics["clarification"]["selected"]["threshold"],
        "ece": metrics["calibration"]["ece"],
        "brier": metrics["calibration"]["brier_score"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    await close_agent_resources()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="仅用于冒烟验证，0 表示全量")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
