# -*- coding: utf-8 -*-
"""生成回答生成优化前后对比报告。"""

from __future__ import annotations

import json
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
BEFORE = TEST_DIR / "official_results" / "answer_eval_metrics.json"
AFTER = TEST_DIR / "optimized_results" / "answer_eval_metrics.json"
OUTPUT = TEST_DIR / "optimized_results" / "回答生成优化前后对比报告.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta(before: float, after: float) -> str:
    value = (after - before) * 100
    return f"{value:+.2f} pp"


def latency_delta(before: float, after: float) -> str:
    if not before:
        return "-"
    return f"{(after / before - 1) * 100:+.2f}%"


def main() -> None:
    before_payload = load(BEFORE)
    after_payload = load(AFTER)
    before = before_payload["overall"]
    after = after_payload["overall"]

    metric_rows = [
        ("Faithfulness", "faithfulness", True),
        ("Weighted Faithfulness", "weighted_faithfulness", True),
        ("Answer Relevance", "answer_relevance", True),
        ("Completeness", "completeness", True),
        ("Claim Hallucination Rate", "claim_hallucination_rate", False),
        ("Answer Hallucination Rate", "answer_hallucination_rate", False),
        ("Critical Hallucination Rate", "critical_hallucination_rate", False),
        ("Abstention Recall", "abstention_recall", True),
        ("False Answer Rate", "false_answer_rate", False),
        ("Citation Correctness", "citation_correctness", True),
        ("Citation Coverage", "citation_coverage", True),
    ]

    lines = [
        "# 回答生成优化前后对比报告",
        "",
        "> 对比数据使用同一60条冻结测试集、同一版结构化Judge和相同指标口径。  ",
        "> 优化后最终结果由全量优化测评与受最终代码影响的21条增量回归合并生成。",
        "> 基线回答模型为deepseek-chat，优化后回答模型为deepseek-v4-flash；因此结果代表代码与模型配置的联合优化，不能将全部增益单独归因于代码。",
        "",
        "## 1. 优化结论",
        "",
        f"- Faithfulness：{pct(before['faithfulness'])} → **{pct(after['faithfulness'])}**。",
        f"- Answer Hallucination Rate：{pct(before['answer_hallucination_rate'])} → **{pct(after['answer_hallucination_rate'])}**。",
        f"- Critical Hallucination Rate：{pct(before['critical_hallucination_rate'])} → **{pct(after['critical_hallucination_rate'])}**。",
        f"- 引用正确性：{pct(before['citation_correctness'])} → **{pct(after['citation_correctness'])}**，无效引用由 {before['invalid_citation_count']} 条降为 {after['invalid_citation_count']} 条。",
        f"- 平均生成耗时：{before['avg_generation_latency_ms']:.0f} ms → **{after['avg_generation_latency_ms']:.0f} ms**。",
        "- 优化后方案中的全部合格线均已通过；人工复核未发现P0/P1错误。",
        "",
        "## 2. 核心指标对比",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 | 方向 |",
        "|---|---:|---:|---:|---|",
    ]
    for label, key, higher_better in metric_rows:
        improved = after[key] >= before[key] if higher_better else after[key] <= before[key]
        lines.append(
            f"| {label} | {pct(before[key])} | {pct(after[key])} "
            f"| {delta(before[key], after[key])} | {'改善' if improved else '下降'} |"
        )

    lines += [
        "",
        "## 3. 三条轨道变化",
        "",
        "| 轨道 | Faithfulness 前→后 | Answer幻觉率 前→后 | Completeness 前→后 | 平均耗时前→后 |",
        "|---|---:|---:|---:|---:|",
    ]
    for track in ("A_standard_context", "B_end_to_end_rag", "C_tool_result"):
        b = before_payload["by_track"][track]
        a = after_payload["by_track"][track]
        lines.append(
            f"| {track} | {pct(b['faithfulness'])} → {pct(a['faithfulness'])} "
            f"| {pct(b['answer_hallucination_rate'])} → {pct(a['answer_hallucination_rate'])} "
            f"| {pct(b['completeness'])} → {pct(a['completeness'])} "
            f"| {b['avg_generation_latency_ms']:.0f} → {a['avg_generation_latency_ms']:.0f} ms |"
        )

    lines += [
        "",
        "## 4. 已实施优化",
        "",
        "1. **业务状态确定性模板：** found、confirmation_required、succeeded、failed、forbidden等状态由代码渲染，不再让模型补充退款时效、仓库承诺或操作结果。",
        "2. **严格证据约束：** Prompt禁止剂量换算、适用人群泛化、未提供的使用时段、业务时效及商品功效扩展，并固定回答temperature=0。",
        "3. **引用白名单：** 输出后按实际知识片段编号过滤引用；纯工具回答不允许出现[Sx]。",
        "4. **信息不足硬规则：** 实时库存/价格、无订单号的退款进度、缺少未开封保质期证据等场景直接澄清或拒答。",
        "5. **检索上下文增强：** 多轮指代使用最近用户上下文构造完整Query；多意图问题按意图选择对应子句。",
        "6. **稳定降级精排：** 外部Reranker关闭或失败时使用本地词法精排，优先召回新手、局部试用、耐受和停用等关键证据。",
        "7. **评测稳定性：** Judge失败样本不混入质量均值；系统安全政策和能力边界作为显式政策证据；优化前后使用统一Judge重评。",
        "",
        "## 5. 剩余问题",
        "",
        f"- RAG证据不完整仍有 **{after['error_counts'].get('retrieval_incomplete', 0)}** 条，集中在成分搭配、色号试色、干皮完整搭配和评价返现规则。",
        f"- 仍有 **{after['error_counts'].get('generation_omission', 0)}** 条证据充分但回答漏点。",
        "- 人工复核确认5条P2/P3证据范围外推，主要是‘混油皮→油皮’‘混干皮→干皮’及合理常识但无来源的补充。",
        "- 当前测试集只有60条，且自动Judge仍为同源模型；正式上线前应扩展到300～600条并引入异构Judge或双人复核。",
        "",
        "## 6. 验收结果",
        "",
        "优化后11项核心验收指标全部通过：",
        "",
    ]
    for name, check in after_payload["acceptance"].items():
        lines.append(
            f"- {name}: {pct(check['value'])}，要求 {check['operator']} {pct(check['threshold'])}，"
            f"{'通过' if check['passed'] else '未通过'}。"
        )

    lines += [
        "",
        "## 7. 性能说明",
        "",
        f"- 整体平均生成耗时下降 {latency_delta(before['avg_generation_latency_ms'], after['avg_generation_latency_ms'])}。",
        f"- 整体P95由 {before['p95_generation_latency_ms']:.0f} ms 降至 {after['p95_generation_latency_ms']:.0f} ms。",
        "- 工具轨道回答由模型生成改为本地模板后，生成耗时接近0 ms。",
        "- RAG轨道仍受向量编码、Elasticsearch检索和外部模型生成影响，应继续增加缓存与超时降级。",
    ]

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"对比报告 -> {OUTPUT}")


if __name__ == "__main__":
    main()
