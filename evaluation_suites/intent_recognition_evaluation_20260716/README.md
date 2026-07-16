# 意图识别专项测评归档（2026-07-16）

本目录集中保存本次意图识别专项测评的方案、冻结测试集、执行脚本和最终结果。原始文件仍保留在项目原目录中，本目录为独立归档副本。

## 目录结构

```text
intent_recognition_evaluation_20260716/
├─ 01_意图识别专项测评方案.md
├─ benchmark/
│  └─ intent_benchmark_v1.jsonl       708 条冻结测评样本
├─ scripts/
│  ├─ build_intent_benchmark.py       测试集生成脚本快照
│  └─ run_intent_evaluation.py        测评执行脚本快照
└─ results/
   ├─ 意图识别专项测评报告.md
   ├─ 意图识别优化建议.md
   ├─ intent_eval_metrics.json
   ├─ intent_eval_predictions.jsonl
   └─ level1_confusion_matrix.csv
```

## 核心结果

- Level-1 Accuracy：76.41%
- Level-1 Macro-F1：75.50%
- Level-2 / Level-3 Accuracy：4.66%
- 多轮上下文 Level-1 Accuracy：60.00%
- 易混淆样本 Level-1 Accuracy：83.33%
- JSON 可解析率：100.00%
- 合法层级前缀率：5.08%

`scripts/` 中保存的是本次运行时的代码快照。需要重新执行时，优先使用项目目录 `evaluation_suites/01_intent_recognition_evaluation/` 下的脚本，以保证能够正确引用 Agent 源码和配置。
