# 回答生成专项测评专题

本目录集中保存回答生成专项测评的方案、冻结基准集、执行脚本、人工复核记录和测评结果。

## 目录内容

- `03_answer_generation_evaluation方案.md`：原始测评方案。
- `answer_generation_benchmark_v1.jsonl`：60条三轨道冻结基准集。
- `build_answer_benchmark.py`：基准集构建脚本。
- `run_answer_evaluation.py`：真实回答生成与结构化Judge测评脚本。
- `rebuild_answer_report.py`：不重复调用模型，基于已有逐条结果重建指标和报告。
- `rejudge_answer_results.py`：不重复生成回答，使用当前统一Judge重新评分已有回答。
- `merge_incremental_results.py`：把指定样本或轨道的增量复测结果合并回完整报告。
- `build_optimization_comparison.py`：生成优化前后指标、轨道和性能对比报告。
- `answer_generation_manual_review_v1.json`：人工复核结论。
- `evaluation_run_manifest.json`：优化前后回答模型、Judge模型和结果生成方式记录。
- `official_results/`：完整指标、逐条预测、复核队列、测评报告及优化建议。
- `optimized_results/`：优化后完整报告、指标、逐条结果和优化前后对比报告。
- `optimization_process/`：最终代码影响范围内的增量复测结果。
- `smoke_tests/`：A/B/C三条轨道的小样本连通性验证结果。

## 重新执行

在项目根目录运行：

```powershell
.venv\Scripts\python.exe "evaluation_suites\03_answer_generation_evaluation\build_answer_benchmark.py"
.venv\Scripts\python.exe "evaluation_suites\03_answer_generation_evaluation\run_answer_evaluation.py" --concurrency 3
```

只根据已保存的回答和Judge结果重建报告：

```powershell
.venv\Scripts\python.exe "evaluation_suites\03_answer_generation_evaluation\rebuild_answer_report.py"
```
