# 05 性能、成本与人工转接专项

本目录集中保存测评方案、优化前基线、优化实现快照、实验过程和优化后结果。

## 最终结论

- 测评请求成功率：19/19（100%）；故障降级成功率：100%。
- 人工转接：24 条专项样本 Precision / Recall 均为 100%，无漏转、无误转。
- 端到端延迟：P50 2.160 秒、P95 6.017 秒；较初始基线分别下降 83.36% 和 71.82%。
- 平均延迟下降 81.01%，4 并发吞吐提升 422.42% 至 1.531 RPS。
- Token 消耗下降 83.09%，推理成本下降 82.17%。
- 知识问答 P95 6.632 秒、混合链路 P95 5.736 秒，均达到原方案目标。
- ES 检索 P95 降至 0.438 秒，达到 500ms 目标。
- 当前知识源与 Elasticsearch 索引均为 25,520 条；17,453 条元数据已完成快速同步，二次校验 0 条变更。

核心结果见 `optimized/优化后性能成本与人工转接专项测评报告.md`，优化前后变化见 `optimized/优化前后对比报告.md`，后续工作见 `optimized/优化后剩余优化建议.md`。

## 目录

- `05_performance_cost_handoff_evaluation方案.md`：测评口径与验收标准。
- `baseline/`：优化前原始数据、报告和初始优化建议。
- `experiments/`：各轮优化实验与中间结果。
- `optimized/`：最终优化后原始数据、对比报告和剩余建议。
- `scripts/`：可复现的主测评与优化实验脚本。
- `implementation/`：本次涉及的核心实现与测试快照。

## 复现

在项目根目录执行：

```powershell
.venv\Scripts\python.exe "evaluation_suites\05_performance_cost_handoff_evaluation\scripts\run_performance_cost_handoff.py" --concurrency 4
```

质量回归：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\e2e_acceptance.py
```

最终验证结果：57 项自动化测试通过，4/4 端到端验收通过。
