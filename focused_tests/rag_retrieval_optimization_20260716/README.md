# RAG 检索focused_tests与优化

本目录独立保存本次 RAG 检索基线测试、优化实现、回归结果和最终报告，不修改项目现有的评测产物。

## 目录结构

```text
benchmark/       扩展后的弱监督检索基准集
implementation/  可独立复用的 Query、过滤、去重和轻量重排优化
scripts/         基准集构建与评测执行脚本
results/         参数扫描、逐条预测、失败案例和汇总指标
reports/         最终优化报告
```

## 执行方式

在项目根目录执行：

```powershell
.venv\Scripts\python.exe "focused_tests\rag_retrieval_optimization_20260716\scripts\build_benchmark.py"
.venv\Scripts\python.exe "focused_tests\rag_retrieval_optimization_20260716\scripts\run_rag_optimization.py"
```

评测依赖当前运行中的 Elasticsearch 索引 `customer_service_knowledge_v1`，不会调用 DeepSeek。
