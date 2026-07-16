# 美妆电商智能客服 Agent

这是一个面向美妆电商场景的任务型客服 Agent 原型，包含三级意图识别、多源 RAG、Elasticsearch 混合检索、多轮状态、业务工具确认、安全防护与人工转接。

## 项目结构

```text
beauty-customer-service-agent/
├─ agent_pipeline.py         Agent 核心编排：意图、路由、检索、工具、答案生成
├─ agent_safety.py           敏感信息脱敏、注入检测、护肤风险分级
├─ conversation_state.py     多轮消息、槽位与任务状态持久化
├─ business_tools/           订单查询与售后操作工具，含确认、权限和幂等
├─ handoff_store.py          人工转接工单持久化
├─ agent_observability.py    结构化日志与运行指标
├─ es_store/                 多源知识转换、ES 增量入库与混合检索
├─ vector_store/             本地 BGE 编码器（ES 入库和查询共用）
├─ intention_prompt/          美妆客服三级意图 Prompt
├─ data/                      公开原始数据与清洗后的知识、订单数据
├─ chunk/                     customer_dialogues 切分实验与评测产物
├─ qa_pairs/                 对话 QA 抽取脚本及结果
├─ evaluation_suites/        编号化专项离线测评
│  ├─ 01_intent_recognition_evaluation/
│  ├─ 02_rag_retrieval_evaluation/
│  ├─ 03_answer_generation_evaluation/
│  ├─ 04_tool_call_task_completion_evaluation/
│  └─ 05_performance_cost_handoff_evaluation/
├─ tests/                    安全、记忆、工具、RAG、Web 自动化测试
├─ web_app.py + web/         FastAPI 服务与演示页面
├─ scripts/                  初始化、检查、启动和端到端验收脚本
├─ diagrams/                 技术流程、在线时序和状态机图
└─ runtime/                  运行时日志和 SQLite 状态（自动生成，不入库）
```

## 核心链路

```text
用户问题
  -> 敏感信息脱敏 / 提示词注入检测 / 护肤风险分级
  -> 多意图识别与置信度路由
  -> 多源 RAG / 业务工具 / 混合执行
  -> 来源引用与安全边界生成
  -> 低置信度或高风险人工转接
```

## Web 演示

首次运行：

```powershell
.\scripts\init.ps1
# 编辑 .env，填写 DEEPSEEK_API_KEY
.\scripts\start_web.ps1
```

浏览器打开 `http://127.0.0.1:8000`。页面左侧用于模拟客户对话，右侧展示意图、路由、知识来源、业务工具、安全状态、工单与请求追踪。

如果 Elasticsearch 已经运行，可以跳过容器启动：

```powershell
.\scripts\start_web.ps1 -SkipDockerStart
```

## API

- `POST /api/chat`：执行一次 Agent 对话。
- `POST /api/handoff`：手动创建人工转接工单。
- `GET /api/handoffs`：查看最近工单。
- `GET /api/metrics`：查看请求量、延迟、路由和安全指标。
- `GET /health`：检查 DeepSeek 配置、Elasticsearch 与运行限制。
- `GET /docs`：FastAPI 自动接口文档。

示例请求：

```json
{
  "conversation_id": "demo-001",
  "question": "订单 MOCK202606260003 的快递到哪里了？",
  "messages": [],
  "knowledge_top_k": 3
}
```

## 运行治理

- 默认使用经过专项回归验证的 Hybrid 检索：上下文 Query 增强、口语归一化、来源感知重排和近重复过滤；RRF/MMR 仍可通过环境变量切换。
- 模型连接池与 BGE 模型进程内复用，高频查询向量使用 LRU 缓存。
- API 使用并发信号量、请求超时和按客户端 IP 的滑动窗口限流。
- 日志以 JSON 写入标准输出和 `runtime/agent.log`，不记录明文手机号、邮箱、身份证、银行卡或地址。
- 严重护肤不良反应跳过模型直接提示就医并创建人工工单；普通反应附加安全提示并转人工。
- 检测到提示词注入时不会进入意图识别、检索或工具执行链路。

## 验证

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.\scripts\check.ps1
```

所有订单和业务状态均为 Mock 数据，网页中的业务操作不代表真实电商系统已执行。
