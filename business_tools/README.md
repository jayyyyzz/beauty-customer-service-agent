# 业务工具层

该模块把原来的订单 CSV 只读查询升级为可执行、可审计的本地业务服务。

## 已支持工具

- `query_order`：查询订单和物流，无需二次确认。
- `urge_shipment`：创建催发货工单。
- `request_refund`：提交退款审核。
- `cancel_order`：取消允许取消的未发货订单。
- `update_address`：修改未发货订单地址。
- `request_invoice`：申请个人或企业电子发票。

结构化 JSON Schema 位于 `tool_definitions.py`，可直接用于后续 LLM Function Calling。

## 安全执行模型

1. 使用 `ActorContext` 校验用户是否拥有目标订单；客服和管理员角色可代客处理。
2. 所有变更操作先返回 `confirmation_required`，包括脱敏后的操作摘要。
3. 用户确认时回传 `confirmation_token` 和 `idempotency_key`，服务才提交事务。
4. 相同幂等键重复请求只回放第一次结果，不会重复取消、退款或开票。
5. 订单变更和调用结果写入 SQLite 的 `orders`、`tool_operations` 表。
6. SQLite 锁冲突使用指数退避重试；每次确认前重新校验最新订单状态。

原始 `order_mock_data.csv` 不会被修改，首次运行时仅用于初始化 `runtime/business_tools.db`。

## 本地验证

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\demo_business_tools.py
```

## Agent 两阶段调用

首次请求通过 `handle_user_question(..., actor_context=..., tool_context=...)` 传入动作和参数。
返回值中的 `api_data.next_tool_context` 应由可信的服务端会话保存，不应让模型自行改写。
用户明确确认后，将该对象原样作为下一轮 `tool_context` 传回，即可执行变更。
