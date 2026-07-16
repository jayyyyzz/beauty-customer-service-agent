# 多轮记忆与任务状态机

## 已实现能力

- SQLite 持久化保存会话消息、共享槽位和任务执行结果。
- 一次输入可拆分为多个独立意图，每个意图对应一个任务。
- 物流等订单任务缺少 `order_id` 时进入 `waiting_user`，不会错误调用工具。
- 等待期间可以先处理其他问题；之后补充订单号会恢复原任务。
- 最终答案会读取最近 20 条完整历史消息，支持“这个、刚才那款”等上下文表达。

默认状态文件为 `runtime/conversation_state.db`，可通过环境变量覆盖：

```text
AGENT_STATE_DB=runtime/conversation_state.db
```

调用方必须为同一段会话持续传入稳定的 `conversation_id`。订单工具还需要传入已登录用户的 `user_id`：

```python
history_dialogue = {
    "conversation_id": "conversation_001",
    "user_id": "mock_user_004",
    "messages": [],
}
```

## 演示脚本

使用同一个 `conversation_id` 连续调用三次：

1. `帮我查一下快递到哪里了` → 任务暂停，要求补充订单号。
2. `那先告诉我护肤顺序` → 新任务完成，物流任务仍在等待。
3. `订单号是 MOCK202606260003` → 自动恢复第一步的物流任务。

返回值中可观察：

- `tasks`：本轮创建或恢复的任务；
- `pending_tasks`：仍在等待用户信息的任务；
- `slots`：会话共享槽位；
- `resumed`：本轮是否恢复了历史任务。
