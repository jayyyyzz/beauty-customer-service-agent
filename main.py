# -*- coding: utf-8 -*-
import asyncio
import json

from agent_pipeline import handle_user_question


async def main():
    history_dialogue = {
        "conversation_id": "test_001",
        "user_id": "mock_user_004",
        "messages": [
            {
                "role": "buyer",
                "content": "我最近脸有点干，还容易泛红",
            },
            {
                "role": "seller",
                "content": "可以看看这款修护精华，主打舒缓保湿",
            },
        ],
    }

    # 知识库类示例:
    question = "护肤流程有吗"

    # 业务 API 类示例，可以改成这一句测试订单查询:
    # question = "订单 MOCK202606260003 的快递到哪里了？"

    result = await handle_user_question(
        history_dialogue=history_dialogue,
        question=question,
    )

    print("\n========== 意图识别 ==========")
    print(json.dumps(result["intent"], ensure_ascii=False, indent=2))

    print("\n========== 路由 ==========")
    print(result["route"])

    print("\n========== 多轮任务状态 ==========")
    print(json.dumps({
        "conversation_id": result.get("conversation_id"),
        "resumed": result.get("resumed"),
        "slots": result.get("slots", {}),
        "tasks": result.get("tasks", []),
        "pending_tasks": result.get("pending_tasks", []),
    }, ensure_ascii=False, indent=2))

    print("\n========== 知识库召回 ==========")
    print(json.dumps(result["knowledge_docs"], ensure_ascii=False, indent=2))

    print("\n========== 业务 API 结果 ==========")
    print(json.dumps(result["api_data"], ensure_ascii=False, indent=2))

    print("\n========== 来源引用 ==========")
    print(json.dumps(result.get("citations", []), ensure_ascii=False, indent=2))

    print("\n========== 最终回复 ==========")
    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
