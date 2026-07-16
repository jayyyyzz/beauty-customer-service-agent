# -*- coding: utf-8 -*-
"""无需调用 LLM/ES 的业务工具两阶段执行演示。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from business_tools import ActorContext, BusinessToolService


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        service = BusinessToolService(
            ROOT / "data" / "processed" / "order_mock_data.csv",
            Path(temp_dir) / "business_tools_demo.db",
        )
        actor = ActorContext("mock_user_003", "customer", "mock_user_003")
        arguments = {"reason": "演示：用户不再需要该商品"}

        prepared = service.execute(
            "cancel_order",
            "MOCK202606260002",
            actor=actor,
            arguments=arguments,
            idempotency_key="demo-cancel-order-0002",
        )
        print("\n第一阶段：准备操作")
        print(json.dumps(prepared, ensure_ascii=False, indent=2))

        if prepared.get("status") != "confirmation_required":
            return

        confirmed = service.execute(
            "cancel_order",
            "MOCK202606260002",
            actor=actor,
            arguments=arguments,
            idempotency_key=prepared["idempotency_key"],
            confirmation_token=prepared["confirmation_token"],
        )
        print("\n第二阶段：确认执行")
        print(json.dumps(confirmed, ensure_ascii=False, indent=2))

        replayed = service.execute(
            "cancel_order",
            "MOCK202606260002",
            actor=actor,
            arguments=arguments,
            idempotency_key=prepared["idempotency_key"],
            confirmation_token=prepared["confirmation_token"],
        )
        print("\n重复请求：幂等回放")
        print(json.dumps(replayed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
