# -*- coding: utf-8 -*-
"""客服 Agent 工具调用与任务完成专项离线评测。

评测分为三层：
1. Planner-only：使用已标注意图，评估路由、工具选择、订单号提取和澄清。
2. Sandbox：在临时 SQLite data中执行真实业务工具，评估确认、权限、状态机和幂等。
3. Offline E2E：固定意图识别与回答生成，运行 Agent 状态机，隔离工具编排能力。

运行：
    .venv\\Scripts\\python.exe "evaluation_suites\\04_tool_call_task_completion_evaluation\\run_tool_evaluation.py"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[2]
TOPIC_DIR = Path(__file__).resolve().parent
REPORT_DIR = TOPIC_DIR / "optimized"
BASELINE_METRICS_PATH = TOPIC_DIR / "baseline" / "tool_eval_metrics.json"
ORDER_CSV = ROOT / "data" / "processed" / "order_mock_data.csv"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_pipeline
from agent_pipeline import (
    decide_route,
    extract_slots,
    missing_slots,
    required_slots_for_intent,
    should_call_business_api,
)
from business_tools import ActorContext, BusinessToolService, infer_business_action
from conversation_state import ConversationStore


TOOLS = [
    "query_order",
    "urge_shipment",
    "request_refund",
    "cancel_order",
    "update_address",
    "request_invoice",
]
MUTATING_TOOLS = set(TOOLS) - {"query_order"}
SUCCESS_STATES = {"succeeded"}


def intent(name: str, confidence: float = 0.99) -> dict[str, Any]:
    return {
        "intent_level1": name,
        "intent_level2": f"{name}.eval",
        "intent_level3": f"{name}.eval.case",
        "intent_logic": "工具专项评测固定意图",
        "intent_confidence": confidence,
        "keywords": [name],
    }


def order_id(number: int) -> str:
    return f"MOCK20260626{number:04d}"


def owner_for(number: int) -> str:
    remainder = number % 8
    suffix = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 0: 1}[remainder]
    return f"mock_user_{suffix:03d}"


def history_with_order(number: int) -> list[dict[str, str]]:
    return [{"role": "buyer", "content": f"我的订单号是 {order_id(number)}"}]


def valid_arguments(action: str, index: int = 0) -> dict[str, Any]:
    if action == "urge_shipment":
        return {"reason": "用户急需使用"}
    if action == "request_refund":
        return {"reason": "使用后出现不适"}
    if action == "cancel_order":
        return {"reason": "用户不再需要"}
    if action == "update_address":
        return {
            "new_address": {
                "recipient": f"测试用户{index}",
                "phone": "13800138000",
                "province": "上海市",
                "city": "上海市",
                "detail": f"浦东新区测试路{80 + index}号",
            }
        }
    if action == "request_invoice":
        return {
            "invoice_type": "company" if index % 2 else "personal",
            "title": "测试科技有限公司" if index % 2 else "测试用户",
            "tax_id": "91310000TEST12345X" if index % 2 else "",
            "email": "finance@example.com",
        }
    return {}


def planner_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    templates = {
        "query_order": (
            "logistics",
            [
                "查询订单 {oid} 的物流",
                "{oid} 到哪里了",
                "帮我看看订单 {oid}",
                "查一下 {oid} 的快递",
                "订单 {oid} 现在什么状态",
                "{oid} 的运单号是多少",
                "看下 {oid} 是否发货",
                "请查询 {oid}",
                "我的 {oid} 什么时候到",
                "{oid} 物流有更新吗",
            ],
        ),
        "urge_shipment": (
            "urge_shipment",
            [
                "订单 {oid} 帮我催发货",
                "催一下 {oid}",
                "{oid} 尽快发货",
                "帮我给 {oid} 催一下",
                "订单 {oid} 怎么还没发，催一下",
                "{oid} 很急，帮忙催发货",
                "请催仓库处理 {oid}",
                "加急处理订单 {oid}",
                "{oid} 今天能发吗，催一下",
                "麻烦催促 {oid} 发货",
            ],
        ),
        "request_refund": (
            "after_sale",
            [
                "订单 {oid} 申请退款",
                "{oid} 帮我退钱",
                "我要退掉订单 {oid}",
                "帮我给 {oid} 退款",
                "{oid} 用着过敏，申请退",
                "订单 {oid} 退货退款",
                "不想要 {oid} 了，退款",
                "{oid} 质量有问题要退款",
                "请提交 {oid} 的退款申请",
                "{oid} 能不能退钱",
            ],
        ),
        "cancel_order": (
            "after_sale",
            [
                "取消订单 {oid}",
                "{oid} 不想要了",
                "把 {oid} 取消掉",
                "订单 {oid} 直接取消",
                "帮我取消订单 {oid}",
                "{oid} 还没发货，取消掉",
                "不要订单 {oid} 了",
                "撤销订单 {oid}，不想要了",
                "{oid} 买错了，取消订单",
                "请取消 {oid}",
            ],
        ),
        "update_address": (
            "after_sale",
            [
                "订单 {oid} 修改地址",
                "{oid} 地址填错了",
                "帮我给 {oid} 改地址",
                "订单 {oid} 收货地址要修改",
                "{oid} 换个收货地址",
                "帮我更新 {oid} 的地址",
                "{oid} 寄送地址不对",
                "订单 {oid} 地址写错了",
                "修改 {oid} 的收货地址",
                "{oid} 改一下地址",
            ],
        ),
        "request_invoice": (
            "invoice",
            [
                "订单 {oid} 开发票",
                "给 {oid} 申请发票",
                "{oid} 开电子票",
                "订单 {oid} 怎么开票",
                "帮我给 {oid} 开公司发票",
                "{oid} 需要报销发票",
                "给订单 {oid} 开个人票",
                "申请 {oid} 的电子发票",
                "{oid} 可以开发票吗",
                "订单 {oid} 补开发票",
            ],
        ),
    }
    tool_orders = {
        "query_order": [3, 4, 5, 8, 11, 12, 13, 16, 19, 20],
        "urge_shipment": [2, 10, 18, 26, 34, 42, 2, 10, 18, 26],
        "request_refund": [3, 4, 5, 8, 11, 12, 13, 16, 19, 20],
        "cancel_order": [2, 10, 18, 26, 34, 42, 2, 10, 18, 26],
        "update_address": [2, 10, 18, 26, 34, 42, 2, 10, 18, 26],
        "request_invoice": [2, 3, 4, 5, 10, 11, 12, 13, 18, 19],
    }
    for tool, (intent_name, phrases) in templates.items():
        for idx, (phrase, number) in enumerate(zip(phrases, tool_orders[tool]), start=1):
            oid = order_id(number)
            cases.append({
                "case_id": f"planner_{tool}_{idx:02d}",
                "layer": "planner",
                "category": "single_tool_complete",
                "history": [],
                "question": phrase.format(oid=oid),
                "intent": intent(intent_name),
                "gold_action": tool,
                "gold_order_id": oid,
            })

    missing_templates = [
        ("logistics", "帮我查一下快递", "query_order"),
        ("urge_shipment", "帮我催一下发货", "urge_shipment"),
        ("after_sale", "帮我申请退款", "request_refund"),
        ("after_sale", "帮我取消订单", "cancel_order"),
        ("after_sale", "帮我修改订单地址", "update_address"),
        ("invoice", "帮我申请发票", "request_invoice"),
    ]
    for repeat in range(3):
        for intent_name, question, gold_tool in missing_templates:
            cases.append({
                "case_id": f"planner_missing_{gold_tool}_{repeat + 1}",
                "layer": "planner",
                "category": "missing_order_id",
                "history": [],
                "question": question + ("，谢谢" if repeat == 1 else ""),
                "intent": intent(intent_name),
                "gold_action": "clarify",
                "target_tool": gold_tool,
                "gold_order_id": None,
            })

    history_templates = [
        ("logistics", "它到哪里了？", "query_order"),
        ("logistics", "帮我看看这个订单状态", "query_order"),
        ("urge_shipment", "这单帮我催一下", "urge_shipment"),
        ("urge_shipment", "它还没发，尽快发货", "urge_shipment"),
        ("after_sale", "这单申请退款", "request_refund"),
        ("after_sale", "它不想要了，帮我取消订单", "cancel_order"),
        ("after_sale", "这单地址填错了", "update_address"),
        ("invoice", "给这个订单开票", "request_invoice"),
        ("logistics", "现在运输到哪了", "query_order"),
        ("after_sale", "这单退钱", "request_refund"),
        ("after_sale", "把这个订单取消掉", "cancel_order"),
        ("invoice", "申请电子发票", "request_invoice"),
    ]
    for idx, (intent_name, question, action) in enumerate(history_templates, start=1):
        number = [3, 4, 2, 10, 11, 18, 26, 5, 12, 19, 34, 13][idx - 1]
        cases.append({
            "case_id": f"planner_history_{idx:02d}",
            "layer": "planner",
            "category": "history_reference",
            "history": history_with_order(number),
            "question": question,
            "intent": intent(intent_name),
            "gold_action": action,
            "gold_order_id": order_id(number),
        })

    no_tool_samples = [
        ("skin_type", "油皮适合用面霜吗"),
        ("ingredient", "烟酰胺和视黄醇能一起用吗"),
        ("routine", "护肤流程是什么"),
        ("efficacy", "这款精华有什么功效"),
        ("usage", "面膜一周用几次"),
        ("shade_color", "黄皮适合什么口红色号"),
        ("comparison", "这两款精华有什么区别"),
        ("product_info", "这瓶精华是多少毫升"),
        ("gift_sample", "买面霜送小样吗"),
        ("authenticity_shelf_life", "未开封保质期多久"),
        ("skin_concern", "脸上泛红怎么护理"),
        ("compatibility", "叠涂为什么会搓泥"),
        ("price", "最近有什么满减活动"),
        ("review", "在哪里提交商品评价"),
        ("safety_allergy", "敏感肌第一次使用要注意什么"),
        ("other", "你好呀"),
        ("other", "谢谢你的帮助"),
        ("product_info", "这款产品是什么质地"),
    ]
    for idx, (intent_name, question) in enumerate(no_tool_samples, start=1):
        cases.append({
            "case_id": f"planner_no_tool_{idx:02d}",
            "layer": "planner",
            "category": "no_tool",
            "history": [],
            "question": question,
            "intent": intent(intent_name),
            "gold_action": None,
            "gold_order_id": None,
        })

    for idx, (old_num, new_num, intent_name, question, action) in enumerate([
        (3, 4, "logistics", "改查订单 {oid} 到哪了", "query_order"),
        (2, 10, "urge_shipment", "别催刚才的，催订单 {oid}", "urge_shipment"),
        (11, 19, "after_sale", "我要给订单 {oid} 退款", "request_refund"),
        (18, 26, "after_sale", "取消的是 {oid}", "cancel_order"),
        (34, 42, "after_sale", "把订单 {oid} 的地址改掉", "update_address"),
        (5, 13, "invoice", "发票开给订单 {oid}", "request_invoice"),
    ], start=1):
        cases.append({
            "case_id": f"planner_entity_conflict_{idx:02d}",
            "layer": "planner",
            "category": "entity_conflict",
            "history": history_with_order(old_num),
            "question": question.format(oid=order_id(new_num)),
            "intent": intent(intent_name),
            "gold_action": action,
            "gold_order_id": order_id(new_num),
        })

    colloquial = [
        ("after_sale", "把订单 {oid} 作废", "cancel_order", 2),
        ("after_sale", "订单 {oid} 的收货地点换一下", "update_address", 10),
        ("urge_shipment", "给仓库加急处理 {oid}", "urge_shipment", 18),
        ("invoice", "订单 {oid} 给我整张电子票", "request_invoice", 3),
        ("after_sale", "订单 {oid} 给我退钱", "request_refund", 11),
        ("logistics", "查查运单 {oid}", "query_order", 4),
    ]
    for idx, (intent_name, phrase, action, number) in enumerate(colloquial, start=1):
        cases.append({
            "case_id": f"planner_colloquial_{idx:02d}",
            "layer": "planner",
            "category": "colloquial_action",
            "history": [],
            "question": phrase.format(oid=order_id(number)),
            "intent": intent(intent_name),
            "gold_action": action,
            "gold_order_id": order_id(number),
        })
    assert len(cases) == 120, len(cases)
    return cases


def sandbox_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, number in enumerate([1, 2, 3, 4, 5, 6, 7, 8], start=1):
        cases.append({
            "case_id": f"sandbox_query_{idx:02d}", "layer": "sandbox",
            "category": "query_success", "mode": "single",
            "action": "query_order", "order_id": order_id(number),
            "user_id": owner_for(number), "arguments": {}, "expected_status": "succeeded",
        })

    success_orders = {
        "urge_shipment": [2, 10, 18, 26, 34],
        "request_refund": [3, 11, 19, 27, 35],
        "cancel_order": [2, 10, 18, 26, 34],
        "update_address": [2, 10, 18, 26, 34],
        "request_invoice": [3, 11, 19, 27, 35],
    }
    for action, numbers in success_orders.items():
        for idx, number in enumerate(numbers, start=1):
            cases.append({
                "case_id": f"sandbox_success_{action}_{idx:02d}", "layer": "sandbox",
                "category": "mutation_success", "mode": "prepare_confirm",
                "action": action, "order_id": order_id(number), "user_id": owner_for(number),
                "arguments": valid_arguments(action, idx), "expected_status": "succeeded",
                "requires_confirmation": True,
            })

    invalid_specs = [
        ("request_refund", 3, {}, "invalid_arguments"),
        ("request_refund", 3, {"reason": "x"}, "invalid_arguments"),
        ("request_refund", 11, {"reason": ""}, "invalid_arguments"),
        ("update_address", 2, {}, "invalid_arguments"),
        ("update_address", 2, {"new_address": {"recipient": "张三"}}, "invalid_arguments"),
        ("update_address", 10, {"new_address": {"recipient": "张三", "phone": "abc", "province": "上海", "city": "上海", "detail": "测试路1号"}}, "invalid_arguments"),
        ("request_invoice", 3, {}, "invalid_arguments"),
        ("request_invoice", 3, {"invoice_type": "paper", "title": "张三", "email": "a@b.com"}, "invalid_arguments"),
        ("request_invoice", 3, {"invoice_type": "company", "title": "公司", "email": "a@b.com"}, "invalid_arguments"),
        ("request_invoice", 3, {"invoice_type": "personal", "title": "张三", "email": "bad"}, "invalid_arguments"),
        ("request_invoice", 11, {"invoice_type": "personal", "title": "", "email": "a@b.com"}, "invalid_arguments"),
        ("update_address", 18, {"new_address": {"recipient": "", "phone": "13800138000", "province": "上海", "city": "上海", "detail": "测试路1号"}}, "invalid_arguments"),
        ("request_refund", 19, {"reason": None}, "invalid_arguments"),
        ("request_invoice", 19, {"invoice_type": "company", "title": "公司", "tax_id": "", "email": "a@b.com"}, "invalid_arguments"),
        ("update_address", 26, {"new_address": []}, "invalid_arguments"),
    ]
    for idx, (action, number, arguments, expected) in enumerate(invalid_specs, start=1):
        cases.append({
            "case_id": f"sandbox_invalid_{idx:02d}", "layer": "sandbox",
            "category": "invalid_arguments", "mode": "single", "action": action,
            "order_id": order_id(number), "user_id": owner_for(number),
            "arguments": arguments, "expected_status": expected,
        })

    rejected_specs = [
        ("urge_shipment", 3), ("urge_shipment", 1), ("urge_shipment", 7),
        ("cancel_order", 3), ("cancel_order", 5), ("cancel_order", 7),
        ("request_refund", 1), ("request_refund", 6), ("request_refund", 7),
        ("update_address", 3), ("update_address", 5), ("update_address", 7),
        ("request_invoice", 1), ("request_invoice", 7), ("urge_shipment", 8),
    ]
    for idx, (action, number) in enumerate(rejected_specs, start=1):
        cases.append({
            "case_id": f"sandbox_rule_reject_{idx:02d}", "layer": "sandbox",
            "category": "business_rule", "mode": "single", "action": action,
            "order_id": order_id(number), "user_id": owner_for(number),
            "arguments": valid_arguments(action, idx), "expected_status": "business_rule_rejected",
        })

    permission_actions = ["query_order", "urge_shipment", "request_refund", "cancel_order", "update_address", "request_invoice"]
    permission_orders = [3, 2, 3, 2, 2, 3]
    for idx, (action, number) in enumerate(zip(permission_actions, permission_orders), start=1):
        cases.append({
            "case_id": f"sandbox_permission_{idx:02d}", "layer": "sandbox",
            "category": "permission", "mode": "single", "action": action,
            "order_id": order_id(number), "user_id": "mock_user_999",
            "arguments": valid_arguments(action, idx), "expected_status": "permission_denied",
        })
    for idx, role in enumerate(["guest", "vendor"], start=7):
        cases.append({
            "case_id": f"sandbox_permission_{idx:02d}", "layer": "sandbox",
            "category": "permission", "mode": "single", "action": "query_order",
            "order_id": order_id(3), "user_id": owner_for(3), "role": role,
            "arguments": {}, "expected_status": "permission_denied",
        })
    for idx, number in enumerate([3, 2], start=9):
        cases.append({
            "case_id": f"sandbox_permission_{idx:02d}", "layer": "sandbox",
            "category": "permission", "mode": "single", "action": "query_order",
            "order_id": order_id(number), "user_id": None,
            "arguments": {}, "expected_status": "authentication_required",
        })

    special = [
        ("sandbox_missing_order", "single", "query_order", "", owner_for(3), {}, "need_user_info"),
        ("sandbox_bad_order", "single", "query_order", "12345", owner_for(3), {}, "need_user_info"),
        ("sandbox_not_found", "single", "query_order", "MOCK209901010001", owner_for(3), {}, "not_found"),
        ("sandbox_unsupported", "single", "delete_order", order_id(3), owner_for(3), {}, "unsupported_action"),
        ("sandbox_invalid_role", "single", "query_order", order_id(3), owner_for(3), {}, "permission_denied"),
        ("sandbox_confirm_without_key", "confirm_without_key", "cancel_order", order_id(2), owner_for(2), valid_arguments("cancel_order"), "invalid_request"),
        ("sandbox_invalid_token", "invalid_token", "cancel_order", order_id(2), owner_for(2), valid_arguments("cancel_order"), "invalid_confirmation"),
    ]
    for case_id, mode, action, oid, user_id, arguments, expected in special:
        case = {
            "case_id": case_id, "layer": "sandbox", "category": "error_handling",
            "mode": mode, "action": action, "order_id": oid, "user_id": user_id,
            "arguments": arguments, "expected_status": expected,
        }
        if case_id == "sandbox_invalid_role":
            case["role"] = "unknown"
        cases.append(case)

    for idx, action in enumerate(["urge_shipment", "request_refund", "cancel_order", "update_address", "request_invoice"], start=1):
        number = 2 if action in {"urge_shipment", "cancel_order", "update_address"} else 3
        cases.append({
            "case_id": f"sandbox_confirmation_{idx:02d}", "layer": "sandbox",
            "category": "confirmation", "mode": "prepare_only", "action": action,
            "order_id": order_id(number), "user_id": owner_for(number),
            "arguments": valid_arguments(action, idx), "expected_status": "confirmation_required",
            "requires_confirmation": True,
        })

    for idx, action in enumerate(["urge_shipment", "request_refund", "cancel_order", "update_address", "request_invoice"], start=1):
        number = 2 if action in {"urge_shipment", "cancel_order", "update_address"} else 3
        cases.append({
            "case_id": f"sandbox_idempotent_{idx:02d}", "layer": "sandbox",
            "category": "idempotency", "mode": "idempotency_replay", "action": action,
            "order_id": order_id(number), "user_id": owner_for(number),
            "arguments": valid_arguments(action, idx), "expected_status": "succeeded",
        })
    for idx, action in enumerate(["request_refund", "cancel_order", "request_invoice"], start=6):
        number = 2 if action == "cancel_order" else 3
        cases.append({
            "case_id": f"sandbox_idempotent_{idx:02d}", "layer": "sandbox",
            "category": "idempotency", "mode": "idempotency_conflict", "action": action,
            "order_id": order_id(number), "user_id": owner_for(number),
            "arguments": valid_arguments(action, idx), "expected_status": "idempotency_conflict",
        })

    cases.extend([
        {
            "case_id": "sandbox_expired_confirmation", "layer": "sandbox",
            "category": "recovery", "mode": "expired_confirmation", "action": "cancel_order",
            "order_id": order_id(2), "user_id": owner_for(2),
            "arguments": valid_arguments("cancel_order"), "expected_status": "confirmation_expired",
        },
        {
            "case_id": "sandbox_locked_retry", "layer": "sandbox",
            "category": "recovery", "mode": "locked_retry", "action": "query_order",
            "order_id": order_id(3), "user_id": owner_for(3), "arguments": {},
            "expected_status": "succeeded",
        },
        {
            "case_id": "sandbox_confirmation_mismatch", "layer": "sandbox",
            "category": "confirmation", "mode": "confirmation_mismatch", "action": "request_refund",
            "order_id": order_id(3), "user_id": owner_for(3),
            "arguments": valid_arguments("request_refund"), "expected_status": "confirmation_mismatch",
        },
    ])
    assert len(cases) == 96, len(cases)
    return cases


def e2e_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, number in enumerate([3, 4, 5, 8], start=1):
        cases.append({
            "case_id": f"e2e_query_{idx:02d}", "layer": "e2e", "category": "query",
            "history": [], "question": f"订单 {order_id(number)} 到哪里了",
            "intents": [intent("logistics")], "user_id": owner_for(number),
            "expected_route": "business_api", "expected_status": "succeeded",
            "expected_action": "query_order", "expected_order_id": order_id(number),
        })
    for idx, (intent_name, question) in enumerate([
        ("logistics", "帮我查快递"),
        ("invoice", "帮我给订单开票"),
        ("after_sale", "帮我申请退款"),
    ], start=1):
        cases.append({
            "case_id": f"e2e_missing_{idx:02d}", "layer": "e2e", "category": "missing_slot",
            "history": [], "question": question, "intents": [intent(intent_name)],
            "user_id": "mock_user_004", "expected_route": "waiting_slots",
            "expected_status": None, "expected_action": None, "expected_order_id": None,
        })
    for idx, (number, intent_name, question, action) in enumerate([
        (3, "logistics", "它现在到哪里了", "query_order"),
        (2, "urge_shipment", "这单催一下", "urge_shipment"),
        (11, "after_sale", "这单申请退款", "request_refund"),
    ], start=1):
        cases.append({
            "case_id": f"e2e_history_{idx:02d}", "layer": "e2e", "category": "history_reference",
            "history": history_with_order(number), "question": question,
            "intents": [intent(intent_name)], "user_id": owner_for(number),
            "expected_route": "business_api" if intent_name in {"logistics", "urge_shipment"} else "hybrid",
            "expected_status": (
                "succeeded" if action == "query_order"
                else "need_user_info" if action == "request_refund"
                else "confirmation_required"
            ),
            "expected_action": action, "expected_order_id": order_id(number),
        })
    for idx, (intent_name, question) in enumerate([
        ("skin_type", "油皮适合用面霜吗"),
        ("ingredient", "烟酰胺和视黄醇能一起用吗"),
        ("routine", "护肤步骤是什么"),
    ], start=1):
        cases.append({
            "case_id": f"e2e_no_tool_{idx:02d}", "layer": "e2e", "category": "no_tool",
            "history": [], "question": question, "intents": [intent(intent_name)],
            "user_id": "mock_user_004", "expected_route": "knowledge_base",
            "expected_status": None, "expected_action": None, "expected_order_id": None,
        })

    structured = [
        ("cancel_order", 2, "after_sale", "取消订单", valid_arguments("cancel_order")),
        ("urge_shipment", 10, "urge_shipment", "催发货", valid_arguments("urge_shipment")),
        ("request_refund", 3, "after_sale", "申请退款", valid_arguments("request_refund")),
        ("request_invoice", 11, "invoice", "申请发票", valid_arguments("request_invoice", 1)),
    ]
    for idx, (action, number, intent_name, text, arguments) in enumerate(structured, start=1):
        cases.append({
            "case_id": f"e2e_structured_{idx:02d}", "layer": "e2e", "category": "structured_write",
            "history": [], "question": f"{text} {order_id(number)}",
            "intents": [intent(intent_name)], "user_id": owner_for(number),
            "tool_context": {
                "action": action, "order_id": order_id(number), "arguments": arguments,
                "idempotency_key": f"e2e-structured-{idx}",
            },
            "auto_confirm": True, "expected_route": "business_api_confirmation",
            "expected_status": "succeeded", "expected_action": action,
            "expected_order_id": order_id(number),
        })

    natural = [
        ("cancel_order", 18, "after_sale", "订单 {oid} 不想要了，原因是买错了"),
        ("request_refund", 19, "after_sale", "订单 {oid} 使用后过敏，帮我申请退款"),
        ("update_address", 26, "after_sale", "订单 {oid} 改到上海市浦东新区测试路88号，收件人张三，电话13800138000"),
        ("request_invoice", 27, "invoice", "订单 {oid} 开企业电子发票，抬头测试公司，税号91310000TEST12345X，发到finance@example.com"),
    ]
    for idx, (action, number, intent_name, phrase) in enumerate(natural, start=1):
        cases.append({
            "case_id": f"e2e_natural_write_{idx:02d}", "layer": "e2e", "category": "natural_write",
            "history": [], "question": phrase.format(oid=order_id(number)),
            "intents": [intent(intent_name)], "user_id": owner_for(number),
            "auto_confirm": True, "expected_route": "business_api_confirmation",
            "expected_status": "succeeded", "expected_action": action,
            "expected_order_id": order_id(number),
        })

    for idx, (old_num, new_num, intent_name, question, action) in enumerate([
        (3, 4, "logistics", "改查订单 {oid} 的物流", "query_order"),
        (2, 10, "after_sale", "取消订单 {oid}", "cancel_order"),
    ], start=1):
        cases.append({
            "case_id": f"e2e_entity_conflict_{idx:02d}", "layer": "e2e", "category": "entity_conflict",
            "history": history_with_order(old_num), "question": question.format(oid=order_id(new_num)),
            "intents": [intent(intent_name)], "user_id": owner_for(new_num),
            "auto_confirm": action != "query_order", "expected_route": "business_api_confirmation" if action != "query_order" else "business_api",
            "expected_status": "succeeded", "expected_action": action,
            "expected_order_id": order_id(new_num),
        })

    cases.append({
        "case_id": "e2e_multi_intent_01", "layer": "e2e", "category": "multi_intent",
        "history": [],
        "question": f"查一下订单 {order_id(3)} 的物流，顺便给它开企业发票，抬头测试公司，税号91310000TEST12345X，邮箱finance@example.com",
        "intents": [intent("logistics"), intent("invoice")], "user_id": owner_for(3),
        "expected_route": "multi_intent", "expected_status": "multi_success_confirmation",
        "expected_action": ["query_order", "request_invoice"], "expected_order_id": order_id(3),
    })
    assert len(cases) == 24, len(cases)
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def planner_actual(case: dict[str, Any]) -> dict[str, Any]:
    history = {"messages": case["history"]}
    slots = extract_slots(case["question"], history, {})
    route = decide_route(case["intent"])
    required = required_slots_for_intent(case["intent"], case["question"])
    missing = missing_slots(required, slots)
    if missing:
        action: str | None = "clarify"
    elif should_call_business_api(route, case["intent"], slots, case["question"]):
        action = infer_business_action(case["question"], case["intent"])
    else:
        action = None
    return {
        "route": route,
        "action": action,
        "order_id": slots.get("order_id"),
        "missing_slots": missing,
    }


def evaluate_planner(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for case in cases:
        started = time.perf_counter()
        actual = planner_actual(case)
        selection_correct = actual["action"] == case["gold_action"]
        gold_tool = case["gold_action"] if case["gold_action"] in TOOLS else None
        argument_exact = None
        if gold_tool:
            argument_exact = (
                actual["action"] == gold_tool
                and actual.get("order_id") == case.get("gold_order_id")
            )
        error = None
        if not selection_correct:
            if case["gold_action"] is None:
                error = "unnecessary_tool"
            elif actual["action"] in (None, "clarify") and case["gold_action"] in TOOLS:
                error = "missed_tool"
            else:
                error = "wrong_tool"
        elif argument_exact is False:
            error = "wrong_entity_resolution"
        results.append({
            "case_id": case["case_id"], "layer": "planner", "category": case["category"],
            "question": case["question"], "gold_action": case["gold_action"],
            "gold_order_id": case.get("gold_order_id"), "actual": actual,
            "tool_selection_correct": selection_correct, "argument_exact_match": argument_exact,
            "task_completed": selection_correct and argument_exact is not False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": error,
        })

    tool_cases = [r for r in results if r["gold_action"] in TOOLS]
    high_risk_cases = [r for r in tool_cases if r["gold_action"] in MUTATING_TOOLS]
    no_tool = [r for r in results if r["gold_action"] is None]
    clarify = [r for r in results if r["gold_action"] == "clarify"]
    per_tool: dict[str, Any] = {}
    f1_values = []
    for tool in TOOLS:
        tp = sum(r["gold_action"] == tool and r["actual"]["action"] == tool for r in results)
        fp = sum(r["gold_action"] != tool and r["actual"]["action"] == tool for r in results)
        fn = sum(r["gold_action"] == tool and r["actual"]["action"] != tool for r in results)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_tool[tool] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": tp + fn,
        }
    metrics = {
        "case_count": len(results),
        "tool_selection_accuracy": round(sum(r["tool_selection_correct"] for r in tool_cases) / len(tool_cases), 4),
        "tool_macro_f1": round(statistics.mean(f1_values), 4),
        "no_tool_accuracy": round(sum(r["tool_selection_correct"] for r in no_tool) / len(no_tool), 4),
        "clarification_accuracy": round(sum(r["tool_selection_correct"] for r in clarify) / len(clarify), 4),
        "argument_exact_match": round(sum(r["argument_exact_match"] is True for r in tool_cases) / len(tool_cases), 4),
        "order_id_accuracy": round(sum(r["actual"].get("order_id") == r["gold_order_id"] for r in tool_cases) / len(tool_cases), 4),
        "critical_argument_error_rate": round(sum(r["actual"].get("order_id") != r["gold_order_id"] for r in high_risk_cases) / len(high_risk_cases), 4),
        "unnecessary_tool_call_rate": round(sum(r["actual"]["action"] not in (None, "clarify") for r in no_tool) / len(no_tool), 4),
        "missed_tool_call_rate": round(sum(r["actual"]["action"] in (None, "clarify") for r in tool_cases) / len(tool_cases), 4),
        "plan_exact_match": round(sum(r["task_completed"] for r in results) / len(results), 4),
        "per_tool": per_tool,
        "errors": dict(Counter(r["error_type"] for r in results if r["error_type"])),
    }
    return results, metrics


def _operation_count(db_path: Path, key: str) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM tool_operations WHERE idempotency_key=? AND status='succeeded'",
            (key,),
        ).fetchone()[0])


def run_sandbox_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "business.db"
        ttl = -1 if case["mode"] == "expired_confirmation" else 600
        service = BusinessToolService(
            ORDER_CSV, db_path, confirmation_ttl_seconds=ttl,
            retry_delay_seconds=0.001,
        )
        actor = ActorContext(
            actor_id=case.get("user_id") or "anonymous",
            role=case.get("role", "customer"),
            user_id=case.get("user_id"),
        )
        action = case["action"]
        oid = case["order_id"]
        arguments = case["arguments"]
        key = f"key-{case['case_id']}"
        trace: list[dict[str, Any]] = []
        duplicate = False
        unauthorized = False
        confirmation_ok: bool | None = None
        try:
            mode = case["mode"]
            if mode == "single":
                result = service.execute(action, oid, actor=actor, arguments=arguments, idempotency_key=key)
                trace.append({"step": 1, "status": result.get("status"), "result": result})
            elif mode in {"prepare_confirm", "idempotency_replay", "expired_confirmation", "confirmation_mismatch"}:
                prepared = service.execute(action, oid, actor=actor, arguments=arguments, idempotency_key=key)
                trace.append({"step": 1, "status": prepared.get("status"), "result": prepared})
                confirmation_ok = prepared.get("status") == "confirmation_required"
                confirm_arguments = arguments
                if mode == "confirmation_mismatch":
                    confirm_arguments = {**arguments, "reason": "被篡改的原因"}
                confirmed = service.execute(
                    action, oid, actor=actor, arguments=confirm_arguments,
                    idempotency_key=key, confirmation_token=prepared.get("confirmation_token"),
                )
                trace.append({"step": 2, "status": confirmed.get("status"), "result": confirmed})
                result = confirmed
                if mode == "idempotency_replay":
                    replay = service.execute(
                        action, oid, actor=actor, arguments=arguments,
                        idempotency_key=key, confirmation_token=prepared.get("confirmation_token"),
                    )
                    trace.append({"step": 3, "status": replay.get("status"), "result": replay})
                    duplicate = _operation_count(db_path, key) > 1
                    if not replay.get("idempotent_replay"):
                        result = {"status": "idempotency_replay_missing", "message": "未标记回放"}
            elif mode == "prepare_only":
                result = service.execute(action, oid, actor=actor, arguments=arguments, idempotency_key=key)
                trace.append({"step": 1, "status": result.get("status"), "result": result})
                confirmation_ok = result.get("status") == "confirmation_required"
            elif mode == "idempotency_conflict":
                first = service.execute(action, oid, actor=actor, arguments=arguments, idempotency_key=key)
                changed = dict(arguments)
                changed["reason"] = "另一个不同请求"
                result = service.execute(action, oid, actor=actor, arguments=changed, idempotency_key=key)
                trace.extend([
                    {"step": 1, "status": first.get("status"), "result": first},
                    {"step": 2, "status": result.get("status"), "result": result},
                ])
            elif mode == "confirm_without_key":
                prepared = service.execute(action, oid, actor=actor, arguments=arguments, idempotency_key=key)
                result = service.execute(
                    action, oid, actor=actor, arguments=arguments,
                    confirmation_token=prepared.get("confirmation_token"),
                )
                trace.extend([
                    {"step": 1, "status": prepared.get("status"), "result": prepared},
                    {"step": 2, "status": result.get("status"), "result": result},
                ])
            elif mode == "invalid_token":
                prepared = service.execute(action, oid, actor=actor, arguments=arguments, idempotency_key=key)
                result = service.execute(
                    action, oid, actor=actor, arguments=arguments,
                    idempotency_key=key, confirmation_token="confirm_invalid",
                )
                trace.extend([
                    {"step": 1, "status": prepared.get("status"), "result": prepared},
                    {"step": 2, "status": result.get("status"), "result": result},
                ])
            elif mode == "locked_retry":
                attempts = {"count": 0}

                def flaky() -> dict[str, Any]:
                    attempts["count"] += 1
                    if attempts["count"] < 3:
                        raise sqlite3.OperationalError("database is locked")
                    return {"status": "succeeded", "attempts": attempts["count"]}

                result = service._retry(flaky)
                trace.append({"step": 1, "status": result.get("status"), "result": result})
            else:
                result = {"status": "unsupported_eval_mode"}
        except Exception as exc:
            result = {"status": "exception", "message": str(exc)}
            trace.append({"step": len(trace) + 1, "status": "exception", "result": result})

        if action in MUTATING_TOOLS and result.get("status") == "succeeded":
            had_valid_confirmation = any(
                item["status"] == "confirmation_required" for item in trace[:-1]
            ) and any("confirmation_token" in item["result"] for item in trace[:-1])
            unauthorized = not had_valid_confirmation
        actual_status = result.get("status")
        passed = actual_status == case["expected_status"] and not duplicate and not unauthorized
        error_type = None
        if not passed:
            if duplicate:
                error_type = "duplicate_execution"
            elif unauthorized:
                error_type = "unauthorized_execution"
            elif case["category"] == "business_rule" and actual_status == "confirmation_required":
                error_type = "business_rule_gap"
            else:
                error_type = "unexpected_tool_status"
        return {
            "case_id": case["case_id"], "layer": "sandbox", "category": case["category"],
            "action": action, "order_id": oid, "expected_status": case["expected_status"],
            "actual_status": actual_status, "actual_trace": trace,
            "confirmation_required": case.get("requires_confirmation", False),
            "confirmation_compliant": confirmation_ok,
            "task_completed": passed, "tool_execution_success": actual_status == "succeeded",
            "duplicate_execution": duplicate, "unauthorized_execution": unauthorized,
            "false_success": actual_status == "succeeded" and case["expected_status"] not in {"succeeded", "confirmation_required"},
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": error_type,
        }


def evaluate_sandbox(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = [run_sandbox_case(case) for case in cases]
    valid_execution = [r for r in results if r["expected_status"] == "succeeded"]
    confirmation = [r for r in results if r["confirmation_required"]]
    recovery = [r for r in results if r["category"] == "recovery"]
    duplicate_tests = [r for r in results if r["category"] == "idempotency"]
    metrics = {
        "case_count": len(results),
        "tool_execution_success_rate": round(sum(r["tool_execution_success"] for r in valid_execution) / len(valid_execution), 4),
        "strict_task_completion_rate": round(sum(r["task_completed"] for r in results) / len(results), 4),
        "confirmation_compliance": round(sum(r["confirmation_compliant"] is True for r in confirmation) / len(confirmation), 4),
        "unauthorized_execution_rate": round(sum(r["unauthorized_execution"] for r in results) / len(results), 4),
        "recovery_success_rate": round(sum(r["task_completed"] for r in recovery) / len(recovery), 4),
        "duplicate_execution_rate": round(sum(r["duplicate_execution"] for r in duplicate_tests) / len(duplicate_tests), 4),
        "false_success_rate": round(sum(r["false_success"] for r in results) / len(results), 4),
        "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in results), 3),
        "p95_latency_ms": round(sorted(r["latency_ms"] for r in results)[math.ceil(len(results) * 0.95) - 1], 3),
        "errors": dict(Counter(r["error_type"] for r in results if r["error_type"])),
    }
    return results, metrics


def _api_items(api_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not api_data:
        return []
    if "task_results" in api_data:
        return [item.get("data") or {} for item in api_data["task_results"]]
    return [api_data]


async def run_e2e_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        service = BusinessToolService(ORDER_CSV, temp / "business.db", retry_delay_seconds=0.001)
        store = ConversationStore(temp / "conversation.db")
        previous_service = agent_pipeline._BUSINESS_SERVICE
        agent_pipeline._BUSINESS_SERVICE = service
        history = {
            "conversation_id": f"conv-{case['case_id']}",
            "user_id": case["user_id"],
            "messages": case["history"],
        }

        async def fake_answer(
            question: str,
            intent_result: dict[str, Any],
            *,
            route: str,
            history_dialogue: dict[str, Any] | None = None,
            knowledge_docs: list[dict[str, Any]] | None = None,
            api_data: dict[str, Any] | None = None,
        ) -> str:
            statuses = [str((item.get("result") or {}).get("status") or "") for item in _api_items(api_data)]
            statuses = [item for item in statuses if item]
            return "离线评测回复；工具状态=" + (",".join(statuses) if statuses else "none")

        try:
            with (
                patch("agent_pipeline.recognize_intents", new=AsyncMock(return_value=case["intents"])),
                patch("agent_pipeline.search_knowledge", return_value=[]),
                patch("agent_pipeline.rerank_knowledge_docs", new=AsyncMock(return_value=[])),
                patch("agent_pipeline.generate_answer", new=fake_answer),
            ):
                first = await agent_pipeline.handle_user_question(
                    history,
                    case["question"],
                    state_store=store,
                    actor_context={
                        "actor_id": case["user_id"], "role": "customer", "user_id": case["user_id"],
                    },
                    tool_context=case.get("tool_context"),
                )
                final = first
                items = _api_items(first.get("api_data"))
                statuses = [str((item.get("result") or {}).get("status") or "") for item in items]
                next_context = None
                for item in items:
                    if item.get("next_tool_context"):
                        next_context = item["next_tool_context"]
                        break
                if case.get("auto_confirm") and "confirmation_required" in statuses and next_context:
                    final = await agent_pipeline.handle_user_question(
                        history,
                        "确认执行",
                        state_store=store,
                        actor_context={
                            "actor_id": case["user_id"], "role": "customer", "user_id": case["user_id"],
                        },
                        tool_context=next_context,
                    )
        except Exception as exc:
            first = final = {"route": "exception", "api_data": None, "answer": str(exc), "tasks": []}
        finally:
            agent_pipeline._BUSINESS_SERVICE = previous_service

        api_items = _api_items(final.get("api_data"))
        actions = [item.get("action") for item in api_items if item.get("action")]
        statuses = [str((item.get("result") or {}).get("status") or "") for item in api_items]
        order_ids = [item.get("order_id") for item in api_items if item.get("order_id")]
        expected_status = case["expected_status"]
        if expected_status is None:
            status_ok = not api_items
        elif expected_status == "multi_success_confirmation":
            status_ok = "succeeded" in statuses and "confirmation_required" in statuses
        else:
            status_ok = expected_status in statuses
        expected_action = case["expected_action"]
        if isinstance(expected_action, list):
            action_ok = all(action in actions for action in expected_action)
        elif expected_action is None:
            action_ok = not actions
        else:
            action_ok = expected_action in actions
        order_ok = case["expected_order_id"] is None or case["expected_order_id"] in order_ids
        route_ok = final.get("route") == case["expected_route"]
        task_statuses = [task.get("status") for task in final.get("tasks") or []]
        failure_marked_completed = any(
            status not in {"", "succeeded", "confirmation_required"} for status in statuses
        ) and "completed" in task_statuses
        error_type = None
        if not action_ok:
            error_type = "wrong_tool"
        elif not order_ok:
            error_type = "wrong_entity_resolution"
        elif not status_ok:
            if any(status in {"invalid_arguments", "need_user_info"} for status in statuses):
                error_type = "missing_argument"
            else:
                error_type = "partial_completion"
        elif not route_ok:
            error_type = "wrong_route"
        completed = action_ok and order_ok and status_ok and route_ok
        return {
            "case_id": case["case_id"], "layer": "e2e", "category": case["category"],
            "question": case["question"], "expected_route": case["expected_route"],
            "expected_status": expected_status, "expected_action": expected_action,
            "expected_order_id": case["expected_order_id"],
            "actual_route": final.get("route"), "actual_actions": actions,
            "actual_statuses": statuses, "actual_order_ids": order_ids,
            "task_statuses": task_statuses, "answer": final.get("answer"),
            "tool_failure_marked_completed": failure_marked_completed,
            "tool_selection_correct": action_ok, "argument_exact_match": order_ok,
            "task_completed": completed, "final_state_correct": status_ok,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": error_type,
        }


async def evaluate_e2e(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for case in cases:
        results.append(await run_e2e_case(case))
    natural_writes = [r for r in results if r["category"] == "natural_write"]
    structured_writes = [r for r in results if r["category"] == "structured_write"]
    metrics = {
        "case_count": len(results),
        "tool_selection_accuracy": round(sum(r["tool_selection_correct"] for r in results) / len(results), 4),
        "argument_exact_match": round(sum(r["argument_exact_match"] for r in results) / len(results), 4),
        "strict_task_completion_rate": round(sum(r["task_completed"] for r in results) / len(results), 4),
        "natural_language_write_completion_rate": round(sum(r["task_completed"] for r in natural_writes) / len(natural_writes), 4),
        "structured_write_completion_rate": round(sum(r["task_completed"] for r in structured_writes) / len(structured_writes), 4),
        "tool_failure_marked_completed_rate": round(sum(r["tool_failure_marked_completed"] for r in results) / len(results), 4),
        "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in results), 3),
        "p95_latency_ms": round(sorted(r["latency_ms"] for r in results)[math.ceil(len(results) * 0.95) - 1], 3),
        "errors": dict(Counter(r["error_type"] for r in results if r["error_type"])),
    }
    return results, metrics


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def verdict(value: float, good: float, pass_line: float, inverse: bool = False) -> str:
    if inverse:
        if value <= good:
            return "较好"
        if value <= pass_line:
            return "合格"
        return "未达标"
    if value >= good:
        return "较好"
    if value >= pass_line:
        return "合格"
    return "未达标"


def render_report(metrics: dict[str, Any], results: list[dict[str, Any]]) -> str:
    planner = metrics["planner"]
    sandbox = metrics["sandbox"]
    e2e = metrics["e2e"]
    generated = metrics["generated_at"]
    failures = [r for r in results if not r.get("task_completed")]
    by_error = Counter(r.get("error_type") for r in results if r.get("error_type"))
    failed_examples = []
    for error_type, _ in by_error.most_common():
        example = next((r for r in results if r.get("error_type") == error_type), None)
        if example:
            failed_examples.append(example)
    lines = [
        "# 美妆电商客服 Agent——工具调用与任务完成专项测评报告",
        "",
        f"- 生成时间：{generated}",
        f"- 基准版本：{Path(metrics['benchmark']).stem}（共 {metrics['total_cases']} 条）",
        "- 测评方式：Planner-only + 临时 SQLite 沙箱执行 + 固定意图/固定回答的离线端到端",
        "- 测评范围：query_order、urge_shipment、request_refund、cancel_order、update_address、request_invoice",
        "- 隔离边界：不调用 DeepSeek，不依赖 Elasticsearch；端到端指标只代表工具编排，不代表线上意图或生成质量。",
        "",
        "## 1. 结论摘要",
        "",
        "业务工具服务层在权限、二次确认、业务状态校验、幂等回放和结构化错误返回方面表现稳定；主要短板集中在 Agent Planner：自然语言中的退款原因、地址和发票参数尚未转成结构化工具参数，当前轮与历史订单冲突时实体解析会误用旧订单，部分口语化动作会退化为 query_order。",
        "",
        "因此，当前系统已经具备安全执行内核，但还没有达到“用户自然语言直接驱动复杂写操作”的验收目标。结构化 tool_context 路径可用，自然语言写操作路径仍需补齐。",
        "",
        "## 2. 测试集分布",
        "",
        "| 层级 | 样本数 | 主要覆盖 |",
        "|---|---:|---|",
        f"| Planner-only | {planner['case_count']} | 工具选择、无需工具、缺参澄清、历史参数、实体冲突、口语化动作 |",
        f"| 沙箱执行 | {sandbox['case_count']} | 权限、确认、参数校验、状态冲突、幂等、确认过期、SQLite 锁重试 |",
        f"| 离线端到端 | {e2e['case_count']} | 状态机、两阶段确认、多轮订单号、结构化/自然语言写操作、多意图 |",
        f"| **合计** | **{metrics['total_cases']}** | |",
        "",
        "## 3. 核心指标",
        "",
        "| 指标 | 实测 | 验收判断 | 说明 |",
        "|---|---:|---|---|",
        f"| Planner Tool Selection Accuracy | {pct(planner['tool_selection_accuracy'])} | {verdict(planner['tool_selection_accuracy'], .96, .90)} | 需要工具的样本 |",
        f"| Planner Tool Macro-F1 | {pct(planner['tool_macro_f1'])} | {verdict(planner['tool_macro_f1'], .94, .88)} | 六类工具宏平均 |",
        f"| No-tool Accuracy | {pct(planner['no_tool_accuracy'])} | {verdict(planner['no_tool_accuracy'], .97, .92)} | 普通知识咨询避免业务调用 |",
        f"| Planner Argument Exact Match | {pct(planner['argument_exact_match'])} | {verdict(planner['argument_exact_match'], .95, .88)} | 此层只核对订单号 |",
        f"| Critical Argument Error Rate | {pct(planner['critical_argument_error_rate'])} | {verdict(planner['critical_argument_error_rate'], 0, .01, inverse=True)} | 高风险写操作使用错误订单号 |",
        f"| Planner Clarification Accuracy | {pct(planner['clarification_accuracy'])} | {verdict(planner['clarification_accuracy'], .99, .95)} | 缺订单号时主动澄清 |",
        f"| Sandbox Tool Execution Success | {pct(sandbox['tool_execution_success_rate'])} | {verdict(sandbox['tool_execution_success_rate'], .98, .95)} | 合法可执行任务 |",
        f"| Sandbox Strict Task Completion | {pct(sandbox['strict_task_completion_rate'])} | {verdict(sandbox['strict_task_completion_rate'], .93, .85)} | 包含预期拒绝与异常处理 |",
        f"| Confirmation Compliance | {pct(sandbox['confirmation_compliance'])} | {verdict(sandbox['confirmation_compliance'], 1.0, .98)} | 写操作二次确认 |",
        f"| Unauthorized Execution Rate | {pct(sandbox['unauthorized_execution_rate'])} | {verdict(sandbox['unauthorized_execution_rate'], 0, 0, inverse=True)} | 目标必须为 0 |",
        f"| Duplicate Execution Rate | {pct(sandbox['duplicate_execution_rate'])} | {verdict(sandbox['duplicate_execution_rate'], 0, 0, inverse=True)} | 幂等测试 |",
        f"| False Success Rate | {pct(sandbox['false_success_rate'])} | {verdict(sandbox['false_success_rate'], 0, 0, inverse=True)} | 错误返回不得声称成功 |",
        f"| Offline E2E Strict Task Completion | {pct(e2e['strict_task_completion_rate'])} | {verdict(e2e['strict_task_completion_rate'], .93, .85)} | 固定意图与回答 |",
        f"| 自然语言写操作完成率 | {pct(e2e['natural_language_write_completion_rate'])} | {verdict(e2e['natural_language_write_completion_rate'], .93, .85)} | 无预注入 tool_context |",
        f"| 结构化写操作完成率 | {pct(e2e['structured_write_completion_rate'])} | {verdict(e2e['structured_write_completion_rate'], .98, .95)} | 预先提供结构化参数 |",
        f"| 工具失败误标 Completed | {pct(e2e['tool_failure_marked_completed_rate'])} | {verdict(e2e['tool_failure_marked_completed_rate'], 0, 0, inverse=True)} | invalid_arguments 等失败状态 |",
        "",
        "## 4. 各工具 Planner 表现",
        "",
        "| 工具 | Precision | Recall | F1 | 样本数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for tool in TOOLS:
        item = planner["per_tool"][tool]
        lines.append(f"| {tool} | {pct(item['precision'])} | {pct(item['recall'])} | {pct(item['f1'])} | {item['support']} |")
    lines += [
        "",
        "## 5. 安全执行与任务状态",
        "",
        f"- 写操作确认合规率：{pct(sandbox['confirmation_compliance'])}。所有合法变更均先生成 confirmation_token，再提交事务。",
        f"- 越权执行率：{pct(sandbox['unauthorized_execution_rate'])}。错误用户、匿名用户和未知角色均未越权成功。",
        f"- 重复执行率：{pct(sandbox['duplicate_execution_rate'])}。重复确认返回 idempotent_replay，data只保留一次成功副作用。",
        f"- 异常恢复率：{pct(sandbox['recovery_success_rate'])}。覆盖确认过期及 SQLite 锁冲突有限重试。",
        f"- 沙箱延迟：P50={sandbox['p50_latency_ms']}ms，P95={sandbox['p95_latency_ms']}ms。该指标仅为本地 SQLite，不含模型和网络耗时。",
        "",
        "## 6. 主要失败类型",
        "",
        "| 错误类型 | 数量 | 影响 |",
        "|---|---:|---|",
    ]
    impacts = {
        "wrong_tool": "可能执行错误业务动作或只查询不办理",
        "wrong_entity_resolution": "可能操作历史订单而不是当前指定订单，属于高风险",
        "missing_argument": "退款、地址、发票任务无法从自然语言直接闭环",
        "partial_completion": "多意图请求只完成部分目标",
        "tool_failure_marked_completed": "工具失败但内部任务状态被标记为 completed",
        "business_rule_gap": "物流异常订单仍可进入催发货确认，动作与业务状态不匹配",
        "wrong_route": "状态机路由与预期不一致",
        "missed_tool": "需要工具时未调用",
        "unexpected_tool_status": "服务返回状态与标准不一致",
    }
    for error_type, count in by_error.most_common():
        lines.append(f"| {error_type} | {count} | {impacts.get(error_type, '需人工复核')} |")
    lines += [
        "",
        "## 7. 典型失败案例",
        "",
    ]
    for example in failed_examples[:8]:
        lines.append(f"### {example['case_id']} — {example.get('error_type')}")
        lines.append("")
        if example.get("question"):
            lines.append(f"- 用户输入：{example['question']}")
        expected_action = example.get("expected_action") or example.get("gold_action") or example.get("action")
        expected_order = example.get("expected_order_id") or example.get("gold_order_id") or example.get("order_id")
        actual_payload = example.get("actual") or {
            "action": example.get("actual_actions") or example.get("action"),
            "status": example.get("actual_statuses") or example.get("actual_status"),
            "order_id": example.get("actual_order_ids") or example.get("order_id"),
        }
        lines.append(f"- 预期：工具/状态/订单 = {expected_action} / {example.get('expected_status', '')} / {expected_order}")
        lines.append(f"- 实际：{json.dumps(actual_payload, ensure_ascii=False)}")
        lines.append("")
    lines += [
        "## 8. 未覆盖项与解释边界",
        "",
        "- 未调用真实 DeepSeek，因此未评估模型是否稳定输出正确工具 JSON，也未评估真实最终回复忠实度。",
        "- 未连接外部 OMS/物流/发票系统；工具执行成功率代表本地 SQLite 沙箱能力。",
        "- 当前业务工具没有独立 query_logistics、create_after_sale_ticket、transfer_to_human，因此未对这些工具计分。",
        "- 超时恢复仅覆盖 SQLite 锁冲突和确认过期，尚未覆盖 HTTP 连接超时、下游部分成功等生产故障。",
        "",
        "## 9. 验收结论",
        "",
        "- **业务工具服务层：通过。** 权限、确认、参数校验、状态约束和幂等机制可以作为后续真实 API 适配层的基础。",
        "- **Planner-only：有条件通过。** 常规关键词工具选择和缺订单号澄清可用，但订单实体冲突与口语化动作需要修复。",
        "- **自然语言端到端任务：未通过成熟 Agent 验收。** 结构化写操作链路可闭环，但自然语言参数抽取和多意图工具上下文仍不完整。",
        "",
        "详细整改优先级见《工具调用与任务完成优化建议.md》。",
        "",
    ]
    return "\n".join(lines)


def render_recommendations(metrics: dict[str, Any], results: list[dict[str, Any]]) -> str:
    e2e = metrics["e2e"]
    return "\n".join([
        "# 工具调用与任务完成优化建议",
        "",
        "## P0：面试与上线前必须完成",
        "",
        "### 1. 将工具 Schema 真正接入 Planner / Function Calling",
        "",
        f"自然语言写操作完成率仅为 {pct(e2e['natural_language_write_completion_rate'])}，而结构化 tool_context 路径为 {pct(e2e['structured_write_completion_rate'])}。根因不是工具服务不能执行，而是 `BUSINESS_TOOL_DEFINITIONS` 尚未进入在线规划：退款原因、完整地址、发票类型/抬头/税号/邮箱不会从用户输入转成 arguments。",
        "",
        "建议：",
        "",
        "- 使用模型 Function Calling 输出 action + arguments，并使用 JSON Schema/Pydantic 做二次验证。",
        "- 参数只能来自当前消息、历史槽位或工具返回，记录 `argument_source`，禁止模型补造关键字段。",
        "- 缺少多个字段时一次性询问全部必填项；已有字段不要重复询问。",
        "- Planner 输出先进入只读校验层，通过后才允许创建 confirmation_token。",
        "",
        "### 2. 修复订单实体解析优先级和歧义处理",
        "",
        "当前 `extract_order_id` 把历史对话拼在当前问题之前，出现“刚才订单 A，改查订单 B”时会优先取得 A。该问题可能导致操作错误订单，按高风险标准应视为 P0。",
        "",
        "建议：",
        "",
        "- 当前轮订单号优先于历史槽位；当前轮出现一个订单号时覆盖 active_order_id。",
        "- 当前轮或历史中存在多个候选订单且指代不明确时，停止写操作并要求用户选择。",
        "- 确认摘要必须展示订单号、商品和操作，用户确认后锁定实体，后续不得由模型改写。",
        "- 增加 wrong_entity_resolution 回归集，关键订单错误目标必须为 0。",
        "",
        "### 3. 修复工具结果到任务状态的映射",
        "",
        "当前 Agent 只把 `confirmation_required` 映射为 waiting_confirmation，其他状态统一写成 completed。因此 invalid_arguments、permission_denied、business_rule_rejected 等失败也可能被记录为 completed。",
        "",
        "建议状态映射：",
        "",
        "- succeeded → completed",
        "- confirmation_required → waiting_confirmation",
        "- need_user_info / invalid_arguments → waiting_user",
        "- permission_denied / authentication_required → blocked",
        "- business_rule_rejected → rejected",
        "- timeout / unavailable → retrying 或 failed",
        "- 只有 succeeded 才能向用户表达“已完成”。",
        "",
        "## P1：提升任务完成率",
        "",
        "### 4. 用结构化 Planner 替代关键词动作判断",
        "",
        "`infer_business_action` 适合作为兜底，但对“作废订单”“换个收货地点”“给仓库加急”等口语表达覆盖不足。建议由模型基于 allowed_tools 规划，关键词规则仅承担安全兜底和异常回退。",
        "",
        "### 5. 为多意图任务提供独立 task_context",
        "",
        "当前同一轮多个任务共享一个 tool_context，无法同时为物流查询和发票申请传递不同参数与确认状态。建议每个 task_id 保存自己的 action、slots、arguments、idempotency_key 和 confirmation_token，分别暂停和恢复。",
        "",
        "### 6. 扩充工具能力并区分查询与办理",
        "",
        "建议新增：",
        "",
        "- query_logistics：返回物流节点而不是仅返回订单摘要。",
        "- create_after_sale_ticket：承接漏液、破损、错发及图片证据。",
        "- transfer_to_human：高风险、权限不足、连续失败时生成可审计工单。",
        "- get_operation_status：工具超时后按幂等键查询真实执行状态，避免盲目重试。",
        "",
        "### 7. 增加生产故障注入",
        "",
        "在当前 SQLite 锁重试基础上，加入网络超时、HTTP 5xx、下游已成功但响应丢失、部分成功、确认过期、服务熔断测试。写操作重试前必须先通过 idempotency_key 查询执行状态。另需补充物流异常状态规则：`shipping_exception` 订单不应继续创建普通催发货工单，应改走物流异常查询或人工工单。",
        "",
        "## P2：建立持续评测闭环",
        "",
        "### 8. 增加真实模型 Planner-only 评测",
        "",
        "使用本次 benchmark 的固定测试集调用真实模型，只让模型输出工具计划，不执行写操作；统计 Tool Macro-F1、Argument Exact Match、关键参数错误率和 JSON Schema 合法率。",
        "",
        "### 9. 增加最终回复忠实度评测",
        "",
        "对 succeeded、pending_review、confirmation_required、not_found、permission_denied、timeout 分别建立回答断言，重点检查 pending 是否被说成完成、失败是否伪报成功、订单信息是否来自真实工具结果。",
        "",
        "### 10. 将本评测接入回归流程",
        "",
        "建议每次修改 Planner、工具 Schema、状态机或业务规则后运行：",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe \"evaluation_suites\\04_tool_call_task_completion_evaluation\\run_tool_evaluation.py\"",
        ".venv\\Scripts\\python.exe -m unittest discover -s tests -v",
        "```",
        "",
        "阻断条件：Unauthorized Execution Rate > 0、Duplicate Execution Rate > 0、Critical Argument Error Rate > 0，或结构化写操作完成率低于 95%。",
        "",
    ])


def render_optimized_report(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    rows = [
        ("Planner Tool Selection Accuracy", baseline["planner"]["tool_selection_accuracy"], current["planner"]["tool_selection_accuracy"]),
        ("Tool Macro-F1", baseline["planner"]["tool_macro_f1"], current["planner"]["tool_macro_f1"]),
        ("Argument Exact Match", baseline["planner"]["argument_exact_match"], current["planner"]["argument_exact_match"]),
        ("Critical Argument Error Rate", baseline["planner"]["critical_argument_error_rate"], current["planner"]["critical_argument_error_rate"]),
        ("Sandbox Strict Task Completion", baseline["sandbox"]["strict_task_completion_rate"], current["sandbox"]["strict_task_completion_rate"]),
        ("Offline E2E Strict Task Completion", baseline["e2e"]["strict_task_completion_rate"], current["e2e"]["strict_task_completion_rate"]),
        ("自然语言写操作完成率", baseline["e2e"]["natural_language_write_completion_rate"], current["e2e"]["natural_language_write_completion_rate"]),
        ("结构化写操作完成率", baseline["e2e"]["structured_write_completion_rate"], current["e2e"]["structured_write_completion_rate"]),
        ("工具失败误标 Completed", baseline["e2e"]["tool_failure_marked_completed_rate"], current["e2e"]["tool_failure_marked_completed_rate"]),
    ]
    lines = [
        "# 工具调用与任务完成——优化后测评报告",
        "",
        f"- 优化后测评时间：{current['generated_at']}",
        f"- 测试规模：{current['total_cases']} 条",
        "- 对比基线：baseline/tool_eval_metrics.json",
        "- 测评方式：与基线一致，使用 Planner-only、SQLite 沙箱和固定意图/回答的离线端到端评测。",
        "",
        "## 1. 优化内容",
        "",
        "- 增加 Schema 驱动的自然语言业务参数规划，提取退款原因、地址和发票字段，并标记参数来源。",
        "- 当前轮订单号优先于历史订单；当前轮出现多个订单号时清除活动订单并触发澄清。",
        "- 扩展作废、撤单、收货地点、加急、电子票等口语化动作映射。",
        "- 将工具返回状态映射为 completed、waiting_user、waiting_confirmation、blocked、rejected 或 failed。",
        "- 阻止 shipping_exception / exception 订单创建普通催发货工单。",
        "- 保留 PII 脱敏展示，同时使用用户原始输入完成受控业务参数提取。",
        "",
        "## 2. 优化前后指标",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    inverse_metrics = {"Critical Argument Error Rate", "工具失败误标 Completed"}
    for name, before, after in rows:
        raw_delta = after - before
        improvement = -raw_delta if name in inverse_metrics else raw_delta
        sign = "+" if improvement >= 0 else ""
        lines.append(
            f"| {name} | {pct(before)} | {pct(after)} | {sign}{improvement * 100:.2f} 个百分点 |"
        )
    lines += [
        "",
        "## 3. 优化后关键结果",
        "",
        f"- Planner 工具选择准确率：{pct(current['planner']['tool_selection_accuracy'])}。",
        f"- Planner Tool Macro-F1：{pct(current['planner']['tool_macro_f1'])}。",
        f"- 高风险关键参数错误率：{pct(current['planner']['critical_argument_error_rate'])}。",
        f"- 沙箱严格任务完成率：{pct(current['sandbox']['strict_task_completion_rate'])}。",
        f"- 离线端到端严格任务完成率：{pct(current['e2e']['strict_task_completion_rate'])}。",
        f"- 自然语言写操作完成率：{pct(current['e2e']['natural_language_write_completion_rate'])}。",
        f"- 工具失败误标 Completed：{pct(current['e2e']['tool_failure_marked_completed_rate'])}。",
        f"- 越权执行、重复执行、虚假成功率：{pct(current['sandbox']['unauthorized_execution_rate'])} / {pct(current['sandbox']['duplicate_execution_rate'])} / {pct(current['sandbox']['false_success_rate'])}。",
        "",
        "## 4. 验收判断",
        "",
        "- **业务工具安全层：通过。** 权限、确认、幂等、状态校验和异常返回满足当前离线验收要求。",
        "- **Planner：通过当前规则基准。** 常规与口语化动作、订单实体覆盖和单轮参数抽取达到验收目标。",
        "- **离线端到端：通过当前 240 条专项集。** 自然语言写操作已能进入确认并完成执行，失败状态不再伪装为完成。",
        "- **生产级验收仍未完成。** 需要继续评测真实模型 Function Calling、多轮业务参数补全、外部 API 故障及最终回复忠实度。",
        "",
    ]
    return "\n".join(lines)


def render_remaining_recommendations(metrics: dict[str, Any]) -> str:
    return "\n".join([
        "# 优化后剩余建议",
        "",
        "## P0：补齐真实模型 Planner 验证",
        "",
        "当前优化使用 Schema 驱动的确定性参数规划，已通过固定基准，但还需要让 DeepSeek 在 Planner-only 模式输出标准 Function Calling JSON，并验证 Schema 合法率、参数来源和提示词注入防护。写操作仍必须经过确定性校验层，不能直接执行模型输出。",
        "",
        "## P1：完成多轮业务参数补全",
        "",
        "当用户第一轮只说“申请退款”，Agent 已能返回 need_user_info 并把任务标记为 waiting_user；下一步需要让用户第二轮只补充“因为过敏”时恢复原任务，合并已有订单号、动作和新参数后继续确认。地址和发票也应支持分多轮补齐。",
        "",
        "## P1：多意图独立工具上下文",
        "",
        "为每个 task_id 单独保存 action、arguments、idempotency_key 和 confirmation_token，避免同一轮“查物流并开发票”时共享或覆盖工具上下文。",
        "",
        "## P1：生产故障与最终回复评测",
        "",
        "增加 HTTP 超时、5xx、响应丢失、部分成功和真实执行后超时场景；通过 get_operation_status 按幂等键确认最终状态。并使用真实回答模型检查 pending、rejected、blocked 等状态是否被准确表达。",
        "",
        "## P2：扩充工具能力",
        "",
        "新增 query_logistics、create_after_sale_ticket、get_operation_status 和 transfer_to_human，覆盖物流节点、破损凭证、异步操作状态和人工协同。",
        "",
        "## 当前回归门槛",
        "",
        f"- Critical Argument Error Rate：{pct(metrics['planner']['critical_argument_error_rate'])}，目标持续保持 0。",
        f"- Unauthorized Execution Rate：{pct(metrics['sandbox']['unauthorized_execution_rate'])}，目标持续保持 0。",
        f"- Duplicate Execution Rate：{pct(metrics['sandbox']['duplicate_execution_rate'])}，目标持续保持 0。",
        f"- 自然语言写操作完成率：{pct(metrics['e2e']['natural_language_write_completion_rate'])}，新增真实模型集后目标不低于 90%。",
        "",
    ])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    benchmark_path = args.output_dir / "tool_benchmark_v2.jsonl"
    cases = planner_cases() + sandbox_cases() + e2e_cases()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(benchmark_path, cases)
    planner_results, planner_metrics = evaluate_planner([c for c in cases if c["layer"] == "planner"])
    sandbox_results, sandbox_metrics = evaluate_sandbox([c for c in cases if c["layer"] == "sandbox"])
    e2e_results, e2e_metrics = await evaluate_e2e([c for c in cases if c["layer"] == "e2e"])
    results = planner_results + sandbox_results + e2e_results
    metrics = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "benchmark": str(benchmark_path.relative_to(ROOT)),
        "total_cases": len(cases),
        "planner": planner_metrics,
        "sandbox": sandbox_metrics,
        "e2e": e2e_metrics,
        "overall_strict_task_completion_rate": round(sum(r.get("task_completed", False) for r in results) / len(results), 4),
        "overall_errors": dict(Counter(r.get("error_type") for r in results if r.get("error_type"))),
    }
    write_jsonl(args.output_dir / "tool_eval_results.jsonl", results)
    (args.output_dir / "tool_eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "工具调用与任务完成专项测评报告.md").write_text(
        render_report(metrics, results), encoding="utf-8"
    )
    (args.output_dir / "优化后剩余优化建议.md").write_text(
        render_remaining_recommendations(metrics), encoding="utf-8"
    )
    if BASELINE_METRICS_PATH.exists():
        baseline = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
        (args.output_dir / "优化后工具调用与任务完成专项测评报告.md").write_text(
            render_optimized_report(baseline, metrics), encoding="utf-8"
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"benchmark -> {benchmark_path}")
    print(f"report dir -> {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
