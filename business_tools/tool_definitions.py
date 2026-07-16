# -*- coding: utf-8 -*-
"""业务工具的结构化参数定义与轻量动作规划。"""

from __future__ import annotations

import re
from typing import Any


def _order_tool(name: str, description: str, extra_properties: dict[str, Any] | None = None,
                required: list[str] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "order_id": {
            "type": "string",
            "pattern": "^MOCK[0-9]{12,}$",
            "description": "订单号",
        }
    }
    properties.update(extra_properties or {})
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["order_id", *(required or [])],
                "additionalProperties": False,
            },
        },
    }


BUSINESS_TOOL_DEFINITIONS = [
    _order_tool("query_order", "查询订单、支付、履约及物流状态"),
    _order_tool(
        "urge_shipment",
        "对已支付但尚未发货的订单发起催发货",
        {"reason": {"type": "string", "maxLength": 200}},
    ),
    _order_tool(
        "request_refund",
        "对符合退款条件的订单发起退款申请",
        {"reason": {"type": "string", "minLength": 2, "maxLength": 500}},
        ["reason"],
    ),
    _order_tool(
        "cancel_order",
        "取消尚未发货且允许取消的订单",
        {"reason": {"type": "string", "maxLength": 500}},
    ),
    _order_tool(
        "update_address",
        "修改尚未发货订单的收货地址",
        {
            "new_address": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "minLength": 1, "maxLength": 50},
                    "phone": {"type": "string", "minLength": 7, "maxLength": 20},
                    "province": {"type": "string", "minLength": 1, "maxLength": 30},
                    "city": {"type": "string", "minLength": 1, "maxLength": 30},
                    "detail": {"type": "string", "minLength": 3, "maxLength": 200},
                },
                "required": ["recipient", "phone", "province", "city", "detail"],
                "additionalProperties": False,
            }
        },
        ["new_address"],
    ),
    _order_tool(
        "request_invoice",
        "为已支付订单申请电子发票",
        {
            "invoice_type": {"type": "string", "enum": ["personal", "company"]},
            "title": {"type": "string", "minLength": 1, "maxLength": 100},
            "tax_id": {"type": "string", "maxLength": 30},
            "email": {"type": "string", "format": "email"},
        },
        ["invoice_type", "title", "email"],
    ),
]


def infer_business_action(question: str, intent_result: dict[str, Any]) -> str:
    """根据显式语义选择业务工具；服务层仍负责权限与业务规则校验。"""
    text = question.lower()
    intent = str(intent_result.get("intent_level1") or "")

    if any(word in text for word in (
        "修改地址", "改地址", "换地址", "收货地址", "收货地点", "寄送地址",
        "送货地址", "地址填错", "地址写错", "地址不对", "地点换一下",
    )) or (
        intent == "after_sale"
        and any(word in text for word in ("收件人", "联系电话", "改到", "更新地址", "改一下地址"))
    ) or (
        "地址" in text and any(word in text for word in ("改", "换", "更新", "填错", "写错", "不对"))
    ):
        return "update_address"
    if any(word in text for word in (
        "退款", "退钱", "退货退款", "申请退", "退掉订单", "退掉这单",
        "我要退掉", "办理退货",
    )):
        return "request_refund"
    if any(word in text for word in (
        "取消订单", "取消掉", "取消这单", "撤销订单", "撤单", "作废",
        "不想要了", "不要了", "不想要这单", "不要订单",
    )) or (intent == "after_sale" and "取消" in text):
        return "cancel_order"
    if intent == "invoice" or any(word in text for word in (
        "开发票", "开票", "申请发票", "电子票", "报销发票", "补开发票",
    )):
        return "request_invoice"
    if intent == "urge_shipment" or any(word in text for word in (
        "催发货", "催一下", "尽快发货", "催仓库", "催促", "加急处理",
        "给仓库加急", "赶紧发货",
    )):
        return "urge_shipment"
    return "query_order"


def _first_match(patterns: tuple[str, ...], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip(" ，。；;：:")
    return ""


def extract_business_arguments(question: str, action: str) -> dict[str, Any]:
    """从当前用户原文提取可追溯的工具参数，不猜测未出现的关键字段。"""
    text = str(question or "").strip()
    if action == "request_refund":
        reason = _first_match(
            (
                r"(?:退款原因|退货原因|原因)(?:是|为|[:：])?\s*([^，。；;]{2,80})",
                r"因为\s*([^，。；;]{2,80})",
            ),
            text,
        )
        if not reason:
            reason_phrases = (
                "使用后过敏", "用了过敏", "出现过敏", "使用后不适", "皮肤不适",
                "商品漏液", "收到漏液", "包装破损", "商品破损", "质量有问题",
                "买错了", "不喜欢", "不合适", "不再需要",
            )
            reason = next((phrase for phrase in reason_phrases if phrase in text), "")
        return {"reason": reason} if reason else {}

    if action == "cancel_order":
        reason = _first_match(
            (r"(?:取消原因|原因)(?:是|为|[:：])?\s*([^，。；;]{2,80})",), text
        )
        if not reason:
            reason = next(
                (phrase for phrase in ("买错了", "不想要了", "不再需要", "重复下单") if phrase in text),
                "",
            )
        return {"reason": reason} if reason else {}

    if action == "urge_shipment":
        reason = _first_match(
            (r"(?:加急原因|原因)(?:是|为|[:：])?\s*([^，。；;]{2,80})",), text
        )
        return {"reason": reason} if reason else {}

    if action == "update_address":
        recipient = _first_match(
            (r"收件人(?:是|为|[:：])?\s*([\u4e00-\u9fa5A-Za-z·]{2,30})",), text
        )
        phone_match = re.search(r"(?<!\d)(1\d{10}|\d{7,20})(?!\d)", text)
        phone = phone_match.group(1) if phone_match else ""
        municipality = next(
            (name for name in ("北京市", "上海市", "天津市", "重庆市") if name in text),
            "",
        )
        province = municipality or _first_match(
            (r"([\u4e00-\u9fa5]{2,12}(?:省|自治区))",), text
        )
        city = municipality or _first_match(
            (r"([\u4e00-\u9fa5]{2,12}市)",), text
        )
        detail = _first_match(
            (
                r"(?:改到|改为|修改为|新地址(?:是|为|[:：])?)\s*([^，；;]{3,200})",
                r"(?:收货地址|寄送地址|送货地址)(?:是|为|[:：])?\s*([^，；;]{3,200})",
            ),
            text,
        )
        address = {
            key: value
            for key, value in {
                "recipient": recipient,
                "phone": phone,
                "province": province,
                "city": city,
                "detail": detail,
            }.items()
            if value
        }
        return {"new_address": address} if address else {}

    if action == "request_invoice":
        invoice_type = ""
        if any(word in text for word in ("企业", "公司", "专票")):
            invoice_type = "company"
        elif any(word in text for word in ("个人", "个人票", "普通发票", "电子票")):
            invoice_type = "personal"
        title = _first_match(
            (r"(?:发票抬头|抬头)(?:是|为|[:：])?\s*([^，。；;]{1,100})",), text
        )
        tax_id = _first_match(
            (r"(?:税号|纳税人识别号)(?:是|为|[:：])?\s*([A-Za-z0-9]{8,30})",), text
        )
        email_match = re.search(r"[^@\s，。；;]+@[^@\s，。；;]+\.[A-Za-z]{2,}", text)
        email = email_match.group(0) if email_match else ""
        return {
            key: value
            for key, value in {
                "invoice_type": invoice_type,
                "title": title,
                "tax_id": tax_id,
                "email": email,
            }.items()
            if value
        }
    return {}


def missing_business_arguments(action: str, arguments: dict[str, Any]) -> list[str]:
    """依据工具 Schema 返回缺失的业务参数字段。"""
    if action == "request_refund":
        return [] if len(str(arguments.get("reason") or "").strip()) >= 2 else ["reason"]
    if action == "update_address":
        address = arguments.get("new_address")
        required = ("recipient", "phone", "province", "city", "detail")
        if not isinstance(address, dict):
            return [f"new_address.{field}" for field in required]
        return [f"new_address.{field}" for field in required if not str(address.get(field) or "").strip()]
    if action == "request_invoice":
        required = ["invoice_type", "title", "email"]
        if arguments.get("invoice_type") == "company":
            required.append("tax_id")
        return [field for field in required if not str(arguments.get(field) or "").strip()]
    return []
