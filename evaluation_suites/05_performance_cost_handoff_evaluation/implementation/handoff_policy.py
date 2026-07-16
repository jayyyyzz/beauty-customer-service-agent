# -*- coding: utf-8 -*-
"""Deterministic human-handoff policy for high-risk customer-service cases."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HandoffDecision:
    should_handoff: bool
    reason: str | None = None
    priority: str = "normal"
    signals: tuple[str, ...] = ()
    user_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_USER_REQUEST_RE = re.compile(
    r"(转|接|换).{0,6}(人工|真人)|人工客服|真人客服|不想.{0,8}(机器人|自动回复)|拒绝.{0,8}(机器人|自动回复)",
    re.I,
)
_ACCOUNT_SECURITY_RE = re.compile(
    r"账号被盗|账户被盗|盗刷|异常支付|不是我付|身份被冒用|诈骗|欺诈|支付安全",
    re.I,
)
_REGULATORY_RE = re.compile(
    r"消协|消费者协会|12315|监管机构|市场监督|律师|起诉|法院|媒体曝光|报警",
    re.I,
)
_REPEATED_FAILURE_RE = re.compile(
    r"(已经|都).{0,8}(两|三|四|2|3|4|多)次|反复|一直.{0,6}(失败|查不到|没结果)|别再让我重复|升级处理",
    re.I,
)
_HIGH_VALUE_RE = re.compile(
    r"(?:[1-9]\d{3,}|一千|两千|二千|三千|几千|上万).{0,12}(元|退款|赔偿|争议)|(退款|赔偿|争议).{0,12}(?:[1-9]\d{3,}|一千|两千|二千|三千|几千|上万)",
    re.I,
)


def _history_text(history_dialogue: dict[str, Any] | None) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in ((history_dialogue or {}).get("messages") or [])[-20:]
    )


def assess_handoff_policy(
    question: str,
    history_dialogue: dict[str, Any] | None = None,
    *,
    safety_level: str = "normal",
    safety_signals: tuple[str, ...] = (),
) -> HandoffDecision:
    """Apply explicit risk rules before LLM routing.

    The order is intentional: urgent safety and account security must not be
    diluted by a generic user-requested or complaint label.
    """
    current = str(question or "")
    history = _history_text(history_dialogue)
    combined = f"{history}\n{current}"

    if safety_level == "urgent":
        return HandoffDecision(
            True,
            "urgent_skincare_safety",
            "urgent",
            tuple(safety_signals),
            "已为您创建紧急人工协助工单。",
        )
    if _ACCOUNT_SECURITY_RE.search(current):
        return HandoffDecision(
            True,
            "account_security",
            "urgent",
            ("account_security",),
            "为保护您的账户与资金安全，已立即升级人工客服处理。",
        )
    if _USER_REQUEST_RE.search(current):
        return HandoffDecision(
            True,
            "user_requested",
            "high",
            ("explicit_human_request",),
            "已按您的要求转接人工客服，并保留本次对话信息。",
        )
    if safety_level == "caution":
        return HandoffDecision(
            True,
            "skincare_reaction",
            "high",
            tuple(safety_signals),
            "已创建人工售后协助工单。",
        )
    if _REGULATORY_RE.search(current):
        return HandoffDecision(
            True,
            "complaint_escalation",
            "high",
            ("regulatory_or_legal_escalation",),
            "您的投诉已升级人工专员处理，并保留当前诉求。",
        )
    if _HIGH_VALUE_RE.search(current):
        return HandoffDecision(
            True,
            "high_value_dispute",
            "high",
            ("high_value_dispute",),
            "该问题涉及较高金额或责任争议，已升级人工专员核实。",
        )
    if _REPEATED_FAILURE_RE.search(combined) and any(
        marker in combined for marker in ("失败", "查不到", "没结果", "重复", "升级")
    ):
        return HandoffDecision(
            True,
            "repeated_failure",
            "high",
            ("repeated_unresolved",),
            "已保留您此前提供的信息并升级人工处理，不会要求您重复描述。",
        )
    return HandoffDecision(False)

