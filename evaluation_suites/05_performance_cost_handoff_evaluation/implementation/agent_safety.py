# -*- coding: utf-8 -*-
"""客服 Agent 输入脱敏、提示词注入防护与护肤安全边界。"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9.-])"
)
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)")
_BANK_CARD_RE = re.compile(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)")
_ADDRESS_RE = re.compile(
    r"(?:收货)?地址\s*[:：]?\s*[^\n，。；;]{6,80}",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class InjectionAssessment:
    blocked: bool
    score: int
    signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkincareSafetyAssessment:
    level: str
    handoff_required: bool
    signals: tuple[str, ...]
    guidance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_INJECTION_RULES: tuple[tuple[str, re.Pattern[str], int], ...] = (
    (
        "override_instructions",
        re.compile(r"忽略.{0,12}(之前|上面|系统).{0,12}(指令|提示|规则)|ignore.{0,20}(previous|system).{0,12}instructions?", re.I),
        3,
    ),
    (
        "reveal_prompt",
        re.compile(r"(输出|泄露|展示|告诉我).{0,12}(系统提示词|开发者消息|隐藏指令|system prompt)", re.I),
        3,
    ),
    (
        "role_override",
        re.compile(r"你现在是|切换身份|扮演.{0,12}(开发者|系统|管理员)|act as.{0,20}(system|developer|admin)", re.I),
        2,
    ),
    (
        "prompt_markup",
        re.compile(r"</?(system|developer|assistant)>|\[/?INST\]|BEGIN SYSTEM PROMPT", re.I),
        2,
    ),
    (
        "security_bypass",
        re.compile(r"越狱|jailbreak|绕过.{0,10}(安全|限制|规则)|关闭.{0,10}(防护|审查)", re.I),
        2,
    ),
)


_URGENT_SKIN_SIGNALS = (
    "呼吸困难", "喘不过气", "喉咙肿", "嘴唇肿", "眼睛肿", "面部肿胀",
    "全身荨麻疹", "大面积水疱", "大面积脱皮", "昏厥", "休克",
)
_REACTION_SIGNALS = (
    "过敏", "红肿", "刺痛", "灼热", "瘙痒", "发痒", "泛红", "爆痘", "脱皮",
)
_REACTION_CONTEXT_SIGNALS = (
    "用了", "使用后", "涂了", "擦了", "敷完", "刚用", "之后", "过敏了",
    "出现", "突然", "持续", "不适", "申请售后", "想退", "能退吗",
)
_SPECIAL_POPULATION_SIGNALS = (
    "孕妇", "怀孕", "备孕", "哺乳", "医美后", "刷酸后", "激光后", "伤口",
)


def redact_sensitive_text(text: str) -> tuple[str, list[str]]:
    """返回脱敏文本及命中的敏感字段类型；订单号不会被掩码。"""
    value = str(text or "")
    matched: list[str] = []

    def replace_phone(match: re.Match[str]) -> str:
        matched.append("phone")
        return f"{match.group(1)}****{match.group(3)}"

    def replace_email(match: re.Match[str]) -> str:
        matched.append("email")
        return f"{match.group(1)}***{match.group(2)}"

    def replace_id(match: re.Match[str]) -> str:
        matched.append("id_card")
        return f"{match.group(1)}********{match.group(2)}"

    def replace_card(match: re.Match[str]) -> str:
        matched.append("bank_card")
        return f"{match.group(1)}********{match.group(2)}"

    value = _EMAIL_RE.sub(replace_email, value)
    value = _PHONE_RE.sub(replace_phone, value)
    value = _ID_CARD_RE.sub(replace_id, value)
    value = _BANK_CARD_RE.sub(replace_card, value)
    if _ADDRESS_RE.search(value):
        matched.append("address")
        value = _ADDRESS_RE.sub("地址：[已脱敏]", value)
    return value, sorted(set(matched))


def redact_payload(value: Any) -> Any:
    """递归清洗用于日志、追踪和人工工单的结构化数据。"""
    if isinstance(value, str):
        return redact_sensitive_text(value)[0]
    if isinstance(value, dict):
        return {str(key): redact_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    return value


def sanitize_history(history_dialogue: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cleaned = dict(history_dialogue or {})
    messages = []
    pii_types: set[str] = set()
    for message in cleaned.get("messages") or []:
        copied = dict(message)
        copied["content"], matches = redact_sensitive_text(str(copied.get("content") or ""))
        pii_types.update(matches)
        messages.append(copied)
    cleaned["messages"] = messages[-30:]
    return cleaned, sorted(pii_types)


def assess_prompt_injection(text: str) -> InjectionAssessment:
    score = 0
    signals: list[str] = []
    for name, pattern, weight in _INJECTION_RULES:
        if pattern.search(str(text or "")):
            score += weight
            signals.append(name)
    return InjectionAssessment(blocked=score >= 3, score=score, signals=tuple(signals))


def assess_skincare_safety(text: str) -> SkincareSafetyAssessment:
    content = str(text or "")
    urgent = tuple(signal for signal in _URGENT_SKIN_SIGNALS if signal in content)
    if urgent:
        return SkincareSafetyAssessment(
            level="urgent",
            handoff_required=True,
            signals=urgent,
            guidance=(
                "请立即停止使用相关产品，并用清水温和冲洗。若出现呼吸困难、面部或喉咙肿胀、"
                "大面积水疱等情况，请立即联系急救或前往医疗机构；在线客服不能替代医生诊断。"
            ),
        )

    reactions = tuple(signal for signal in _REACTION_SIGNALS if signal in content)
    has_reaction_context = any(signal in content for signal in _REACTION_CONTEXT_SIGNALS)
    if reactions and has_reaction_context:
        return SkincareSafetyAssessment(
            level="caution",
            handoff_required=True,
            signals=reactions,
            guidance=(
                "建议先停用可疑产品，避免继续叠加功效型成分，并记录使用时间与反应。"
                "若不适持续、加重或影响眼周，请及时就医；客服可继续协助售后，但不能作医疗诊断。"
            ),
        )

    special = tuple(signal for signal in _SPECIAL_POPULATION_SIGNALS if signal in content)
    if special:
        return SkincareSafetyAssessment(
            level="advisory",
            handoff_required=False,
            signals=special,
            guidance=(
                "孕期、哺乳期、医美后或皮肤屏障受损时，请先核对完整成分并咨询医生；"
                "首次使用应小范围斑贴测试，不对安全性或疗效作绝对承诺。"
            ),
        )

    return SkincareSafetyAssessment(
        level="normal", handoff_required=False, signals=(), guidance=""
    )


def apply_skincare_safety_boundary(
    question: str,
    answer: str,
    assessment: SkincareSafetyAssessment | None = None,
) -> str:
    assessment = assessment or assess_skincare_safety(question)
    if assessment.level == "normal" or not assessment.guidance:
        return answer.strip()
    if assessment.level == "urgent":
        return assessment.guidance
    if assessment.guidance in answer:
        return answer.strip()
    return f"{answer.strip()}\n\n安全提示：{assessment.guidance}".strip()
