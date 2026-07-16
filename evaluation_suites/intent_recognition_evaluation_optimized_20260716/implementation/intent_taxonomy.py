# -*- coding: utf-8 -*-
"""美妆电商客服意图体系的机器可校验定义。"""

from __future__ import annotations

from collections import defaultdict


_PATH_TEXT = """
price.discount.coupon
price.discount.promotion
price.discount.member
price.inquiry.current
price.inquiry.set
price.inquiry.compare
product_info.specification.capacity
product_info.specification.package
product_info.specification.set_content
product_info.availability.stock
product_info.availability.restock
skin_type.fit.oily
skin_type.fit.dry
skin_type.fit.combination
skin_type.fit.sensitive
skin_type.advice.season
skin_type.advice.age
skin_concern.acne.pimple
skin_concern.acne.closed_comedone
skin_concern.acne.blackhead
skin_concern.repair.barrier
skin_concern.repair.redness
skin_concern.tone.dullness
skin_concern.tone.spot
skin_concern.texture.pore
skin_concern.texture.dryness
skin_concern.texture.oiliness
ingredient.composition.active
ingredient.composition.concentration
ingredient.composition.full_list
ingredient.safety.alcohol
ingredient.safety.fragrance
ingredient.safety.preservative
ingredient.special.acid
ingredient.special.retinol
ingredient.special.niacinamide
ingredient.special.vitamin_c
efficacy.claim.moisturizing
efficacy.claim.repair
efficacy.claim.soothing
efficacy.claim.brightening
efficacy.claim.anti_aging
efficacy.claim.oil_control
efficacy.claim.acne_care
efficacy.claim.sun_protection
efficacy.timeline.how_long
efficacy.timeline.persistence
usage.method.dosage
usage.method.frequency
usage.method.order
usage.scene.morning
usage.scene.night
usage.scene.makeup_remove
usage.tolerance.build
usage.tolerance.irritation
routine.step.cleanser
routine.step.toner
routine.step.essence
routine.step.cream
routine.step.sunscreen
routine.design.morning
routine.design.night
routine.design.simple
compatibility.ingredient.conflict
compatibility.ingredient.retinol_acid
compatibility.ingredient.vc_niacinamide
compatibility.product.layering
compatibility.product.pilling
compatibility.product.comedogenic
shade_color.shade.foundation
shade_color.shade.concealer
shade_color.shade.lipstick
shade_color.effect.finish
shade_color.effect.coverage
shade_color.effect.longevity
shade_color.effect.color_difference
authenticity_shelf_life.authenticity.channel
authenticity_shelf_life.authenticity.anti_fake
authenticity_shelf_life.shelf_life.expiration
authenticity_shelf_life.shelf_life.production_date
authenticity_shelf_life.shelf_life.open_after
safety_allergy.sensitive.skin_test
safety_allergy.sensitive.pregnancy
safety_allergy.sensitive.medical
safety_allergy.reaction.stinging
safety_allergy.reaction.redness
safety_allergy.reaction.allergy
safety_allergy.reaction.breakout
quality_issue.abnormal.texture
quality_issue.abnormal.smell
quality_issue.abnormal.oxidation
quality_issue.abnormal.leakage
quality_issue.abnormal.damaged
quality_issue.complaint.return
quality_issue.complaint.compensation
comparison.product.similar
comparison.product.upgrade
comparison.product.suitability
comparison.store.price
comparison.store.authenticity
logistics.status.tracking
logistics.status.location
logistics.time.shipping
logistics.time.delivery
urge_shipment.normal.request
urge_shipment.normal.reason
urge_shipment.urgent.complaint
urge_shipment.urgent.deadline
logistics_delay.abnormal.stuck
logistics_delay.abnormal.lost
logistics_delay.force_majeure.weather
logistics_delay.force_majeure.holiday
after_sale.refund.progress
after_sale.refund.account
after_sale.return.policy
after_sale.return.address
after_sale.return.condition
after_sale.exchange.request
after_sale.exchange.process
after_sale.exchange.cost
invoice.type.electronic
invoice.type.paper
invoice.type.special
invoice.process.application
invoice.process.modification
gift_sample.gift.rule
gift_sample.gift.missing
gift_sample.gift.package
gift_sample.sample.request
gift_sample.sample.trial_pack
review.policy.amount
review.policy.condition
review.process.submission
review.process.payment
other.irrelevant
other.unclear
"""


INTENT_LEVEL3_PATHS = frozenset(_PATH_TEXT.split())
INTENT_LEVEL1_VALUES = frozenset(path.split(".", 1)[0] for path in INTENT_LEVEL3_PATHS)

_grouped: dict[str, list[str]] = defaultdict(list)
for _path in sorted(INTENT_LEVEL3_PATHS):
    _grouped[_path.split(".", 1)[0]].append(_path)
INTENT_PATHS_BY_LEVEL1 = {key: tuple(values) for key, values in _grouped.items()}


INTENT_LEVEL1_DESCRIPTIONS = {
    "price": "价格、优惠券、满减、会员价和套装价格",
    "product_info": "容量、包装、套装内容、库存和补货",
    "skin_type": "商品是否适合某种肤质、年龄或季节",
    "skin_concern": "用户自身的痘痘、暗沉、毛孔、干燥、屏障等问题",
    "ingredient": "成分表、浓度、酒精香精、A醇酸类等",
    "efficacy": "商品能否提供保湿、修护、提亮、抗老等功效及见效时间",
    "usage": "单个产品的用量、频率、顺序、早晚使用、卸妆和耐受",
    "routine": "整套护肤步骤或早晚流程设计",
    "compatibility": "多个成分或产品能否叠加、冲突、搓泥或闷痘",
    "shade_color": "粉底、遮瑕、口红的色号和妆效",
    "authenticity_shelf_life": "正品、防伪、生产日期、有效期和开封期限",
    "safety_allergy": "孕期医美后安全性，以及已发生的刺痛、泛红、过敏和爆痘",
    "quality_issue": "商品本体漏液、破损、异味、氧化、分层及质量投诉",
    "comparison": "两款商品、版本、店铺价格或正品差异对比",
    "logistics": "中性查询发货时间、快递单号、位置和预计送达",
    "urge_shipment": "尚未发货且用户明确催促、投诉或给出截止时间",
    "logistics_delay": "已经发出后物流停滞、丢失或因天气节假日延误",
    "after_sale": "商品无质量主诉下的退款、退货、换货政策与进度",
    "invoice": "发票类型、申请和修改",
    "gift_sample": "赠品、小样、漏发和礼盒包装",
    "review": "好评返现规则、截图提交与返现发放",
    "other": "领域外问题或缺少上下文、无法判断的请求",
}


BOUNDARY_RULES = (
    "中性询问多久发货属于 logistics；明确要求赶紧发、投诉发货慢或给出截止时间属于 urge_shipment。",
    "包裹已经发出后长时间不更新、疑似丢失属于 logistics_delay；仅查询当前位置属于 logistics。",
    "用户描述商品漏液、破损、异味、变色、分层时优先 quality_issue；商品本身无异常、只问退换流程时属于 after_sale。",
    "用户描述自己存在暗沉、痘痘、毛孔等问题属于 skin_concern；询问某产品能否改善这些问题属于 efficacy。",
    "询问单个产品怎么用、用量或先后顺序属于 usage；要求设计一整套早晚护肤步骤属于 routine。",
    "询问产品是否适合敏感肌属于 skin_type；询问孕期、医美后安全或已经刺痛过敏属于 safety_allergy。",
    "询问活动减多少钱属于 price；询问活动赠送什么或赠品漏发属于 gift_sample。",
    "评价后返现金额、条件和到账进度属于 review，不属于普通 after_sale。",
    "两款产品有什么区别、哪个更适合属于 comparison；询问单一商品规格或库存属于 product_info。",
    "同一句话包含多个可独立处理的诉求时分别输出；同一一级意图下两个不同三级诉求也要分别保留。",
)


def level2_from_level3(path: str) -> str:
    parts = path.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else path


def level1_from_level3(path: str) -> str:
    return path.split(".", 1)[0]


def is_valid_level3(path: str) -> bool:
    return path in INTENT_LEVEL3_PATHS


def taxonomy_prompt() -> str:
    sections = []
    for level1 in sorted(INTENT_PATHS_BY_LEVEL1):
        description = INTENT_LEVEL1_DESCRIPTIONS[level1]
        paths = ", ".join(INTENT_PATHS_BY_LEVEL1[level1])
        sections.append(f"- {level1}（{description}）：{paths}")
    return "\n".join(sections)

