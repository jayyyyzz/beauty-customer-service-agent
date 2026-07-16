# -*- coding: utf-8 -*-
"""构建冻结的美妆客服意图识别专项基准集 v1。

数据由人工定义的业务语句与三级标签模板生成，不使用待测模型生成 gold 标签。
生成后默认写入当前测评目录的 intent_benchmark_v1.jsonl。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parent
ROOT = EVALUATION_DIR.parents[1]
OUT = EVALUATION_DIR / "intent_benchmark_v1.jsonl"


# 每个一级意图 5 个已人工指定三级路径的基础语句；使用 4 种自然包装生成 20 条基础样本。
INTENT_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "price": [
        ("有可以领的优惠券吗？", "price.discount", "price.discount.coupon"),
        ("这次活动满多少可以减？", "price.discount", "price.discount.promotion"),
        ("会员买这款是什么价格？", "price.discount", "price.discount.member"),
        ("这瓶现在卖多少钱？", "price.inquiry", "price.inquiry.current"),
        ("水乳套装一共多少钱？", "price.inquiry", "price.inquiry.set"),
    ],
    "product_info": [
        ("这瓶精华是多少毫升？", "product_info.specification", "product_info.specification.capacity"),
        ("新包装和旧包装有什么区别？", "product_info.specification", "product_info.specification.package"),
        ("这个礼盒里面都有什么？", "product_info.specification", "product_info.specification.set_content"),
        ("这款面霜现在有现货吗？", "product_info.availability", "product_info.availability.stock"),
        ("缺货的色号什么时候补货？", "product_info.availability", "product_info.availability.restock"),
    ],
    "skin_type": [
        ("油皮适合用这款面霜吗？", "skin_type.fit", "skin_type.fit.oily"),
        ("我是大干皮，用它够滋润吗？", "skin_type.fit", "skin_type.fit.dry"),
        ("混合皮可以用这套水乳吗？", "skin_type.fit", "skin_type.fit.combination"),
        ("敏感肌能用这款精华吗？", "skin_type.fit", "skin_type.fit.sensitive"),
        ("四十岁用这款会不会不合适？", "skin_type.advice", "skin_type.advice.age"),
    ],
    "skin_concern": [
        ("最近总长痘应该选什么产品？", "skin_concern.acne", "skin_concern.acne.pimple"),
        ("脸上闭口很多怎么护理？", "skin_concern.acne", "skin_concern.acne.closed_comedone"),
        ("皮肤屏障受损应该怎么修护？", "skin_concern.repair", "skin_concern.repair.barrier"),
        ("脸特别暗沉应该怎么改善？", "skin_concern.tone", "skin_concern.tone.dullness"),
        ("鼻子两边毛孔很大怎么办？", "skin_concern.texture", "skin_concern.texture.pore"),
    ],
    "ingredient": [
        ("这款精华的核心成分是什么？", "ingredient.composition", "ingredient.composition.active"),
        ("烟酰胺添加浓度是多少？", "ingredient.composition", "ingredient.composition.concentration"),
        ("可以发一下完整成分表吗？", "ingredient.composition", "ingredient.composition.full_list"),
        ("这个里面含酒精吗？", "ingredient.safety", "ingredient.safety.alcohol"),
        ("这款用的是A醇还是A醛？", "ingredient.special", "ingredient.special.retinol"),
    ],
    "efficacy": [
        ("这款面霜主要是补水保湿的吗？", "efficacy.claim", "efficacy.claim.moisturizing"),
        ("这个精华有修护屏障的效果吗？", "efficacy.claim", "efficacy.claim.repair"),
        ("这款可以美白提亮吗？", "efficacy.claim", "efficacy.claim.brightening"),
        ("它能淡纹抗老吗？", "efficacy.claim", "efficacy.claim.anti_aging"),
        ("一般坚持用多久能看到效果？", "efficacy.timeline", "efficacy.timeline.how_long"),
    ],
    "usage": [
        ("这款精华一次用几滴？", "usage.method", "usage.method.dosage"),
        ("这个一周用几次合适？", "usage.method", "usage.method.frequency"),
        ("精华和面霜哪个先用？", "usage.method", "usage.method.order"),
        ("这款防晒晚上需要卸妆吗？", "usage.scene", "usage.scene.makeup_remove"),
        ("第一次用A醇怎么建立耐受？", "usage.tolerance", "usage.tolerance.build"),
    ],
    "routine": [
        ("洁面之后下一步用什么？", "routine.step", "routine.step.toner"),
        ("精华应该放在护肤第几步？", "routine.step", "routine.step.essence"),
        ("早上的护肤步骤怎么安排？", "routine.design", "routine.design.morning"),
        ("晚上完整护肤流程是什么？", "routine.design", "routine.design.night"),
        ("能帮我搭一套精简护肤流程吗？", "routine.design", "routine.design.simple"),
    ],
    "compatibility": [
        ("A醇和刷酸产品能一起用吗？", "compatibility.ingredient", "compatibility.ingredient.retinol_acid"),
        ("维C和烟酰胺可以叠加吗？", "compatibility.ingredient", "compatibility.ingredient.vc_niacinamide"),
        ("这两个精华可以一起涂吗？", "compatibility.product", "compatibility.product.layering"),
        ("这款和粉底叠加会搓泥吗？", "compatibility.product", "compatibility.product.pilling"),
        ("油皮叠加这两款会不会闷痘？", "compatibility.product", "compatibility.product.comedogenic"),
    ],
    "shade_color": [
        ("黄二白适合哪个粉底色号？", "shade_color.shade", "shade_color.shade.foundation"),
        ("黑眼圈应该选什么遮瑕色号？", "shade_color.shade", "shade_color.shade.concealer"),
        ("冷白皮适合哪个口红色号？", "shade_color.shade", "shade_color.shade.lipstick"),
        ("这款粉底是哑光还是水光妆效？", "shade_color.effect", "shade_color.effect.finish"),
        ("这款底妆能持妆几个小时？", "shade_color.effect", "shade_color.effect.longevity"),
    ],
    "authenticity_shelf_life": [
        ("你们店里的商品保证正品吗？", "authenticity_shelf_life.authenticity", "authenticity_shelf_life.authenticity.channel"),
        ("收到以后怎么查防伪码？", "authenticity_shelf_life.authenticity", "authenticity_shelf_life.authenticity.anti_fake"),
        ("这瓶产品的有效期到什么时候？", "authenticity_shelf_life.shelf_life", "authenticity_shelf_life.shelf_life.expiration"),
        ("能帮我看一下生产日期吗？", "authenticity_shelf_life.shelf_life", "authenticity_shelf_life.shelf_life.production_date"),
        ("开封以后可以保存多久？", "authenticity_shelf_life.shelf_life", "authenticity_shelf_life.shelf_life.open_after"),
    ],
    "safety_allergy": [
        ("第一次用要先做斑贴测试吗？", "safety_allergy.sensitive", "safety_allergy.sensitive.skin_test"),
        ("怀孕期间可以用这款产品吗？", "safety_allergy.sensitive", "safety_allergy.sensitive.pregnancy"),
        ("刚做完医美能用这个精华吗？", "safety_allergy.sensitive", "safety_allergy.sensitive.medical"),
        ("涂完以后脸上一直刺痛怎么办？", "safety_allergy.reaction", "safety_allergy.reaction.stinging"),
        ("用了以后全脸过敏了怎么办？", "safety_allergy.reaction", "safety_allergy.reaction.allergy"),
    ],
    "quality_issue": [
        ("收到的面霜膏体已经分层了。", "quality_issue.abnormal", "quality_issue.abnormal.texture"),
        ("新开的精华闻起来有一股怪味。", "quality_issue.abnormal", "quality_issue.abnormal.smell"),
        ("这瓶精华颜色氧化变黄了。", "quality_issue.abnormal", "quality_issue.abnormal.oxidation"),
        ("收到货发现瓶口一直漏液。", "quality_issue.abnormal", "quality_issue.abnormal.leakage"),
        ("快递盒完好但里面的瓶子碎了。", "quality_issue.abnormal", "quality_issue.abnormal.damaged"),
    ],
    "comparison": [
        ("这两款修护精华有什么区别？", "comparison.product", "comparison.product.similar"),
        ("新版和老版配方有什么变化？", "comparison.product", "comparison.product.upgrade"),
        ("这两瓶分别适合什么肤质？", "comparison.product", "comparison.product.suitability"),
        ("为什么你们店比别家贵？", "comparison.store", "comparison.store.price"),
        ("旗舰店和专柜的货是一样的吗？", "comparison.store", "comparison.store.authenticity"),
    ],
    "logistics": [
        ("我的快递单号是多少？", "logistics.status", "logistics.status.tracking"),
        ("订单现在运输到哪里了？", "logistics.status", "logistics.status.location"),
        ("正常下单后多久可以发货？", "logistics.time", "logistics.time.shipping"),
        ("寄到上海预计几天能到？", "logistics.time", "logistics.time.delivery"),
        ("帮我查询一下当前物流状态。", "logistics.status", "logistics.status.location"),
    ],
    "urge_shipment": [
        ("订单还没发，麻烦帮我催一下。", "urge_shipment.normal", "urge_shipment.normal.request"),
        ("都两天了为什么还不发货？", "urge_shipment.normal", "urge_shipment.normal.reason"),
        ("发货太慢了，我要投诉。", "urge_shipment.urgent", "urge_shipment.urgent.complaint"),
        ("明天出差要用，今天必须发出。", "urge_shipment.urgent", "urge_shipment.urgent.deadline"),
        ("能不能让仓库现在就给我发货？", "urge_shipment.normal", "urge_shipment.normal.request"),
    ],
    "logistics_delay": [
        ("快递三天没有更新物流了。", "logistics_delay.abnormal", "logistics_delay.abnormal.stuck"),
        ("包裹显示发出但好像丢了。", "logistics_delay.abnormal", "logistics_delay.abnormal.lost"),
        ("物流一直停在中转站不动。", "logistics_delay.abnormal", "logistics_delay.abnormal.stuck"),
        ("是不是因为暴雨所以快递延误了？", "logistics_delay.force_majeure", "logistics_delay.force_majeure.weather"),
        ("春节期间快递延迟多久？", "logistics_delay.force_majeure", "logistics_delay.force_majeure.holiday"),
    ],
    "after_sale": [
        ("申请退款后钱什么时候到账？", "after_sale.refund", "after_sale.refund.progress"),
        ("退款会退回到哪个账户？", "after_sale.refund", "after_sale.refund.account"),
        ("你们支持七天无理由退货吗？", "after_sale.return", "after_sale.return.policy"),
        ("退货应该寄到哪个地址？", "after_sale.return", "after_sale.return.address"),
        ("想把商品换成另一个色号怎么操作？", "after_sale.exchange", "after_sale.exchange.process"),
    ],
    "invoice": [
        ("可以给我开电子发票吗？", "invoice.type", "invoice.type.electronic"),
        ("公司报销需要纸质发票。", "invoice.type", "invoice.type.paper"),
        ("你们可以开增值税专票吗？", "invoice.type", "invoice.type.special"),
        ("订单完成后在哪里申请开票？", "invoice.process", "invoice.process.application"),
        ("发票抬头填错了怎么修改？", "invoice.process", "invoice.process.modification"),
    ],
    "gift_sample": [
        ("这次活动会送什么赠品？", "gift_sample.gift", "gift_sample.gift.rule"),
        ("包裹里少了活动赠品怎么办？", "gift_sample.gift", "gift_sample.gift.missing"),
        ("送朋友可以帮忙包装成礼盒吗？", "gift_sample.gift", "gift_sample.gift.package"),
        ("可以先申请一个试用装吗？", "gift_sample.sample", "gift_sample.sample.request"),
        ("你们有没有小样套装可以买？", "gift_sample.sample", "gift_sample.sample.trial_pack"),
    ],
    "review": [
        ("好评以后可以返多少钱？", "review.policy", "review.policy.amount"),
        ("参加好评返现需要满足什么条件？", "review.policy", "review.policy.condition"),
        ("评价截图应该提交到哪里？", "review.process", "review.process.submission"),
        ("好评已经提交了，返现什么时候发？", "review.process", "review.process.payment"),
        ("五星评价返现活动还有效吗？", "review.policy", "review.policy.condition"),
    ],
    "other": [
        ("今天天气怎么样？", "other.irrelevant", "other.irrelevant"),
        ("帮我写一段旅游攻略。", "other.irrelevant", "other.irrelevant"),
        ("你会不会解数学题？", "other.irrelevant", "other.irrelevant"),
        ("这个到底怎么样？", "other.unclear", "other.unclear"),
        ("帮我处理一下。", "other.unclear", "other.unclear"),
    ],
}


WRAPPERS = [
    "{q}",
    "想问一下，{q}",
    "亲，{q}",
    "麻烦回复一下：{q}",
]


CONTEXT_CASES: list[tuple[list[dict[str, str]], str, str, str, str]] = [
    ([{"role": "buyer", "content": "我刚领到一张店铺券"}], "这个能和满减一起用吗？", "price", "price.discount", "price.discount.coupon"),
    ([{"role": "seller", "content": "您看的修护礼盒有三个单品"}], "那里面具体是哪几个？", "product_info", "product_info.specification", "product_info.specification.set_content"),
    ([{"role": "buyer", "content": "我是混油皮，T区特别油"}], "那这套适合我吗？", "skin_type", "skin_type.fit", "skin_type.fit.combination"),
    ([{"role": "buyer", "content": "最近脸上反复长闭口"}], "这种情况怎么护理？", "skin_concern", "skin_concern.acne", "skin_concern.acne.closed_comedone"),
    ([{"role": "seller", "content": "这款主打烟酰胺提亮"}], "它的浓度是多少？", "ingredient", "ingredient.composition", "ingredient.composition.concentration"),
    ([{"role": "buyer", "content": "我想改善脸上的暗沉"}], "这个真的有用吗？", "efficacy", "efficacy.claim", "efficacy.claim.brightening"),
    ([{"role": "seller", "content": "您可以使用这瓶A醇精华"}], "第一次应该怎么用？", "usage", "usage.tolerance", "usage.tolerance.build"),
    ([{"role": "buyer", "content": "我有洁面、水、精华和面霜"}], "晚上按什么步骤用？", "routine", "routine.design", "routine.design.night"),
    ([{"role": "buyer", "content": "我晚上已经在用A醇"}], "还能同时刷酸吗？", "compatibility", "compatibility.ingredient", "compatibility.ingredient.retinol_acid"),
    ([{"role": "seller", "content": "您看的粉底有自然色和象牙白"}], "黄二白选哪个？", "shade_color", "shade_color.shade", "shade_color.shade.foundation"),
    ([{"role": "buyer", "content": "这瓶已经开封半年了"}], "现在还能继续用吗？", "authenticity_shelf_life", "authenticity_shelf_life.shelf_life", "authenticity_shelf_life.shelf_life.open_after"),
    ([{"role": "buyer", "content": "我昨晚第一次用了这款精华"}], "现在脸特别刺痛怎么办？", "safety_allergy", "safety_allergy.reaction", "safety_allergy.reaction.stinging"),
    ([{"role": "buyer", "content": "刚收到一瓶精华"}, {"role": "seller", "content": "请问商品有什么问题？"}], "它一直从瓶口往外漏。", "quality_issue", "quality_issue.abnormal", "quality_issue.abnormal.leakage"),
    ([{"role": "seller", "content": "您正在看修护精华和舒缓精华"}], "这两个哪个更适合敏感肌？", "comparison", "comparison.product", "comparison.product.suitability"),
    ([{"role": "buyer", "content": "订单 MOCK202606260003 已经发出了"}], "它现在到哪了？", "logistics", "logistics.status", "logistics.status.location"),
    ([{"role": "buyer", "content": "我昨天上午下的订单还没出库"}], "能帮我催一下吗？", "urge_shipment", "urge_shipment.normal", "urge_shipment.normal.request"),
    ([{"role": "seller", "content": "您的包裹三天前已经到达中转站"}], "为什么到现在都没有动？", "logistics_delay", "logistics_delay.abnormal", "logistics_delay.abnormal.stuck"),
    ([{"role": "buyer", "content": "我已经提交了退款申请"}], "这个钱什么时候回来？", "after_sale", "after_sale.refund", "after_sale.refund.progress"),
    ([{"role": "buyer", "content": "刚才申请的电子票抬头写错了"}], "现在还能改吗？", "invoice", "invoice.process", "invoice.process.modification"),
    ([{"role": "buyer", "content": "活动页面写着会送三件小样"}], "为什么我的包裹里没有？", "gift_sample", "gift_sample.gift", "gift_sample.gift.missing"),
    ([{"role": "buyer", "content": "我昨天已经晒图好评了"}], "怎么还没收到钱？", "review", "review.process", "review.process.payment"),
    ([{"role": "buyer", "content": "我脸上最近特别干还起皮"}], "应该重点解决什么问题？", "skin_concern", "skin_concern.texture", "skin_concern.texture.dryness"),
    ([{"role": "seller", "content": "这款产品含有维C"}], "能和烟酰胺一起用吗？", "compatibility", "compatibility.ingredient", "compatibility.ingredient.vc_niacinamide"),
    ([{"role": "buyer", "content": "订单一直显示待发货"}], "今天必须给我寄出来，我明天要用。", "urge_shipment", "urge_shipment.urgent", "urge_shipment.urgent.deadline"),
    ([{"role": "buyer", "content": "快递已经七天没更新了"}], "是不是弄丢了？", "logistics_delay", "logistics_delay.abnormal", "logistics_delay.abnormal.lost"),
    ([{"role": "buyer", "content": "收到的口红外壳裂了"}], "这种情况能赔吗？", "quality_issue", "quality_issue.complaint", "quality_issue.complaint.compensation"),
    ([{"role": "buyer", "content": "我准备退掉这瓶没拆封的精华"}], "退货寄到哪里？", "after_sale", "after_sale.return", "after_sale.return.address"),
    ([{"role": "buyer", "content": "我是孕妇，最近皮肤很干"}], "前面那款面霜我能用吗？", "safety_allergy", "safety_allergy.sensitive", "safety_allergy.sensitive.pregnancy"),
    ([{"role": "seller", "content": "您看的两款分别是旧版和升级版"}], "主要升级了什么？", "comparison", "comparison.product", "comparison.product.upgrade"),
    ([{"role": "buyer", "content": "我买的是轻薄防晒"}], "晚上洗脸前还要卸吗？", "usage", "usage.scene", "usage.scene.makeup_remove"),
]


# 易混淆边界：每条用两种表达包装，共 60 条。
CONFUSION_CASES: list[tuple[str, str, str, str]] = [
    ("下单以后通常多久发货？", "logistics", "logistics.time", "logistics.time.shipping"),
    ("已经等两天了，马上给我发货。", "urge_shipment", "urge_shipment.normal", "urge_shipment.normal.request"),
    ("快递预计什么时候送到？", "logistics", "logistics.time", "logistics.time.delivery"),
    ("快递五天不更新，帮我查异常。", "logistics_delay", "logistics_delay.abnormal", "logistics_delay.abnormal.stuck"),
    ("退款已经通过但还没到账。", "after_sale", "after_sale.refund", "after_sale.refund.progress"),
    ("物流延误导致我没收到商品。", "logistics_delay", "logistics_delay.abnormal", "logistics_delay.abnormal.stuck"),
    ("上脸以后皮肤红肿发痒。", "safety_allergy", "safety_allergy.reaction", "safety_allergy.reaction.allergy"),
    ("没使用就发现膏体颜色异常。", "quality_issue", "quality_issue.abnormal", "quality_issue.abnormal.oxidation"),
    ("我脸色暗沉，应该怎么护理？", "skin_concern", "skin_concern.tone", "skin_concern.tone.dullness"),
    ("这款精华能不能提亮肤色？", "efficacy", "efficacy.claim", "efficacy.claim.brightening"),
    ("这瓶精华具体排在哪一步？", "usage", "usage.method", "usage.method.order"),
    ("帮我设计水乳精华面霜的完整顺序。", "routine", "routine.design", "routine.design.simple"),
    ("满三百会送什么小样？", "gift_sample", "gift_sample.gift", "gift_sample.gift.rule"),
    ("满三百能减多少钱？", "price", "price.discount", "price.discount.promotion"),
    ("漏液的商品要怎么申请退货？", "quality_issue", "quality_issue.complaint", "quality_issue.complaint.return"),
    ("商品没问题，但我不想要了怎么退？", "after_sale", "after_sale.return", "after_sale.return.policy"),
    ("A醇和酸叠加会不会刺激？", "compatibility", "compatibility.ingredient", "compatibility.ingredient.retinol_acid"),
    ("用了A醇以后已经刺痛发红。", "safety_allergy", "safety_allergy.reaction", "safety_allergy.reaction.stinging"),
    ("这个套装包含哪几瓶？", "product_info", "product_info.specification", "product_info.specification.set_content"),
    ("这个套装现在卖多少钱？", "price", "price.inquiry", "price.inquiry.set"),
    ("黄皮应该选哪个粉底色号？", "shade_color", "shade_color.shade", "shade_color.shade.foundation"),
    ("油性皮肤适合用这个粉底吗？", "skin_type", "skin_type.fit", "skin_type.fit.oily"),
    ("精华刚开封就有酸臭味。", "quality_issue", "quality_issue.abnormal", "quality_issue.abnormal.smell"),
    ("这个批次是不是正品？", "authenticity_shelf_life", "authenticity_shelf_life.authenticity", "authenticity_shelf_life.authenticity.anti_fake"),
    ("好评后赠送的小样是什么？", "gift_sample", "gift_sample.gift", "gift_sample.gift.rule"),
    ("好评后返现金额是多少？", "review", "review.policy", "review.policy.amount"),
    ("敏感肌用之前需要注意什么？", "safety_allergy", "safety_allergy.sensitive", "safety_allergy.sensitive.skin_test"),
    ("这款产品适不适合敏感肌？", "skin_type", "skin_type.fit", "skin_type.fit.sensitive"),
    ("成分表里有没有香精？", "ingredient", "ingredient.safety", "ingredient.safety.fragrance"),
    ("这款闻起来有香味正常吗？", "quality_issue", "quality_issue.abnormal", "quality_issue.abnormal.smell"),
]


UNCLEAR_QUESTIONS = [
    "这个怎么样？", "帮我弄一下。", "还是之前那个。", "可以处理吗？", "我该怎么办？",
    "就是那个东西。", "有问题。", "你看着办吧。", "能不能快点？", "怎么回事？",
    "我想问一下。", "不太对劲。", "帮我查查。", "这个要怎么搞？", "给个说法。",
    "为什么会这样？", "然后呢？", "现在怎么办？", "你们能解决不？", "麻烦处理。",
]


MULTI_INTENT_CASES: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("这款适合油皮吗，另外现在有优惠券吗？", [("skin_type", "skin_type.fit", "skin_type.fit.oily"), ("price", "price.discount", "price.discount.coupon")]),
    ("订单到哪了，顺便帮我开电子发票。", [("logistics", "logistics.status", "logistics.status.location"), ("invoice", "invoice.type", "invoice.type.electronic")]),
    ("收到的瓶子漏液了，我要申请退货。", [("quality_issue", "quality_issue.abnormal", "quality_issue.abnormal.leakage"), ("after_sale", "after_sale.return", "after_sale.return.policy")]),
    ("这个精华含酒精吗，敏感肌能不能用？", [("ingredient", "ingredient.safety", "ingredient.safety.alcohol"), ("skin_type", "skin_type.fit", "skin_type.fit.sensitive")]),
    ("A醇怎么建立耐受，能和酸一起用吗？", [("usage", "usage.tolerance", "usage.tolerance.build"), ("compatibility", "compatibility.ingredient", "compatibility.ingredient.retinol_acid")]),
    ("快递一周没更新，帮我查一下并申请补偿。", [("logistics_delay", "logistics_delay.abnormal", "logistics_delay.abnormal.stuck"), ("after_sale", "after_sale.refund", "after_sale.refund.progress")]),
    ("这瓶多少钱，套装里又包含什么？", [("price", "price.inquiry", "price.inquiry.current"), ("product_info", "product_info.specification", "product_info.specification.set_content")]),
    ("我脸很干，这款有没有保湿效果？", [("skin_concern", "skin_concern.texture", "skin_concern.texture.dryness"), ("efficacy", "efficacy.claim", "efficacy.claim.moisturizing")]),
    ("这个色号适合黄皮吗，持妆怎么样？", [("shade_color", "shade_color.shade", "shade_color.shade.foundation"), ("shade_color", "shade_color.effect", "shade_color.effect.longevity")]),
    ("赠品漏发了，好评返现也没到账。", [("gift_sample", "gift_sample.gift", "gift_sample.gift.missing"), ("review", "review.process", "review.process.payment")]),
]


def add_case(cases: list[dict[str, Any]], **case: Any) -> None:
    case["case_id"] = f"intent_{len(cases) + 1:04d}"
    cases.append(case)


def main() -> None:
    cases: list[dict[str, Any]] = []

    for level1, specs in INTENT_SPECS.items():
        for question, level2, level3 in specs:
            for wrapper in WRAPPERS:
                add_case(
                    cases,
                    history=[],
                    question=wrapper.format(q=question),
                    gold_intent_level1=level1,
                    gold_intent_level2=level2,
                    gold_intent_level3=level3,
                    gold_intents=[{"level1": level1, "level2": level2, "level3": level3}],
                    should_clarify=(level3 == "other.unclear"),
                    difficulty="easy" if wrapper == "{q}" else "medium",
                    scenario="base",
                    has_noise=False,
                    emotion="neutral",
                    is_multi_intent=False,
                )

        # 每个一级意图额外 4 条口语/噪声样本。
        for index, (question, level2, level3) in enumerate(specs[:4]):
            noisy_prefix = ["？？？ ", "人呢，", "急急急！！", "客服客服，"][index]
            noisy_question = noisy_prefix + question.replace("？", "??")
            add_case(
                cases,
                history=[],
                question=noisy_question,
                gold_intent_level1=level1,
                gold_intent_level2=level2,
                gold_intent_level3=level3,
                gold_intents=[{"level1": level1, "level2": level2, "level3": level3}],
                should_clarify=(level3 == "other.unclear"),
                difficulty="medium",
                scenario="noise",
                has_noise=True,
                emotion="negative" if index in {1, 2} else "neutral",
                is_multi_intent=False,
            )

    for history, question, level1, level2, level3 in CONTEXT_CASES:
        for wrapper in ("{q}", "那，{q}"):
            add_case(
                cases,
                history=history,
                question=wrapper.format(q=question),
                gold_intent_level1=level1,
                gold_intent_level2=level2,
                gold_intent_level3=level3,
                gold_intents=[{"level1": level1, "level2": level2, "level3": level3}],
                should_clarify=False,
                difficulty="hard",
                scenario="context",
                has_noise=False,
                emotion="neutral",
                is_multi_intent=False,
            )

    for question, level1, level2, level3 in CONFUSION_CASES:
        for wrapper in ("{q}", "请准确判断：{q}"):
            add_case(
                cases,
                history=[],
                question=wrapper.format(q=question),
                gold_intent_level1=level1,
                gold_intent_level2=level2,
                gold_intent_level3=level3,
                gold_intents=[{"level1": level1, "level2": level2, "level3": level3}],
                should_clarify=False,
                difficulty="hard",
                scenario="confusable",
                has_noise=False,
                emotion="neutral",
                is_multi_intent=False,
            )

    for question in UNCLEAR_QUESTIONS:
        for wrapper in ("{q}", "亲，{q}"):
            add_case(
                cases,
                history=[],
                question=wrapper.format(q=question),
                gold_intent_level1="other",
                gold_intent_level2="other.unclear",
                gold_intent_level3="other.unclear",
                gold_intents=[{"level1": "other", "level2": "other.unclear", "level3": "other.unclear"}],
                acceptable_intents=["other", "after_sale"],
                should_clarify=True,
                difficulty="hard",
                scenario="clarify",
                has_noise=False,
                emotion="neutral",
                is_multi_intent=False,
            )

    for question, gold_intents in MULTI_INTENT_CASES:
        for wrapper in ("{q}", "麻烦一起处理：{q}"):
            primary = gold_intents[0]
            add_case(
                cases,
                history=[],
                question=wrapper.format(q=question),
                gold_intent_level1=primary[0],
                gold_intent_level2=primary[1],
                gold_intent_level3=primary[2],
                gold_intents=[{"level1": x[0], "level2": x[1], "level3": x[2]} for x in gold_intents],
                should_clarify=False,
                difficulty="hard",
                scenario="multi_intent",
                has_noise=False,
                emotion="neutral",
                is_multi_intent=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for case in cases:
        counts[case["scenario"]] = counts.get(case["scenario"], 0) + 1
    print(f"wrote {len(cases)} cases -> {OUT}")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
