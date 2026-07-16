# -*- coding: utf-8 -*-
"""生成可复现的客服 RAG 第一阶段检索基准集。"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "retrieval_benchmark.jsonl"


BASE_CASES = [
    {
        "key": "allergy_return",
        "name": "过敏后退货",
        "intent": "after_sale",
        "groups": [["过敏", "红疹", "刺痛", "泛红"], ["退货", "退款", "售后", "能退"]],
        "variants": [
            ("商品使用后过敏可以退货吗？", ["过敏", "退货"], "standard", "easy", []),
            ("精华抹完脸又红又痒，这都拆了还能退不", ["过敏", "红痒", "拆封退货"], "colloquial_noise", "medium", []),
            ("那这瓶还能退吗？", ["精华", "过敏", "退货"], "multi_turn", "hard", ["我用了这瓶精华以后脸上起了红疹，还有刺痛。"]),
        ],
    },
    {
        "key": "logistics_delay",
        "name": "物流延误规则",
        "intent": "logistics_delay",
        "groups": [["快递", "物流"], ["没到", "延误", "异常", "未更新", "丢件"]],
        "variants": [
            ("快递一直没到或物流长时间未更新怎么办？", ["快递", "未更新", "延误"], "standard", "easy", []),
            ("物流卡三天了咋整啊，别又给我丢件了", ["物流", "三天未更新", "丢件"], "colloquial_noise", "medium", []),
            ("它怎么还在原地？", ["快递", "物流停滞", "未更新"], "multi_turn", "hard", ["我前几天买的东西已经发货了。"]),
        ],
    },
    {
        "key": "invoice",
        "name": "发票申请",
        "intent": "invoice",
        "groups": [["发票", "开票"], ["抬头", "税号", "订单", "申请"]],
        "variants": [
            ("订单发票怎么申请？", ["订单", "发票", "申请"], "standard", "easy", []),
            ("能开票不，公司抬头和税号填哪", ["开票", "公司抬头", "税号"], "colloquial_noise", "medium", []),
            ("这个怎么开？", ["订单", "电子发票", "开票"], "multi_turn", "hard", ["公司报销需要电子发票。"]),
        ],
    },
    {
        "key": "skin_type",
        "name": "油皮适配",
        "intent": "skin_type",
        "groups": [["油皮", "油性", "出油"], ["适合", "能用", "肤质", "闷痘"]],
        "variants": [
            ("油性皮肤适合使用这款面霜吗？", ["油皮", "面霜", "适合"], "standard", "easy", []),
            ("大油田用它会不会糊一脸还闷痘", ["油皮", "闷痘", "肤质适配"], "colloquial_noise", "medium", []),
            ("那我这种皮肤能用吗？", ["油皮", "面霜", "能用"], "multi_turn", "hard", ["我T区特别容易出油，还经常闷痘。"]),
        ],
    },
    {
        "key": "ingredient",
        "name": "成分浓度",
        "intent": "ingredient",
        "groups": [["成分", "烟酰胺", "浓度"], ["含量", "%", "配方", "耐受"]],
        "variants": [
            ("这款产品的烟酰胺浓度是多少？", ["烟酰胺", "浓度", "成分"], "standard", "easy", []),
            ("烟酰安加了几个点啊，敏感皮扛得住不", ["烟酰胺", "浓度", "耐受"], "colloquial_noise", "medium", []),
            ("它含量高吗？", ["烟酰胺", "含量", "浓度"], "multi_turn", "hard", ["我在看一款含烟酰胺的美白精华。"]),
        ],
    },
    {
        "key": "compatibility",
        "name": "成分搭配",
        "intent": "compatibility",
        "groups": [["视黄醇", "A醇"], ["果酸", "酸类"], ["搭配", "一起", "冲突", "同用"]],
        "variants": [
            ("视黄醇能和果酸一起使用吗？", ["视黄醇", "果酸", "搭配"], "standard", "easy", []),
            ("a醇跟果酸能不能同一晚叠，怕烂脸", ["A醇", "果酸", "同用"], "colloquial_noise", "medium", []),
            ("那能跟这个一起用吗？", ["视黄醇", "果酸", "成分冲突"], "multi_turn", "hard", ["我晚上在用视黄醇精华，最近还买了果酸产品。"]),
        ],
    },
    {
        "key": "quality_issue",
        "name": "破损漏液",
        "intent": "quality_issue",
        "groups": [["漏液", "破损", "质量", "坏"], ["补发", "退款", "换货", "处理"]],
        "variants": [
            ("收到商品破损漏液应该怎么处理？", ["破损", "漏液", "售后"], "standard", "easy", []),
            ("瓶口全漏了包装也压烂了，赶紧给个说法", ["漏液", "包装破损", "补发"], "colloquial_noise", "medium", []),
            ("这种情况能补吗？", ["商品破损", "漏液", "补发"], "multi_turn", "hard", ["快递刚到，瓶子裂了，液体漏得到处都是。"]),
        ],
    },
    {
        "key": "dispatch",
        "name": "发货时效",
        "intent": "logistics",
        "groups": [["发货", "发出"], ["多久", "时间", "小时", "天"]],
        "variants": [
            ("现货订单通常多久发货？", ["现货", "多久", "发货"], "standard", "easy", []),
            ("今天拍啥时候能给我发，着急用", ["今天下单", "发货时间", "着急"], "colloquial_noise", "medium", []),
            ("大概什么时候能出库？", ["订单", "发货", "出库时间"], "multi_turn", "hard", ["我刚下单了一瓶精华，是现货。"]),
        ],
    },
    {
        "key": "shelf_life",
        "name": "开封保质期",
        "intent": "authenticity_shelf_life",
        "groups": [["开封", "开盖"], ["保质期", "多久", "月", "保存"]],
        "variants": [
            ("护肤品开封后保质期通常是多久？", ["开封", "保质期", "多久"], "standard", "easy", []),
            ("开盖半年还能往脸上抹吗", ["开封半年", "保质期", "能用"], "colloquial_noise", "medium", []),
            ("这个打开后能放多长时间？", ["开封", "保存", "保质期"], "multi_turn", "hard", ["我去年买了一瓶面霜，一直没有开封。"]),
        ],
    },
    {
        "key": "shade",
        "name": "黄皮口红色号",
        "intent": "shade_color",
        "groups": [["黄皮", "肤色", "黄黑皮"], ["口红", "色号", "显白"]],
        "variants": [
            ("黄皮适合选择什么口红色号？", ["黄皮", "口红", "色号"], "standard", "easy", []),
            ("黄黑皮求个不荧光还显白的唇色", ["黄黑皮", "显白", "口红色号"], "colloquial_noise", "medium", []),
            ("那我选哪支？", ["黄皮", "口红", "显白色号"], "multi_turn", "hard", ["我肤色偏黄，担心豆沙色显得没精神。"]),
        ],
    },
    {
        "key": "promotion",
        "name": "满减优惠",
        "intent": "price",
        "groups": [["优惠", "满减", "活动", "优惠券"], ["价格", "便宜", "折扣", "叠加"]],
        "variants": [
            ("现在有哪些满减优惠活动？", ["满减", "优惠", "活动"], "standard", "easy", []),
            ("券能叠不，怎么凑单最便宜", ["优惠券", "叠加", "便宜"], "colloquial_noise", "medium", []),
            ("这个还能再减吗？", ["商品价格", "满减", "优惠券"], "multi_turn", "hard", ["详情页显示这款商品正在参加店铺活动。"]),
        ],
    },
    {
        "key": "usage_order",
        "name": "护肤使用顺序",
        "intent": "routine",
        "groups": [["顺序", "步骤", "先", "后"], ["精华", "面霜", "洁面", "防晒"]],
        "variants": [
            ("洁面、精华、面霜和防晒的使用顺序是什么？", ["护肤步骤", "精华", "面霜", "防晒"], "standard", "easy", []),
            ("水乳精华到底谁先谁后，每次都整乱", ["水乳精华", "先后顺序", "护肤流程"], "colloquial_noise", "medium", []),
            ("它应该放在哪一步？", ["精华", "面霜前", "使用顺序"], "multi_turn", "hard", ["我刚买了一瓶精华，平时会用水、乳液和面霜。"]),
        ],
    },
    {
        "key": "dosage",
        "name": "产品用量",
        "intent": "usage",
        "groups": [["用量", "多少", "几滴", "黄豆"], ["一次", "每次", "涂"]],
        "variants": [
            ("精华每次使用多少量比较合适？", ["精华", "每次", "用量"], "standard", "easy", []),
            ("一次挤一大坨是不是太多了，总搓泥", ["一次用量", "太多", "搓泥"], "colloquial_noise", "medium", []),
            ("一次这么多够吗？", ["精华", "一次", "使用量"], "multi_turn", "hard", ["我每次大约只用一滴精华。"]),
        ],
    },
    {
        "key": "efficacy",
        "name": "保湿修护功效",
        "intent": "efficacy",
        "groups": [["功效", "保湿", "修护", "补水"], ["干燥", "泛红", "屏障", "改善"]],
        "variants": [
            ("这款产品对干燥泛红有保湿修护功效吗？", ["干燥", "泛红", "保湿修护"], "standard", "easy", []),
            ("脸干得起皮还红，它能救一下屏障不", ["干燥起皮", "泛红", "屏障修护"], "colloquial_noise", "medium", []),
            ("它主要改善什么？", ["保湿", "修护", "干燥泛红"], "multi_turn", "hard", ["客服给我推荐了一款主打修护的面霜。"]),
        ],
    },
    {
        "key": "gift",
        "name": "送礼推荐",
        "intent": "gift_sample",
        "groups": [["送礼", "礼物", "礼盒", "送妈妈"], ["推荐", "合适", "包装"]],
        "variants": [
            ("想送妈妈护肤品，有适合送礼的礼盒推荐吗？", ["送妈妈", "护肤品", "礼盒推荐"], "standard", "easy", []),
            ("给女朋友买，包装得体点别太踩雷", ["女朋友礼物", "包装", "推荐"], "colloquial_noise", "medium", []),
            ("这个拿来送人合适吗？", ["送礼", "礼盒", "合适"], "multi_turn", "hard", ["我在看店里的护肤套装，准备送朋友生日礼物。"]),
        ],
    },
    {
        "key": "review",
        "name": "评价反馈",
        "intent": "review",
        "groups": [["评价", "评论", "反馈"], ["返现", "奖励", "使用感", "好评"]],
        "variants": [
            ("商品评价后有返现或奖励活动吗？", ["评价", "返现", "活动"], "standard", "easy", []),
            ("晒图好评给不给红包呀", ["晒图评价", "好评", "红包返现"], "colloquial_noise", "medium", []),
            ("这个反馈提交后有奖励吗？", ["商品评价", "反馈", "奖励"], "multi_turn", "hard", ["我已经写好了产品使用感，准备发布评价。"]),
        ],
    },
    {
        "key": "authenticity",
        "name": "正品防伪",
        "intent": "authenticity_shelf_life",
        "groups": [["正品", "防伪", "真假", "备案"], ["查询", "验证", "批次", "保证"]],
        "variants": [
            ("怎么查询商品是不是正品？", ["正品", "防伪", "查询"], "standard", "easy", []),
            ("怕买到假货，批次码到底去哪验", ["假货", "批次码", "验证"], "colloquial_noise", "medium", []),
            ("这个能查真伪吗？", ["商品", "真伪", "防伪查询"], "multi_turn", "hard", ["我收到商品后发现包装上的批次码不太清楚。"]),
        ],
    },
    {
        "key": "brand_policy",
        "name": "品牌售后政策",
        "intent": "after_sale",
        "groups": [["ColourPop"], ["退货", "换货", "replacement", "returns"], ["14天", "14 days", "政策", "官方"]],
        "variants": [
            ("ColourPop 的商品破损后可以换货吗？", ["ColourPop", "破损", "换货政策"], "standard", "medium", []),
            ("colourpop摔坏的货14天内给不给换", ["ColourPop", "14天", "破损换货"], "colloquial_noise", "hard", []),
            ("这个牌子的官方规则怎么说？", ["ColourPop", "破损", "退换货政策"], "multi_turn", "hard", ["我买的是 ColourPop，收到时商品在运输中破损了。"]),
        ],
    },
    {
        "key": "product_exact",
        "name": "英文商品精确查询",
        "intent": "product_info",
        "groups": [["Almay Smart Shade Butter Kiss Lipstick"], ["8.99", "lipstick", "Berry", "Nude"]],
        "variants": [
            ("What is the price of Almay Smart Shade Butter Kiss Lipstick?", ["Almay Smart Shade Butter Kiss Lipstick", "price"], "standard", "easy", []),
            ("almay butter kiss lipstick shades and price", ["Almay Butter Kiss Lipstick", "shades", "price"], "colloquial_noise", "medium", []),
            ("How much is it and what shades are available?", ["Almay Smart Shade Butter Kiss Lipstick", "price", "shades"], "multi_turn", "hard", ["I am looking at Almay Smart Shade Butter Kiss Lipstick."]),
        ],
    },
]


NO_ANSWER_CASES = [
    ("临床双盲实验具体招募了多少名受试者？", "efficacy", ["临床双盲", "受试者人数"], "clinical_sample_size"),
    ("这款精华的碳足迹精确是多少千克二氧化碳？", "product_info", ["碳足迹", "二氧化碳"], "carbon_footprint"),
    ("门店今天下午三点还有多少瓶实时库存？", "product_info", ["门店", "实时库存"], "realtime_stock"),
    ("这个批次每一项微生物实验的原始报告给我", "safety_allergy", ["批次", "微生物原始报告"], "lab_raw_report"),
    ("孕妇用了以后胎儿会不会有问题，请直接诊断", "safety_allergy", ["孕妇", "胎儿诊断"], "medical_diagnosis"),
    ("这款防晒的UVA-PF实验实测精确数值是多少？", "efficacy", ["UVA-PF", "实测数值"], "uva_pf"),
    ("品牌创始人下周私人行程是什么？", "product_info", ["创始人", "私人行程"], "private_schedule"),
    ("把尚未发布的下一季度新品配方完整发给我", "ingredient", ["未发布新品", "完整配方"], "unreleased_formula"),
]


def build_cases() -> list[dict]:
    cases: list[dict] = []
    sequence = 1
    for base in BASE_CASES:
        for variant_index, (question, keywords, query_type, difficulty, history) in enumerate(base["variants"]):
            cases.append({
                "query_id": f"retrieval_{sequence:04d}",
                "case_key": base["key"],
                "name": base["name"],
                "history": [{"role": "buyer", "content": item} for item in history],
                "question": question,
                "intent": base["intent"],
                "keywords": keywords,
                "difficulty": difficulty,
                "query_type": query_type,
                "has_answer": True,
                "required_groups": base["groups"],
                "split": "calibration" if variant_index < 2 else "test",
            })
            sequence += 1

    for index, (question, intent, keywords, key) in enumerate(NO_ANSWER_CASES):
        cases.append({
            "query_id": f"retrieval_{sequence:04d}",
            "case_key": key,
            "name": "无答案查询",
            "history": [],
            "question": question,
            "intent": intent,
            "keywords": keywords,
            "difficulty": "hard",
            "query_type": "no_answer",
            "has_answer": False,
            "required_groups": [],
            "split": "calibration" if index < len(NO_ANSWER_CASES) // 2 else "test",
        })
        sequence += 1
    return cases


def main() -> None:
    cases = build_cases()
    with OUTPUT.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"已生成 {len(cases)} 条检索基准 -> {OUTPUT}")


if __name__ == "__main__":
    main()
