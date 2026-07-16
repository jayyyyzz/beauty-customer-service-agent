# -*- coding: utf-8 -*-
"""在现有 65 条基准上扩展同义改写、易混淆长问法和无答案样本。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[1]
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.build_retrieval_benchmark import BASE_CASES, NO_ANSWER_CASES


EXTRA_VARIANTS = {
    "allergy_return": ("用完爆红刺痛，开封商品怎么走退款售后？", "我不是问普通七天无理由，东西已经拆封而且脸部过敏，这种情况怎么退？"),
    "logistics_delay": ("快递轨迹几天不动，异常件应该怎么处理？", "不是问仓库什么时候发货，是已经发出后物流一直停在原地怎么办？"),
    "invoice": ("电子发票在哪里填企业名称和纳税人识别号？", "我不是咨询退款，订单报销需要公司抬头发票，应该从哪里申请？"),
    "skin_type": ("出油多又容易长痘，这种肤质能用这款面霜吗？", "虽然脸颊有点干，但T区很油，我主要想确认会不会闷痘。"),
    "ingredient": ("烟酰胺具体添加比例和耐受建议是什么？", "我不是问产品功效，敏感肌只想知道烟酰安浓度高不高。"),
    "compatibility": ("A醇和酸类需要分开使用吗？", "不是问单个成分作用，我晚上已经用视黄醇，还能叠加果酸吗？"),
    "quality_issue": ("商品运输破裂并漏液，可以补发还是退款？", "不是普通退货，刚签收瓶身就碎了并且漏液，需要按质量问题处理。"),
    "dispatch": ("现货付款后一般几个小时或几天出库？", "物流还没有产生，我问的是下单后多久发出，不是运输需要几天。"),
    "shelf_life": ("开封后的护肤品建议在几个月内用完？", "不是问未开封有效期，我已经打开瓶子，想知道还能保存多久。"),
    "shade": ("暖黄肤色选哪种口红更显白？", "我不是问粉底色号，想选一支黄黑皮不显荧光的口红。"),
    "promotion": ("店铺券和满减活动能同时使用吗？", "不是问商品原价，我想知道优惠券能不能叠加以及怎么凑满减。"),
    "usage_order": ("水、精华、乳液、面霜和防晒应该按什么顺序涂？", "我不是问每次用多少，想确认精华是在乳液前还是面霜后。"),
    "dosage": ("精华一次用几滴才不会过量？", "不是问早晚频率，我是想知道单次挤多少，最近总是搓泥。"),
    "efficacy": ("干燥起皮和泛红时，这款产品主要能提供什么修护？", "不是问美白效果，我主要关注补水、舒缓和屏障修护。"),
    "gift": ("送长辈的护肤礼盒有什么稳妥推荐？", "不是给自己用，准备送朋友生日礼物，想要包装体面并且不容易踩雷。"),
    "review": ("发布商品评价或晒图后有没有奖励？", "不是问商品好不好用，我想确认提交使用反馈有没有返现活动。"),
    "authenticity": ("批次码和防伪信息应该在哪里核验？", "不是问保质期，我担心是假货，想查包装上的批次码。"),
    "brand_policy": ("ColourPop 运输破损的官方退换规则是什么？", "不是店铺通用规则，我问的是 ColourPop 收货破损后的官方处理期限。"),
    "product_exact": ("How much does the Almay Smart Shade Butter Kiss Lipstick cost and which colors are offered?", "I am not asking about shipping; I need the price and available shades for Almay Smart Shade Butter Kiss Lipstick."),
}

EXTRA_NO_ANSWER = [
    ("能告诉我仓库此刻准确还剩多少件吗？", "product_info", ["仓库", "实时库存"], "warehouse_live_stock"),
    ("把防晒人体功效试验的逐人原始数据发给我", "efficacy", ["人体功效试验", "逐人原始数据"], "efficacy_raw_records"),
    ("下季度还没发布的新品完整成分百分比是多少？", "ingredient", ["未发布新品", "成分百分比"], "future_formula"),
    ("你直接诊断一下我是不是激素脸", "safety_allergy", ["直接诊断", "激素脸"], "medical_diagnosis_skin"),
    ("这款面霜完整碳排放核算报告在哪里？", "product_info", ["碳排放", "核算报告"], "carbon_report"),
    ("品牌创始人的家庭住址可以告诉我吗？", "product_info", ["创始人", "家庭住址"], "private_address"),
    ("给我这个批次重金属检测的全部原始数据", "safety_allergy", ["批次", "重金属原始数据"], "heavy_metal_raw"),
    ("这款新品尚未公开的内部配方能发一下吗？", "ingredient", ["未公开", "内部配方"], "confidential_formula"),
]


def build_cases() -> list[dict]:
    cases: list[dict] = []
    sequence = 1
    for base in BASE_CASES:
        variants = list(base["variants"])
        synonym, confusing = EXTRA_VARIANTS[base["key"]]
        variants.extend([
            (synonym, base["variants"][0][1], "synonym", "medium", []),
            (confusing, base["variants"][0][1], "confusing_long", "hard", []),
        ])
        for variant_index, (question, keywords, query_type, difficulty, history) in enumerate(variants):
            cases.append({
                "query_id": f"retrieval_v2_{sequence:04d}",
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
                "split": "calibration" if query_type in {"standard", "synonym"} else "test",
            })
            sequence += 1

    no_answer = list(NO_ANSWER_CASES) + EXTRA_NO_ANSWER
    for index, (question, intent, keywords, key) in enumerate(no_answer):
        cases.append({
            "query_id": f"retrieval_v2_{sequence:04d}",
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
            "split": "calibration" if index % 2 == 0 else "test",
        })
        sequence += 1
    return cases


def main() -> None:
    output = PACKAGE_ROOT / "benchmark" / "retrieval_benchmark_v2.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    with output.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"已生成 {len(cases)} 条检索基准 -> {output}")


if __name__ == "__main__":
    main()
