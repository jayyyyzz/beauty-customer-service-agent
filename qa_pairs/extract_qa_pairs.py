# -*- coding: utf-8 -*-
"""
从方案二（qa_topic）的chunk中提取完整QA对

规则：
1. 只解析 core_text，不碰 overlap_text —— overlap是前一chunk复制来的上文，
   解析它会导致同一QA对被提取两次（core_text在全量chunk中天然无重复）
2. 一个QA对 = 连续买家消息块(问) + 紧随的连续商家消息块(答)
3. 清洗：仅按【明确寒暄词表】丢弃填充消息（"在吗/？？？/好的谢谢"等催复寒暄）
   和客服应答语（"在的在的，请讲~"），多条碎消息合并成一句。
   注意：不使用"长度<=4即过滤"的启发式，避免误伤"我是干皮"这类短的有效信息
4. 清洗后问或答为空的QA对丢弃（如纯寒暄轮）

输出 qa_pairs.jsonl：
  qa_id, session_id, chunk_id, topic, question, answer, question_raw, answer_raw
"""
import json
from qa_common import FILLER_SET

IN_PATH = "qa_topic_chunks.jsonl"
OUT_JSONL = "qa_pairs.jsonl"

# 客服侧的应答/寒暄语（无实质信息，is_filler只覆盖买家侧）
SELLER_FILLERS = {
    "亲亲在的哦~", "在的呢，您说~", "您好，客服小美很高兴为您服务~", "在的在的，请讲~",
    "不客气哦~祝您生活愉快❤", "好的亲~后续有任何问题随时联系在线客服哦",
    "感谢您的咨询，祝您生活愉快❤", "不客气~有其他问题随时找我哦❤",
    "不客气哦~祝您使用愉快，有问题随时来找客服小美❤", "么么哒~期待您的下次光临❤",
}

# 买家侧明确寒暄词表（在qa_common.FILLER_SET基础上补充生成器里的短确认语）
BUYER_FILLERS = FILLER_SET | {
    "好嘞", "好哒", "嗯嗯", "谢谢", "谢啦", "好的呢", "收到", "嗯，没别的问题了",
}

def clean(parts, seller=False):
    """仅按明确词表过滤填充消息，合并碎消息（不按长度启发式过滤）"""
    fillers = SELLER_FILLERS if seller else BUYER_FILLERS
    kept = [p for p in parts if p.strip() not in fillers]
    return " ".join(kept)

def iter_qa_rounds(core_text):
    """按行解析core_text，切出 买家块+商家块 = 1个QA对"""
    buyer, seller = [], []
    for line in core_text.split("\n"):
        if line.startswith("买家: "):
            if seller:                     # 新一轮买家发问 -> 结算上一轮
                yield buyer, seller
                buyer, seller = [], []
            buyer.append(line[len("买家: "):])
        elif line.startswith("商家: "):
            seller.append(line[len("商家: "):])
    if buyer or seller:
        yield buyer, seller

def main():
    chunks = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    pairs, dropped = [], 0
    for c in chunks:
        for buyer, seller in iter_qa_rounds(c["core_text"]):
            q, a = clean(buyer), clean(seller, seller=True)
            if not q or not a:             # 清洗后问或答为空 -> 丢弃
                dropped += 1
                continue
            pairs.append({
                "qa_id": f"qa_{len(pairs):05d}",
                "session_id": c["session_id"],
                "chunk_id": c["chunk_id"],
                "topic": c["main_topic"],
                "question": q,
                "answer": a,
                "question_raw": " | ".join(buyer),
                "answer_raw": " | ".join(seller),
            })
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"提取QA对 {len(pairs)} 个（丢弃纯寒暄/空轮 {dropped} 个）-> {OUT_JSONL}")
    from collections import Counter
    print("主题分布:", Counter(p["topic"] for p in pairs).most_common())

if __name__ == "__main__":
    main()
