# -*- coding: utf-8 -*-
"""
chunk质量测评：对三个切分方案打分并输出对比报告

═══════════════ 质量评分公式 ═══════════════

单chunk质量分（0~100）：
    Q_chunk = 100 × (0.30×C + 0.25×P + 0.20×L + 0.15×D + 0.10×(1-R))

  C 问答完整性 = 含"买家消息且商家消息"的QA轮数 / 总QA轮数      （硬约束，权重最高）
  P 主题纯净度 = 众数主题QA轮数 / 总QA轮数                       （检索时避免语义稀释）
  L 长度合规度 = 1                    if 150 ≤ len ≤ 800
              = len/150              if len < 150   （过短：语义不足）
              = max(0, 1-(len-800)/800) if len > 800 （过长：embedding稀释）
  D 信息密度   = 非填充消息数 / 总消息数    （"在吗/？？？/好的"等为填充）
  R 冗余率     = overlap字符数 / chunk总字符数（overlap有价值但要惩罚过度冗余）

方案总分：
    Q_scheme = mean(Q_chunk) − Penalty
    Penalty  = 10×破损QA比例（问答被拆开的轮占比） + 5×超长chunk占比(>1200字)

等级：A ≥ 85 > B ≥ 75 > C ≥ 65 > D
═════════════════════════════════════════════
"""
import json
import statistics as st
from collections import Counter

W_C, W_P, W_L, W_D, W_R = 0.30, 0.25, 0.20, 0.15, 0.10
LEN_LO, LEN_HI, LEN_MAX = 150, 800, 1200

def score_chunk(c):
    rounds = c["core_rounds"]
    n = len(rounds)
    # C 问答完整性
    complete = sum(1 for r in rounds if r["n_buyer"] > 0 and r["n_seller"] > 0)
    C = complete / n if n else 0
    # P 主题纯净度（"其他/寒暄"轮不算破坏纯净度，视为附属内容）
    topics = [r["topic"] for r in rounds if r["topic"] not in ("其他", "寒暄闲聊")] or [rounds[0]["topic"]]
    P = Counter(topics).most_common(1)[0][1] / len(topics)
    # L 长度合规度
    ln = c["char_len"]
    if ln < LEN_LO:
        L = ln / LEN_LO
    elif ln <= LEN_HI:
        L = 1.0
    else:
        L = max(0.0, 1 - (ln - LEN_HI) / LEN_HI)
    # D 信息密度
    n_msgs = sum(r["n_msgs"] for r in rounds)
    n_filler = sum(r["n_filler"] for r in rounds)
    D = (n_msgs - n_filler) / n_msgs if n_msgs else 0
    # R 冗余率
    R = len(c.get("overlap_text", "")) / max(1, len(c["text"]))
    q = 100 * (W_C * C + W_P * P + W_L * L + W_D * D + W_R * (1 - R))
    return q, dict(C=C, P=P, L=L, D=D, R=R)

def grade(x):
    return "A" if x >= 85 else "B" if x >= 75 else "C" if x >= 65 else "D"

def evaluate(path, name):
    chunks = [json.loads(l) for l in open(path, encoding="utf-8")]
    per, subs = [], []
    for c in chunks:
        q, s = score_chunk(c)
        per.append(q); subs.append(s)
    n = len(chunks)
    broken = sum(1 for c in chunks for r in c["core_rounds"] if not (r["n_buyer"] and r["n_seller"]))
    total_rounds = sum(len(c["core_rounds"]) for c in chunks)
    overlong = sum(1 for c in chunks if c["char_len"] > LEN_MAX)
    penalty = 10 * (broken / total_rounds) + 5 * (overlong / n)
    final = st.mean(per) - penalty
    lens = [c["char_len"] for c in chunks]
    return {
        "方案": name, "chunk数": n,
        "平均分": round(st.mean(per), 2), "总分(扣罚后)": round(final, 2), "等级": grade(final),
        "问答完整性C": round(st.mean(s["C"] for s in subs), 4),
        "主题纯净度P": round(st.mean(s["P"] for s in subs), 4),
        "长度合规度L": round(st.mean(s["L"] for s in subs), 4),
        "信息密度D": round(st.mean(s["D"] for s in subs), 4),
        "冗余率R": round(st.mean(s["R"] for s in subs), 4),
        "破损QA轮占比": round(broken / total_rounds, 4),
        "超长chunk占比": round(overlong / n, 4),
        "长度min/中位/max": f"{min(lens)}/{int(st.median(lens))}/{max(lens)}",
        "低分chunk数(<65)": sum(1 for q in per if q < 65),
    }

SCHEMES = [
    ("qa_pair_chunks.jsonl", "方案一 qa_pair（单QA轮+前轮overlap）"),
    ("qa_topic_chunks.jsonl", "方案二 qa_topic（同主题聚合+跨块overlap）"),
    ("qa_window_chunks.jsonl", "方案三 qa_window（滑窗3轮/步长2）"),
]

if __name__ == "__main__":
    results = [evaluate(p, n) for p, n in SCHEMES]
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # markdown 报告
    keys = list(results[0].keys())[1:]
    lines = ["# Chunk 切分方案测评报告", "",
             "数据源：beauty_skincare_ecommerce_dialogue_10000.csv（10000条消息 / 1809会话 / 3669个QA轮）", "",
             "## 评分公式", "",
             "```",
             "Q_chunk  = 100 × (0.30×C + 0.25×P + 0.20×L + 0.15×D + 0.10×(1−R))",
             "  C 问答完整性 = 完整QA轮数/总QA轮数（买家问题与商家回答同chunk）",
             "  P 主题纯净度 = 众数主题轮数/总轮数",
             "  L 长度合规度 = 1 if 150≤len≤800; len/150 if 过短; 1−(len−800)/800 if 过长",
             "  D 信息密度   = 非填充消息数/总消息数（『在吗/？？？/好的』为填充）",
             "  R 冗余率     = overlap字符数/chunk总字符数",
             "Q_scheme = mean(Q_chunk) − 10×破损QA轮占比 − 5×超长chunk占比(>1200字)",
             "等级: A≥85 > B≥75 > C≥65 > D",
             "```", "",
             "## 测评结果", "",
             "| 指标 | " + " | ".join(r["方案"] for r in results) + " |",
             "|---|" + "---|" * len(results)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(r[k]) for r in results) + " |")
    best = max(results, key=lambda r: r["总分(扣罚后)"])
    lines += ["", "## 结论", "",
              f"**推荐方案：{best['方案']}**（总分 {best['总分(扣罚后)']}，等级 {best['等级']}）", "",
              "- 方案一 qa_pair：粒度最细、定位最准，适合FAQ型精准问答，但chunk数量多、overlap冗余占比最高。",
              "- 方案二 qa_topic：以『单主题问答片段』为单位，语义完整且主题纯净，embedding质量最好，是RAG知识库的推荐方案。",
              "- 方案三 qa_window：上下文连续性最好，适合需要多轮上下文的对话理解任务，但窗口跨主题导致纯净度下降。"]
    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    for r in results:
        print(f"{r['方案']}: chunk数={r['chunk数']} 平均分={r['平均分']} 总分={r['总分(扣罚后)']} 等级={r['等级']}")
    print("报告 -> evaluation_report.md / evaluation_results.json")
