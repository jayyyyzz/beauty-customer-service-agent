# -*- coding: utf-8 -*-
"""
方案一 qa_pair：最小粒度切分
- 1 chunk = 1 个完整QA轮（连续买家消息 + 商家回答）
- overlap = 前一个QA轮全文（作为【上文】前缀，提供跨轮指代的上下文）
特点：粒度最细、检索定位最准，但chunk数量最多、冗余率最高
"""
import qa_common as q

def split(all_rounds):
    chunks = []
    for sid, rounds in sorted(all_rounds.items()):
        for i, rd in enumerate(rounds):
            overlap_text = rounds[i - 1]["text"] if i > 0 else ""
            core_text = rd["text"]
            text = (f"【上文】\n{overlap_text}\n【正文】\n{core_text}") if overlap_text else core_text
            chunks.append({
                "chunk_id": f"qa_pair_{len(chunks):05d}",
                "scheme": "qa_pair",
                "session_id": sid,
                "core_rounds": [q.round_meta(rd)],
                "topics": [rd["topic"]],
                "core_text": core_text,
                "overlap_text": overlap_text,
                "text": text,
                "char_len": len(text),
            })
    return chunks

if __name__ == "__main__":
    rounds = q.build_rounds(q.load_messages())
    chunks = split(rounds)
    q.save_chunks(chunks, "qa_pair_chunks.jsonl")
