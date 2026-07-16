# -*- coding: utf-8 -*-
"""
方案二 qa_topic：会话内主题聚合切分（RAG推荐候选）
- 同一会话内，连续且主题相同的QA轮聚合为1个chunk（"其他"轮粘滞并入前一主题块）
- overlap = 前一个主题块的最后1个QA轮（作为【上文】前缀，衔接话题切换处的指代）
特点：chunk = 完整的"单主题问答片段"，主题纯净、语义完整，冗余适中
"""
import qa_common as q

MAX_ROUNDS_PER_CHUNK = 5  # 防止超长主题块（如反复拉扯的售后）撑爆chunk

def split(all_rounds):
    chunks = []
    for sid, rounds in sorted(all_rounds.items()):
        # 1) 按连续同主题分组，"其他/寒暄"并入前一组
        groups = []
        for rd in rounds:
            if groups and (
                rd["topic"] == groups[-1][-1]["topic"]
                or rd["topic"] in ("其他", "寒暄闲聊")
            ) and len(groups[-1]) < MAX_ROUNDS_PER_CHUNK:
                groups[-1].append(rd)
            else:
                groups.append([rd])
        # 2) 组 -> chunk，跨组做1轮overlap
        for gi, grp in enumerate(groups):
            overlap_text = groups[gi - 1][-1]["text"] if gi > 0 else ""
            core_text = "\n".join(rd["text"] for rd in grp)
            text = (f"【上文】\n{overlap_text}\n【正文】\n{core_text}") if overlap_text else core_text
            main_topic = grp[0]["topic"]
            chunks.append({
                "chunk_id": f"qa_topic_{len(chunks):05d}",
                "scheme": "qa_topic",
                "session_id": sid,
                "core_rounds": [q.round_meta(rd) for rd in grp],
                "topics": [rd["topic"] for rd in grp],
                "main_topic": main_topic,
                "core_text": core_text,
                "overlap_text": overlap_text,
                "text": text,
                "char_len": len(text),
            })
    return chunks

if __name__ == "__main__":
    rounds = q.build_rounds(q.load_messages())
    chunks = split(rounds)
    q.save_chunks(chunks, "qa_topic_chunks.jsonl")
