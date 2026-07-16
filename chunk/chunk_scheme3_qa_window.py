# -*- coding: utf-8 -*-
"""
方案三 qa_window：QA轮滑动窗口切分
- 窗口 = 3 个QA轮，步长 = 2（相邻chunk天然共享1个QA轮，即overlap=1轮）
- 会话不足3轮则整会话为1个chunk
特点：chunk大小均匀、上下文连续性最好，但窗口可能横跨不同主题
"""
import qa_common as q

WINDOW = 3  # 每个chunk包含的QA轮数
STEP = 2    # 步长（WINDOW - STEP = overlap轮数）

def split(all_rounds):
    chunks = []
    for sid, rounds in sorted(all_rounds.items()):
        n = len(rounds)
        starts = [0] if n <= WINDOW else list(range(0, n - WINDOW + 1, STEP))
        # 确保尾部覆盖：最后一个窗口必须包含最后一轮
        if n > WINDOW and starts[-1] + WINDOW < n:
            starts.append(n - WINDOW)
        for wi, s in enumerate(starts):
            grp = rounds[s:s + WINDOW]
            # 与前一窗口重叠的轮 = overlap，其余 = core
            ov_n = 0 if wi == 0 else max(0, (starts[wi - 1] + WINDOW) - s)
            ov_rounds, core_rounds = grp[:ov_n], grp[ov_n:]
            overlap_text = "\n".join(rd["text"] for rd in ov_rounds)
            core_text = "\n".join(rd["text"] for rd in core_rounds)
            text = "\n".join(rd["text"] for rd in grp)
            chunks.append({
                "chunk_id": f"qa_window_{len(chunks):05d}",
                "scheme": "qa_window",
                "session_id": sid,
                "core_rounds": [q.round_meta(rd) for rd in grp],  # 窗口内全部轮都参与语义
                "topics": [rd["topic"] for rd in grp],
                "core_text": core_text,
                "overlap_text": overlap_text,
                "text": text,
                "char_len": len(text),
            })
    return chunks

if __name__ == "__main__":
    rounds = q.build_rounds(q.load_messages())
    chunks = split(rounds)
    q.save_chunks(chunks, "qa_window_chunks.jsonl")
