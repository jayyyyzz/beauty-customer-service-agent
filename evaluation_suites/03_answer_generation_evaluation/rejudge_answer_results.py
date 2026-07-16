# -*- coding: utf-8 -*-
"""使用当前统一Judge重新评估已保存的回答，不重复执行回答生成。"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from run_answer_evaluation import (
    close_agent_resources,
    derive_claim_metrics,
    judge_answer,
)


TEST_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


async def rejudge_one(row: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        started = time.perf_counter()
        judge, usage, error = await judge_answer(
            row,
            row.get("answer") or "",
            row.get("knowledge_docs") or [],
            row.get("api_data"),
        )
        row["judge"] = judge
        row["judge_usage"] = usage
        row["judge_error"] = error
        row["judge_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        row["claim_metrics"] = derive_claim_metrics(judge.get("claims") or [])
        return row


async def async_main(args: argparse.Namespace) -> None:
    details_path = args.output_dir / "answer_eval_predictions.jsonl"
    rows = load_jsonl(details_path)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(rejudge_one(row, semaphore)) for row in rows]
    results = []
    for index, task in enumerate(asyncio.as_completed(tasks), start=1):
        row = await task
        results.append(row)
        print(
            f"[{index:02d}/{len(tasks):02d}] {row['case_id']} "
            f"judge_error={bool(row.get('judge_error'))}",
            flush=True,
        )
    results.sort(key=lambda row: row["case_id"])
    with details_path.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    await close_agent_resources()

    subprocess.run(
        [
            sys.executable,
            str(TEST_DIR / "rebuild_answer_report.py"),
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
        cwd=TEST_DIR.parents[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
