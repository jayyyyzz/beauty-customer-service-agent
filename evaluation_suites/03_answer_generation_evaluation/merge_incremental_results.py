# -*- coding: utf-8 -*-
"""将增量复测样本合并回完整逐条结果，并重建报告。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    target_path = args.target_dir / "answer_eval_predictions.jsonl"
    rows = {row["case_id"]: row for row in load_jsonl(target_path)}
    replaced = []
    for source_dir in args.source_dir:
        source_path = source_dir / "answer_eval_predictions.jsonl"
        for row in load_jsonl(source_path):
            rows[row["case_id"]] = row
            replaced.append(row["case_id"])

    ordered = [rows[case_id] for case_id in sorted(rows)]
    with target_path.open("w", encoding="utf-8") as file:
        for row in ordered:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print("已替换样本:", ", ".join(sorted(set(replaced))))

    subprocess.run(
        [
            sys.executable,
            str(TEST_DIR / "rebuild_answer_report.py"),
            "--output-dir",
            str(args.target_dir),
        ],
        check=True,
        cwd=TEST_DIR.parents[1],
    )


if __name__ == "__main__":
    main()
