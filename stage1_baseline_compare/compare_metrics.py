from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage1_baseline_compare.shared import print_metric_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved quick baseline metrics.")
    parser.add_argument("--output_root", default="stage1_baseline_compare/outputs")
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    results = [
        load_metrics(root / "old_marked_input" / "metrics.json"),
        load_metrics(root / "new_target_only" / "metrics.json"),
    ]
    print_metric_table(results)


if __name__ == "__main__":
    main()

