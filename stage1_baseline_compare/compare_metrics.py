from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage1_baseline_compare.shared import print_metric_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved quick baseline metrics.")
    parser.add_argument("--output_root", default="stage1_baseline_compare/outputs")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["old_marked_input", "new_target_only", "hybrid_marked_mlp", "neural_m"],
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    results = []
    for model_name in args.models:
        metrics_path = root / model_name / "metrics.json"
        if metrics_path.exists():
            results.append(load_metrics(metrics_path))
        else:
            print(f"Skipping missing metrics: {metrics_path}")

    if not results:
        raise FileNotFoundError(f"No metrics files found under {root}")

    print_metric_table(results)


if __name__ == "__main__":
    main()
