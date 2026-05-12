from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent

ALL_MODELS = [
    "baseline_mlp",
    "lang_marked_mlp",
    "lang_marked_perturbation_mlp",
    "perturbation_mlp",
    "target_perturbation_mlp",
]


def main() -> None:
    requested = sys.argv[1:]
    if not requested:
        models = ALL_MODELS
    else:
        invalid = [m for m in requested if m not in ALL_MODELS]
        if invalid:
            print(f"Unknown model(s): {invalid}")
            print(f"Available: {ALL_MODELS}")
            sys.exit(1)
        models = requested

    print(f"Models to run: {models}")

    for model_name in models:
        train_path = MODEL_ROOT / model_name / "train.py"
        if not train_path.exists():
            print(f"Skipping {model_name}: train.py not found")
            continue

        print(f"\n{'='*60}")
        print(f"Running {model_name}")
        print(f"{'='*60}")
        subprocess.run([sys.executable, str(train_path)], check=True)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'model':<25} {'mean_f1':>10} {'best_seed':>10} {'best_f1':>10}")
    print("-" * 58)
    for model_name in models:
        metric_files = sorted((MODEL_ROOT / model_name).glob("outputs_*/metrics.json"), key=lambda p: p.stat().st_mtime)
        if not metric_files:
            print(f"{model_name:<25} {'—':>10} {'—':>10} {'—':>10}")
            continue
        metrics_path = metric_files[-1]
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        mean_f1 = data.get("mean_val_f1", 0)
        best_seed = data.get("best_seed", "—")
        best_f1 = data.get("best_val_f1", 0)
        print(f"{model_name:<25} {mean_f1:>10.4f} {best_seed:>10} {best_f1:>10.4f}")


if __name__ == "__main__":
    main()
