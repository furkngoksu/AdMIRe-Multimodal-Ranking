from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = REPO_ROOT / "stage1_idiom_signal/data/unique_mwe_target_label_language.csv"
DEFAULT_LLM_RESULTS_DIR = REPO_ROOT / "llm_results"

MODEL_MODULES = [
    "old_marked_input",
    "new_target_only",
    "hybrid_marked_mlp",
    "neural_m",
    "hybrid_llm_fusion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all five Stage 1 idiom-signal model train scripts.")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--llm_results_dir", default=str(DEFAULT_LLM_RESULTS_DIR))
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--final_epochs", type=int, default=None)
    parser.add_argument("--tune_epochs", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--neural_batch_size", type=int, default=64)
    parser.add_argument("--fusion_batch_size", type=int, default=64)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--neural_weight_decay", type=float, default=1e-4)
    parser.add_argument("--fusion_weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--limit", type=int, default=512, help="Use 0 for the full CSV/all matched LLM records.")
    parser.add_argument("--skip_neural_m", action="store_true")
    parser.add_argument("--skip_hybrid_llm_fusion", action="store_true")
    parser.add_argument("--skip_tuning", action="store_true")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    return parser.parse_args()


def model_output_dir(model_name: str) -> Path:
    return MODEL_ROOT / model_name / "outputs"


def base_command(args: argparse.Namespace, model_name: str) -> list[str]:
    final_epochs = args.final_epochs if args.final_epochs is not None else args.epochs
    command = [
        sys.executable,
        "-m",
        f"stage1_idiom_signal.models.{model_name}.train",
        "--csv_path",
        args.csv_path,
        "--output_dir",
        str(model_output_dir(model_name)),
        "--model_name",
        args.model_name,
        "--epochs",
        str(final_epochs),
        "--tune_epochs",
        str(args.tune_epochs),
        "--n_trials",
        str(args.n_trials),
        "--lr",
        str(args.lr),
        "--max_length",
        str(args.max_length),
        "--val_ratio",
        str(args.val_ratio),
        "--limit",
        str(args.limit),
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.skip_tuning:
        command.append("--skip_tuning")
    if args.unfreeze_encoder and model_name in {"old_marked_input", "new_target_only", "hybrid_marked_mlp"}:
        command.append("--unfreeze_encoder")
    return command


def command_for_model(args: argparse.Namespace, model_name: str) -> list[str]:
    command = base_command(args, model_name)

    if model_name in {"old_marked_input", "new_target_only", "hybrid_marked_mlp"}:
        command.extend(["--batch_size", str(args.batch_size), "--weight_decay", str(args.weight_decay)])

    if model_name == "neural_m":
        command.extend(
            [
                "--llm_results_dir",
                args.llm_results_dir,
                "--batch_size",
                str(args.neural_batch_size),
                "--encode_batch_size",
                str(args.encode_batch_size),
                "--weight_decay",
                str(args.neural_weight_decay),
            ]
        )

    if model_name == "hybrid_llm_fusion":
        command.extend(
            [
                "--llm_results_dir",
                args.llm_results_dir,
                "--batch_size",
                str(args.fusion_batch_size),
                "--encode_batch_size",
                str(args.encode_batch_size),
                "--weight_decay",
                str(args.fusion_weight_decay),
            ]
        )

    return command


def load_summary(model_name: str) -> dict[str, Any] | None:
    path = model_output_dir(model_name) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_metric_table(results: list[dict[str, Any]]) -> None:
    print("")
    print("model\tbest_seed\tbest_f1\tmean_f1\tstd_f1\toutput_dir")
    for result in results:
        print(
            f"{result['name']}\t"
            f"{result.get('best_seed')}\t"
            f"{float(result.get('best_val_f1', 0.0)):.4f}\t"
            f"{float(result.get('mean_val_f1', 0.0)):.4f}\t"
            f"{float(result.get('std_val_f1', 0.0)):.4f}\t"
            f"{model_output_dir(result['name'])}"
        )


def main() -> None:
    args = parse_args()
    selected_models = []
    for model_name in MODEL_MODULES:
        if model_name == "neural_m" and args.skip_neural_m:
            continue
        if model_name == "hybrid_llm_fusion" and args.skip_hybrid_llm_fusion:
            continue
        selected_models.append(model_name)

    for model_name in selected_models:
        command = command_for_model(args, model_name)
        print("")
        print(f"Running {model_name}")
        print(" ".join(command))
        subprocess.run(command, check=True)

    results = [result for model_name in selected_models if (result := load_summary(model_name)) is not None]
    print_metric_table(results)


if __name__ == "__main__":
    main()
