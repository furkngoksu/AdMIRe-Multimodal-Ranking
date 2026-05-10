from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stage1_idiom_signal.models.hybrid_llm_fusion.run_model import (
    DEFAULT_CACHE_PATH as DEFAULT_FUSION_CACHE_PATH,
    build_or_load_fusion_cache,
    train_hybrid_llm_fusion,
)
from stage1_idiom_signal.models.hybrid_marked_mlp.run_model import train_hybrid_marked_mlp
from stage1_idiom_signal.models.neural_m.run_model import (
    DEFAULT_CACHE_PATH as DEFAULT_NEURAL_CACHE_PATH,
    DEFAULT_LLM_RESULTS_DIR,
    build_or_load_cache,
    train_neural_m,
)
from stage1_idiom_signal.models.new_target_only.run_model import train_new_target_only
from stage1_idiom_signal.models.old_marked_input.run_model import train_old_marked_input


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = REPO_ROOT / "stage1_idiom_signal/data/unique_mwe_target_label_language.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all five Stage 1 idiom-signal models.")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--llm_results_dir", default=str(DEFAULT_LLM_RESULTS_DIR))
    parser.add_argument("--neural_cache_path", default=str(DEFAULT_NEURAL_CACHE_PATH))
    parser.add_argument("--fusion_cache_path", default=str(DEFAULT_FUSION_CACHE_PATH))
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--neural_batch_size", type=int, default=64)
    parser.add_argument("--fusion_batch_size", type=int, default=64)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--neural_weight_decay", type=float, default=1e-4)
    parser.add_argument("--fusion_weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=512, help="Use 0 for the full CSV/all matched LLM records.")
    parser.add_argument("--skip_neural_m", action="store_true")
    parser.add_argument("--skip_hybrid_llm_fusion", action="store_true")
    parser.add_argument("--force_rebuild_neural_cache", action="store_true")
    parser.add_argument("--force_rebuild_fusion_cache", action="store_true")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    return parser.parse_args()


def model_output_dir(model_name: str) -> Path:
    return MODEL_ROOT / model_name / "outputs"


def print_metric_table(results: list[dict[str, Any]]) -> None:
    print("")
    print("model\tbest_epoch\tval_f1\tval_acc\tval_auc\toutput_dir")
    for result in results:
        metrics = result.get("best_metrics") or {}
        auc = metrics.get("roc_auc")
        auc_text = "None" if auc is None else f"{float(auc):.4f}"
        output_dir = model_output_dir(result["name"])
        print(
            f"{result['name']}\t"
            f"{result['best_epoch']}\t"
            f"{float(result['best_val_f1']):.4f}\t"
            f"{float(metrics.get('accuracy', 0.0)):.4f}\t"
            f"{auc_text}\t"
            f"{output_dir}"
        )


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []

    results.append(
        train_old_marked_input(
            csv_path=args.csv_path,
            output_dir=model_output_dir("old_marked_input"),
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
            unfreeze_encoder=args.unfreeze_encoder,
        )
    )

    results.append(
        train_new_target_only(
            csv_path=args.csv_path,
            output_dir=model_output_dir("new_target_only"),
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
            hidden_1=512,
            hidden_2=128,
            dropout_1=0.2,
            dropout_2=0.2,
            unfreeze_encoder=args.unfreeze_encoder,
        )
    )

    results.append(
        train_hybrid_marked_mlp(
            csv_path=args.csv_path,
            output_dir=model_output_dir("hybrid_marked_mlp"),
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
            hidden_1=512,
            hidden_2=128,
            dropout_1=0.2,
            dropout_2=0.2,
            unfreeze_encoder=args.unfreeze_encoder,
        )
    )

    if not args.skip_neural_m:
        neural_cache = build_or_load_cache(
            cache_path=Path(args.neural_cache_path),
            csv_path=Path(args.csv_path),
            llm_results_dir=Path(args.llm_results_dir),
            model_name=args.model_name,
            max_length=args.max_length,
            encode_batch_size=args.encode_batch_size,
            limit=args.limit,
            seed=args.seed,
            force_rebuild=args.force_rebuild_neural_cache,
        )
        results.append(
            train_neural_m(
                cache=neural_cache,
                output_dir=model_output_dir("neural_m"),
                epochs=args.epochs,
                batch_size=args.neural_batch_size,
                lr=args.lr,
                weight_decay=args.neural_weight_decay,
                val_ratio=args.val_ratio,
                seed=args.seed,
                hidden_1=512,
                hidden_2=128,
                dropout_1=0.2,
                dropout_2=0.2,
            )
        )

    if not args.skip_hybrid_llm_fusion:
        fusion_cache = build_or_load_fusion_cache(
            cache_path=Path(args.fusion_cache_path),
            csv_path=Path(args.csv_path),
            llm_results_dir=Path(args.llm_results_dir),
            model_name=args.model_name,
            max_length=args.max_length,
            encode_batch_size=args.encode_batch_size,
            limit=args.limit,
            seed=args.seed,
            force_rebuild=args.force_rebuild_fusion_cache,
        )
        results.append(
            train_hybrid_llm_fusion(
                cache=fusion_cache,
                output_dir=model_output_dir("hybrid_llm_fusion"),
                epochs=args.epochs,
                batch_size=args.fusion_batch_size,
                lr=args.lr,
                weight_decay=args.fusion_weight_decay,
                val_ratio=args.val_ratio,
                seed=args.seed,
                hidden_1=1024,
                hidden_2=256,
                dropout_1=0.3,
                dropout_2=0.3,
            )
        )

    print_metric_table(results)


if __name__ == "__main__":
    main()
