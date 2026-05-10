from __future__ import annotations

import argparse
from pathlib import Path

from stage1_idiom_signal.models.hybrid_llm_fusion.train import (
    DEFAULT_CACHE_PATH as DEFAULT_FUSION_CACHE_PATH,
    build_or_load_fusion_cache,
    train_hybrid_llm_fusion,
)
from stage1_idiom_signal.models.neural_m.train import (
    DEFAULT_CACHE_PATH,
    DEFAULT_LLM_RESULTS_DIR,
    build_or_load_cache,
    train_neural_m,
)
from stage1_idiom_signal.common import (
    DEFAULT_CSV_PATH,
    build_new_target_only_input,
    build_old_marked_input,
    print_metric_table,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all quick Stage 1 comparison models.")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--llm_results_dir", default=str(DEFAULT_LLM_RESULTS_DIR))
    parser.add_argument("--neural_cache_path", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--fusion_cache_path", default=str(DEFAULT_FUSION_CACHE_PATH))
    parser.add_argument("--output_root", default="stage1_idiom_signal/outputs")
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


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    results = []

    results.append(
        run_training(
            name="old_marked_input",
            csv_path=args.csv_path,
            output_dir=output_root / "old_marked_input",
            text_builder=build_old_marked_input,
            input_format="[LANG] + sentence with <mwe> tags + [MWE]",
            head_type="old_simple",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
            model_name=args.model_name,
            unfreeze_encoder=args.unfreeze_encoder,
        )
    )

    results.append(
        run_training(
            name="new_target_only",
            csv_path=args.csv_path,
            output_dir=output_root / "new_target_only",
            text_builder=build_new_target_only_input,
            input_format="raw Target sentence only",
            head_type="new_mlp",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
            model_name=args.model_name,
            unfreeze_encoder=args.unfreeze_encoder,
            hidden_1=512,
            hidden_2=128,
            dropout_1=0.2,
            dropout_2=0.2,
        )
    )

    results.append(
        run_training(
            name="hybrid_marked_mlp",
            csv_path=args.csv_path,
            output_dir=output_root / "hybrid_marked_mlp",
            text_builder=build_old_marked_input,
            input_format="[LANG] + sentence with <mwe> tags + [MWE]",
            head_type="new_mlp",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
            model_name=args.model_name,
            unfreeze_encoder=args.unfreeze_encoder,
            hidden_1=512,
            hidden_2=128,
            dropout_1=0.2,
            dropout_2=0.2,
        )
    )

    if not args.skip_neural_m:
        cache = build_or_load_cache(
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
                cache=cache,
                output_dir=output_root / "neural_m",
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
        cache = build_or_load_fusion_cache(
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
                cache=cache,
                output_dir=output_root / "hybrid_llm_fusion",
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
