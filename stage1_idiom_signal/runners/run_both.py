from __future__ import annotations

import argparse
from pathlib import Path

from stage1_idiom_signal.common import (
    DEFAULT_CSV_PATH,
    build_new_target_only_input,
    build_old_marked_input,
    print_metric_table,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run old and current Stage 1 baselines quickly.")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--output_root", default="stage1_idiom_signal/outputs")
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=512, help="Use 0 for the full CSV.")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    old_result = run_training(
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

    new_result = run_training(
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

    print_metric_table([old_result, new_result])


if __name__ == "__main__":
    main()

