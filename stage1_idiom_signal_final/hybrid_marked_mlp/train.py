from __future__ import annotations

import argparse

from stage1_idiom_signal_final.shared import (
    DEFAULT_CSV_PATH,
    build_old_marked_input,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid marked-input Stage 1 baseline.")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--output_dir", default="stage1_idiom_signal_final/outputs/hybrid_marked_mlp")
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=512, help="Use 0 for the full CSV.")
    parser.add_argument("--hidden_1", type=int, default=512)
    parser.add_argument("--hidden_2", type=int, default=128)
    parser.add_argument("--dropout_1", type=float, default=0.2)
    parser.add_argument("--dropout_2", type=float, default=0.2)
    parser.add_argument("--unfreeze_encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        name="hybrid_marked_mlp",
        csv_path=args.csv_path,
        output_dir=args.output_dir,
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
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        dropout_1=args.dropout_1,
        dropout_2=args.dropout_2,
    )


if __name__ == "__main__":
    main()

