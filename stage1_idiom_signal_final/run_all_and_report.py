from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from stage1_idiom_signal_final.visualize_results import create_report


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run all Stage 1 comparison models, then create tables and plots.",
        add_help=False,
    )
    parser.add_argument("--output_root", default="stage1_idiom_signal_final/outputs")
    parser.add_argument("--report_dir", default="stage1_idiom_signal_final/reports")
    parser.add_argument("--help", action="store_true")
    args, remaining = parser.parse_known_args()

    if args.help:
        print("Usage example:")
        print(
            "python -m stage1_idiom_signal_final.run_all_and_report "
            "--epochs 40 --limit 0 --batch_size 16 --neural_batch_size 64 --fusion_batch_size 64"
        )
        print("")
        subprocess.run([sys.executable, "-m", "stage1_idiom_signal_final.run_all", "--help"], check=False)
        sys.exit(0)

    return args, remaining


def main() -> None:
    args, remaining = parse_args()
    command = [
        sys.executable,
        "-m",
        "stage1_idiom_signal_final.run_all",
        "--output_root",
        args.output_root,
        *remaining,
    ]
    subprocess.run(command, check=True)
    create_report(output_root=Path(args.output_root), report_dir=Path(args.report_dir))


if __name__ == "__main__":
    main()

