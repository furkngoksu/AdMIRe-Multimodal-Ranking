from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.idiom_signal import compute_m, load_idiom_signal


EXAMPLES = [
    (
        "EN",
        "spilled the beans",
        "He spilled the beans about the plan.",
    ),
    (
        "EN",
        "kicked the ball",
        "She kicked the ball.",
    ),
    (
        "EN",
        "break the ice",
        "The joke helped break the ice before the meeting.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo Stage 1 idiomaticity scores.")
    parser.add_argument("--model_dir", default="outputs/idiom_signal")
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_idiom_signal(args.model_dir)

    for language, mwe, sentence in EXAMPLES:
        score = compute_m(
            sentence=sentence,
            mwe=mwe,
            language=language,
            max_length=args.max_length,
        )
        print(f"M={score:.4f}\tlanguage={language}\tmwe={mwe}\tsentence={sentence}")


if __name__ == "__main__":
    main()
