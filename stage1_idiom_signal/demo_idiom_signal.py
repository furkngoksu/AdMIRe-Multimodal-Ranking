from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--json_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_idiom_signal(args.model_dir)

    records = []
    for language, mwe, sentence in EXAMPLES:
        score = compute_m(
            sentence=sentence,
            mwe=mwe,
            language=language,
            max_length=args.max_length,
        )
        records.append(
            {
                "language": language,
                "mwe": mwe,
                "sentence": sentence,
                "m_score": score,
            }
        )
        print(f"M={score:.4f}\tlanguage={language}\tmwe={mwe}\tsentence={sentence}")

    if args.json_path is not None:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
