from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.idiom_signal import load_idiom_signal


MODEL_DIR = REPO_ROOT / "outputs/idiom_signal_40ep"
OUTPUT_TSV = REPO_ROOT / "outputs/m_scores/admire_train_scores.tsv"
BATCH_SIZE = 32
MAX_LENGTH = 128

ADMIRE_FILES = [
    ("EN", "train", REPO_ROOT / "admire2_training/EN/train/subtask_a_train.tsv"),
    ("PT", "train", REPO_ROOT / "admire2_training/PT/train/subtask_a_train.tsv"),
    ("EN", "dev", REPO_ROOT / "admire2_training/EN/dev/subtask_a_dev.tsv"),
    ("PT", "dev", REPO_ROOT / "admire2_training/PT/dev/subtask_a_dev.tsv"),
    ("EN", "test", REPO_ROOT / "admire2_training/EN/test/subtask_a_test.tsv"),
    ("PT", "test", REPO_ROOT / "admire2_training/PT/test/subtask_a_test.tsv"),
    ("EN", "xeval", REPO_ROOT / "admire2_training/EN/xeval/subtask_a_xe.tsv"),
    ("PT", "xeval", REPO_ROOT / "admire2_training/PT/xeval/subtask_a_xp.tsv"),
]


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_tsv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows cannot be empty.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "language",
                "split",
                "row_index",
                "compound",
                "sentence",
                "m_prob",
                "m_input",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def batched(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    ]


def build_sample_id(language: str, split: str, row_index: int) -> str:
    return f"{language}_{split}_{row_index}"


def score_rows(
    model,
    tsv_path: str | Path,
    language: str,
    split: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    rows = read_tsv(tsv_path)
    records: list[dict[str, Any]] = []
    batches = batched(rows, batch_size=batch_size)

    print(
        f"[M export] scoring {language}/{split}: "
        f"rows={len(rows)} batches={len(batches)} file={tsv_path}"
    )

    with torch.no_grad():
        for batch_index, batch_rows in enumerate(batches, start=1):
            sentences = [row["sentence"] for row in batch_rows]
            compounds = [row["compound"] for row in batch_rows]
            languages = [language] * len(batch_rows)

            m_probs = model(
                sentences=sentences,
                mwes=compounds,
                languages=languages,
                max_length=max_length,
                return_probabilities=True,
            ).squeeze(-1)

            for row, m_prob_tensor in zip(batch_rows, m_probs):
                row_index = len(records)
                m_prob = float(m_prob_tensor.detach().cpu().item())
                records.append(
                    {
                        "sample_id": build_sample_id(
                            language=language,
                            split=split,
                            row_index=row_index,
                        ),
                        "language": language,
                        "split": split,
                        "row_index": row_index,
                        "compound": row["compound"],
                        "sentence": row["sentence"],
                        "m_prob": f"{m_prob:.8f}",
                        "m_input": f"{(2.0 * m_prob - 1.0):.8f}",
                    }
                )

            print(
                f"[M export] {language}/{split} "
                f"batch={batch_index}/{len(batches)} done"
            )

    print(f"[M export] finished {language}/{split}: scores={len(records)}")
    return records


def export_scores(model) -> None:
    records: list[dict[str, Any]] = []

    for language, split, tsv_path in ADMIRE_FILES:
        before = len(records)
        records.extend(
            score_rows(
                model=model,
                tsv_path=tsv_path,
                language=language,
                split=split,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
            )
        )
        added = len(records) - before
        print(f"[M export] added {added} rows from {language}/{split}")

    write_tsv(OUTPUT_TSV, records)
    print(f"[M export] wrote total={len(records)} rows to {OUTPUT_TSV}")


def main() -> None:
    print("[M export] loading trained M model")
    print(f"[M export] model_dir={MODEL_DIR}")
    model = load_idiom_signal(MODEL_DIR)
    print("[M export] model loaded")
    export_scores(model=model)


if __name__ == "__main__":
    main()
