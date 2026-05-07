from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.config import PIPELINE_DATA_CONFIG


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def build_sample(
    row: dict[str, str],
    row_index: int,
    image_root: str | Path,
    language: str,
    num_candidates: int = 5,
) -> dict[str, Any]:
    compound = row["compound"]
    compound_dir = Path(image_root) / compound

    candidate_names: list[str] = []
    captions: list[str] = []
    image_paths: list[str] = []

    for idx in range(1, num_candidates + 1):
        image_name = row[f"image{idx}_name"]
        candidate_names.append(image_name)
        captions.append(row[f"image{idx}_caption"])
        image_paths.append(str(compound_dir / image_name))

    return {
        "sample_id": f"{language}_{row_index}",
        "language": language,
        "sentence": row["sentence"],
        "compound": compound,
        "candidate_names": candidate_names,
        "captions": captions,
        "image_paths": image_paths,
    }


def load_samples(
    tsv_path: str | Path,
    image_root: str | Path,
    language: str,
    num_candidates: int = 5,
) -> list[dict[str, Any]]:
    rows = read_tsv(tsv_path)
    return [
        build_sample(
            row=row,
            row_index=row_index,
            image_root=image_root,
            language=language,
            num_candidates=num_candidates,
        )
        for row_index, row in enumerate(rows)
    ]


def get_samples() -> list[dict[str, Any]]:
    config = PIPELINE_DATA_CONFIG
    return load_samples(
        tsv_path=config.tsv_path,
        image_root=config.image_root,
        language=config.language,
        num_candidates=config.num_candidates,
    )


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("samples cannot be empty.")

    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "languages": [sample["language"] for sample in samples],
        "sentences": [sample["sentence"] for sample in samples],
        "compounds": [sample["compound"] for sample in samples],
        "candidate_names": [sample["candidate_names"] for sample in samples],
        "captions": [sample["captions"] for sample in samples],
        "image_paths": [sample["image_paths"] for sample in samples],
    }


def get_dataloader() -> DataLoader:
    config = PIPELINE_DATA_CONFIG
    return DataLoader(
        get_samples(),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
