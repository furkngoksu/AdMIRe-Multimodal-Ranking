from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any

import torch


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def parse_expected_order(value: str) -> list[str]:
    if value.strip() == "":
        raise ValueError("Training rows must include expected_order.")
    return [str(name) for name in ast.literal_eval(value)]


def read_m_scores(path: str | Path) -> dict[str, float]:
    rows = read_tsv(path)
    return {
        row["sample_id"]: float(row["m_prob"])
        for row in rows
    }


def build_gold_labels(
    candidate_names: list[str],
    expected_order: list[str],
) -> tuple[list[int], list[int]]:
    name_to_index = {name: index for index, name in enumerate(candidate_names)}
    order_indices = [name_to_index[name] for name in expected_order]

    relevance = [0] * len(candidate_names)
    for rank, candidate_index in enumerate(order_indices):
        relevance[candidate_index] = len(candidate_names) - rank

    return order_indices, relevance


def resolve_compound_dir(image_root: str | Path, compound: str) -> Path:
    """Resolve compound folder, including apostrophe-to-underscore dataset naming."""
    root = Path(image_root)

    direct = root / compound
    if direct.exists():
        return direct

    underscored = root / compound.replace("'s", "_s").replace("'", "_")
    if underscored.exists():
        return underscored

    raise FileNotFoundError(f"Could not find compound folder: {root / compound}")


def resolve_image_path(image_root: str | Path, compound: str, image_name: str) -> str:
    image_path = resolve_compound_dir(image_root, compound) / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image file: {image_path}")

    return str(image_path)


def build_train_sample(
    row: dict[str, str],
    row_index: int,
    image_root: str | Path,
    language: str,
    split: str,
    m_scores: dict[str, float],
    num_candidates: int = 5,
) -> dict[str, Any]:
    compound = row["compound"]
    sample_id = f"{language}_{split}_{row_index}"

    candidate_names: list[str] = []
    captions: list[str] = []
    image_paths: list[str] = []

    for idx in range(1, num_candidates + 1):
        image_name = row[f"image{idx}_name"]
        candidate_names.append(image_name)
        captions.append(row[f"image{idx}_caption"])
        image_paths.append(resolve_image_path(image_root, compound, image_name))

    expected_order = parse_expected_order(row["expected_order"])
    gold_order_indices, gold_relevance = build_gold_labels(
        candidate_names=candidate_names,
        expected_order=expected_order,
    )

    return {
        "sample_id": sample_id,
        "language": language,
        "split": split,
        "sentence": row["sentence"],
        "compound": compound,
        "m_prob": m_scores[sample_id],
        "sentence_type": row.get("sentence_type"),
        "candidate_names": candidate_names,
        "captions": captions,
        "image_paths": image_paths,
        "expected_order": expected_order,
        "gold_order_indices": gold_order_indices,
        "gold_relevance": gold_relevance,
    }


def load_train_samples(
    tsv_path: str | Path,
    image_root: str | Path,
    language: str,
    split: str,
    m_scores: dict[str, float],
    num_candidates: int = 5,
) -> list[dict[str, Any]]:
    rows = read_tsv(tsv_path)
    return [
        build_train_sample(
            row=row,
            row_index=row_index,
            image_root=image_root,
            language=language,
            split=split,
            m_scores=m_scores,
            num_candidates=num_candidates,
        )
        for row_index, row in enumerate(rows)
    ]


def collate_train_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("samples cannot be empty.")

    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "languages": [sample["language"] for sample in samples],
        "splits": [sample["split"] for sample in samples],
        "sentences": [sample["sentence"] for sample in samples],
        "compounds": [sample["compound"] for sample in samples],
        "m_prob": torch.tensor(
            [[sample["m_prob"]] for sample in samples],
            dtype=torch.float,
        ),
        "sentence_types": [sample["sentence_type"] for sample in samples],
        "candidate_names": [sample["candidate_names"] for sample in samples],
        "captions": [sample["captions"] for sample in samples],
        "image_paths": [sample["image_paths"] for sample in samples],
        "expected_order": [sample["expected_order"] for sample in samples],
        "gold_order_indices": torch.tensor(
            [sample["gold_order_indices"] for sample in samples],
            dtype=torch.long,
        ),
        "gold_relevance": torch.tensor(
            [sample["gold_relevance"] for sample in samples],
            dtype=torch.float,
        ),
    }
