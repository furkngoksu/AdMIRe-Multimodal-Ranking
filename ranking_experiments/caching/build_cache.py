from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.caption_encoder import CaptionEncoder
from src.config import get_device
from src.image_encoder import ImageEncoder
from src.query_encoder import QueryEncoder
from ranking_experiments.data_loader import (
    collate_train_samples,
    load_train_samples,
    read_m_scores,
)


CACHE_PATH = REPO_ROOT / "ranking_experiments/caching/cache/admire_rank_cache.pt"
M_SCORES_PATH = REPO_ROOT / "signal_inference/admire_train_scores.tsv"
CACHE_BATCH_SIZE = 4

SPLIT_FILES = {
    "train": [
        (
            "EN",
            "train",
            REPO_ROOT / "admire2_training/EN/train/subtask_a_train.tsv",
            REPO_ROOT / "admire2_training/EN/train",
        ),
        (
            "PT",
            "train",
            REPO_ROOT / "admire2_training/PT/train/subtask_a_train.tsv",
            REPO_ROOT / "admire2_training/PT/train",
        ),
    ],
    "dev": [
        (
            "EN",
            "dev",
            REPO_ROOT / "admire2_training/EN/dev/subtask_a_dev.tsv",
            REPO_ROOT / "admire2_training/EN/dev",
        ),
        (
            "PT",
            "dev",
            REPO_ROOT / "admire2_training/PT/dev/subtask_a_dev.tsv",
            REPO_ROOT / "admire2_training/PT/dev",
        ),
    ],
    "test": [
        (
            "EN",
            "test",
            REPO_ROOT / "admire2_training/EN/test/subtask_a_test.tsv",
            REPO_ROOT / "admire2_training/EN/test",
        ),
        (
            "PT",
            "test",
            REPO_ROOT / "admire2_training/PT/test/subtask_a_test.tsv",
            REPO_ROOT / "admire2_training/PT/test",
        ),
    ],
    "xeval": [
        (
            "EN",
            "xeval",
            REPO_ROOT / "admire2_training/EN/xeval/subtask_a_xe.tsv",
            REPO_ROOT / "admire2_training/EN/xeval",
        ),
        (
            "PT",
            "xeval",
            REPO_ROOT / "admire2_training/PT/xeval/subtask_a_xp.tsv",
            REPO_ROOT / "admire2_training/PT/xeval",
        ),
    ],
}


def load_split_samples(
    files: list[tuple[str, str, Path, Path]],
    m_scores: dict[str, float],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for language, split, tsv_path, image_root in files:
        samples.extend(
            load_train_samples(
                tsv_path=tsv_path,
                image_root=image_root,
                language=language,
                split=split,
                m_scores=m_scores,
            )
        )

    return samples


def encode_split(
    split_name: str,
    samples: list[dict[str, Any]],
    query_encoder: QueryEncoder,
    caption_encoder: CaptionEncoder,
    image_encoder: ImageEncoder,
) -> dict[str, Any]:
    loader = DataLoader(
        samples,
        batch_size=CACHE_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_train_samples,
    )

    q_raw_batches: list[torch.Tensor] = []
    t_batches: list[torch.Tensor] = []
    i_batches: list[torch.Tensor] = []
    m_prob_batches: list[torch.Tensor] = []
    m_gold_batches: list[torch.Tensor] = []
    relevance_batches: list[torch.Tensor] = []
    order_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        progress = tqdm(loader, desc=f"cache {split_name}", dynamic_ncols=True)
        for batch in progress:
            q_raw_batches.append(
                query_encoder.encode_raw(
                    sentences=batch["sentences"],
                    compounds=batch["compounds"],
                ).detach().cpu()
            )
            t_batches.append(caption_encoder(batch["captions"]).detach().cpu())
            i_batches.append(image_encoder(batch["image_paths"]).detach().cpu())
            m_prob_batches.append(batch["m_prob"].detach().cpu())
            m_gold_batches.append(batch["m_gold"].detach().cpu())
            relevance_batches.append(batch["gold_relevance"].detach().cpu())
            order_batches.append(batch["gold_order_indices"].detach().cpu())
            sample_ids.extend(batch["sample_ids"])

    encoded = {
        "sample_ids": sample_ids,
        "q_raw": torch.cat(q_raw_batches, dim=0),
        "t": torch.cat(t_batches, dim=0),
        "i": torch.cat(i_batches, dim=0),
        "m_prob": torch.cat(m_prob_batches, dim=0),
        "m_gold": torch.cat(m_gold_batches, dim=0),
        "gold_relevance": torch.cat(relevance_batches, dim=0),
        "gold_order": torch.cat(order_batches, dim=0),
    }

    print(
        f"[cache] {split_name}: "
        f"q_raw={tuple(encoded['q_raw'].shape)} "
        f"t={tuple(encoded['t'].shape)} "
        f"i={tuple(encoded['i'].shape)} "
        f"m_prob={tuple(encoded['m_prob'].shape)} "
        f"gold_relevance={tuple(encoded['gold_relevance'].shape)}"
    )
    return encoded


def main() -> None:
    device = get_device()
    m_scores = read_m_scores(M_SCORES_PATH)

    query_encoder = QueryEncoder().to(device).eval()
    caption_encoder = CaptionEncoder().to(device).eval()
    image_encoder = ImageEncoder().to(device).eval()

    splits: dict[str, Any] = {}
    for split_name, files in SPLIT_FILES.items():
        samples = load_split_samples(files=files, m_scores=m_scores)
        print(f"[cache] loading {split_name}: samples={len(samples)} device={device}")
        splits[split_name] = encode_split(
            split_name=split_name,
            samples=samples,
            query_encoder=query_encoder,
            caption_encoder=caption_encoder,
            image_encoder=image_encoder,
        )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "splits": splits,
            "meta": {
                "cache_batch_size": CACHE_BATCH_SIZE,
                "q_raw_dim": splits["train"]["q_raw"].size(-1),
                "shared_dim": splits["train"]["t"].size(-1),
            },
        },
        CACHE_PATH,
    )
    print(f"[cache] wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
