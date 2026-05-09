from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.candidate_fusion import GatedCandidateFusion
from src.caption_encoder import CaptionEncoder
from src.config import QueryFusionConfig, get_device
from src.image_encoder import ImageEncoder
from src.query_encoder import QueryEncoder
from src.query_fusion import QueryFusion
from src.ranker import CosineRanker
from src.weights import load_weights, save_weights
from training.losses import listwise_ranking_loss
from training.metrics import ndcg_at_k, top1_accuracy
from training.train_data import collate_train_samples, load_train_samples, read_m_scores


M_EMBEDDING_DIMS = (0, 8, 16, 32, 64)
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

M_SCORES_PATH = REPO_ROOT / "outputs/m_scores/admire_train_scores.tsv"

TRAIN_FILES = [
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
]

DEV_FILES = [
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
]

TEST_FILES = [
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
]

XEVAL_FILES = [
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
]


def load_samples(files: list[tuple[str, str, Path, Path]]) -> list[dict[str, Any]]:
    m_scores = read_m_scores(M_SCORES_PATH)
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


def build_loader(
    samples: list[dict[str, Any]],
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        samples,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_train_samples,
    )


def forward_scores(
    batch: dict[str, Any],
    query_encoder: QueryEncoder,
    query_fusion: QueryFusion,
    caption_encoder: CaptionEncoder,
    image_encoder: ImageEncoder,
    candidate_fusion: GatedCandidateFusion,
    ranker: CosineRanker,
    device: torch.device,
) -> torch.Tensor:
    q = query_encoder(batch["sentences"], batch["compounds"])
    q_prime = query_fusion(q, batch["m_prob"].to(device))

    t = caption_encoder(batch["captions"])
    i = image_encoder(batch["image_paths"])
    c = candidate_fusion(i, t)

    return ranker(q_prime, c)


def train_one_epoch(
    loader: DataLoader,
    query_encoder: QueryEncoder,
    query_fusion: QueryFusion,
    caption_encoder: CaptionEncoder,
    image_encoder: ImageEncoder,
    candidate_fusion: GatedCandidateFusion,
    ranker: CosineRanker,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    query_encoder.projection.train()
    query_fusion.train()
    candidate_fusion.train()

    total_loss = 0.0
    total_examples = 0

    progress = tqdm(
        loader,
        desc=f"ranking train m{query_fusion.config.m_embedding_dim} {epoch}/{EPOCHS}",
        dynamic_ncols=True,
    )
    for batch in progress:
        gold_relevance = batch["gold_relevance"].to(device)

        optimizer.zero_grad(set_to_none=True)
        scores = forward_scores(
            batch=batch,
            query_encoder=query_encoder,
            query_fusion=query_fusion,
            caption_encoder=caption_encoder,
            image_encoder=image_encoder,
            candidate_fusion=candidate_fusion,
            ranker=ranker,
            device=device,
        )
        loss = listwise_ranking_loss(scores=scores, gold_relevance=gold_relevance)
        loss.backward()
        optimizer.step()

        batch_size = gold_relevance.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_examples


def evaluate(
    loader: DataLoader,
    split_name: str,
    query_encoder: QueryEncoder,
    query_fusion: QueryFusion,
    caption_encoder: CaptionEncoder,
    image_encoder: ImageEncoder,
    candidate_fusion: GatedCandidateFusion,
    ranker: CosineRanker,
    device: torch.device,
) -> dict[str, float]:
    query_encoder.eval()
    query_fusion.eval()
    caption_encoder.eval()
    image_encoder.eval()
    candidate_fusion.eval()
    ranker.eval()

    losses: list[float] = []
    top1_values: list[float] = []
    ndcg_values: list[float] = []

    with torch.no_grad():
        progress = tqdm(loader, desc=f"ranking {split_name}", dynamic_ncols=True)
        for batch in progress:
            gold_relevance = batch["gold_relevance"].to(device)
            gold_order_indices = batch["gold_order_indices"].to(device)

            scores = forward_scores(
                batch=batch,
                query_encoder=query_encoder,
                query_fusion=query_fusion,
                caption_encoder=caption_encoder,
                image_encoder=image_encoder,
                candidate_fusion=candidate_fusion,
                ranker=ranker,
                device=device,
            )
            loss = listwise_ranking_loss(scores=scores, gold_relevance=gold_relevance)

            losses.append(loss.item())
            top1_values.append(
                top1_accuracy(
                    scores=scores,
                    gold_order_indices=gold_order_indices,
                )
            )
            ndcg_values.append(
                ndcg_at_k(
                    scores=scores,
                    gold_relevance=gold_relevance,
                )
            )

            progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": sum(losses) / len(losses),
        "top1": sum(top1_values) / len(top1_values),
        "ndcg": sum(ndcg_values) / len(ndcg_values),
    }


def save_results(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def run_training(
    m_embedding_dim: int,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    xeval_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = REPO_ROOT / f"checkpoints/ranking_m{m_embedding_dim}.pt"
    results_path = REPO_ROOT / f"outputs/results/ranking_m{m_embedding_dim}.json"

    print(
        f"starting ranking run "
        f"m_embedding_dim={m_embedding_dim} device={device}"
    )

    query_encoder = QueryEncoder().to(device)
    query_fusion = QueryFusion(
        QueryFusionConfig(m_embedding_dim=m_embedding_dim)
    ).to(device)
    caption_encoder = CaptionEncoder().to(device)
    image_encoder = ImageEncoder().to(device)
    candidate_fusion = GatedCandidateFusion().to(device)
    ranker = CosineRanker().to(device)

    trainable_parameters = []
    trainable_parameters.extend(query_encoder.projection.parameters())
    trainable_parameters.extend(query_fusion.parameters())
    trainable_parameters.extend(candidate_fusion.parameters())

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    history: list[dict[str, Any]] = []
    best_ndcg = -1.0
    best_metrics: dict[str, float] | None = None
    trained_modules = {
        "query_projection": query_encoder.projection,
        "query_fusion": query_fusion,
        "candidate_fusion": candidate_fusion,
    }

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            loader=train_loader,
            query_encoder=query_encoder,
            query_fusion=query_fusion,
            caption_encoder=caption_encoder,
            image_encoder=image_encoder,
            candidate_fusion=candidate_fusion,
            ranker=ranker,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        dev_metrics = evaluate(
            loader=dev_loader,
            split_name="dev",
            query_encoder=query_encoder,
            query_fusion=query_fusion,
            caption_encoder=caption_encoder,
            image_encoder=image_encoder,
            candidate_fusion=candidate_fusion,
            ranker=ranker,
            device=device,
        )

        print(
            f"epoch={epoch}/{EPOCHS} "
            f"m={m_embedding_dim} "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_top1={dev_metrics['top1']:.4f} "
            f"dev_ndcg={dev_metrics['ndcg']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev": dev_metrics,
            }
        )

        if dev_metrics["ndcg"] > best_ndcg:
            best_ndcg = dev_metrics["ndcg"]
            best_metrics = dev_metrics
            save_weights(
                modules=trained_modules,
                path=str(checkpoint_path),
            )

    load_weights(
        modules=trained_modules,
        path=str(checkpoint_path),
        device=device,
    )
    test_metrics = evaluate(
        loader=test_loader,
        split_name="test",
        query_encoder=query_encoder,
        query_fusion=query_fusion,
        caption_encoder=caption_encoder,
        image_encoder=image_encoder,
        candidate_fusion=candidate_fusion,
        ranker=ranker,
        device=device,
    )
    xeval_metrics = evaluate(
        loader=xeval_loader,
        split_name="xeval",
        query_encoder=query_encoder,
        query_fusion=query_fusion,
        caption_encoder=caption_encoder,
        image_encoder=image_encoder,
        candidate_fusion=candidate_fusion,
        ranker=ranker,
        device=device,
    )

    result_payload = {
        "m_embedding_dim": m_embedding_dim,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "checkpoint_path": str(checkpoint_path),
        "best_dev": best_metrics,
        "test": test_metrics,
        "xeval": xeval_metrics,
        "history": history,
    }
    save_results(result_payload, results_path)

    print(f"saved best weights to {checkpoint_path}")
    print(f"saved results to {results_path}")
    print(
        f"final m={m_embedding_dim} "
        f"test_top1={test_metrics['top1']:.4f} "
        f"test_ndcg={test_metrics['ndcg']:.4f} "
        f"xeval_top1={xeval_metrics['top1']:.4f} "
        f"xeval_ndcg={xeval_metrics['ndcg']:.4f}"
    )

    return result_payload


def main() -> None:
    device = get_device()

    train_samples = load_samples(TRAIN_FILES)
    dev_samples = load_samples(DEV_FILES)
    test_samples = load_samples(TEST_FILES)
    xeval_samples = load_samples(XEVAL_FILES)
    train_loader = build_loader(train_samples, shuffle=True)
    dev_loader = build_loader(dev_samples, shuffle=False)
    test_loader = build_loader(test_samples, shuffle=False)
    xeval_loader = build_loader(xeval_samples, shuffle=False)

    print(
        f"ranking data "
        f"train={len(train_samples)} "
        f"dev={len(dev_samples)} "
        f"test={len(test_samples)} "
        f"xeval={len(xeval_samples)} "
        f"m_embedding_dims={M_EMBEDDING_DIMS} "
        f"device={device}"
    )

    all_results = [
        run_training(
            m_embedding_dim=m_embedding_dim,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            xeval_loader=xeval_loader,
            device=device,
        )
        for m_embedding_dim in M_EMBEDDING_DIMS
    ]

    summary_path = REPO_ROOT / "outputs/results/ranking_ablation_summary.json"
    save_results(
        {
            "m_embedding_dims": list(M_EMBEDDING_DIMS),
            "selection_metric": "dev_ndcg",
            "runs": all_results,
        },
        summary_path,
    )
    print(f"saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
