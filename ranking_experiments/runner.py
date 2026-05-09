from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.candidate_fusion import GatedCandidateFusion
from src.config import QueryFusionConfig, get_device
from src.query_fusion import QueryFusion
from src.ranker import CosineRanker
from src.weights import load_weights, save_weights
from ranking_experiments.losses import listwise_ranking_loss
from ranking_experiments.metrics import ndcg_at_k, top1_accuracy


CACHE_PATH = REPO_ROOT / "ranking_experiments/cache/admire_rank_cache.pt"
SHARED_DIM = 512


@dataclass(frozen=True)
class RankingExperimentConfig:
    name: str
    query_projection: str
    m_embedding_dim: int
    seed: int
    m_source: str = "predicted"  # "predicted", "oracle", or "none"
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class CachedRankingDataset(Dataset):
    def __init__(self, split_payload: dict[str, Any]) -> None:
        self.payload = split_payload

    def __len__(self) -> int:
        return int(self.payload["q_raw"].size(0))

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "sample_id": self.payload["sample_ids"][index],
            "q_raw": self.payload["q_raw"][index],
            "t": self.payload["t"][index],
            "i": self.payload["i"][index],
            "m_prob": self.payload["m_prob"][index],
            "m_oracle": self.payload["m_oracle"][index],
            "gold_relevance": self.payload["gold_relevance"][index],
            "gold_order": self.payload["gold_order"][index],
        }


class FrozenPCAProjection(nn.Module):
    """Frozen PCA projection from cached Q_raw to the shared query dimension."""

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)

    def forward(self, q_raw: torch.Tensor) -> torch.Tensor:
        projected = (q_raw - self.mean) @ self.components.t()

        if projected.size(-1) < self.output_dim:
            padding = projected.new_zeros(
                projected.size(0),
                self.output_dim - projected.size(-1),
            )
            projected = torch.cat([projected, padding], dim=-1)

        return projected[:, : self.output_dim]


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}. "
            "Run python3 ranking_experiments/build_cache.py first."
        )
    try:
        return torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(CACHE_PATH, map_location="cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_loader(
    split_payload: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        CachedRankingDataset(split_payload),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def fit_pca_projection(
    q_raw: torch.Tensor,
    output_dim: int,
    device: torch.device,
) -> FrozenPCAProjection:
    mean = q_raw.mean(dim=0, keepdim=True)
    centered = q_raw - mean
    max_components = max(1, min(output_dim, centered.size(0) - 1, centered.size(1)))

    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    components = vh[:max_components]

    print(
        f"Q PCA fitted: raw_dim={q_raw.size(1)} "
        f"samples={q_raw.size(0)} "
        f"components={max_components} "
        f"output_dim={output_dim}"
    )

    return FrozenPCAProjection(
        mean=mean.to(device),
        components=components.to(device),
        output_dim=output_dim,
    )


def build_query_projection(
    experiment: RankingExperimentConfig,
    train_q_raw: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    if experiment.query_projection == "linear_trainable":
        projection = nn.Linear(train_q_raw.size(-1), SHARED_DIM).to(device)
        return projection, {
            "query_projection": "linear_trainable",
            "trainable": True,
            "raw_dim": train_q_raw.size(-1),
            "output_dim": SHARED_DIM,
        }

    if experiment.query_projection == "pca_frozen":
        projection = fit_pca_projection(
            q_raw=train_q_raw,
            output_dim=SHARED_DIM,
            device=device,
        )
        return projection, {
            "query_projection": "pca_frozen",
            "trainable": False,
            "raw_dim": train_q_raw.size(-1),
            "samples": train_q_raw.size(0),
            "components": projection.components.size(0),
            "output_dim": projection.output_dim,
        }

    raise ValueError(f"Unknown query projection: {experiment.query_projection}")


def forward_scores(
    batch: dict[str, torch.Tensor],
    m_source: str,
    query_projection: nn.Module,
    query_fusion: QueryFusion,
    candidate_fusion: GatedCandidateFusion,
    ranker: CosineRanker,
    device: torch.device,
) -> torch.Tensor:
    q_raw = batch["q_raw"].to(device)
    if m_source == "oracle":
        m_tensor = batch["m_oracle"].to(device)
    else:
        m_tensor = batch["m_prob"].to(device)
        
    image_repr = batch["i"].to(device)
    caption_repr = batch["t"].to(device)

    q = F.normalize(query_projection(q_raw), p=2, dim=-1)
    q_prime = query_fusion(q, m_tensor)
    candidate_repr = candidate_fusion(image_repr, caption_repr)

    return ranker(q_prime, candidate_repr)


def train_one_epoch(
    loader: DataLoader,
    experiment: RankingExperimentConfig,
    query_projection: nn.Module,
    query_fusion: QueryFusion,
    candidate_fusion: GatedCandidateFusion,
    ranker: CosineRanker,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    if experiment.query_projection == "linear_trainable":
        query_projection.train()
    else:
        query_projection.eval()
    query_fusion.train()
    candidate_fusion.train()

    total_loss = 0.0
    total_examples = 0

    progress = tqdm(
        loader,
        desc=f"{experiment.name} train {epoch}/{experiment.epochs}",
        dynamic_ncols=True,
    )
    for batch in progress:
        gold_relevance = batch["gold_relevance"].to(device)

        optimizer.zero_grad(set_to_none=True)
        scores = forward_scores(
            batch=batch,
            m_source=experiment.m_source,
            query_projection=query_projection,
            query_fusion=query_fusion,
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
    m_source: str,
    query_projection: nn.Module,
    query_fusion: QueryFusion,
    candidate_fusion: GatedCandidateFusion,
    ranker: CosineRanker,
    device: torch.device,
) -> dict[str, float]:
    query_projection.eval()
    query_fusion.eval()
    candidate_fusion.eval()
    ranker.eval()

    losses: list[float] = []
    top1_values: list[float] = []
    ndcg_values: list[float] = []

    with torch.no_grad():
        progress = tqdm(loader, desc=f"ranking {split_name}", dynamic_ncols=True)
        for batch in progress:
            gold_relevance = batch["gold_relevance"].to(device)
            gold_order = batch["gold_order"].to(device)

            scores = forward_scores(
                batch=batch,
                m_source=m_source,
                query_projection=query_projection,
                query_fusion=query_fusion,
                candidate_fusion=candidate_fusion,
                ranker=ranker,
                device=device,
            )
            loss = listwise_ranking_loss(scores=scores, gold_relevance=gold_relevance)

            losses.append(loss.item())
            top1_values.append(top1_accuracy(scores=scores, gold_order_indices=gold_order))
            ndcg_values.append(ndcg_at_k(scores=scores, gold_relevance=gold_relevance))
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": sum(losses) / len(losses),
        "top1": sum(top1_values) / len(top1_values),
        "ndcg": sum(ndcg_values) / len(ndcg_values),
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def run_training(
    experiment: RankingExperimentConfig,
    output_dir: Path,
    loaders: dict[str, DataLoader],
    train_q_raw: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    weights_path = output_dir / "weights.pt"
    results_path = output_dir / "results.json"
    config_path = output_dir / "config.json"

    print(
        f"starting ranking run "
        f"name={experiment.name} "
        f"query_projection={experiment.query_projection} "
        f"m_embedding_dim={experiment.m_embedding_dim} "
        f"seed={experiment.seed} "
        f"device={device}"
    )

    set_seed(experiment.seed)

    query_projection, projection_info = build_query_projection(
        experiment=experiment,
        train_q_raw=train_q_raw,
        device=device,
    )
    query_fusion = QueryFusion(
        QueryFusionConfig(m_embedding_dim=experiment.m_embedding_dim)
    ).to(device)
    candidate_fusion = GatedCandidateFusion().to(device)
    ranker = CosineRanker().to(device)

    trainable_parameters = []
    trainable_modules = []
    if experiment.query_projection == "linear_trainable":
        trainable_parameters.extend(query_projection.parameters())
        trainable_modules.append("query_projection")
    trainable_parameters.extend(query_fusion.parameters())
    trainable_parameters.extend(candidate_fusion.parameters())
    trainable_modules.extend(["query_fusion", "candidate_fusion"])

    print(
        f"query_projection={experiment.query_projection} "
        f"trainable_modules={trainable_modules}"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=experiment.learning_rate,
    )

    history: list[dict[str, Any]] = []
    best_ndcg = -1.0
    best_metrics: dict[str, float] | None = None
    trained_modules = {
        "query_projection": query_projection,
        "query_fusion": query_fusion,
        "candidate_fusion": candidate_fusion,
    }

    for epoch in range(1, experiment.epochs + 1):
        train_loss = train_one_epoch(
            loader=loaders["train"],
            experiment=experiment,
            query_projection=query_projection,
            query_fusion=query_fusion,
            candidate_fusion=candidate_fusion,
            ranker=ranker,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        dev_metrics = evaluate(
            loader=loaders["dev"],
            split_name="dev",
            m_source=experiment.m_source,
            query_projection=query_projection,
            query_fusion=query_fusion,
            candidate_fusion=candidate_fusion,
            ranker=ranker,
            device=device,
        )

        print(
            f"epoch={epoch}/{experiment.epochs} "
            f"run={experiment.name} "
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
                path=str(weights_path),
            )

    load_weights(
        modules=trained_modules,
        path=str(weights_path),
        device=device,
    )
    test_metrics = evaluate(
        loader=loaders["test"],
        split_name="test",
        m_source=experiment.m_source,
        query_projection=query_projection,
        query_fusion=query_fusion,
        candidate_fusion=candidate_fusion,
        ranker=ranker,
        device=device,
    )
    xeval_metrics = evaluate(
        loader=loaders["xeval"],
        split_name="xeval",
        m_source=experiment.m_source,
        query_projection=query_projection,
        query_fusion=query_fusion,
        candidate_fusion=candidate_fusion,
        ranker=ranker,
        device=device,
    )

    result_payload = {
        "experiment": experiment.to_dict(),
        "projection_info": projection_info,
        "weights_path": str(weights_path),
        "best_dev": best_metrics,
        "test": test_metrics,
        "xeval": xeval_metrics,
        "history": history,
    }
    save_json(
        {
            "experiment": experiment.to_dict(),
            "projection_info": projection_info,
        },
        config_path,
    )
    save_json(result_payload, results_path)

    print(f"saved best weights to {weights_path}")
    print(f"saved results to {results_path}")
    print(
        f"final run={experiment.name} "
        f"seed={experiment.seed} "
        f"test_top1={test_metrics['top1']:.4f} "
        f"test_ndcg={test_metrics['ndcg']:.4f} "
        f"xeval_top1={xeval_metrics['top1']:.4f} "
        f"xeval_ndcg={xeval_metrics['ndcg']:.4f}"
    )

    return result_payload


def build_loaders(
    cache: dict[str, Any],
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    splits = cache["splits"]
    return {
        "train": build_loader(
            splits["train"],
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        ),
        "dev": build_loader(
            splits["dev"],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
        "test": build_loader(
            splits["test"],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
        "xeval": build_loader(
            splits["xeval"],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
    }


def aggregate_runs(
    results: list[dict[str, Any]],
    m_dims: tuple[int, ...],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []

    for m_dim in m_dims:
        runs = [
            result
            for result in results
            if result["experiment"]["m_embedding_dim"] == m_dim
        ]

        aggregate = {
            "m_dim": m_dim,
            "num_seeds": len(runs),
            "best_epoch": mean(
                float(
                    max(
                        run["history"],
                        key=lambda item: item["dev"]["ndcg"],
                    )["epoch"]
                )
                for run in runs
            ),
        }

        for split_name in ("best_dev", "test", "xeval"):
            aggregate[split_name] = {}
            for metric_name in ("loss", "top1", "ndcg"):
                aggregate[split_name][metric_name] = mean(
                    float(run[split_name][metric_name])
                    for run in runs
                )

        aggregates.append(aggregate)

    return aggregates


def run_ablation(
    family_name: str,
    query_projection: str,
    m_dims: tuple[int, ...],
    seeds: tuple[int, ...],
    m_source: str = "predicted",
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
) -> list[dict[str, Any]]:
    device = get_device()
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    split_sizes = {
        split_name: int(split_payload["q_raw"].size(0))
        for split_name, split_payload in cache["splits"].items()
    }

    print(
        f"ranking cache loaded: "
        f"train={split_sizes['train']} "
        f"dev={split_sizes['dev']} "
        f"test={split_sizes['test']} "
        f"xeval={split_sizes['xeval']} "
        f"family={family_name} "
        f"seeds={seeds} "
        f"device={device}"
    )

    results: list[dict[str, Any]] = []
    for m_dim in m_dims:
        for seed in seeds:
            run_name = f"m{m_dim}"
            experiment = RankingExperimentConfig(
                name=run_name,
                query_projection=query_projection,
                m_embedding_dim=m_dim,
                seed=seed,
                m_source=m_source,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            loaders = build_loaders(
                cache=cache,
                batch_size=batch_size,
                seed=seed,
            )
            output_dir = (
                REPO_ROOT
                / "ranking_experiments"
                / family_name
                / "output"
                / run_name
                / f"seed{seed}"
            )
            results.append(
                run_training(
                    experiment=experiment,
                    output_dir=output_dir,
                    loaders=loaders,
                    train_q_raw=train_q_raw,
                    device=device,
                )
            )

    summary_path = REPO_ROOT / "ranking_experiments" / family_name / "output/summary.json"
    save_json(
        {
            "family_name": family_name,
            "query_projection": query_projection,
            "m_dims": list(m_dims),
            "seeds": list(seeds),
            "selection_metric": "dev_ndcg",
            "runs": results,
            "aggregate": aggregate_runs(results, m_dims),
        },
        summary_path,
    )
    print(f"saved ablation summary to {summary_path}")
    return results
