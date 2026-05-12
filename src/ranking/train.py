from __future__ import annotations

import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ranking_experiments.data_loader import (
    collate_train_samples,
    load_train_samples,
    read_m_scores,
)
from src.candidate_fusion import GatedCandidateFusion
from src.caption_encoder import CaptionEncoder
from src.config import (
    QUERY_ENCODER_CONFIG,
    QueryFusionConfig,
    RANKING_MODEL_CONFIG,
    RANKING_PATHS,
    STAGE1_SIGNAL_CONFIG,
    TRAINING_DATA_CONFIG,
    get_device,
)
from src.image_encoder import ImageEncoder
from src.query_encoder import QueryEncoder
from src.query_fusion import build_query_fusion
from src.ranker import CosineRanker
from src.utils import save_json


SEEDS = (42, 100, 150, 200, 250)
N_TRIALS = 30


@dataclass(frozen=True)
class TrainConfig:
    name: str = "ranker"

    fusion_type: str = RANKING_MODEL_CONFIG.fusion_type
    m_embedding_dim: int = RANKING_MODEL_CONFIG.m_embedding_dim
    logit_scale_learnable: bool = RANKING_MODEL_CONFIG.logit_scale_learnable

    epochs: int = RANKING_MODEL_CONFIG.epochs
    batch_size: int = RANKING_MODEL_CONFIG.batch_size
    learning_rate: float = RANKING_MODEL_CONFIG.learning_rate
    weight_decay: float = RANKING_MODEL_CONFIG.weight_decay
    patience: int = RANKING_MODEL_CONFIG.early_stopping_patience

    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ListDataset(Dataset):
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


class RankingDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __len__(self) -> int:
        return self.payload["q_raw"].size(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "sample_id": self.payload["sample_ids"][index],
            "q_raw": self.payload["q_raw"][index],
            "i": self.payload["i"][index],
            "t": self.payload["t"][index],
            "m_prob": self.payload["m_prob"][index],
            "m_gold": self.payload["m_gold"][index],
            "gold_relevance": self.payload["gold_relevance"][index],
            "gold_order": self.payload["gold_order"][index],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    payload: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        RankingDataset(payload),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def load_samples(
    files: tuple[tuple[str, str, Path, Path], ...],
    m_scores: dict[str, float],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for language, split, tsv_path, image_root in files:
        split_samples = load_train_samples(
            tsv_path=tsv_path,
            image_root=image_root,
            language=language,
            split=split,
            m_scores=m_scores,
        )
        print(f"  {language}/{split}: {len(split_samples)} samples")
        samples.extend(split_samples)

    return samples


def encode_split(
    split_name: str,
    samples: list[dict[str, Any]],
    query_encoder: QueryEncoder,
    caption_encoder: CaptionEncoder,
    image_encoder: ImageEncoder,
) -> dict[str, Any]:
    loader = DataLoader(
        ListDataset(samples),
        batch_size=RANKING_MODEL_CONFIG.cache_batch_size,
        shuffle=False,
        collate_fn=collate_train_samples,
    )

    encoded: dict[str, Any] = {
        "sample_ids": [],
        "q_raw": [],
        "i": [],
        "t": [],
        "m_prob": [],
        "m_gold": [],
        "gold_relevance": [],
        "gold_order": [],
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"encoding {split_name}", dynamic_ncols=True):
            encoded["sample_ids"].extend(batch["sample_ids"])
            encoded["q_raw"].append(
                query_encoder.encode_raw(
                    batch["sentences"],
                    batch["compounds"],
                ).cpu()
            )
            encoded["i"].append(image_encoder(batch["image_paths"]).cpu())
            encoded["t"].append(caption_encoder(batch["captions"]).cpu())
            encoded["m_prob"].append(batch["m_prob"].cpu())
            encoded["m_gold"].append(batch["m_gold"].cpu())
            encoded["gold_relevance"].append(batch["gold_relevance"].cpu())
            encoded["gold_order"].append(batch["gold_order_indices"].cpu())

    return {
        key: value if key == "sample_ids" else torch.cat(value, dim=0)
        for key, value in encoded.items()
    }


def print_cache_summary(cache: dict[str, Any]) -> None:
    print("\n[CACHE SUMMARY]")
    print(f"Path    : {RANKING_PATHS.cache_path}")
    print(f"M source: {STAGE1_SIGNAL_CONFIG.train_scores_path}")
    print("Training uses predicted M: batch['m_prob']")
    print("Gold M is debug-only     : batch['m_gold']")

    for split_name, payload in cache["splits"].items():
        m_prob = payload["m_prob"].float()

        print(
            f"  {split_name:<5} "
            f"n={len(payload['sample_ids']):<5} "
            f"q={tuple(payload['q_raw'].shape)} "
            f"i={tuple(payload['i'].shape)} "
            f"t={tuple(payload['t'].shape)} "
            f"M(mean={m_prob.mean():.3f}, std={m_prob.std():.3f}, "
            f"min={m_prob.min():.3f}, max={m_prob.max():.3f})"
        )
        print(f"        first ids: {payload['sample_ids'][:3]}")


def build_cache(device: torch.device) -> dict[str, Any]:
    print("\n[BUILD CACHE]")
    print(f"M scores: {STAGE1_SIGNAL_CONFIG.train_scores_path}")

    m_scores = read_m_scores(STAGE1_SIGNAL_CONFIG.train_scores_path)

    query_encoder = QueryEncoder(QUERY_ENCODER_CONFIG).to(device).eval()
    caption_encoder = CaptionEncoder().to(device).eval()
    image_encoder = ImageEncoder().to(device).eval()

    splits: dict[str, Any] = {}

    for split_name, files in TRAINING_DATA_CONFIG.split_files.items():
        print(f"\nSplit: {split_name}")
        samples = load_samples(files, m_scores)

        splits[split_name] = encode_split(
            split_name=split_name,
            samples=samples,
            query_encoder=query_encoder,
            caption_encoder=caption_encoder,
            image_encoder=image_encoder,
        )

    cache = {
        "splits": splits,
        "meta": {
            "m_source": str(STAGE1_SIGNAL_CONFIG.train_scores_path),
            "m_used_for_training": "m_prob",
            "q_raw_dim": int(splits["train"]["q_raw"].size(-1)),
            "shared_dim": int(splits["train"]["i"].size(-1)),
        },
    }

    RANKING_PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cache, RANKING_PATHS.cache_path)

    print(f"\nSaved cache: {RANKING_PATHS.cache_path}")

    return cache


def load_or_build_cache(device: torch.device) -> dict[str, Any]:
    should_build = (
        RANKING_MODEL_CONFIG.force_rebuild_cache
        or not RANKING_PATHS.cache_path.exists()
    )

    if should_build:
        cache = build_cache(device)
    else:
        print(f"\nLoading cache: {RANKING_PATHS.cache_path}")
        cache = torch.load(RANKING_PATHS.cache_path, map_location="cpu")

    print_cache_summary(cache)
    return cache


def combined_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    gold_order: torch.Tensor,
) -> torch.Tensor:
    target_distribution = F.softmax(relevance.float(), dim=-1)
    log_probs = F.log_softmax(scores, dim=-1)
    lw_loss = -(target_distribution * log_probs).sum(dim=-1).mean()

    ce_loss = F.cross_entropy(scores, gold_order[:, 0])

    return 0.3 * lw_loss + 0.7 * ce_loss


def top1(scores: torch.Tensor, gold_order: torch.Tensor) -> float:
    predicted_top = scores.argmax(dim=-1)
    gold_top = gold_order[:, 0]

    return (predicted_top == gold_top).float().mean().item()


def ndcg(scores: torch.Tensor, relevance: torch.Tensor, k: int = 5) -> float:
    k = min(k, scores.size(1))

    ranked_indices = scores.argsort(dim=-1, descending=True)[:, :k]
    ranked_relevance = relevance.gather(1, ranked_indices)
    ideal_relevance = relevance.sort(dim=-1, descending=True).values[:, :k]

    discounts = torch.log2(
        torch.arange(k, device=scores.device).float() + 2.0
    )

    dcg = ((2.0**ranked_relevance - 1.0) / discounts).sum(dim=-1)
    idcg = ((2.0**ideal_relevance - 1.0) / discounts).sum(dim=-1)

    return (dcg / idcg.clamp(min=1e-9)).mean().item()


def build_modules(
    config: TrainConfig,
    q_dim: int,
    device: torch.device,
) -> dict[str, nn.Module]:
    query_projection = nn.Sequential(
        nn.Linear(q_dim, RANKING_MODEL_CONFIG.query_dim),
        nn.LayerNorm(RANKING_MODEL_CONFIG.query_dim),
        nn.GELU(),
        nn.Dropout(RANKING_MODEL_CONFIG.query_dropout),
    ).to(device)

    query_fusion = build_query_fusion(
        QueryFusionConfig(
            shared_dim=RANKING_MODEL_CONFIG.query_dim,
            m_embedding_dim=config.m_embedding_dim,
            fusion_type=config.fusion_type,
        )
    ).to(device)

    return {
        "query_projection": query_projection,
        "query_fusion": query_fusion,
        "candidate_fusion": GatedCandidateFusion().to(device),
        "ranker": CosineRanker(
            learnable=config.logit_scale_learnable,
        ).to(device),
    }


def forward_pass(
    batch: dict[str, torch.Tensor],
    modules: dict[str, nn.Module],
    device: torch.device,
) -> torch.Tensor:
    q_raw = batch["q_raw"].to(device)

    # Predicted Stage-1 idiomaticity signal. Gold M is not used for training.
    predicted_m = batch["m_prob"].to(device)

    q_projected = modules["query_projection"](q_raw)
    q_fused = modules["query_fusion"](q_projected, predicted_m)

    candidate_repr = modules["candidate_fusion"](
        batch["i"].to(device),
        batch["t"].to(device),
    )

    return modules["ranker"](q_fused, candidate_repr)


def train_epoch(
    modules: dict[str, nn.Module],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    for module in modules.values():
        module.train()

    total_loss = 0.0
    total_count = 0

    for batch in tqdm(
        loader,
        desc=f"train {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    ):
        relevance = batch["gold_relevance"].to(device)
        gold_order = batch["gold_order"].to(device)

        optimizer.zero_grad(set_to_none=True)
        scores = forward_pass(batch, modules, device)
        loss = combined_loss(scores, relevance, gold_order)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * relevance.size(0)
        total_count += relevance.size(0)

    return total_loss / max(total_count, 1)


def evaluate(
    modules: dict[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    for module in modules.values():
        module.eval()

    losses: list[float] = []
    top1_scores: list[float] = []
    ndcg_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            relevance = batch["gold_relevance"].to(device)
            gold_order = batch["gold_order"].to(device)

            scores = forward_pass(batch, modules, device)

            losses.append(combined_loss(scores, relevance, gold_order).item())
            top1_scores.append(top1(scores, gold_order))
            ndcg_scores.append(ndcg(scores, relevance))

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "top1": sum(top1_scores) / max(len(top1_scores), 1),
        "ndcg": sum(ndcg_scores) / max(len(ndcg_scores), 1),
    }


def run_training(
    config: TrainConfig,
    cache: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    set_seed(config.seed)

    modules = build_modules(config, cache["meta"]["q_raw_dim"], device)

    optimizer = torch.optim.AdamW(
        [
            parameter
            for module in modules.values()
            for parameter in module.parameters()
            if parameter.requires_grad
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    loaders = {
        split_name: build_loader(
            payload=payload,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            seed=config.seed,
        )
        for split_name, payload in cache["splits"].items()
    }

    best_ndcg = -1.0
    best_state: dict[str, Any] | None = None
    best_dev: dict[str, float] | None = None
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            modules=modules,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=config.epochs,
        )

        dev = evaluate(modules, loaders["dev"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev": dev})

        print(
            f"[{config.name}] epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"dev_top1={dev['top1']:.4f} "
            f"dev_ndcg={dev['ndcg']:.4f}"
        )

        if dev["ndcg"] > best_ndcg:
            best_ndcg = dev["ndcg"]
            best_dev = dev
            best_epoch = epoch
            best_state = {
                name: module.state_dict()
                for name, module in modules.items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint was selected.")

    for name, module in modules.items():
        module.load_state_dict(best_state[name])

    return {
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_dev": best_dev,
        "test": evaluate(modules, loaders["test"], device),
        "xeval": evaluate(modules, loaders["xeval"], device),
        "history": history,
        "state_dicts": best_state,
    }


def objective(
    trial: optuna.Trial,
    cache: dict[str, Any],
    device: torch.device,
) -> float:
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-5,
            1.5e-4,
            log=True,
        ),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.01, 0.05]),
    }

    scores: list[float] = []

    for seed in SEEDS:
        config = TrainConfig(
            name=f"trial_{trial.number}_seed_{seed}",
            seed=seed,
            **params,
        )
        result = run_training(config, cache, device)
        scores.append(result["best_dev"]["ndcg"])

    return sum(scores) / len(scores)


def tune(cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, cache, device),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    return {
        "batch_size": study.best_params["batch_size"],
        "learning_rate": study.best_params["learning_rate"],
        "weight_decay": study.best_params["weight_decay"],
    }


def final_train(
    best_params: dict[str, Any],
    cache: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    dev_scores: list[float] = []

    for seed in SEEDS:
        config = TrainConfig(
            name=f"final_seed_{seed}",
            seed=seed,
            **best_params,
        )
        result = run_training(config, cache, device)
        results.append(result)
        dev_scores.append(result["best_dev"]["ndcg"])

    best_idx = dev_scores.index(max(dev_scores))

    return {
        "best_params": best_params,
        "results": results,
        "best_idx": best_idx,
        "dev_scores": dev_scores,
        "mean_dev_ndcg": sum(dev_scores) / len(dev_scores),
    }


def save_results(result: dict[str, Any]) -> None:
    RANKING_PATHS.output_dir.mkdir(parents=True, exist_ok=True)

    best_idx = result["best_idx"]
    best_seed = SEEDS[best_idx]
    best_weights = result["results"][best_idx]["state_dicts"]

    torch.save(best_weights, RANKING_PATHS.weights_path)
    save_json(result["best_params"], RANKING_PATHS.best_params_path)

    summary = {
        "cache_path": str(RANKING_PATHS.cache_path),
        "m_scores_path": str(STAGE1_SIGNAL_CONFIG.train_scores_path),
        "m_used_for_training": "predicted m_prob",
        "ranking_model_config": {
            "query_dim": RANKING_MODEL_CONFIG.query_dim,
            "query_dropout": RANKING_MODEL_CONFIG.query_dropout,
            "fusion_type": RANKING_MODEL_CONFIG.fusion_type,
            "m_embedding_dim": RANKING_MODEL_CONFIG.m_embedding_dim,
            "logit_scale_learnable": RANKING_MODEL_CONFIG.logit_scale_learnable,
        },
        "best_params": result["best_params"],
        "best_seed": best_seed,
        "mean_dev_ndcg": result["mean_dev_ndcg"],
        "dev_scores": {
            str(seed): score
            for seed, score in zip(SEEDS, result["dev_scores"])
        },
        "best_dev": result["results"][best_idx]["best_dev"],
        "test": result["results"][best_idx]["test"],
        "xeval": result["results"][best_idx]["xeval"],
    }

    save_json(summary, RANKING_PATHS.summary_path)

    print("\n[SAVED]")
    print(f"Weights: {RANKING_PATHS.weights_path}")
    print(f"Params : {RANKING_PATHS.best_params_path}")
    print(f"Summary: {RANKING_PATHS.summary_path}")


def main() -> None:
    device = get_device()

    print("\n" + "=" * 70)
    print("[START] Ranking training")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Cache         : {RANKING_PATHS.cache_path}")
    print(f"M scores      : {STAGE1_SIGNAL_CONFIG.train_scores_path}")
    print(f"Output dir    : {RANKING_PATHS.output_dir}")
    print(f"Fusion        : {RANKING_MODEL_CONFIG.fusion_type}")
    print(f"M dim         : {RANKING_MODEL_CONFIG.m_embedding_dim}")
    print(f"Query dim     : {RANKING_MODEL_CONFIG.query_dim}")
    print(f"Rebuild cache : {RANKING_MODEL_CONFIG.force_rebuild_cache}")
    print("=" * 70)

    cache = load_or_build_cache(device)
    best_params = tune(cache, device)
    result = final_train(best_params, cache, device)

    save_results(result)

    print("\n[DONE] Ranking training finished")


if __name__ == "__main__":
    main()