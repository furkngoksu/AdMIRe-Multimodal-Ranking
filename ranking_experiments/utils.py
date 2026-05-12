from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.utils import RankingExperimentConfig, load_cache, load_weights, save_weights

SHARED_DIM = 512


def listwise_ranking_loss(
    scores: torch.Tensor,
    gold_relevance: torch.Tensor,
) -> torch.Tensor:
    """
    Listwise ranking loss for one query and five candidates.

    Args:
        scores: [B, K] model scores.
        gold_relevance: [B, K] per-candidate relevance values.

    Returns:
        Scalar loss.
    """
    if scores.shape != gold_relevance.shape:
        raise ValueError("scores and gold_relevance must have the same shape.")

    target_distribution = F.softmax(gold_relevance.float(), dim=-1)
    log_scores = F.log_softmax(scores, dim=-1)

    return -(target_distribution * log_scores).sum(dim=-1).mean()


def top1_accuracy(
    scores: torch.Tensor,
    gold_order_indices: torch.Tensor,
) -> float:
    """Top-1 image accuracy for ranked candidate scores."""
    predicted_top = scores.argmax(dim=-1)
    gold_top = gold_order_indices[:, 0]

    return (predicted_top == gold_top).float().mean().item()


def ndcg_at_k(
    scores: torch.Tensor,
    gold_relevance: torch.Tensor,
    k: int = 5,
) -> float:
    """Mean nDCG@k for candidate ranking scores."""
    if scores.shape != gold_relevance.shape:
        raise ValueError("scores and gold_relevance must have the same shape.")

    k = min(k, scores.size(1))

    ranked_indices = scores.argsort(dim=-1, descending=True)[:, :k]
    ranked_relevance = gold_relevance.gather(1, ranked_indices)

    ideal_relevance = gold_relevance.sort(dim=-1, descending=True).values[:, :k]

    discounts = torch.log2(
        torch.arange(k, device=scores.device, dtype=torch.float) + 2.0
    )

    dcg = ((2.0**ranked_relevance - 1.0) / discounts).sum(dim=-1)
    ideal_dcg = ((2.0**ideal_relevance - 1.0) / discounts).sum(dim=-1)

    return (dcg / ideal_dcg.clamp(min=1e-9)).mean().item()


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
            "m_gold": self.payload["m_gold"][index],
            "gold_relevance": self.payload["gold_relevance"][index],
            "gold_order": self.payload["gold_order"][index],
        }

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def build_loader(split_payload: dict[str, Any], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(CachedRankingDataset(split_payload), batch_size=batch_size, shuffle=shuffle, generator=generator)

def build_query_projection(experiment: RankingExperimentConfig, train_q_raw: torch.Tensor, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    # Only linear_trainable is supported after removing legacy PCA
    projection = nn.Sequential(
        nn.Linear(train_q_raw.size(-1), SHARED_DIM),
        nn.LayerNorm(SHARED_DIM),
        nn.GELU(),
        nn.Dropout(0.1),
    ).to(device)
    return projection, {"query_projection": "linear_trainable", "trainable": True, "raw_dim": train_q_raw.size(-1), "output_dim": SHARED_DIM}

def forward_scores(batch: dict[str, torch.Tensor], m_source: str, query_projection: nn.Module, query_fusion: nn.Module, candidate_fusion: nn.Module, ranker: nn.Module, device: torch.device) -> torch.Tensor:
    q_raw = batch["q_raw"].to(device)
    m_tensor = batch["m_prob"].to(device) if m_source == "predicted" else batch["m_gold"].to(device)
    q = query_projection(q_raw)
    q_prime = query_fusion(q, m_tensor)
    candidate_repr = candidate_fusion(batch["i"].to(device), batch["t"].to(device))
    return ranker(q_prime, candidate_repr)

def train_one_epoch(modules: dict[str, nn.Module], loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, experiment: RankingExperimentConfig) -> float:
    query_projection, query_fusion, candidate_fusion, ranker = modules["query_projection"], modules["query_fusion"], modules["candidate_fusion"], modules.get("ranker")
    if experiment.query_projection == "linear_trainable": query_projection.train()
    else: query_projection.eval()
    query_fusion.train(); candidate_fusion.train()
    total_loss, total_examples = 0.0, 0
    progress = tqdm(loader, desc=f"{experiment.name} train {epoch}/{experiment.epochs}", dynamic_ncols=True)
    for batch in progress:
        gold_relevance = batch["gold_relevance"].to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = forward_scores(batch, experiment.m_source, query_projection, query_fusion, candidate_fusion, ranker, device)
        loss = listwise_ranking_loss(scores, gold_relevance)
        loss.backward()
        optimizer.step()
        
        # Removed clamping to allow model to find its own optimum logit scale
        
        total_loss += loss.item() * gold_relevance.size(0); total_examples += gold_relevance.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / total_examples

def evaluate(modules: dict[str, nn.Module], loader: DataLoader, split_name: str, m_source: str, device: torch.device) -> dict[str, float]:
    query_projection, query_fusion, candidate_fusion, ranker = modules["query_projection"], modules["query_fusion"], modules["candidate_fusion"], modules["ranker"]
    query_projection.eval(); query_fusion.eval(); candidate_fusion.eval(); ranker.eval()
    losses, top1_values, ndcg_values = [], [], []
    with torch.no_grad():
        progress = tqdm(loader, desc=f"ranking {split_name}", dynamic_ncols=True)
        for batch in progress:
            gold_relevance, gold_order = batch["gold_relevance"].to(device), batch["gold_order"].to(device)
            scores = forward_scores(batch, m_source, query_projection, query_fusion, candidate_fusion, ranker, device)
            losses.append(listwise_ranking_loss(scores, gold_relevance).item())
            top1_values.append(top1_accuracy(scores, gold_order)); ndcg_values.append(ndcg_at_k(scores, gold_relevance))
    return {"loss": sum(losses)/len(losses), "top1": sum(top1_values)/len(top1_values), "ndcg": sum(ndcg_values)/len(ndcg_values)}

def save_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(payload, f, indent=2, ensure_ascii=False)
