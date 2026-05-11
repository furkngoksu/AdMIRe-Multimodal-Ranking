from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ranking_experiments.losses import listwise_ranking_loss
from ranking_experiments.metrics import ndcg_at_k, top1_accuracy
from src.utils import RankingExperimentConfig, load_cache, save_json, build_query_projection, load_weights, save_weights

SHARED_DIM = 512

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

class FrozenPCAProjection(nn.Module):
    def __init__(self, mean: torch.Tensor, components: torch.Tensor, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)

    def forward(self, q_raw: torch.Tensor) -> torch.Tensor:
        projected = (q_raw - self.mean) @ self.components.t()
        if projected.size(-1) < self.output_dim:
            padding = projected.new_zeros(projected.size(0), self.output_dim - projected.size(-1))
            projected = torch.cat([projected, padding], dim=-1)
        return projected[:, : self.output_dim]

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def build_loader(split_payload: dict[str, Any], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(CachedRankingDataset(split_payload), batch_size=batch_size, shuffle=shuffle, generator=generator)

def fit_pca_projection(q_raw: torch.Tensor, output_dim: int, device: torch.device) -> FrozenPCAProjection:
    mean = q_raw.mean(dim=0, keepdim=True)
    centered = q_raw - mean
    max_components = max(1, min(output_dim, centered.size(0) - 1, centered.size(1)))
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    components = vh[:max_components]
    return FrozenPCAProjection(mean=mean.to(device), components=components.to(device), output_dim=output_dim)

def build_query_projection(experiment: RankingExperimentConfig, train_q_raw: torch.Tensor, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if experiment.query_projection == "linear_trainable":
        projection = nn.Sequential(
            nn.Linear(train_q_raw.size(-1), SHARED_DIM),
            nn.LayerNorm(SHARED_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        ).to(device)
        return projection, {"query_projection": "linear_trainable", "trainable": True, "raw_dim": train_q_raw.size(-1), "output_dim": SHARED_DIM}
    projection = fit_pca_projection(q_raw=train_q_raw, output_dim=SHARED_DIM, device=device)
    return projection, {"query_projection": "pca_frozen", "trainable": False, "raw_dim": train_q_raw.size(-1), "samples": train_q_raw.size(0), "components": projection.components.size(0), "output_dim": projection.output_dim}

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
