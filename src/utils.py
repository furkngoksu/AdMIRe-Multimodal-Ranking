from __future__ import annotations
import csv
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

def read_m_scores(path: Path | str) -> dict[str, float]:
    """Read M scores from a TSV file and return a mapping of sample_id to score."""
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            scores[row["sample_id"]] = float(row["m_prob"])
    return scores

@dataclass(frozen=True)
class RankingExperimentConfig:
    name: str
    m_embedding_dim: int
    seed: int
    query_projection: str = "pca_frozen"
    m_source: str = "predicted"
    fusion_type: str = "concat"
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    logit_scale_learnable: bool = False
    model_name: str = "xlm-roberta-base"
    shared_dim: int = 512
    max_length: int = 128
    debug: bool = False

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

def load_cache():
    project_root = Path(__file__).resolve().parents[1]
    cache_path = project_root / "ranking_experiments/caching/cache/admire_rank_cache.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}. Run caching/build_cache.py first.")
    return torch.load(cache_path, map_location="cpu")

def save_json(data: Any, path: Path | str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def build_query_projection(config: RankingExperimentConfig, train_q_raw: torch.Tensor, device: torch.device):
    from src.query_encoder import FrozenPCAProjection
    from sklearn.decomposition import PCA
    import numpy as np
    
    n_features = train_q_raw.shape[1]
    target_dim = 512
    
    if config.query_projection == "pca_frozen":
        # No patches, just standard PCA fit
        pca = PCA(n_components=min(train_q_raw.shape[0], target_dim))
        pca.fit(train_q_raw.cpu().numpy())
        
        # Use existing FrozenPCAProjection from query_encoder.py
        projection = FrozenPCAProjection(
            mean=torch.from_numpy(pca.mean_).float().unsqueeze(0).to(device),
            components=torch.from_numpy(pca.components_).float().to(device),
            output_dim=target_dim
        )
        return projection, target_dim
    else:
        projection = nn.Linear(n_features, target_dim).to(device)
        return projection, target_dim

def save_weights(modules: dict[str, nn.Module], path: str | Path):
    weights = {name: module.state_dict() for name, module in modules.items()}
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, path_obj)

def load_weights(modules: dict[str, nn.Module], path: str | Path, device: torch.device):
    weights = torch.load(path, map_location=device)
    for name, module in modules.items():
        module.load_state_dict(weights[name])
