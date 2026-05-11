import os
import sys
from pathlib import Path
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ranking_experiments.utils import (
    RankingExperimentConfig,
    train_one_epoch,
    evaluate,
    set_seed,
    build_loader,
    build_query_projection,
    save_json,
    load_weights,
    save_weights,
)
from src.config import QueryFusionConfig, get_device
from src.query_fusion import build_query_fusion
from src.candidate_fusion import GatedCandidateFusion
from src.ranker import CosineRanker

def load_cache():
    cache_path = project_root / "ranking_experiments/caching/cache/admire_rank_cache.pt"
    return torch.load(cache_path, map_location="cpu")

def run_experiment(config: RankingExperimentConfig, device: torch.device):
    print(f"\n[experiment] Starting: {config.name}")
    set_seed(config.seed)
    
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    
    query_projection, _ = build_query_projection(config, train_q_raw, device)
    query_fusion = build_query_fusion(QueryFusionConfig(m_embedding_dim=config.m_embedding_dim, fusion_type=config.fusion_type)).to(device)
    candidate_fusion, ranker = GatedCandidateFusion().to(device), CosineRanker().to(device)
    
    loaders = {
        "train": build_loader(cache["splits"]["train"], config.batch_size, True, config.seed),
        "dev": build_loader(cache["splits"]["dev"], config.batch_size, False, config.seed),
        "test": build_loader(cache["splits"]["test"], config.batch_size, False, config.seed),
    }
    
    trained_modules = {"query_projection": query_projection, "query_fusion": query_fusion, "candidate_fusion": candidate_fusion}
    optimizer = torch.optim.AdamW(list(query_fusion.parameters()) + list(candidate_fusion.parameters()), lr=config.learning_rate)
    
    best_ndcg = -1.0
    best_metrics = None
    
    for epoch in range(1, config.epochs + 1):
        train_one_epoch(trained_modules, loaders["train"], optimizer, device)
        dev_metrics = evaluate(trained_modules, ranker, loaders["dev"], device)
        if dev_metrics["ndcg"] > best_ndcg:
            best_ndcg = dev_metrics["ndcg"]
            best_metrics = dev_metrics
            
    test_metrics = evaluate(trained_modules, ranker, loaders["test"], device)
    return {"name": config.name, "best_dev": best_metrics, "test": test_metrics}

def main():
    device = get_device()
    output_dir = project_root / "ranking_experiments/fusion_ablation/output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for f_type in ["concat", "gated", "film", "film_full"]:
        for m_dim in [64]:
            config = RankingExperimentConfig(name=f"{f_type}_m{m_dim}", m_embedding_dim=m_dim, seed=42, fusion_type=f_type, m_source="gold")
            results.append(run_experiment(config, device))
            
    save_json({"runs": results}, output_dir / "summary.json")

if __name__ == "__main__":
    main()
