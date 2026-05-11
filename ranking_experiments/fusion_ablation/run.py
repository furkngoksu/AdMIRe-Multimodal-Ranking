import os
import sys
import json
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

def run_experiment(config: RankingExperimentConfig, output_dir: Path, device: torch.device):
    print(f"\n[experiment] Starting: {config.name}")
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    
    query_projection, _ = build_query_projection(config, train_q_raw, device)
    query_fusion = build_query_fusion(QueryFusionConfig(m_embedding_dim=config.m_embedding_dim, fusion_type=config.fusion_type)).to(device)
    candidate_fusion, ranker = GatedCandidateFusion().to(device), CosineRanker().to(device)
    
    loaders = {
        "train": build_loader(cache["splits"]["train"], config.batch_size, True, config.seed),
        "dev": build_loader(cache["splits"]["dev"], config.batch_size, False, config.seed),
        "test": build_loader(cache["splits"]["test"], config.batch_size, False, config.seed),
        "xeval": build_loader(cache["splits"]["xeval"], config.batch_size, False, config.seed),
    }
    
    trained_modules = {"query_projection": query_projection, "query_fusion": query_fusion, "candidate_fusion": candidate_fusion, "ranker": ranker}
    optimizer = torch.optim.AdamW(list(query_fusion.parameters()) + list(candidate_fusion.parameters()), lr=config.learning_rate)
    
    best_ndcg = -1.0
    best_metrics = None
    best_state_dicts = None
    
    for epoch in range(1, config.epochs + 1):
        train_one_epoch(trained_modules, loaders["train"], optimizer, device, epoch, config)
        dev_metrics = evaluate(trained_modules, loaders["dev"], "dev", config.m_source, device)
        if dev_metrics["ndcg"] > best_ndcg:
            best_ndcg = dev_metrics["ndcg"]
            best_metrics = dev_metrics
            best_state_dicts = {name: mod.state_dict() for name, mod in trained_modules.items()}

    # Final evaluation on best model
    if best_state_dicts:
        for name, mod in trained_modules.items(): mod.load_state_dict(best_state_dicts[name])
            
    test_metrics = evaluate(trained_modules, loaders["test"], "test", config.m_source, device)
    xeval_metrics = evaluate(trained_modules, loaders["xeval"], "xeval", config.m_source, device)
    
    # Save model and results for this specific config
    save_weights(trained_modules, output_dir / "weights.pt")
    
    # Notebook'un hem 'aggregate' hem de 'runs' bekleyen hücrelerini doyurma
    single_run = {
        "best_dev": best_metrics, 
        "test": test_metrics, 
        "xeval": xeval_metrics,
        "config": config.to_dict()
    }
    
    res_payload = {
        "name": config.name, 
        "runs": [single_run],
        "aggregate": [single_run]
    }
    
    save_json(res_payload, output_dir / "summary.json")
    
    return res_payload

def main():
    device = get_device()
    base_dir = project_root / "ranking_experiments/fusion_ablation"
    
    results = []
    configs = [
        ("concat", "concat_m64"),
        ("gated", "gated_m64"),
        ("film", "film_scale_m64"),
        ("film_full", "film_full_m64")
    ]
    
    for f_type, name in configs:
        config = RankingExperimentConfig(name=name, m_embedding_dim=64, seed=42, fusion_type=f_type, m_source="gold")
        output_path = base_dir / name / "output"
        results.append(run_experiment(config, output_path, device))
            
    # Genel özet
    save_json({"runs": results}, base_dir / "summary.json")

if __name__ == "__main__":
    main()
