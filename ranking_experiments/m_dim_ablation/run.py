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

# --- GLOBAL CONFIGURATION ---
# Trainable Linear Projection Head is hard-coded to theoretically resolve the Modality Gap
PROJECTION_TYPE = "linear_trainable"
LEARNABLE_LOGIT_SCALE = True
BASE_DIR = project_root / "ranking_experiments/m_dim_ablation"

SEEDS = (7, 13, 21, 42, 100)
M_DIMS = (0, 8, 16, 32, 64, 128, 256)


def load_cache():
    cache_path = project_root / "ranking_experiments/caching/cache/admire_rank_cache.pt"
    return torch.load(cache_path, map_location="cpu")


def run_experiment(config: RankingExperimentConfig, output_dir: Path, device: torch.device) -> dict:
    print(f"\n[experiment] Starting: {config.name} (seed={config.seed}, m_dim={config.m_embedding_dim})")
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    
    query_projection, _ = build_query_projection(config, train_q_raw, device)
    query_fusion = build_query_fusion(QueryFusionConfig(m_embedding_dim=config.m_embedding_dim, fusion_type=config.fusion_type)).to(device)
    candidate_fusion = GatedCandidateFusion().to(device)
    ranker = CosineRanker(learnable=config.logit_scale_learnable).to(device)
    
    loaders = {
        "train": build_loader(cache["splits"]["train"], config.batch_size, True, config.seed),
        "dev": build_loader(cache["splits"]["dev"], config.batch_size, False, config.seed),
        "test": build_loader(cache["splits"]["test"], config.batch_size, False, config.seed),
        "xeval": build_loader(cache["splits"]["xeval"], config.batch_size, False, config.seed),
    }
    
    trained_modules = {
        "query_projection": query_projection,
        "query_fusion": query_fusion,
        "candidate_fusion": candidate_fusion,
        "ranker": ranker,
    }
    
    # Collect all trainable parameters from all modules
    params = []
    for mod in trained_modules.values():
        params.extend([p for p in mod.parameters() if p.requires_grad])
        
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    
    best_ndcg = -1.0
    best_metrics = None
    best_state_dicts = None
    best_epoch = 1
    history = []
    
    # --- PRODUCTION-GRADE ENHANCEMENT: EARLY STOPPING CONFIGURATION ---
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, config.epochs + 1):
        train_one_epoch(trained_modules, loaders["train"], optimizer, device, epoch, config)
        dev_metrics = evaluate(trained_modules, loaders["dev"], "dev", config.m_source, device)
        history.append({"epoch": epoch, "dev": dev_metrics})
        
        if dev_metrics["ndcg"] > best_ndcg:
            best_ndcg = dev_metrics["ndcg"]
            best_metrics = dev_metrics
            best_epoch = epoch
            best_state_dicts = {name: mod.state_dict() for name, mod in trained_modules.items()}
            patience_counter = 0 # Generalization improved, reset counter
        else:
            patience_counter += 1 # No improvement, advance counter
            if patience_counter >= patience:
                print(f"      [Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
                break # Exit training loop for current seed run

    # Reload optimal weights for final split alignment testing
    if best_state_dicts:
        for name, mod in trained_modules.items():
            mod.load_state_dict(best_state_dicts[name])
            
    test_metrics = evaluate(trained_modules, loaders["test"], "test", config.m_source, device)
    xeval_metrics = evaluate(trained_modules, loaders["xeval"], "xeval", config.m_source, device)
    
    save_weights(trained_modules, output_dir / "weights.pt")
    save_json(config.to_dict(), output_dir / "config.json")
    
    single_run = {
        "experiment": config.to_dict(),
        "history": history,
        "best_dev": best_metrics, 
        "test": test_metrics, 
        "xeval": xeval_metrics,
        "best_epoch": best_epoch,
    }
    save_json(single_run, output_dir / "results.json")
    
    return single_run


def compute_aggregate(runs_for_m_dim: list[dict], m_dim: int) -> dict:
    num_seeds = len(runs_for_m_dim)
    avg_best_epoch = sum(r["best_epoch"] for r in runs_for_m_dim) / num_seeds
    
    agg = {
        "m_dim": m_dim,
        "num_seeds": num_seeds,
        "best_epoch": avg_best_epoch,
        "best_dev": {"loss": 0.0, "top1": 0.0, "ndcg": 0.0},
        "test": {"loss": 0.0, "top1": 0.0, "ndcg": 0.0},
        "xeval": {"loss": 0.0, "top1": 0.0, "ndcg": 0.0},
    }
    
    for split_name in ("best_dev", "test", "xeval"):
        for metric_name in ("loss", "top1", "ndcg"):
            total = sum(r[split_name][metric_name] for r in runs_for_m_dim)
            agg[split_name][metric_name] = total / num_seeds
            
    return agg


def main():
    device = get_device()
    output_base = BASE_DIR / "output"
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_runs = []
    all_aggregates = []
    
    for m_dim in M_DIMS:
        m_dim_runs = []
        
        # Explicitly map m=0 to Baseline without referencing Concat, while assigning champion FiLM Full for active capacity scaling runs
        fusion_type = "concat" if m_dim == 0 else "film_full"
        exp_name = "Baseline (m=0)" if m_dim == 0 else f"m{m_dim}"
        
        for seed in SEEDS:
            config = RankingExperimentConfig(
                name=f"{exp_name}_seed{seed}",
                m_embedding_dim=m_dim,
                seed=seed,
                query_projection=PROJECTION_TYPE,
                m_source="gold",
                fusion_type=fusion_type,
                epochs=20,
                batch_size=4,
                learning_rate=1e-4,
                logit_scale_learnable=LEARNABLE_LOGIT_SCALE,
            )
            
            run_output_dir = output_base / f"m{m_dim}" / f"seed{seed}"
            single_run = run_experiment(config, run_output_dir, device)
            m_dim_runs.append(single_run)
            all_runs.append(single_run)
            
        agg = compute_aggregate(m_dim_runs, m_dim)
        all_aggregates.append(agg)
        
    summary_payload = {
        "runs": all_runs,
        "aggregate": all_aggregates,
    }
    save_json(summary_payload, output_base / "summary.json")
    print(f"\n[success] Flat M-Dimension Capacity Ablation completed. Results written to {output_base / 'summary.json'}")


if __name__ == "__main__":
    main()
