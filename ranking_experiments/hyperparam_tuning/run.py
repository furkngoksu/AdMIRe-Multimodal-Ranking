import json
import os
import sys
from pathlib import Path
import optuna
import torch
import torch.nn as nn

# Project path configuration
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
    save_weights,
)
from src.config import QueryFusionConfig, get_device
from src.query_fusion import build_query_fusion
from src.candidate_fusion import GatedCandidateFusion
from src.ranker import CosineRanker

# --- GLOBAL CONFIGURATION ---
# Set to "linear_trainable" for the new sequential head or "pca_frozen" for the old baseline
PROJECTION_TYPE = "linear_trainable"
LEARNABLE_LOGIT_SCALE = True

SEEDS = (42, 100, 150, 200, 250)
N_TRIALS = 30
TUNING_DIR = project_root / f"ranking_experiments/hyperparam_tuning/{PROJECTION_TYPE.split('_')[0]}"
FINAL_MODEL_DIR = TUNING_DIR / "final_model"
BASELINE_DIR = TUNING_DIR / "baseline"

# Based on Fusion Ablation results, our selected architecture:
SELECTED_ARCHITECTURE = "film_full" 

def load_cache():
    cache_path = project_root / "ranking_experiments/caching/cache/admire_rank_cache.pt"
    return torch.load(cache_path, map_location="cpu")

def build_loaders(cache, batch_size, seed):
    splits = cache["splits"]
    return {
        "train": build_loader(splits["train"], batch_size, True, seed),
        "dev": build_loader(splits["dev"], batch_size, False, seed),
        "test": build_loader(splits["test"], batch_size, False, seed),
        "xeval": build_loader(splits["xeval"], batch_size, False, seed),
    }

def run_training_flow(experiment: RankingExperimentConfig, output_dir: Path, loaders: dict, train_q_raw: torch.Tensor, device: torch.device, save_to_disk: bool = False):
    """Core training logic for both Baseline and Tuned runs."""
    set_seed(experiment.seed)
    
    # Initialize Multimodal Ranking components
    query_projection, _ = build_query_projection(experiment, train_q_raw, device)
    query_fusion = build_query_fusion(QueryFusionConfig(m_embedding_dim=experiment.m_embedding_dim, fusion_type=experiment.fusion_type)).to(device)
    candidate_fusion = GatedCandidateFusion().to(device)
    ranker = CosineRanker(learnable=experiment.logit_scale_learnable).to(device)
    
    trained_modules = {
        "query_projection": query_projection, 
        "query_fusion": query_fusion, 
        "candidate_fusion": candidate_fusion, 
        "ranker": ranker
    }
    
    # Collect all trainable parameters from all modules
    params = []
    for mod in trained_modules.values():
        params.extend([p for p in mod.parameters() if p.requires_grad])
        
    optimizer = torch.optim.AdamW(params, lr=experiment.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=experiment.epochs)
    
    best_ndcg = -1.0
    best_metrics = None
    best_state_dicts = None

    for epoch in range(1, experiment.epochs + 1):
        train_one_epoch(trained_modules, loaders["train"], optimizer, device, epoch, experiment)
        dev_metrics = evaluate(trained_modules, loaders["dev"], "dev", experiment.m_source, device)
        scheduler.step()
        
        if dev_metrics["ndcg"] > best_ndcg:
            best_ndcg = dev_metrics["ndcg"]
            best_metrics = dev_metrics
            best_state_dicts = {name: mod.state_dict() for name, mod in trained_modules.items()}

    # Load best weights for final evaluation
    if best_state_dicts:
        for name, mod in trained_modules.items(): 
            mod.load_state_dict(best_state_dicts[name])
            
    test_metrics = evaluate(trained_modules, loaders["test"], "test", experiment.m_source, device)
    xeval_metrics = evaluate(trained_modules, loaders["xeval"], "xeval", experiment.m_source, device)

    if save_to_disk and best_state_dicts:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_weights(trained_modules, output_dir / "weights.pt")
        save_json({
            "experiment": experiment.to_dict(), 
            "best_dev": best_metrics,
            "test": test_metrics,
            "xeval": xeval_metrics
        }, output_dir / "results.json")
    
    return {
        "experiment": experiment.to_dict(), 
        "best_dev": best_metrics, 
        "test": test_metrics, 
        "xeval": xeval_metrics
    }

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    device = get_device()
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    
    config = RankingExperimentConfig(
        name=f"trial_{trial.number}",
        m_embedding_dim=64,
        seed=42,
        fusion_type=SELECTED_ARCHITECTURE,
        epochs=trial.suggest_int("epochs", 30, 50),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        learning_rate=trial.suggest_float("learning_rate", 8e-5, 5e-4, log=True),
        query_projection=PROJECTION_TYPE,
        logit_scale_learnable=LEARNABLE_LOGIT_SCALE
    )
    
    loaders = build_loaders(cache, config.batch_size, config.seed)
    result = run_training_flow(config, TUNING_DIR / f"trial_{trial.number}", loaders, train_q_raw, device, save_to_disk=False)
    return result["best_dev"]["ndcg"]

def aggregate_results(results):
    best_dev = {"loss": sum(r["best_dev"]["loss"] for r in results) / len(results), "top1": sum(r["best_dev"]["top1"] for r in results) / len(results), "ndcg": sum(r["best_dev"]["ndcg"] for r in results) / len(results)}
    test = {"loss": sum(r["test"]["loss"] for r in results) / len(results), "top1": sum(r["test"]["top1"] for r in results) / len(results), "ndcg": sum(r["test"]["ndcg"] for r in results) / len(results)}
    xeval = {"loss": sum(r["xeval"]["loss"] for r in results) / len(results), "top1": sum(r["xeval"]["top1"] for r in results) / len(results), "ndcg": sum(r["xeval"]["ndcg"] for r in results) / len(results)}
    return {"best_dev": best_dev, "test": test, "xeval": xeval}

def main():
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    device = get_device()

    # --- PHASE 1: BASELINE (Selected Architecture with Default Params) ---
    print(f"\n[PHASE 1] Training Baseline ({SELECTED_ARCHITECTURE} with Default Params)...")
    baseline_runs = []
    for seed in SEEDS:
        exp = RankingExperimentConfig(
            name=f"baseline_s{seed}", 
            m_embedding_dim=64, 
            seed=seed, 
            fusion_type=SELECTED_ARCHITECTURE, 
            epochs=20, 
            batch_size=32, # Increased default batch size
            learning_rate=1e-4,
            query_projection=PROJECTION_TYPE,
            logit_scale_learnable=LEARNABLE_LOGIT_SCALE
        )
        loaders = build_loaders(cache, 32, seed)
        res = run_training_flow(exp, BASELINE_DIR / f"seed{seed}", loaders, train_q_raw, device, save_to_disk=True)
        baseline_runs.append(res)
    save_json({"aggregate": [aggregate_results(baseline_runs)], "runs": baseline_runs}, BASELINE_DIR / "summary.json")

    # --- PHASE 2: OPTUNA TUNING (Selected Architecture) ---
    print(f"\n[PHASE 2] Tuning Hyperparameters for {SELECTED_ARCHITECTURE}...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_params = study.best_params
    save_json({"best_params": best_params, "best_value": study.best_value}, TUNING_DIR / "best_params.json")

    # --- PHASE 3: FINAL TRAINING (Best Hyperparams) ---
    print(f"\n[PHASE 3] Final Training: Optimized {SELECTED_ARCHITECTURE}...")
    final_runs = []
    for seed in SEEDS:
        exp = RankingExperimentConfig(
            name=f"tuned_s{seed}",
            m_embedding_dim=64,
            seed=seed,
            fusion_type=SELECTED_ARCHITECTURE,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            learning_rate=best_params["learning_rate"],
            query_projection=PROJECTION_TYPE,
            logit_scale_learnable=LEARNABLE_LOGIT_SCALE
        )
        loaders = build_loaders(cache, best_params["batch_size"], seed)
        res = run_training_flow(exp, FINAL_MODEL_DIR / f"seed{seed}", loaders, train_q_raw, device, save_to_disk=True)
        final_runs.append(res)

    save_json({"aggregate": [aggregate_results(final_runs)], "runs": final_runs}, FINAL_MODEL_DIR / "summary.json")
    print(f"\n[DONE] Results and models saved to {TUNING_DIR}")

if __name__ == "__main__":
    main()
