import json
import os
import sys
from pathlib import Path
import optuna
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

SEEDS = (42, 100, 150, 200, 250)
N_TRIALS = 30
TUNING_DIR = Path("ranking_experiments/hyperparam_tuning")

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
    set_seed(experiment.seed)
    query_projection, _ = build_query_projection(experiment, train_q_raw, device)
    query_fusion = build_query_fusion(QueryFusionConfig(m_embedding_dim=experiment.m_embedding_dim, fusion_type=experiment.fusion_type)).to(device)
    candidate_fusion, ranker = GatedCandidateFusion().to(device), CosineRanker().to(device)
    
    trainable_params = list(query_fusion.parameters()) + list(candidate_fusion.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=experiment.learning_rate)
    
    best_ndcg = -1.0
    best_metrics, best_state_dicts = None, None
    trained_modules = {"query_projection": query_projection, "query_fusion": query_fusion, "candidate_fusion": candidate_fusion}

    for epoch in range(1, experiment.epochs + 1):
        train_one_epoch(trained_modules, loaders["train"], optimizer, device)
        dev_metrics = evaluate(trained_modules, ranker, loaders["dev"], device)
        if dev_metrics["ndcg"] > best_ndcg:
            best_ndcg = dev_metrics["ndcg"]
            best_metrics = dev_metrics
            best_state_dicts = {name: mod.state_dict() for name, mod in trained_modules.items()}

    if save_to_disk and best_state_dicts:
        save_weights(trained_modules, output_dir / "weights.pt")
        save_json({"experiment": experiment.to_dict(), "best_dev": best_metrics}, output_dir / "results.json")
    
    return {"experiment": experiment.to_dict(), "best_dev": best_metrics, "weights_path": str(output_dir / "weights.pt")}

def objective(trial):
    device = get_device()
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    config = RankingExperimentConfig(
        name=f"trial_{trial.number}",
        m_embedding_dim=64,
        seed=42,
        m_source="gold",
        fusion_type=trial.suggest_categorical("fusion_type", ["concat", "gated", "film", "film_full"]),
        epochs=trial.suggest_int("epochs", 10, 30),
        batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
    )
    loaders = build_loaders(cache, config.batch_size, config.seed)
    result = run_training_flow(config, TUNING_DIR / f"trial_{trial.number}", loaders, train_q_raw, device, save_to_disk=False)
    return result["best_dev"]["ndcg"]

def aggregate_runs(results):
    from statistics import mean
    return {"best_dev": {m: mean(float(run["best_dev"][m]) for run in results) for m in ("loss", "top1", "ndcg")}}

def main():
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    device = get_device()

    print("\n--- Phase 1: Hyperparameter Tuning (Gold M) ---")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    save_json({"best_params": best_params, "score": study.best_value}, TUNING_DIR / "best_params.json")

    print("\n--- Phase 2: Final Training with Best Parameters ---")
    final_results = []
    for seed in SEEDS:
        exp = RankingExperimentConfig(
            name=f"best_model_seed{seed}",
            m_embedding_dim=64,
            seed=seed,
            m_source="gold",
            fusion_type=best_params["fusion_type"],
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            learning_rate=best_params["learning_rate"],
        )
        loaders = build_loaders(cache, best_params["batch_size"], seed)
        res = run_training_flow(exp, TUNING_DIR / "final_model" / f"seed{seed}", loaders, train_q_raw, device, save_to_disk=True)
        final_results.append(res)
    
    save_json({"aggregate": [aggregate_runs(final_results)], "runs": final_results}, TUNING_DIR / "final_model/summary.json")

if __name__ == "__main__":
    main()
