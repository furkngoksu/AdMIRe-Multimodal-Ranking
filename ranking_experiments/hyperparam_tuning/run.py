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
# Fixed to linear_trainable as required for final optimal capacity mapping
PROJECTION_TYPE = "linear_trainable"
LEARNABLE_LOGIT_SCALE = True

SEEDS = (42, 100, 150, 200, 250)
N_TRIALS = 30
TUNING_DIR = project_root / "ranking_experiments/hyperparam_tuning"
FINAL_MODEL_DIR = TUNING_DIR / "final_model"
BASELINE_DIR = TUNING_DIR / "baseline"

# Based on definitive fusion ablation results:
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


def run_training_flow(
    experiment: RankingExperimentConfig,
    output_dir: Path,
    loaders: dict,
    train_q_raw: torch.Tensor,
    device: torch.device,
    save_to_disk: bool = False,
):
    """Core training logic with integrated early stopping for optimal generalization."""
    set_seed(experiment.seed)

    # Initialize Multimodal Ranking components
    query_projection, _ = build_query_projection(experiment, train_q_raw, device)
    query_fusion = build_query_fusion(
        QueryFusionConfig(
            m_embedding_dim=experiment.m_embedding_dim,
            fusion_type=experiment.fusion_type,
        )
    ).to(device)
    candidate_fusion = GatedCandidateFusion().to(device)
    ranker = CosineRanker(learnable=experiment.logit_scale_learnable).to(device)

    trained_modules = {
        "query_projection": query_projection,
        "query_fusion": query_fusion,
        "candidate_fusion": candidate_fusion,
        "ranker": ranker,
    }

    # Collect all trainable parameters
    params = []
    for mod in trained_modules.values():
        params.extend([p for p in mod.parameters() if p.requires_grad])

    optimizer = torch.optim.AdamW(params, lr=experiment.learning_rate, weight_decay=experiment.weight_decay)

    best_ndcg = -1.0
    best_metrics = None
    best_state_dicts = None
    best_epoch = 1
    history = []

    # Patience-based early stopping to prevent over-optimization
    patience = 5
    patience_counter = 0

    for epoch in range(1, experiment.epochs + 1):
        train_one_epoch(
            trained_modules, loaders["train"], optimizer, device, epoch, experiment
        )
        dev_metrics = evaluate(
            trained_modules, loaders["dev"], "dev", experiment.m_source, device
        )
        history.append({"epoch": epoch, "dev": dev_metrics})

        if dev_metrics["ndcg"] > best_ndcg:
            best_ndcg = dev_metrics["ndcg"]
            best_metrics = dev_metrics
            best_epoch = epoch
            best_state_dicts = {
                name: mod.state_dict() for name, mod in trained_modules.items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Silently terminate seed training loop to conserve compute
                break

    # Reload optimal weights for final held-out evaluations
    if best_state_dicts:
        for name, mod in trained_modules.items():
            mod.load_state_dict(best_state_dicts[name])

    test_metrics = evaluate(
        trained_modules, loaders["test"], "test", experiment.m_source, device
    )
    xeval_metrics = evaluate(
        trained_modules, loaders["xeval"], "xeval", experiment.m_source, device
    )

    if save_to_disk and best_state_dicts:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_weights(trained_modules, output_dir / "weights.pt")
        save_json(experiment.to_dict(), output_dir / "config.json")

    single_run = {
        "experiment": experiment.to_dict(),
        "history": history,
        "best_dev": best_metrics,
        "test": test_metrics,
        "xeval": xeval_metrics,
        "best_epoch": best_epoch,
    }

    if save_to_disk:
        save_json(single_run, output_dir / "results.json")

    return single_run


def compute_aggregate(runs_cohort: list[dict], group_label: str) -> dict:
    """Standardized cohort reporting schema for multi-split validation."""
    num_seeds = len(runs_cohort)
    avg_best_epoch = sum(r["best_epoch"] for r in runs_cohort) / num_seeds

    agg = {
        "group_label": group_label,
        "num_seeds": num_seeds,
        "best_epoch": avg_best_epoch,
        "best_dev": {"loss": 0.0, "top1": 0.0, "ndcg": 0.0},
        "test": {"loss": 0.0, "top1": 0.0, "ndcg": 0.0},
        "xeval": {"loss": 0.0, "top1": 0.0, "ndcg": 0.0},
    }

    for split_name in ("best_dev", "test", "xeval"):
        for metric_name in ("loss", "top1", "ndcg"):
            total = sum(r[split_name][metric_name] for r in runs_cohort)
            agg[split_name][metric_name] = total / num_seeds

    return agg


def objective(trial):
    """Optuna objective function searching batch size and learning rate trajectories across all seeds."""
    device = get_device()
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.5e-4, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.05])

    ndcg_scores = []
    for seed in SEEDS:
        config = RankingExperimentConfig(
            name=f"trial_{trial.number}_s{seed}",
            m_embedding_dim=64,
            seed=seed,
            fusion_type=SELECTED_ARCHITECTURE,
            epochs=50,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            query_projection=PROJECTION_TYPE,
            logit_scale_learnable=LEARNABLE_LOGIT_SCALE,
        )

        loaders = build_loaders(cache, config.batch_size, seed)
        result = run_training_flow(
            config,
            TUNING_DIR / f"trial_{trial.number}_s{seed}",
            loaders,
            train_q_raw,
            device,
            save_to_disk=False,
        )
        ndcg_scores.append(result["best_dev"]["ndcg"])

    return sum(ndcg_scores) / len(ndcg_scores)


def main():
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]
    device = get_device()

    # --- PHASE 1: BASELINE (Selected Architecture with Default Initial Params) ---
    print(
        f"\n[PHASE 1] Training Baseline ({SELECTED_ARCHITECTURE} Init Configuration)..."
    )
    baseline_runs = []
    for seed in SEEDS:
        exp = RankingExperimentConfig(
            name=f"baseline_s{seed}",
            m_embedding_dim=64,
            seed=seed,
            fusion_type=SELECTED_ARCHITECTURE,
            epochs=50,
            batch_size=32,
            learning_rate=1e-4,
            query_projection=PROJECTION_TYPE,
            logit_scale_learnable=LEARNABLE_LOGIT_SCALE,
        )
        loaders = build_loaders(cache, exp.batch_size, seed)
        res = run_training_flow(
            exp,
            BASELINE_DIR / f"seed{seed}",
            loaders,
            train_q_raw,
            device,
            save_to_disk=False,
        )
        baseline_runs.append(res)

    save_json(
        {
            "aggregate": [
                compute_aggregate(
                    baseline_runs, f"Init Configuration ({SELECTED_ARCHITECTURE})"
                )
            ],
            "runs": baseline_runs,
        },
        BASELINE_DIR / "summary.json",
    )

    # --- PHASE 2: OPTUNA HYPERPARAM TUNING ---
    print(
        f"\n[PHASE 2] Executing Parameter Optimization Sweep for {SELECTED_ARCHITECTURE}..."
    )
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    save_json(
        {"best_params": best_params, "best_value": study.best_value},
        TUNING_DIR / "best_params.json",
    )

    # --- PHASE 3: FINAL MULTI-SEED RUN WITH TUNED PARAMETERS ---
    print(
        f"\n[PHASE 3] Final Production Sweep: Optimized {SELECTED_ARCHITECTURE}..."
    )
    final_runs = []
    for seed in SEEDS:
        exp = RankingExperimentConfig(
            name=f"tuned_s{seed}",
            m_embedding_dim=64,
            seed=seed,
            fusion_type=SELECTED_ARCHITECTURE,
            epochs=50,  # Bound execution dynamically using early stopping
            batch_size=best_params["batch_size"],
            learning_rate=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
            query_projection=PROJECTION_TYPE,
            logit_scale_learnable=LEARNABLE_LOGIT_SCALE,
        )
        loaders = build_loaders(cache, exp.batch_size, seed)
        res = run_training_flow(
            exp,
            FINAL_MODEL_DIR / f"seed{seed}",
            loaders,
            train_q_raw,
            device,
            save_to_disk=False,
        )
        final_runs.append(res)

    save_json(
        {
            "aggregate": [
                compute_aggregate(
                    final_runs, f"Optimized Configuration ({SELECTED_ARCHITECTURE})"
                )
            ],
            "runs": final_runs,
        },
        FINAL_MODEL_DIR / "summary.json",
    )
    print(f"\n[COMPLETED] Optimization artifacts saved to {TUNING_DIR}")


if __name__ == "__main__":
    main()
