import json
import os
import shutil
import sys
from pathlib import Path

import optuna

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ranking_experiments.runner import run_ablation

SEEDS = (42, 100, 150, 200, 250)
N_TRIALS = 30
TUNING_DIR = Path(project_root) / "ranking_experiments" / "hyperparam_tuning"

def objective(trial: optuna.Trial) -> float:
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    epochs = trial.suggest_categorical("epochs", [20, 30, 40])
    
    family_name = f"hyperparam_tuning/trials/trial_{trial.number}"
    
    # 2. Run Training with 5 seeds
    print(f"\n>>> Starting Trial {trial.number} | lr={lr:.5f}, bs={batch_size}, epochs={epochs}")
    try:
        results = run_ablation(
            family_name=family_name,
            m_dims=(64,),
            seeds=SEEDS,
            m_source="actual",
            fusion_type="concat",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
        )
        
        # 3. Calculate Average Dev nDCG
        dev_ndcgs = [run["best_dev"]["ndcg"] for run in results]
        avg_dev_ndcg = sum(dev_ndcgs) / len(dev_ndcgs)
        
        # 4. Clean up the trial folder to save disk space if not the best
        # Optuna tracks the best. We will clean up all folders. 
        # The user can just train one final time with the best parameters.
        trial_dir = TUNING_DIR / "trials" / f"trial_{trial.number}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
            
        return avg_dev_ndcg

    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        return 0.0

def main():
    print("Starting Hyperparameter Tuning with Optuna")
    print("=" * 60)
    
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create study with a fixed seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        study_name="concat_m64_tuning",
        sampler=sampler,
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Print best results
    print("\n" + "=" * 60)
    print(f"Optimization Finished! Best Trial: {study.best_trial.number}")
    print(f"Best Dev nDCG: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Save best parameters to JSON
    best_params_path = TUNING_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_dev_ndcg": study.best_value,
            "params": study.best_params
        }, f, indent=2)
        
    print(f"\nBest parameters saved to: {best_params_path}")

if __name__ == "__main__":
    main()
