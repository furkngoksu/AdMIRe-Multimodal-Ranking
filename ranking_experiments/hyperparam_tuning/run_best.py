import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ranking_experiments.runner import run_ablation

SEEDS = (42, 100, 150, 200, 250)

def main():
    tuning_dir = Path(project_root) / "ranking_experiments" / "hyperparam_tuning"
    best_params_file = tuning_dir / "best_params.json"
    
    if not best_params_file.exists():
        print("Could not find best_params.json")
        return
        
    with open(best_params_file, "r") as f:
        best_data = json.load(f)
        
    params = best_data["params"]
    print("Running Final Model with Best Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
        
    run_ablation(
        family_name="hyperparam_tuning/best_model",
        m_dims=(64,),
        seeds=SEEDS,
        m_source="gold",
        fusion_type="concat",
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"]
    )
    
    print("\nBest model trained and saved to hyperparam_tuning/best_model/")

if __name__ == "__main__":
    main()
