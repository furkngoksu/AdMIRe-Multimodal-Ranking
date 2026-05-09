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
    print("Starting Fusion Architecture Ablation (Gold M)")
    print("=" * 60)
    
    # 1. Baseline: Concat with M=64 (Our best model so far)
    print("\n--- Running Concat Fusion (M=64) ---")
    run_ablation(
        family_name="fusion_ablation/concat_m64",
        m_dims=(64,),
        seeds=SEEDS,
        m_source="gold",
        fusion_type="concat",
    )
    
    # 2. Gated Fusion (M_dim=0 since it doesn't use linear embedding)
    print("\n--- Running Gated Fusion ---")
    run_ablation(
        family_name="fusion_ablation/gated",
        m_dims=(0,),
        seeds=SEEDS,
        m_source="gold",
        fusion_type="gated",
    )
    
    # 3. FiLM Scale-Only Fusion
    print("\n--- Running FiLM Scale-Only Fusion ---")
    run_ablation(
        family_name="fusion_ablation/film_scale",
        m_dims=(0,),
        seeds=SEEDS,
        m_source="gold",
        fusion_type="film",
    )
    
    # 4. FiLM Full Fusion (Scale + Shift)
    print("\n--- Running FiLM Full Fusion ---")
    run_ablation(
        family_name="fusion_ablation/film_full",
        m_dims=(0,),
        seeds=SEEDS,
        m_source="gold",
        fusion_type="film_full",
    )

if __name__ == "__main__":
    main()
