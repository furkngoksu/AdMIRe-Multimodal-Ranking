import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranking_experiments.runner import run_ablation

if __name__ == "__main__":
    run_ablation(
        family_name="oracle_ablation_pca",
        query_projection="pca_frozen",
        m_dims=(0, 8, 16, 32, 64, 128, 256),
        seeds=(7, 13, 21, 42, 100),
        m_source="oracle",
        epochs=20,
        batch_size=4,
    )
