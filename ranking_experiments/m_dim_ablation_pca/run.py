from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranking_experiments.fusion_ablation.runner import run_ablation


SEEDS = (7, 13, 21, 42, 100)


if __name__ == "__main__":
    run_ablation(
        family_name="m_dim_ablation_pca",
        query_projection="pca_frozen",
        m_dims=(0, 8, 16, 32, 64, 128, 256),
        seeds=SEEDS,
        epochs=20,
        batch_size=4,
        learning_rate=1e-4,
    )
