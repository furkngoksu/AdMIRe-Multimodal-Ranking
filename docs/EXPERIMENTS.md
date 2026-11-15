# Experiments

## Final ranking pipeline

`src/ranking/train.py` uses predicted idiomaticity probabilities, gated image–caption fusion, full FiLM query fusion, and a 512-dimensional signal embedding. The loss is 0.3 × listwise cross-entropy plus 0.7 × top-choice cross-entropy. Hyperparameters are selected with 30 Optuna trials, followed by runs with seeds 42, 100, 150, 200, and 250.

The [training summary](assets/training_summary.json) selects seed 42 and reports development, test, and cross-evaluation metrics from local evaluation.

## Ablation studies

| Study | Entry point | Comparison |
| --- | --- | --- |
| Query fusion | `ranking_experiments/fusion_ablation/run.py` | Concatenation, gating, FiLM scale-only, full FiLM |
| Signal dimension | `ranking_experiments/m_dim_ablation/run.py` | Signal-embedding capacity, including a no-signal baseline |
| Ranking loss | `ranking_experiments/loss_ablation.ipynb` | Notebook-based loss comparisons |

The two script-based studies use gold idiomaticity labels and seeds 7, 13, 21, 42, and 100, so their scores are not directly comparable to the final pipeline, which uses predicted idiomaticity. Each study writes its results to its own `output/summary.json`.

## Notes

The [project proposal](proposal.pdf) also considers SigLIP as an image encoder; the final configuration uses CLIP. LLM rewrites are generated with DeepSeek and Gemini.
