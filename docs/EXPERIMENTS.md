# Experiments

## Final ranking pipeline

`src/ranking/train.py` uses predicted idiomaticity probabilities, gated image–caption fusion, full FiLM query fusion, and a 512-dimensional signal embedding. The loss is 0.3 × listwise cross-entropy plus 0.7 × top-choice cross-entropy. Hyperparameters are selected with 30 Optuna trials, followed by runs with seeds 42, 100, 150, 200, and 250.

The saved summary selects seed 42 and reports development, test, and cross-evaluation metrics. It records local experiments, not official submission scores. Historical metrics are preserved without claiming that the current checkout reproduces them.

## Ablation studies

| Study | Entry point | Comparison |
| --- | --- | --- |
| Query fusion | `ranking_experiments/fusion_ablation/run.py` | Concatenation, gating, FiLM scale-only, full FiLM |
| Signal dimension | `ranking_experiments/m_dim_ablation/run.py` | Signal-embedding capacity, including a no-signal baseline |
| Ranking loss | `ranking_experiments/loss_ablation.ipynb` | Notebook-based loss comparisons |

The two script-based studies use gold idiomaticity labels and seeds 7, 13, 21, 42, and 100. Their results should not be treated as directly comparable to the predicted-signal final pipeline. Each writes its own `output/summary.json`; these generated outputs are not included here.

## Interpretation limits

The proposal describes candidate methods, including SigLIP, whereas the checked-in configuration uses CLIP. It also mentions a different LLM than the current perturbation dispatcher, which routes to DeepSeek and Gemini through provider APIs. Model names in the proposal should not be presented as a record of every executed experiment.

The ablation scripts retain best-model state dictionaries in memory without explicitly copying their tensors. The reported selected epoch therefore needs verification before claiming exact checkpoint reproducibility. Model training was not rerun as part of this documentation cleanup.
