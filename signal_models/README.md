# Idiomaticity models

These classifiers estimate whether a target expression is used literally or idiomatically. Their predicted probability provides the auxiliary signal for multimodal ranking.

| Model | Features |
| --- | --- |
| `baseline_mlp` | Sentence representation |
| `lang_marked_mlp` | Language label and marked target expression |
| `lang_marked_perturbation_mlp` | Marked context and rewrite differences |
| `perturbation_mlp` | Differences between original and rewritten sentence representations |
| `target_perturbation_mlp` | Target-span representation and rewrite differences |

The rewrite variants are lexical substitution, component removal, paraphrasing, and literalization. Model implementations and inference feature construction are in the training scripts and `model_registry.py`.

From the repository root, after preparing the inputs described in the [setup guide](../docs/SETUP.md):

```bash
python signal_models/run_all_models.py target_perturbation_mlp
```

Omit the model argument to train all five variants. Prepared input records are under `data/`. Each model directory contains the saved configuration, training history, and evaluation metrics.
