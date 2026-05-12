# Stage 1 — Idiom Signal Models (M)

This module implements six distinct binary classification models to detect idiomaticity in multilingual contexts. These models generate the "Idiom Signal" ($M$) for the downstream multimodal ranking pipeline.

## Model Philosophy

We distinguish between **literal** and **idiomatic** uses using four core feature strategies:
1. **Contextual Awareness**: Surrounding sentence semantics.
2. **Language Signaling (`lang`)**: Explicit `[LANG=...]` prefix for the encoder.
3. **Structural Marking (`marked`)**: Highlighting the compound via `<mwe>` tags.
4. **Target Extraction (`target`)**: Precise coordinate extraction (Span Pooling) to get the exact target vector without text alteration.
5. **Semantic Break (`perturbation`)**: Measuring semantic gap ($V_{diffs}$) using LLM-generated literal paraphrases.

## Model Variants

| Model | Features | Reading Mechanism | Rationale |
|---|---|---|---|
| **`baseline_mlp`** | Raw Context | Mean Pool | Pure contextual cues; no target information. |
| **`lang_marked_simple`** | Lang + Marked | Mean Pool | Simple head with explicit tags to avoid overfitting. |
| **`lang_marked_mlp`** | Lang + Marked | Mean Pool | Deeper head for non-linear tag-context mapping. |
| **`lang_marked_perturbation_mlp`** | Lang + Marked + Diffs | Mean Pool | **Legacy Flagship.** Context + semantic gap with tags. |
| **`perturbation_mlp`** | Semantic Diffs | Mean Pool | Only looks at the "semantic gap" ($V_{diffs}$). |
| **`target_perturbation_mlp`** | Target + Diffs | **Target Pool** + Mean Pool | **New Flagship.** Precise target vector + semantic gap. |

## Feature Engineering

Definitions in `model_registry.py`:

- **Tagged Input**: `[LANG=tr] <mwe>çürük elma</mwe> [MWE] çürük elma`
- **Target Pooling**: Dynamic coordinate extraction (Span Pooling) over raw tokens.
- **Perturbation Diffs**: $V_{diff} = |V_{target} - V_{literal\_rewrite}|$
- **Target Perturbation Vector**: $[V_{target\_span} ; V_{diff\_lexsub} ; V_{diff\_compdrop} ; V_{diff\_paraphrase} ; V_{diff\_literal}]$

## Directory Structure

- `data/`: Labels and pre-matched LLM perturbations.
- `[model_name]/train.py`: Standalone scripts for training and Optuna tuning.
- `model_registry.py`: Centralized input preparation and feature extraction logic.

## Usage

### Training
To train a specific variant (e.g., the target-based flagship):
```bash
python stage1_idiom_signal/target_perturbation_mlp/train.py
```

### Inference

* **Submission**: `stage1_idiom_signal_final/submission_inference/generate_scores.py`
* **Internal**: `stage1_idiom_signal_final/train_inference/generate_scores.py`

## Output Artifacts

Each model saves:

* `idiom_signal.pt`: Best seed weights.
* `config.json`: Hyperparameters and metadata.
* `metrics.json`: Per-seed and aggregate performance (F1, Accuracy, ROC-AUC).
