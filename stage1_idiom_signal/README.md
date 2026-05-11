# Stage 1 — Idiom Signal Models (M)

This module implements five distinct binary classification models designed to detect idiomaticity in multilingual contexts. These models provide the "Idiom Signal" ($M$) used by the downstream multimodal ranking pipeline.

## Model Philosophy

The core challenge is distinguishing between **literal** and **idiomatic** uses of a compound (e.g., "black sheep" as a literal animal vs. a social outcast). We employ three main strategies:
1. **Contextual Awareness**: Understanding the surrounding sentence.
2. **Structural Marking**: Explicitly flagging the compound to the encoder.
3. **Semantic Break (Perturbation)**: Measuring how much the sentence meaning changes when the compound is replaced by literal paraphrases.

## Model Variants

| Model | Input Logic | Architecture | Rationale |
|---|---|---|---|
| **`baseline`** | Raw Sentence | MLP (BN+GELU) | Captures pure contextual cues without knowing where the idiom is. |
| **`marked_simple`** | Marked Sentence | Simple ReLU | Flags the idiom with `<mwe>` tags. Uses a lightweight head to avoid overfitting on small datasets. |
| **`marked_mlp`** | Marked Sentence | MLP (BN+GELU) | Same as above but with a deeper head (BatchNorm + GELU) for better non-linear mapping. |
| **`perturbation_mlp`**| Semantic Diffs | MLP (BN+GELU) | Ignores context; only looks at the "semantic gap" when replacing the idiom with LLM-generated literal alternatives. |
| **`fusion_mlp`** | Marked + Diffs | MLP (BN+GELU) | **The flagship model.** Combines the marked contextual embedding with the semantic perturbation signals for maximum robustness. |

## Feature Engineering

Each model's "character" is defined in `model_registry.py`:

- **Marked Input**: `[LANG=tr] <mwe>çürük elma</mwe> [MWE] çürük elma`
- **Perturbation Diffs**: $V_{diff} = |V_{target} - V_{literal\_rewrite}|$
- **Fusion Vector**: $[V_{marked} ; V_{diff\_lexsub} ; V_{diff\_compdrop} ; V_{diff\_paraphrase} ; V_{diff\_literal}]$

## Directory Structure

- `data/`: Contains the ground truth labels and pre-matched LLM perturbations.
- `[model_name]/train.py`: Standalone script to train, tune (Optuna), and save each model variant.
- `model_registry.py`: Centralized logic defining input preparation and feature extraction for all variants.

## Usage

### Training
To train or retrain a specific model:
```bash
python stage1_idiom_signal/fusion_mlp/train.py
```

### Inference (Generating Scores)
Inference logic is organized in `stage1_idiom_signal_final/`:
- **Submission**: `stage1_idiom_signal_final/submission_inference/generate_scores.py`
- **Training (EN/PT)**: `stage1_idiom_signal_final/train_inference/generate_scores.py`

## Output Artifacts
Each model saves its best weights (`idiom_signal.pt`), hyperparameters (`config.json`), and performance logs (`metrics.json`) in its respective `outputs/` folder.
