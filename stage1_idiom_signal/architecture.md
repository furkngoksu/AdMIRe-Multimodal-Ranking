# Stage 1 Idiom Signal Architecture

This document describes the Stage 1 baseline for the idiomaticity signal module
M. It is intentionally separate from the Stage 2 image-caption ranking pipeline.

## Goal

Stage 1 trains only M:

```text
Input:  Language, MWE, Target, Label
Output: continuous idiomaticity score M in [0, 1]
```

The binary label is only the training target:

```text
0 = literal / non-idiomatic
1 = idiomatic
```

At inference time the model returns:

```python
M = sigmoid(logit)
```

Stage 2 should later load this trained M model, freeze it, and inject the
continuous score into the query representation.

## Input Formatting

The Stage 1 input text is built as:

```text
[LANG={Language}] {Target with <mwe>...</mwe> around MWE if found} [MWE] {MWE}
```

Example:

```text
[LANG=EN] He <mwe>spilled the beans</mwe> about the plan. [MWE] spilled the beans
```

If the MWE string is not found inside `Target`, the fallback is:

```text
[LANG={Language}] {Target} [MWE] {MWE}
```

## Model

The baseline model is:

```text
formatted text
  -> XLM-R encoder
  -> mean pooling
  -> Linear(hidden_dim -> hidden_dim / 2)
  -> ReLU
  -> Dropout
  -> Linear(hidden_dim / 2 -> 1)
  -> logit
  -> sigmoid(logit) = M
```

By default the XLM-R encoder is frozen. Use `--unfreeze-encoder` only when you
want to fine-tune the encoder too.

## Relation To Stage 2

Stage 2 query construction is separate:

```text
Sentence + Compound
  -> QueryEncoder
  -> Q
```

The current Stage 2 fusion code expects:

```text
Q: [B, 512]
M: [B, 1]
```

So the fusion is:

```text
Q_prime = Q + alpha * Proj(M)
```

where `Proj(M)` maps `[B, 1]` to `[B, 512]`.

## Train Stage 1

From the repository root:

```bash
python -m stage1_idiom_signal.train_idiom_signal \
  --csv_path stage1_idiom_signal/data/unique_mwe_target_label_language.csv \
  --output_dir stage1_idiom_signal/runs/baseline_1epoch \
  --epochs 1 \
  --batch_size 16 \
  --lr 1e-4 \
  --max_length 128 \
  --val_ratio 0.15
```

For a longer baseline:

```bash
python -m stage1_idiom_signal.train_idiom_signal \
  --csv_path stage1_idiom_signal/data/unique_mwe_target_label_language.csv \
  --output_dir stage1_idiom_signal/runs/baseline_5epoch \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-4 \
  --max_length 128 \
  --val_ratio 0.15
```

Saved local artifacts:

```text
stage1_idiom_signal/runs/<run_name>/idiom_signal.pt
stage1_idiom_signal/runs/<run_name>/config.json
stage1_idiom_signal/runs/<run_name>/tokenizer/
stage1_idiom_signal/runs/<run_name>/metrics.json
```

The model weights and tokenizer are ignored by git. Small report JSON files are
kept under `stage1_idiom_signal/reports/`.

## Load And Compute M

```python
from src.idiom_signal import compute_m, load_idiom_signal

model = load_idiom_signal("stage1_idiom_signal/runs/baseline_1epoch")
score = compute_m(
    sentence="He spilled the beans about the plan.",
    mwe="spilled the beans",
    language="EN",
    model=model,
)
print(score)
```

## Demo

```bash
python -m stage1_idiom_signal.demo_idiom_signal \
  --model_dir stage1_idiom_signal/runs/baseline_1epoch \
  --max_length 128 \
  --json_path stage1_idiom_signal/reports/baseline_1epoch_demo_scores.json
```

## Dataset Report

```bash
python -m stage1_idiom_signal.analyze_dataset \
  --csv_path stage1_idiom_signal/data/unique_mwe_target_label_language.csv \
  --output_json stage1_idiom_signal/reports/dataset_distribution.json
```

Current dataset summary:

```text
Total rows: 5370
Label 0: 2931 (54.58%)
Label 1: 2439 (45.42%)
EN: 3880 (72.25%)
PT: 1490 (27.75%)
Unique MWE: 336
MWE found in Target: 5272 (98.18%)
Average Target length: 26.41 words
```
