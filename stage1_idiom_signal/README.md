# Stage 1 Idiom Signal

This folder contains the organized Stage 1 idiomaticity-signal comparison suite.
It keeps all five M-signal variants in one place with the same seed, same split,
same epoch count, and same CSV.

## What It Compares

Old marked-input baseline:

```text
[LANG=EN] sentence with <mwe>...</mwe> [MWE] compound
    -> XLM-R mean pool
    -> Linear(768, 384), ReLU, Dropout, Linear(384, 1)
    -> sigmoid(logit) = M
```

Current target-only quick baseline:

```text
raw Target sentence only
    -> XLM-R mean pool
    -> Linear(768, hidden_1), BatchNorm, GELU, Dropout,
       Linear(hidden_1, hidden_2), BatchNorm, GELU, Dropout,
       Linear(hidden_2, 1)
    -> sigmoid(logit) = M
```

The current quick baseline mirrors the current target-only input idea and MLP
family, but it does not run the full Optuna/cache pipeline.

Hybrid marked-input MLP:

```text
[LANG=EN] sentence with <mwe>...</mwe> [MWE] compound
    -> XLM-R mean pool
    -> larger BatchNorm/GELU/Dropout MLP
    -> sigmoid(logit) = M
```

Neural M:

```text
target, paraphrase, lexical substitution, component drop, literalization
    -> XLM-R mean pool for each text
    -> abs(target - lex_sub), abs(target - comp_drop),
       abs(target - literalize), abs(target - paraphrase)
    -> concat to 3072 dimensions
    -> MLP
    -> sigmoid(logit) = M
```

Neural M uses the already saved `llm_results/*.jsonl` perturbation outputs. It
does not call any LLM API.

Hybrid LLM fusion:

```text
[LANG=EN] sentence with <mwe>...</mwe> [MWE] compound
    -> XLM-R mean pool = 768 dims

LLM perturbation vectors
    -> abs(target - lex_sub), abs(target - comp_drop),
       abs(target - literalize), abs(target - paraphrase)
    -> 3072 dims

concat = 3840 dims
    -> Linear(3840, 1024), BatchNorm, GELU, Dropout(0.3),
       Linear(1024, 256), BatchNorm, GELU, Dropout(0.3),
       Linear(256, 1)
    -> sigmoid(logit) = M
```

## Main Commands

From repo root:

Run all comparison models:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.runners.run_all --epochs 1 --limit 512
```

Run all models and create tables/plots:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.runners.run_all_and_report --epochs 1 --limit 512
```

Use the full CSV by passing `--limit 0`:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.runners.run_all_and_report --epochs 40 --limit 0 --batch_size 16 --neural_batch_size 64 --fusion_batch_size 64
```

## Run Separately

Old:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.models.old_marked_input.train --epochs 1 --limit 512
```

New target-only:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.models.new_target_only.train --epochs 1 --limit 512
```

Hybrid:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.models.hybrid_marked_mlp.train --epochs 1 --limit 512
```

Neural M:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.models.neural_m.train --epochs 1 --limit 512
```

Hybrid LLM fusion:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.models.hybrid_llm_fusion.train --epochs 1 --limit 512
```

Compare saved metrics:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.runners.compare_metrics
```

Create saved tables and plots from existing metrics:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal.visualization.visualize_results
```

## Outputs

Generated outputs are ignored by git:

```text
stage1_idiom_signal/outputs/old_marked_input/metrics.json
stage1_idiom_signal/outputs/old_marked_input/model.pt
stage1_idiom_signal/outputs/new_target_only/metrics.json
stage1_idiom_signal/outputs/new_target_only/model.pt
stage1_idiom_signal/outputs/hybrid_marked_mlp/metrics.json
stage1_idiom_signal/outputs/hybrid_marked_mlp/model.pt
stage1_idiom_signal/outputs/neural_m/metrics.json
stage1_idiom_signal/outputs/neural_m/model.pt
stage1_idiom_signal/outputs/hybrid_llm_fusion/metrics.json
stage1_idiom_signal/outputs/hybrid_llm_fusion/model.pt
stage1_idiom_signal/cache/neural_m_pooled.pt
stage1_idiom_signal/cache/hybrid_llm_fusion_pooled.pt
stage1_idiom_signal/reports/summary.csv
stage1_idiom_signal/reports/summary.md
stage1_idiom_signal/reports/history.csv
stage1_idiom_signal/reports/best_metrics.png
stage1_idiom_signal/reports/val_f1_curve.png
stage1_idiom_signal/reports/val_loss_curve.png
```

## Important Notes

- XLM-R based runners use frozen XLM-R by default.
- `--unfreeze_encoder` exists, but it is slower and not needed for quick checks.
- The same stratified validation split is used for both models.
- Accuracy is computed with threshold `M >= 0.5`.
- Best model selection is by validation F1.
- This folder calls no LLM API.
- `run_all` will build the Neural M cache the first time, then reuse it.
- `run_all` will also build the Hybrid LLM fusion cache the first time.
