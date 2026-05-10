# Stage 1 Baseline Quick Compare

This folder is a small comparison harness for the Stage 1 baseline ideas.
It does not replace the main training code. It lets us run both versions from
one place with the same seed, same split, same epoch count, and same CSV.

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
family, but it does not run the full Optuna/cache pipeline. The full current
pipeline is still in `stage1_idiom_signal/baseline/train.py`.

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

## One Command

From repo root:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.run_both --epochs 1 --limit 512
```

Run all comparison models:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.run_all --epochs 1 --limit 512
```

Use the full CSV by passing `--limit 0`:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.run_both --epochs 3 --limit 0
```

## Run Separately

Old:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.old_marked_input.train --epochs 1 --limit 512
```

New target-only:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.new_target_only.train --epochs 1 --limit 512
```

Hybrid:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.hybrid_marked_mlp.train --epochs 1 --limit 512
```

Neural M:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.neural_m.train --epochs 1 --limit 512
```

Compare saved metrics:

```powershell
.venv\Scripts\python.exe -m stage1_baseline_compare.compare_metrics
```

## Outputs

Generated outputs are ignored by git:

```text
stage1_baseline_compare/outputs/old_marked_input/metrics.json
stage1_baseline_compare/outputs/old_marked_input/model.pt
stage1_baseline_compare/outputs/new_target_only/metrics.json
stage1_baseline_compare/outputs/new_target_only/model.pt
stage1_baseline_compare/outputs/hybrid_marked_mlp/metrics.json
stage1_baseline_compare/outputs/hybrid_marked_mlp/model.pt
stage1_baseline_compare/outputs/neural_m/metrics.json
stage1_baseline_compare/outputs/neural_m/model.pt
stage1_baseline_compare/cache/neural_m_pooled.pt
```

## Important Notes

- Both quick runners use frozen XLM-R by default.
- `--unfreeze_encoder` exists, but it is slower and not needed for quick checks.
- The same stratified validation split is used for both models.
- Accuracy is computed with threshold `M >= 0.5`.
- Best model selection is by validation F1.
- This folder calls no LLM API.
- `run_all` will build the Neural M cache the first time, then reuse it.
