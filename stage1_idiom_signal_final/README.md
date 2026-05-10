# Stage 1 Idiom Signal Final

This folder contains the final organized Stage 1 idiomaticity-signal comparison
suite. It does not modify the existing `stage1_idiom_signal/` code. It lets us
run all five M-signal variants from one place with the same seed, same split,
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

## One Command

From repo root:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.run_both --epochs 1 --limit 512
```

Run all comparison models:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.run_all --epochs 1 --limit 512
```

Run all models and create tables/plots:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.run_all_and_report --epochs 1 --limit 512
```

Use the full CSV by passing `--limit 0`:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.run_both --epochs 3 --limit 0
```

## Run Separately

Old:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.old_marked_input.train --epochs 1 --limit 512
```

New target-only:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.new_target_only.train --epochs 1 --limit 512
```

Hybrid:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.hybrid_marked_mlp.train --epochs 1 --limit 512
```

Neural M:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.neural_m.train --epochs 1 --limit 512
```

Hybrid LLM fusion:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.hybrid_llm_fusion.train --epochs 1 --limit 512
```

Compare saved metrics:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.compare_metrics
```

Create saved tables and plots from existing metrics:

```powershell
.venv\Scripts\python.exe -m stage1_idiom_signal_final.visualize_results
```

## Outputs

Generated outputs are ignored by git:

```text
stage1_idiom_signal_final/outputs/old_marked_input/metrics.json
stage1_idiom_signal_final/outputs/old_marked_input/model.pt
stage1_idiom_signal_final/outputs/new_target_only/metrics.json
stage1_idiom_signal_final/outputs/new_target_only/model.pt
stage1_idiom_signal_final/outputs/hybrid_marked_mlp/metrics.json
stage1_idiom_signal_final/outputs/hybrid_marked_mlp/model.pt
stage1_idiom_signal_final/outputs/neural_m/metrics.json
stage1_idiom_signal_final/outputs/neural_m/model.pt
stage1_idiom_signal_final/outputs/hybrid_llm_fusion/metrics.json
stage1_idiom_signal_final/outputs/hybrid_llm_fusion/model.pt
stage1_idiom_signal_final/cache/neural_m_pooled.pt
stage1_idiom_signal_final/cache/hybrid_llm_fusion_pooled.pt
stage1_idiom_signal_final/reports/summary.csv
stage1_idiom_signal_final/reports/summary.md
stage1_idiom_signal_final/reports/history.csv
stage1_idiom_signal_final/reports/best_metrics.png
stage1_idiom_signal_final/reports/val_f1_curve.png
stage1_idiom_signal_final/reports/val_loss_curve.png
```

## Important Notes

- Both quick runners use frozen XLM-R by default.
- `--unfreeze_encoder` exists, but it is slower and not needed for quick checks.
- The same stratified validation split is used for both models.
- Accuracy is computed with threshold `M >= 0.5`.
- Best model selection is by validation F1.
- This folder calls no LLM API.
- `run_all` will build the Neural M cache the first time, then reuse it.
- `run_all` will also build the Hybrid LLM fusion cache the first time.
