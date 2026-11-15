# Setup and execution

Run commands from the repository root. Use Python 3.10 or later and install `requirements.txt` in a virtual environment. GPU acceleration is recommended for training; the scripts also contain CPU/MPS device selection.

## Required local inputs

| Input | Expected location |
| --- | --- |
| AdMIRe training TSVs and images | `admire2_training/EN/` and `admire2_training/PT/` |
| Submission sentences | `admire2_submission/admire2_sentences/` |
| Submission images | `admire2_submission/admire2_data/` |
| Prepared idiomaticity training records | `signal_models/data/matched_records.json` |
| LLM rewrites for ranking splits | `llm_results_training/*.jsonl` |
| LLM rewrites for submission inputs | `llm_results_submission/*.jsonl` |

The AdMIRe 2 dataset folders are external inputs. Prepared idiomaticity records, LLM rewrites, and generated score TSVs are included in the repository. See `src/config.py` for the exact TSV filenames and split paths.

## Idiomaticity models

With prepared training records available:

```bash
python signal_models/run_all_models.py target_perturbation_mlp
```

Omit the model argument to run all five models. Training includes hyperparameter search and can be expensive. Confirm that the resulting checkpoint matches `STAGE1_SIGNAL_CONFIG.weights_path` in `src/config.py` before generating scores:

```bash
python -m signal_inference.generate_training_m
python -m src.idiom_signal
```

The first command writes training-split scores; the second requires submission inputs and writes submission scores. The loaders fall back to the original sentence when a rewrite is missing, so complete rewrite coverage matters when comparing runs.

## Ranking

Once the training data, images, and generated training scores are present:

```bash
python -m src.ranking.train
```

The final trainer builds its own embedding cache and saves weights and summaries under `src/ranking/output/`. For the separate ablation studies:

```bash
python -m ranking_experiments.caching.build_cache
python -m ranking_experiments.fusion_ablation.run
python -m ranking_experiments.m_dim_ablation.run
```

Ablations use the cache under `ranking_experiments/caching/cache/`, not the final trainer's cache.

## Submission generation

After providing submission data, submission idiomaticity scores, and the trained ranking weights:

```bash
python -m src.run_submission
```

Outputs are written under `final_submission/`.

## Optional rewrite generation

`perturbation/pipeline.py` provides the rewrite and semantic-feature generation utilities. It routes requests to DeepSeek and to Gemini 2.5 Flash via OpenRouter, so `DEEPSEEK_API_KEY` and `OPENROUTER_API_KEY` must be set. Generation incurs API usage costs.
