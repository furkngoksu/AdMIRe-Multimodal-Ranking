# Query-fusion ablation

Compares concatenation, gating, scale-only FiLM, and full FiLM with a 64-dimensional signal embedding.

The study uses gold idiomaticity labels, a trainable query projection, and seeds 7, 13, 21, 42, and 100. These settings differ from the final predicted-signal pipeline.

After preparing the [required inputs](../../docs/SETUP.md), run from the repository root:

```bash
python -m ranking_experiments.caching.build_cache
python -m ranking_experiments.fusion_ablation.run
```

The script writes per-run artifacts and `output/summary.json` inside this directory. These generated files are not bundled. `analyze_results.ipynb` is the existing analysis notebook; its outputs have been cleared for publication.

See [experiment notes](../../docs/EXPERIMENTS.md) for interpretation and reproducibility limits.
