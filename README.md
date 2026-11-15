<div align="center">

# 🧩 Idiomaticity-Aware Multimodal Ranking

**Matching expressions to images by their meaning in context.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Multimodal NLP](https://img.shields.io/badge/Multimodal-NLP-6554C0?style=for-the-badge)

[![Project Proposal](https://img.shields.io/badge/Project-Proposal-3467A1?style=for-the-badge)](docs/proposal.pdf)
[![Original Repository](https://img.shields.io/badge/Original-Team_Repository-181717?style=for-the-badge&logo=github)](https://github.com/Deniztpl/YZV405E_2526_Sisli)

</div>

An expression such as *spill the beans* can describe a physical action or mean revealing a secret. This project uses the surrounding sentence, candidate images, and their captions to rank the interpretation that fits the context.

Developed for the **AdMIRe 2 image-and-text ranking task** as an NLP team project at **Istanbul Technical University**, the system combines multilingual text representations, vision–language embeddings, and a learned idiomaticity signal. Each query is matched against five image–caption candidates.

## ✨ Approach

- **Multilingual text understanding:** XLM-RoBERTa represents both the sentence and the target expression.
- **Vision–language matching:** CLIP image embeddings and multilingual caption embeddings represent the candidates.
- **LLM-generated semantic perturbations:** Rewrites help estimate whether an expression is literal or idiomatic.
- **Learned fusion:** Concatenation, gating, and FiLM variants combine the idiomaticity signal with the query representation.
- **Ranking experiments:** The project compares fusion strategies, signal dimensions, and training losses, with Optuna-based tuning in the final training pipeline.

## 🏗️ Architecture

![Original project architecture diagram](docs/assets/pipeline.png)

The implementation uses `xlm-roberta-base` for text, `clip-ViT-B-32` for images, and `clip-ViT-B-32-multilingual-v1` for captions. Image and caption embeddings are combined through gated fusion, and candidates are ranked by scaled cosine similarity.

The final training script combines a listwise objective with top-choice cross-entropy. See [experiment notes](docs/EXPERIMENTS.md) for the distinction between the final pipeline and the ablation studies.

## 📊 Results

Local evaluation results of the selected final-training run:

| Split | Top-1 accuracy | nDCG@5 |
| --- | ---: | ---: |
| Development | 48.96% | 0.8123 |
| Test | 34.38% | 0.7958 |
| Cross-evaluation | 14.38% | 0.7121 |

Mean development nDCG across five seeds is **0.7894**. These are local evaluation scores, not official leaderboard results. The [training summary](docs/assets/training_summary.json) lists the configuration and per-seed values.

## 📁 Repository

| Directory | Contents |
| --- | --- |
| [`src/`](src/) | Encoders, fusion modules, ranking training, and submission generation |
| [`signal_models/`](signal_models/) | Five idiomaticity classifier variants and saved experiment metrics |
| [`signal_inference/`](signal_inference/) | Generation of idiomaticity scores for ranking data |
| [`perturbation/`](perturbation/) | LLM-based rewrite and semantic-feature generation |
| [`ranking_experiments/`](ranking_experiments/) | Fusion, signal-dimension, and loss ablations |
| [`docs/`](docs/) | Setup, experiments, proposal, and architecture diagram |

## 🚀 Running the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.ranking.train
```

Prepared LLM rewrites and idiomaticity scores are included. The AdMIRe 2 datasets and images are distributed separately by the task organizers. See the [setup guide](docs/SETUP.md) for the full pipeline: idiomaticity models, ranking training, ablations, and submission generation.

## 🤝 Team

**Deniz Topal · Furkan Yasir Goksu · Faruk Akyol**

Istanbul Technical University — YZV405E

Originally developed in the [team repository](https://github.com/Deniztpl/YZV405E_2526_Sisli).
