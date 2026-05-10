from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranking_experiments.runner import (  # noqa: E402
    RankingExperimentConfig,
    build_query_projection,
    load_cache,
)
from ranking_experiments.train_data import read_m_scores  # noqa: E402
from src.candidate_fusion import GatedCandidateFusion  # noqa: E402
from src.caption_encoder import CaptionEncoder  # noqa: E402
from src.config import QueryFusionConfig, get_device  # noqa: E402
from src.image_encoder import ImageEncoder  # noqa: E402
from src.query_encoder import QueryEncoder  # noqa: E402
from src.query_fusion import build_query_fusion  # noqa: E402
from src.ranker import CosineRanker  # noqa: E402
from src.weights import load_weights  # noqa: E402


SUMMARY_PATH = REPO_ROOT / "ranking_experiments/hyperparam_tuning/best_model/output/summary.json"
M_SCORES_PATH = REPO_ROOT / "stage1_idiom_signal_final/output/submission_m_scores.tsv"
SUBMISSION_TSV_DIR = REPO_ROOT / "admire2_submission/admire2_sentences"
SUBMISSION_DATA_DIR = REPO_ROOT / "admire2_submission/admire2_data"
OUTPUT_CSV = REPO_ROOT / "submission.csv"
BATCH_SIZE = 8
NUM_CANDIDATES = 5


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def select_best_run(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary["runs"]
    selection_metric = summary.get("selection_metric", "dev_ndcg")

    if selection_metric == "dev_ndcg":
        return max(runs, key=lambda run: float(run["best_dev"]["ndcg"]))

    if selection_metric == "dev_top1":
        return max(runs, key=lambda run: float(run["best_dev"]["top1"]))

    return runs[-1]


def resolve_image_path(language: str, compound: str, image_name: str) -> Path:
    image_path = SUBMISSION_DATA_DIR / language / compound / image_name
    if image_path.exists():
        return image_path

    normalized_compound = compound.strip()
    image_path = SUBMISSION_DATA_DIR / language / normalized_compound / image_name
    if image_path.exists():
        return image_path

    language_dir = SUBMISSION_DATA_DIR / language
    for compound_dir in language_dir.iterdir():
        if compound_dir.is_dir() and compound_dir.name.casefold() == normalized_compound.casefold():
            image_path = compound_dir / image_name
            if image_path.exists():
                return image_path

    raise FileNotFoundError(
        f"Image not found: language={language} compound={compound} image={image_name}"
    )


def load_ranking_modules(device: torch.device) -> tuple[dict[str, torch.nn.Module], CosineRanker, dict[str, Any]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Ranking summary not found: {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    best_run = select_best_run(summary)
    experiment_config = best_run["experiment"]
    weights_path = Path(best_run["weights_path"])

    if not weights_path.exists():
        raise FileNotFoundError(f"Ranking weights not found: {weights_path}")

    if experiment_config.get("m_source") != "predicted":
        print(
            "[warning] ranking model was trained with "
            f"m_source={experiment_config.get('m_source')}; submission uses predicted M scores."
        )

    cache = load_cache()
    train_q_raw = cache["splits"]["train"]["q_raw"]

    experiment = RankingExperimentConfig(**experiment_config)
    query_projection, projection_info = build_query_projection(
        experiment=experiment,
        train_q_raw=train_q_raw,
        device=device,
    )

    query_fusion = build_query_fusion(
        QueryFusionConfig(
            m_embedding_dim=experiment.m_embedding_dim,
            fusion_type=experiment.fusion_type,
        )
    ).to(device)
    candidate_fusion = GatedCandidateFusion().to(device)
    ranker = CosineRanker().to(device)

    modules = {
        "query_projection": query_projection,
        "query_fusion": query_fusion,
        "candidate_fusion": candidate_fusion,
    }
    load_weights(modules=modules, path=str(weights_path), device=device)

    for module in modules.values():
        module.eval()
    ranker.eval()

    print(
        "[ranking] "
        f"weights={weights_path} "
        f"query_projection={projection_info['query_projection']} "
        f"fusion_type={experiment.fusion_type} "
        f"m_dim={experiment.m_embedding_dim} "
        f"seed={experiment.seed}"
    )

    return modules, ranker, experiment_config


def load_submission_samples(m_scores: dict[str, float]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for tsv_path in sorted(SUBMISSION_TSV_DIR.glob("*.tsv")):
        language = tsv_path.stem.removeprefix("submission_")
        rows = read_tsv(tsv_path)

        for row_index, row in enumerate(rows):
            sample_id = f"{language}_{row_index}"
            if sample_id not in m_scores:
                raise KeyError(f"Missing M score for sample_id={sample_id}")

            candidate_names = [
                row[f"image{candidate_index}_name"]
                for candidate_index in range(1, NUM_CANDIDATES + 1)
            ]
            image_paths = [
                resolve_image_path(
                    language=language,
                    compound=row["compound"].strip(),
                    image_name=image_name,
                )
                for image_name in candidate_names
            ]
            captions = [
                row[f"image{candidate_index}_caption"]
                for candidate_index in range(1, NUM_CANDIDATES + 1)
            ]

            samples.append(
                {
                    "sample_id": sample_id,
                    "language": language,
                    "sentence": row["sentence"].strip(),
                    "compound": row["compound"].strip(),
                    "m_prob": m_scores[sample_id],
                    "candidate_names": candidate_names,
                    "captions": captions,
                    "image_paths": image_paths,
                }
            )

    return samples


def batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    ]


def rank_batch(
    batch: list[dict[str, Any]],
    query_encoder: QueryEncoder,
    image_encoder: ImageEncoder,
    caption_encoder: CaptionEncoder,
    query_projection: torch.nn.Module,
    query_fusion: torch.nn.Module,
    candidate_fusion: GatedCandidateFusion,
    ranker: CosineRanker,
    device: torch.device,
) -> torch.Tensor:
    sentences = [sample["sentence"] for sample in batch]
    compounds = [sample["compound"] for sample in batch]
    image_paths = [sample["image_paths"] for sample in batch]
    captions = [sample["captions"] for sample in batch]
    m_probs = torch.tensor(
        [[sample["m_prob"]] for sample in batch],
        dtype=torch.float,
        device=device,
    )

    q_raw = query_encoder.encode_raw(sentences=sentences, compounds=compounds)
    image_repr = image_encoder(image_paths)
    caption_repr = caption_encoder(captions)

    q = F.normalize(query_projection(q_raw), p=2, dim=-1)
    q_prime = query_fusion(q, m_probs)
    candidate_repr = candidate_fusion(image_repr, caption_repr)

    return ranker(q_prime, candidate_repr)


def main() -> None:
    device = get_device()
    print(f"[system] device={device}")

    modules, ranker, _ = load_ranking_modules(device)
    m_scores = read_m_scores(M_SCORES_PATH)
    samples = load_submission_samples(m_scores)
    print(f"[data] samples={len(samples)} m_scores={len(m_scores)}")

    query_encoder = QueryEncoder().to(device).eval()
    image_encoder = ImageEncoder().to(device).eval()
    caption_encoder = CaptionEncoder().to(device).eval()

    results: list[dict[str, str]] = []

    with torch.no_grad():
        for batch in tqdm(batches(samples, BATCH_SIZE), desc="ranking submission", dynamic_ncols=True):
            scores = rank_batch(
                batch=batch,
                query_encoder=query_encoder,
                image_encoder=image_encoder,
                caption_encoder=caption_encoder,
                query_projection=modules["query_projection"],
                query_fusion=modules["query_fusion"],
                candidate_fusion=modules["candidate_fusion"],
                ranker=ranker,
                device=device,
            )
            ranked_indices = scores.argsort(dim=-1, descending=True).cpu().tolist()

            for sample, order_indices in zip(batch, ranked_indices):
                predicted_order = [
                    sample["candidate_names"][candidate_index]
                    for candidate_index in order_indices
                ]
                results.append(
                    {
                        "sample_id": sample["sample_id"],
                        "expected_order": str(predicted_order),
                    }
                )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "expected_order"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[done] wrote {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
