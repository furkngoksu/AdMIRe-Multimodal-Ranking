from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.candidate_fusion import GatedCandidateFusion
from src.caption_encoder import CaptionEncoder
from src.config import (
    QUERY_ENCODER_CONFIG,
    QUERY_FUSION_CONFIG,
    RANKING_MODEL_CONFIG,
    RANKING_PATHS,
    STAGE1_SIGNAL_CONFIG,
    SUBMISSION_CONFIG,
    get_device,
)
from src.image_encoder import ImageEncoder
from src.query_encoder import QueryEncoder
from src.query_fusion import build_query_fusion
from src.ranker import CosineRanker
from src.utils import read_m_scores

from src.idiom_signal import generate_submission_m_scores

# =============================================================================
# File utilities
# =============================================================================

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(language: str, compound: str, image_name: str) -> Path:
    compound_name = compound.strip()
    direct_path = SUBMISSION_CONFIG.data_dir / language / compound_name / image_name

    if direct_path.exists():
        return direct_path

    language_dir = SUBMISSION_CONFIG.data_dir / language
    if language_dir.exists():
        for compound_dir in language_dir.iterdir():
            if compound_dir.is_dir() and compound_dir.name.casefold() == compound_name.casefold():
                fallback_path = compound_dir / image_name
                if fallback_path.exists():
                    return fallback_path

    raise FileNotFoundError(f"Image not found: {language} / {compound} / {image_name}")


# =============================================================================
# Text utilities
# =============================================================================

def create_visual_sentence(sentence: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return f"[FALLBACK - TARGET NOT FOUND] {sentence}"

    marked_sentence = sentence

    for start, end in sorted(spans, key=lambda span: span[0], reverse=True):
        marked_sentence = (
            marked_sentence[:start]
            + " ->> "
            + marked_sentence[start:end]
            + " <<- "
            + marked_sentence[end:]
        )

    return re.sub(r"\s+", " ", marked_sentence).strip()


# =============================================================================
# Model loading
# =============================================================================

def build_query_projection(input_dim: int, device: torch.device) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, RANKING_MODEL_CONFIG.query_dim),
        nn.LayerNorm(RANKING_MODEL_CONFIG.query_dim),
        nn.GELU(),
        nn.Dropout(RANKING_MODEL_CONFIG.query_dropout),
    ).to(device)


def load_ranking_modules(
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, torch.nn.Module], CosineRanker]:
    print("\n" + "=" * 70)
    print("[LOAD] Ranking model")
    print("=" * 70)
    print(f"Weights              : {RANKING_PATHS.weights_path}")
    print(f"Query dim            : {RANKING_MODEL_CONFIG.query_dim}")
    print(f"Query fusion         : {RANKING_MODEL_CONFIG.fusion_type}")
    print(f"M embedding dim      : {RANKING_MODEL_CONFIG.m_embedding_dim}")
    print(f"Learnable logit scale: {RANKING_MODEL_CONFIG.logit_scale_learnable}")
    print("=" * 70)

    weights = torch.load(RANKING_PATHS.weights_path, map_location=device)

    query_encoder = QueryEncoder(QUERY_ENCODER_CONFIG).to(device).eval()
    image_encoder = ImageEncoder().to(device).eval()
    caption_encoder = CaptionEncoder().to(device).eval()

    q_raw_dim = int(query_encoder.encoder.config.hidden_size * 2)

    query_projection = build_query_projection(q_raw_dim, device)
    query_projection.load_state_dict(weights["query_projection"])
    query_projection.eval()

    query_fusion = build_query_fusion(
        QUERY_FUSION_CONFIG.__class__(
            shared_dim=QUERY_FUSION_CONFIG.shared_dim,
            m_embedding_dim=RANKING_MODEL_CONFIG.m_embedding_dim,
            fusion_type=RANKING_MODEL_CONFIG.fusion_type,
        )
    ).to(device)
    query_fusion.load_state_dict(weights["query_fusion"])
    query_fusion.eval()

    candidate_fusion = GatedCandidateFusion().to(device)
    candidate_fusion.load_state_dict(weights["candidate_fusion"])
    candidate_fusion.eval()

    ranker = CosineRanker(
        learnable=RANKING_MODEL_CONFIG.logit_scale_learnable,
    ).to(device)
    ranker.load_state_dict(weights["ranker"])
    ranker.eval()

    encoders = {
        "query": query_encoder,
        "image": image_encoder,
        "caption": caption_encoder,
    }

    modules = {
        "query_projection": query_projection,
        "query_fusion": query_fusion,
        "candidate_fusion": candidate_fusion,
    }

    print("[OK] Ranking model loaded\n")

    return encoders, modules, ranker


# =============================================================================
# Inference
# =============================================================================

def rank_batch(
    batch: list[dict[str, Any]],
    encoders: dict[str, torch.nn.Module],
    modules: dict[str, torch.nn.Module],
    ranker: CosineRanker,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    sentences = [sample["sentence"] for sample in batch]
    compounds = [sample["compound"] for sample in batch]

    predicted_m = torch.tensor(
        [[sample["m_prob"]] for sample in batch],
        dtype=torch.float,
        device=device,
    )

    q_raw = encoders["query"].encode_raw(sentences, compounds)
    q_projected = modules["query_projection"](q_raw)
    q_fused = modules["query_fusion"](q_projected, predicted_m)

    image_repr = encoders["image"](
        [sample["image_paths"] for sample in batch]
    )
    caption_repr = encoders["caption"](
        [sample["captions"] for sample in batch]
    )

    candidate_repr = modules["candidate_fusion"](image_repr, caption_repr)
    scores = ranker(q_fused, candidate_repr)

    tensor_stats = []
    for index in range(len(batch)):
        tensor_stats.append(
            {
                "q_raw_norm": round(float(torch.norm(q_raw[index]).item()), 4),
                "q_projected_mean": round(float(q_projected[index].mean().item()), 6),
                "q_fused_mean": round(float(q_fused[index].mean().item()), 6),
                "m_prob": round(float(predicted_m[index].item()), 6),
                "scores_max": round(float(scores[index].max().item()), 4),
            }
        )

    return scores, tensor_stats


def build_samples(
    rows: list[dict[str, str]],
    language: str,
    m_scores: dict[str, float],
    query_encoder: QueryEncoder,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        sample_id = f"{language}_{row_index}"

        candidate_names = [
            row[f"image{candidate_index}_name"]
            for candidate_index in range(1, SUBMISSION_CONFIG.num_candidates + 1)
        ]

        spans = query_encoder.find_compound_spans(
            row["sentence"],
            row["compound"],
        )

        samples.append(
            {
                "sample_id": sample_id,
                "original_row": row,
                "sentence": row["sentence"],
                "compound": row["compound"],
                "visual_sentence": create_visual_sentence(row["sentence"], spans),
                "is_fallback": len(spans) == 0,
                "m_prob": float(m_scores[sample_id]),
                "image_paths": [
                    resolve_image_path(language, row["compound"], image_name)
                    for image_name in candidate_names
                ],
                "captions": [
                    row[f"image{candidate_index}_caption"]
                    for candidate_index in range(1, SUBMISSION_CONFIG.num_candidates + 1)
                ],
                "candidate_names": candidate_names,
            }
        )

    return samples


def build_debug_entry(
    sample: dict[str, Any],
    language: str,
    ranked_indices: list[int],
    score_row: torch.Tensor,
    tensor_stats: dict[str, float],
) -> dict[str, Any]:
    ranking_details = []

    for rank_position, candidate_index in enumerate(ranked_indices, start=1):
        ranking_details.append(
            {
                "rank": rank_position,
                "score": round(float(score_row[candidate_index]), 4),
                "image": sample["candidate_names"][candidate_index],
                "caption": sample["captions"][candidate_index],
            }
        )

    return {
        "sample_id": sample["sample_id"],
        "language": language,
        "m_score_signal": round(sample["m_prob"], 4),
        "target_compound": sample["compound"],
        "text_analysis": {
            "original_sentence": sample["sentence"],
            "visual_span_match": sample["visual_sentence"],
            "fallback_triggered": sample["is_fallback"],
        },
        "tensors": tensor_stats,
        "ranking_pipeline": ranking_details,
    }


def process_language_tsv(
    tsv_path: Path,
    m_scores: dict[str, float],
    encoders: dict[str, torch.nn.Module],
    modules: dict[str, torch.nn.Module],
    ranker: CosineRanker,
    device: torch.device,
    debug_logs: list[dict[str, Any]],
) -> Path:
    language = tsv_path.stem.removeprefix("submission_")
    output_language = SUBMISSION_CONFIG.language_code_map.get(language, language)
    output_path = SUBMISSION_CONFIG.output_dir / f"submission_{output_language}.tsv"

    print(f"\n[PROCESS] {language}")

    rows = read_tsv(tsv_path)
    samples = build_samples(
        rows=rows,
        language=language,
        m_scores=m_scores,
        query_encoder=encoders["query"],
    )

    output_rows: list[dict[str, str]] = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(samples), SUBMISSION_CONFIG.batch_size),
            desc=f"Ranking {language}",
            dynamic_ncols=True,
        ):
            batch = samples[start : start + SUBMISSION_CONFIG.batch_size]
            scores, batch_tensor_stats = rank_batch(
                batch=batch,
                encoders=encoders,
                modules=modules,
                ranker=ranker,
                device=device,
            )

            ranked_batch = scores.argsort(dim=-1, descending=True).cpu().tolist()

            for sample, ranked_indices, score_row, tensor_stats in zip(
                batch,
                ranked_batch,
                scores,
                batch_tensor_stats,
            ):
                ordered_candidates = [
                    sample["candidate_names"][candidate_index]
                    for candidate_index in ranked_indices
                ]

                output_row = sample["original_row"].copy()
                output_row["expected_order"] = str(ordered_candidates)
                output_rows.append(output_row)

                debug_logs.append(
                    build_debug_entry(
                        sample=sample,
                        language=language,
                        ranked_indices=ranked_indices,
                        score_row=score_row,
                        tensor_stats=tensor_stats,
                    )
                )

    write_tsv(output_path, output_rows)

    print(f"[OK] {language}: {len(output_rows)} rows -> {output_path.name}")

    return output_path


# =============================================================================
# Output packaging
# =============================================================================

def save_debug_log(debug_logs: list[dict[str, Any]]) -> None:
    with SUBMISSION_CONFIG.debug_log_path.open("w", encoding="utf-8") as file:
        json.dump(debug_logs, file, indent=4, ensure_ascii=False)

    print(f"[OK] Debug log: {SUBMISSION_CONFIG.debug_log_path}")


def create_submission_zip(files: list[Path]) -> None:
    with zipfile.ZipFile(SUBMISSION_CONFIG.zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files:
            zip_file.write(file_path, arcname=file_path.name)

    print(f"[OK] Submission archive: {SUBMISSION_CONFIG.zip_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = get_device()

    print("\n" + "=" * 70)
    print("[START] Full submission pipeline")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Stage-1 weight: {STAGE1_SIGNAL_CONFIG.weights_path}")
    print(f"M scores      : {STAGE1_SIGNAL_CONFIG.submission_scores_path}")
    print(f"Ranking weight: {RANKING_PATHS.weights_path}")
    print(f"TSV directory : {SUBMISSION_CONFIG.tsv_dir}")
    print(f"Image data    : {SUBMISSION_CONFIG.data_dir}")
    print(f"Output dir    : {SUBMISSION_CONFIG.output_dir}")
    print("=" * 70)

    SUBMISSION_CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: generate predicted M scores for the submission set.
    generate_submission_m_scores(device)

    # Stage 2: use predicted M scores inside the multimodal ranking model.
    encoders, modules, ranker = load_ranking_modules(device)
    m_scores = read_m_scores(STAGE1_SIGNAL_CONFIG.submission_scores_path)

    generated_files: list[Path] = []
    debug_logs: list[dict[str, Any]] = []

    for tsv_path in sorted(SUBMISSION_CONFIG.tsv_dir.glob("*.tsv")):
        generated_files.append(
            process_language_tsv(
                tsv_path=tsv_path,
                m_scores=m_scores,
                encoders=encoders,
                modules=modules,
                ranker=ranker,
                device=device,
                debug_logs=debug_logs,
            )
        )

    save_debug_log(debug_logs)
    create_submission_zip(generated_files)

    print("\n[DONE] Full submission pipeline finished")


if __name__ == "__main__":
    main()