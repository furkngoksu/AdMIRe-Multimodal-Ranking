from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- Path configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Project imports ---
from src.utils import (  # noqa: E402
    RankingExperimentConfig,
    read_m_scores,
    load_weights,
)
from src.candidate_fusion import GatedCandidateFusion  # noqa: E402
from src.caption_encoder import CaptionEncoder  # noqa: E402
from src.config import QueryFusionConfig, get_device  # noqa: E402
from src.image_encoder import ImageEncoder  # noqa: E402
from src.query_encoder import QueryEncoder  # noqa: E402
from src.query_fusion import build_query_fusion  # noqa: E402
from src.ranker import CosineRanker  # noqa: E402

# --- GLOBAL CONFIGURATION ---
# Set to "linear_trainable" for the new sequential head or "pca_frozen" for the old baseline
PROJECTION_TYPE = "linear_trainable"

# --- Constants ---
SUMMARY_PATH = REPO_ROOT / f"ranking_experiments/hyperparam_tuning/{PROJECTION_TYPE.split('_')[0]}/final_model/summary.json"
M_SCORES_PATH = REPO_ROOT / "stage1_idiom_signal_final/output/submission_m_scores.tsv"
SUBMISSION_TSV_DIR = REPO_ROOT / "admire2_submission/admire2_sentences"
SUBMISSION_DATA_DIR = REPO_ROOT / "admire2_submission/admire2_data"
OUTPUT_DIR = REPO_ROOT / "final_submission"
DEBUG_LOG_PATH = OUTPUT_DIR / "pipeline_debug.json"

BATCH_SIZE = 8
NUM_CANDIDATES = 5

LANGUAGE_CODE_MAP = {
    "Chinese": "zh",
    "Georgian": "ka",
    "Greek": "el",
    "Igbo": "ig",
    "Kazakh": "kk",
    "Norwegian": "no",
    "Portuguese-Brazil": "pt-br",
    "Portuguese-Portugal": "pt-pt",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish-Ecuador": "es-ec",
    "Turkish": "tr",
    "Uzbek": "uz",
}


# --- File helpers ---
def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file into a list of dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def resolve_image_path(language: str, compound: str, image_name: str) -> Path:
    """Resolve the image path, using a case-insensitive compound fallback if needed."""
    compound_name = compound.strip()
    direct_path = SUBMISSION_DATA_DIR / language / compound_name / image_name

    if direct_path.exists():
        return direct_path

    language_dir = SUBMISSION_DATA_DIR / language
    if language_dir.exists():
        for compound_dir in language_dir.iterdir():
            if not compound_dir.is_dir():
                continue

            if compound_dir.name.casefold() != compound_name.casefold():
                continue

            fallback_path = compound_dir / image_name
            if fallback_path.exists():
                return fallback_path

    raise FileNotFoundError(f"Image not found: {language} / {compound} / {image_name}")


# --- Text helpers ---
def create_visual_sentence(sentence: str, spans: list[tuple[int, int]]) -> str:
    """Mark target spans in the sentence for readable debug output."""
    if not spans:
        return f"[FALLBACK - TARGET NOT FOUND IN SENTENCE] {sentence}"

    visual_sentence = sentence

    # Apply replacements from right to left so earlier indexes remain valid.
    for start, end in sorted(spans, key=lambda span: span[0], reverse=True):
        visual_sentence = (
            visual_sentence[:start]
            + " ->> "
            + visual_sentence[start:end]
            + " <<- "
            + visual_sentence[end:]
        )

    return re.sub(r"\s+", " ", visual_sentence).strip()


# --- Model loading ---
def load_ranking_modules(
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, torch.nn.Module], CosineRanker]:
    """Load the best ranking configuration and initialized modules with fancy logging."""
    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    best_run = max(summary["runs"], key=lambda run: float(run["best_dev"]["ndcg"]))
    subdir = PROJECTION_TYPE.split("_")[0]
    weights_path = REPO_ROOT / f"ranking_experiments/hyperparam_tuning/{subdir}/final_model" / f"seed{best_run['experiment']['seed']}" / "weights.pt"
    
    # Robustly initialize config from dictionary
    exp_dict = best_run["experiment"]
    # Filter keys to match dataclass
    valid_keys = RankingExperimentConfig.__dataclass_fields__.keys()
    filtered_exp = {k: v for k, v in exp_dict.items() if k in valid_keys}
    experiment_config = RankingExperimentConfig(**filtered_exp)

    print("\n" + "="*50)
    print("  🏆 LOADING BEST RANKING MODEL")
    print("="*50)
    print(f"  • Projection:  {PROJECTION_TYPE}")
    print(f"  • Fusion:      {experiment_config.fusion_type}")
    print(f"  • LR:          {experiment_config.learning_rate:.2e}")
    print(f"  • Learnable T: {experiment_config.logit_scale_learnable}")
    print(f"  • Seed:        {experiment_config.seed}")
    print(f"  • Weights:     .../{weights_path.relative_to(REPO_ROOT)}")
    print("="*50 + "\n")

    print(f"[info] Loading best model weights...")
    weights = torch.load(weights_path, map_location=device)

    # 1. Encoders
    query_encoder = QueryEncoder(config=experiment_config).to(device)
    query_encoder.projection.load_state_dict(weights["query_projection"])
    
    image_encoder = ImageEncoder().to(device)
    caption_encoder = CaptionEncoder().to(device)

    # 2. Fusion & Ranking Modules
    query_fusion_config = QueryFusionConfig(
        m_embedding_dim=int(experiment_config.m_embedding_dim),
        fusion_type=experiment_config.fusion_type,
    )
    query_fusion = build_query_fusion(query_fusion_config).to(device)
    query_fusion.load_state_dict(weights["query_fusion"])
    
    candidate_fusion = GatedCandidateFusion().to(device)
    # candidate_fusion has no weights in current architecture (parameter-less gating)

    ranker = CosineRanker(
        learnable=getattr(experiment_config, "logit_scale_learnable", False)
    ).to(device)
    ranker.load_state_dict(weights["ranker"])

    # Prepare return dictionaries
    encoders = {
        "query": query_encoder,
        "image": image_encoder,
        "caption": caption_encoder,
    }
    modules = {
        "query_fusion": query_fusion,
        "candidate_fusion": candidate_fusion,
    }

    for e in encoders.values(): e.eval()
    for m in modules.values(): m.eval()
    ranker.eval()

    return encoders, modules, ranker


# --- Inference ---
def rank_batch(
    batch: list[dict[str, Any]],
    encoders: dict[str, torch.nn.Module],
    modules: dict[str, torch.nn.Module],
    ranker: CosineRanker,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Run inference for a batch and collect compact tensor statistics."""
    sentences = [sample["sentence"] for sample in batch]
    compounds = [sample["compound"] for sample in batch]
    m_probs = torch.tensor(
        [[sample["m_prob"]] for sample in batch],
        dtype=torch.float,
        device=device,
    )

    q_fused_repr = encoders["query"](sentences, compounds)
    image_repr = encoders["image"]([sample["image_paths"] for sample in batch])
    caption_repr = encoders["caption"]([sample["captions"] for sample in batch])

    q_fused = modules["query_fusion"](q_fused_repr, m_probs)
    candidate_repr = modules["candidate_fusion"](image_repr, caption_repr)

    scores = ranker(q_fused, candidate_repr)

    tensor_stats = []
    for index in range(len(batch)):
        tensor_stats.append(
            {
                "q_proj_mean": round(float(q_fused_repr[index].mean().item()), 6),
                "q_fused_mean": round(float(q_fused[index].mean().item()), 6),
                "scores_max": round(float(scores[index].max().item()), 4),
            }
        )

    return scores, tensor_stats


def build_samples(
    rows: list[dict[str, str]],
    language: str,
    m_scores: dict[str, float],
    query_encoder: torch.nn.Module,
) -> list[dict[str, Any]]:
    """Convert raw TSV rows into model-ready samples."""
    samples = []

    for row_index, row in enumerate(rows):
        sample_id = f"{language}_{row_index}"
        candidate_names = [
            row[f"image{candidate_index}_name"]
            for candidate_index in range(1, NUM_CANDIDATES + 1)
        ]

        try:
            spans = query_encoder.find_compound_spans(row["sentence"], row["compound"])
        except Exception:
            spans = []

        samples.append(
            {
                "sample_id": sample_id,
                "original_row": row,
                "sentence": row["sentence"],
                "visual_sentence": create_visual_sentence(row["sentence"], spans),
                "compound": row["compound"],
                "is_fallback": len(spans) == 0,
                "m_prob": float(m_scores[sample_id]),
                "image_paths": [
                    resolve_image_path(language, row["compound"], image_name)
                    for image_name in candidate_names
                ],
                "captions": [
                    row[f"image{candidate_index}_caption"]
                    for candidate_index in range(1, NUM_CANDIDATES + 1)
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
    """Create one readable debug record for the final JSON log."""
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
    """Rank all samples in one language TSV and write the submission file."""
    language = tsv_path.stem.removeprefix("submission_")
    print(f"\n[process] Language: {language}")

    rows = read_tsv(tsv_path)
    samples = build_samples(rows, language, m_scores, encoders["query"])
    final_rows = []

    with torch.no_grad():
        for start in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Ranking {language}"):
            batch = samples[start : start + BATCH_SIZE]
            scores, batch_tensor_stats = rank_batch(batch, encoders, modules, ranker, device)
            batch_ranked_indices = scores.argsort(dim=-1, descending=True).cpu().tolist()

            for sample, ranked_indices, score_row, tensor_stats in zip(
                batch,
                batch_ranked_indices,
                scores,
                batch_tensor_stats,
            ):
                ordered_candidates = [
                    sample["candidate_names"][candidate_index]
                    for candidate_index in ranked_indices
                ]

                output_row = sample["original_row"].copy()
                output_row["expected_order"] = str(ordered_candidates)
                final_rows.append(output_row)

                debug_logs.append(
                    build_debug_entry(
                        sample=sample,
                        language=language,
                        ranked_indices=ranked_indices,
                        score_row=score_row,
                        tensor_stats=tensor_stats,
                    )
                )

    output_path = OUTPUT_DIR / f"submission_{LANGUAGE_CODE_MAP.get(language, language)}.tsv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[done] Processed {len(final_rows)} samples.")
    return output_path


# --- Main pipeline ---
def main() -> None:
    """Run the full submission inference pipeline."""
    device = get_device()
    print(f"[system] Inference device: {device}")

    encoders, modules, ranker = load_ranking_modules(device)
    m_scores = read_m_scores(M_SCORES_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    debug_logs: list[dict[str, Any]] = []

    for tsv_path in sorted(SUBMISSION_TSV_DIR.glob("*.tsv")):
        output_path = process_language_tsv(
            tsv_path=tsv_path,
            m_scores=m_scores,
            encoders=encoders,
            modules=modules,
            ranker=ranker,
            device=device,
            debug_logs=debug_logs,
        )
        generated_files.append(output_path)

    print("\n[log] Writing readable debug file...")
    with DEBUG_LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(debug_logs, file, indent=4, ensure_ascii=False)

    zip_path = OUTPUT_DIR / "submission.zip"
    print(f"\n[zip] Creating submission archive at {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in generated_files:
            zip_file.write(file_path, arcname=file_path.name)

    print(f"\n[complete] Pipeline finished. Debug log saved to: {DEBUG_LOG_PATH}")


if __name__ == "__main__":
    main()