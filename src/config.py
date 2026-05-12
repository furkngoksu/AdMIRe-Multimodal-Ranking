from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Device
# =============================================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================================
# Encoder configs
# =============================================================================

@dataclass(frozen=True)
class QueryEncoderConfig:
    model_name: str = "xlm-roberta-base"
    shared_dim: int = 512
    max_length: int = 128
    query_projection_type: str = "linear"
    debug: bool = False


@dataclass(frozen=True)
class CaptionEncoderConfig:
    model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    shared_dim: int = 512
    max_length: int = 128
    debug: bool = False


@dataclass(frozen=True)
class ImageEncoderConfig:
    model_name: str = "sentence-transformers/clip-ViT-B-32"
    shared_dim: int = 512
    debug: bool = False


# =============================================================================
# Stage-1 idiom signal config
# =============================================================================

@dataclass(frozen=True)
class Stage1SignalConfig:
    model_name: str = "target_perturbation_mlp"
    transformer_name: str = "xlm-roberta-base"

    batch_size: int = 16
    max_length: int = 128

    weights_path: Path = (
        PROJECT_ROOT
        / "signal_models"
        / "target_perturbation_mlp"
        / "outputs_xlm_base_1"
        / "idiom_signal.pt"
    )

    train_llm_dir: Path = PROJECT_ROOT / "llm_results_training"
    submission_llm_dir: Path = PROJECT_ROOT / "llm_results_submission"

    train_scores_path: Path = (
        PROJECT_ROOT
        / "signal_inference"
        / "admire_train_scores.tsv"
    )

    submission_scores_path: Path = (
        PROJECT_ROOT
        / "signal_inference"
        / "submission_m_scores.tsv"
    )


# =============================================================================
# Ranking training / inference config
# =============================================================================

@dataclass(frozen=True)
class QueryFusionConfig:
    shared_dim: int = 512
    m_embedding_dim: int = 256
    fusion_type: str = "film_full"  # "concat", "gated", "film", "film_full"


@dataclass(frozen=True)
class RankingModelConfig:
    query_dim: int = 512
    query_dropout: float = 0.1

    fusion_type: str = "film_full"
    m_embedding_dim: int = 512
    logit_scale_learnable: bool = True

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    early_stopping_patience: int = 5

    cache_batch_size: int = 4
    force_rebuild_cache: bool = True


@dataclass(frozen=True)
class RankingPaths:
    output_dir: Path = PROJECT_ROOT / "src" / "ranking" / "output"

    cache_path: Path = output_dir / "admire_rank_cache.pt"
    weights_path: Path = output_dir / "final_weights.pt"
    best_params_path: Path = output_dir / "best_params.json"
    summary_path: Path = output_dir / "training_summary.json"


# =============================================================================
# Training data config
# =============================================================================

@dataclass(frozen=True)
class TrainingDataConfig:
    train_files: tuple[tuple[str, str, Path, Path], ...] = (
        (
            "EN",
            "train",
            PROJECT_ROOT / "admire2_training" / "EN" / "train" / "subtask_a_train.tsv",
            PROJECT_ROOT / "admire2_training" / "EN" / "train",
        ),
        (
            "PT",
            "train",
            PROJECT_ROOT / "admire2_training" / "PT" / "train" / "subtask_a_train.tsv",
            PROJECT_ROOT / "admire2_training" / "PT" / "train",
        ),
    )

    dev_files: tuple[tuple[str, str, Path, Path], ...] = (
        (
            "EN",
            "dev",
            PROJECT_ROOT / "admire2_training" / "EN" / "dev" / "subtask_a_dev.tsv",
            PROJECT_ROOT / "admire2_training" / "EN" / "dev",
        ),
        (
            "PT",
            "dev",
            PROJECT_ROOT / "admire2_training" / "PT" / "dev" / "subtask_a_dev.tsv",
            PROJECT_ROOT / "admire2_training" / "PT" / "dev",
        ),
    )

    test_files: tuple[tuple[str, str, Path, Path], ...] = (
        (
            "EN",
            "test",
            PROJECT_ROOT / "admire2_training" / "EN" / "test" / "subtask_a_test.tsv",
            PROJECT_ROOT / "admire2_training" / "EN" / "test",
        ),
        (
            "PT",
            "test",
            PROJECT_ROOT / "admire2_training" / "PT" / "test" / "subtask_a_test.tsv",
            PROJECT_ROOT / "admire2_training" / "PT" / "test",
        ),
    )

    xeval_files: tuple[tuple[str, str, Path, Path], ...] = (
        (
            "EN",
            "xeval",
            PROJECT_ROOT / "admire2_training" / "EN" / "xeval" / "subtask_a_xe.tsv",
            PROJECT_ROOT / "admire2_training" / "EN" / "xeval",
        ),
        (
            "PT",
            "xeval",
            PROJECT_ROOT / "admire2_training" / "PT" / "xeval" / "subtask_a_xp.tsv",
            PROJECT_ROOT / "admire2_training" / "PT" / "xeval",
        ),
    )

    @property
    def split_files(self) -> dict[str, tuple[tuple[str, str, Path, Path], ...]]:
        return {
            "train": self.train_files,
            "dev": self.dev_files,
            "test": self.test_files,
            "xeval": self.xeval_files,
        }


# =============================================================================
# Submission config
# =============================================================================

@dataclass(frozen=True)
class SubmissionConfig:
    num_candidates: int = 5
    batch_size: int = 8

    tsv_dir: Path = PROJECT_ROOT / "admire2_submission" / "admire2_sentences"
    data_dir: Path = PROJECT_ROOT / "admire2_submission" / "admire2_data"

    output_dir: Path = PROJECT_ROOT / "final_submission"
    debug_log_path: Path = output_dir / "pipeline_debug.json"
    zip_path: Path = output_dir / "submission.zip"

    language_code_map: dict[str, str] = None

    def __post_init__(self) -> None:
        if self.language_code_map is None:
            object.__setattr__(
                self,
                "language_code_map",
                {
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
                },
            )


# =============================================================================
# Shared instances
# =============================================================================

QUERY_ENCODER_CONFIG = QueryEncoderConfig()
CAPTION_ENCODER_CONFIG = CaptionEncoderConfig()
IMAGE_ENCODER_CONFIG = ImageEncoderConfig()

STAGE1_SIGNAL_CONFIG = Stage1SignalConfig()

QUERY_FUSION_CONFIG = QueryFusionConfig()
RANKING_MODEL_CONFIG = RankingModelConfig()
RANKING_PATHS = RankingPaths()
TRAINING_DATA_CONFIG = TrainingDataConfig()

SUBMISSION_CONFIG = SubmissionConfig()