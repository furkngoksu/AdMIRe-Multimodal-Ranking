from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PipelineDataConfig:
    tsv_path: str = "admire2_submission/admire2_sentences/submission_Turkish.tsv"
    image_root: str = "admire2_submission/admire2_data/Turkish"
    language: str = "Turkish"
    num_candidates: int = 5
    batch_size: int = 1


@dataclass(frozen=True)
class WeightsConfig:
    idiom_signal_dir: str = "stage1_idiom_signal/output"
    ranking_weights_path: str = "ranking_experiments/oracle_ablation_pca/output/m64/seed100/weights.pt"


@dataclass(frozen=True)
class QueryEncoderConfig:
    model_name: str = "xlm-roberta-base"
    shared_dim: int = 512
    max_length: int = 128
    query_projection_type: str = "pca"  # "pca" or "linear"
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


@dataclass(frozen=True)
class QueryFusionConfig:
    shared_dim: int = 512
    m_embedding_dim: int = 64
    fusion_type: str = "concat"  # "concat", "gated", "film"


PIPELINE_DATA_CONFIG = PipelineDataConfig()
WEIGHTS_CONFIG = WeightsConfig()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
