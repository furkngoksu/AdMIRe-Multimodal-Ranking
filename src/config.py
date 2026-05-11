from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PipelineDataConfig:
    tsv_path: str = "admire2_submission/admire2_sentences/submission_Turkish.tsv"
    image_root: str = "admire2_submission/admire2_data/Turkish"
    m_scores_path: str = "stage1_idiom_signal_final/output/submission_m_scores.json"
    language: str = "Turkish"
    num_candidates: int = 5
    batch_size: int = 1


@dataclass
class WeightsConfig:
    ranking_weights_path: str = "ranking_experiments/gold_m_ablation_pca/output/m64/seed100/weights.pt"


@dataclass
class QueryEncoderConfig:
    model_name: str = "xlm-roberta-base"
    shared_dim: int = 512
    max_length: int = 128
    query_projection_type: str = "pca"  # "pca" or "linear"
    debug: bool = True


@dataclass
class CaptionEncoderConfig:
    model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    shared_dim: int = 512
    max_length: int = 128
    debug: bool = False


@dataclass
class ImageEncoderConfig:
    model_name: str = "sentence-transformers/clip-ViT-B-32"
    shared_dim: int = 512
    debug: bool = False


@dataclass
class QueryFusionConfig:
    shared_dim: int = 512
    m_embedding_dim: int = 64
    fusion_type: str = "concat"  # "concat", "gated", "film", "film_full"


PIPELINE_DATA_CONFIG = PipelineDataConfig()
WEIGHTS_CONFIG = WeightsConfig()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
