from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryEncoderConfig:
    model_name: str = "xlm-roberta-base"
    shared_dim: int = 512
    max_length: int = 128
    normalize: bool = True
    debug: bool = False
