from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import QueryEncoderConfig
from src.query_encoder import QueryEncoder


def main() -> None:
    config = QueryEncoderConfig(
        model_name="xlm-roberta-base",
        shared_dim=512,
        max_length=128,
        normalize=True,
        debug=True,
    )

    encoder = QueryEncoder(config)
    encoder.eval()

    sentences = [
        "It took a lot of elbow grease to get the old engine running again.",
        "I am a night owl, so I usually work after midnight.",
        "Even the failed private eye has a heart of gold.",
    ]

    compounds = [
        "elbow grease",
        "night owl",
        "heart of gold",
    ]

    with torch.no_grad():
        encoder(sentences, compounds)

    print("Demo finished.")


if __name__ == "__main__":
    main()
