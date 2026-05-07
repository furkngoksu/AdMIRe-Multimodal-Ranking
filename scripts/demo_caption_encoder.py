from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.caption_encoder import CaptionEncoder
from src.config import CaptionEncoderConfig


def main() -> None:
    config = CaptionEncoderConfig(
        model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        shared_dim=512,
        max_length=128,
        debug=True,
    )

    encoder = CaptionEncoder(config)
    encoder.eval()

    captions = [
        [
            "The image depicts a hand holding a sponge and cleaning a glass cooktop stove.",
            "The image depicts a hand wearing a yellow work glove holding a rusty metal pipe.",
            "The image depicts a hand holding a duster with a long black and orange handle.",
            "The image depicts a person wearing knee pads and hiking shoes.",
            "The image depicts a person holding a bowl of honey.",
        ],
        [
            "The image depicts a cartoon-style illustration of a person working on a computer at night.",
            "The image depicts an owl perched on a branch under a bright moon.",
            "The image depicts a child sitting on a couch in pajamas.",
            "The image depicts a dumbbell used for strength training.",
            "The image depicts an autumn landscape with an owl in a tree.",
        ],
    ]

    with torch.no_grad():
        t = encoder(captions)

    expected_shape = (2, 5, config.shared_dim)
    if tuple(t.shape) != expected_shape:
        raise RuntimeError(f"Unexpected caption vector shape: {tuple(t.shape)}")

    norms = t.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise RuntimeError("Caption vectors are not L2-normalized.")

    print("Caption vector demo finished.")


if __name__ == "__main__":
    main()
