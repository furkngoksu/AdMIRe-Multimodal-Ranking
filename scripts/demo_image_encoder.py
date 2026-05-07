from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ImageEncoderConfig
from src.image_encoder import ImageEncoder


def make_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (224, 224), color=color)


def main() -> None:
    config = ImageEncoderConfig(
        model_name="sentence-transformers/clip-ViT-B-32",
        shared_dim=512,
        normalize=True,
        freeze_encoder=True,
        debug=True,
    )

    encoder = ImageEncoder(config)
    encoder.eval()

    images = [
        [
            make_image((220, 40, 40)),
            make_image((40, 180, 80)),
            make_image((40, 80, 220)),
            make_image((220, 180, 40)),
            make_image((160, 80, 200)),
        ],
        [
            make_image((30, 30, 30)),
            make_image((240, 240, 240)),
            make_image((120, 120, 120)),
            make_image((240, 120, 40)),
            make_image((40, 200, 200)),
        ],
    ]

    with torch.no_grad():
        i = encoder(images)

    expected_shape = (2, 5, config.shared_dim)
    if tuple(i.shape) != expected_shape:
        raise RuntimeError(f"Unexpected image vector shape: {tuple(i.shape)}")

    norms = i.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise RuntimeError("Image vectors are not L2-normalized.")

    print("Image vector demo finished.")


if __name__ == "__main__":
    main()
