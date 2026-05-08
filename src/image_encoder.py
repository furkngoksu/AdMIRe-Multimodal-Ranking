from __future__ import annotations

from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.config import ImageEncoderConfig


ImageInput = Union[str, Path, Image.Image]


class ImageEncoder(nn.Module):
    """Encodes candidate images into image vectors."""

    def __init__(self, config: ImageEncoderConfig = ImageEncoderConfig()) -> None:
        super().__init__()

        self.config = config
        self.encoder = SentenceTransformer(config.model_name)

        raw_dim = self.encoder.get_sentence_embedding_dimension()
        if raw_dim is None:
            raise ValueError("Could not infer image embedding dimension.")

        if raw_dim != config.shared_dim:
            raise ValueError(
                f"Expected image embedding dim {config.shared_dim}, got {raw_dim}."
            )

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    @staticmethod
    def flatten_images(images: List[List[ImageInput]]) -> tuple[List[ImageInput], int, int]:
        """Convert images from [B][K] to [B*K]."""
        if len(images) == 0:
            raise ValueError("images cannot be empty.")

        batch_size = len(images)
        num_candidates = len(images[0])

        if num_candidates == 0:
            raise ValueError("each example must contain at least one image.")

        for row in images:
            if len(row) != num_candidates:
                raise ValueError(
                    "All examples must have the same number of candidate images."
                )

        flat_images = [image for row in images for image in row]
        return flat_images, batch_size, num_candidates

    @staticmethod
    def load_image(image_input: ImageInput) -> Image.Image:
        """Load an image path or normalize an existing PIL image to RGB."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        image_path = Path(image_input)
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def forward(self, images: List[List[ImageInput]]) -> torch.Tensor:
        """Encode candidate images into image vectors [B, K, shared_dim]."""
        flat_images, batch_size, num_candidates = self.flatten_images(images)
        pil_images = [self.load_image(image_input) for image_input in flat_images]

        device = next(self.encoder.parameters()).device

        with torch.no_grad():
            raw_embeddings = self.encoder.encode(
                pil_images,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        raw_embeddings = raw_embeddings.to(device)
        i = F.normalize(raw_embeddings, p=2, dim=-1)
        i = i.view(batch_size, num_candidates, self.config.shared_dim)

        if self.config.debug:
            print(
                "[ImageEncoder] "
                f"batch={batch_size} candidates={num_candidates} "
                f"images_encoded={len(flat_images)} device={i.device}"
            )
            print("[ImageEncoder] images -> I")
            preview = i[0, 0, :6].detach().cpu().tolist()
            preview = [round(value, 4) for value in preview]
            print(f"[ImageEncoder] first image vector preview={preview}")

        return i
