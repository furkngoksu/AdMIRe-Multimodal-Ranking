from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.config import CaptionEncoderConfig


class CaptionEncoder(nn.Module):
    """Encodes candidate captions into caption vectors."""

    def __init__(self, config: CaptionEncoderConfig = CaptionEncoderConfig()) -> None:
        super().__init__()

        self.config = config
        self.encoder = SentenceTransformer(config.model_name)

        self.encoder.max_seq_length = config.max_length

        raw_dim = self.encoder.get_sentence_embedding_dimension()
        if raw_dim is None:
            raise ValueError("Could not infer sentence embedding dimension.")

        self.projection = nn.Linear(raw_dim, config.shared_dim)
        if raw_dim == config.shared_dim:
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(raw_dim))
                self.projection.bias.zero_()

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    @staticmethod
    def flatten_captions(captions: List[List[str]]) -> tuple[List[str], int, int]:
        """Convert captions from [B][K] to [B*K]."""
        if len(captions) == 0:
            raise ValueError("captions cannot be empty.")

        batch_size = len(captions)
        num_candidates = len(captions[0])

        if num_candidates == 0:
            raise ValueError("each example must contain at least one caption.")

        for row in captions:
            if len(row) != num_candidates:
                raise ValueError(
                    "All examples must have the same number of candidate captions."
                )

        flat_captions = [caption for row in captions for caption in row]
        return flat_captions, batch_size, num_candidates

    def forward(self, captions: List[List[str]]) -> torch.Tensor:
        """Encode candidate captions into caption vectors [B, K, shared_dim]."""
        flat_captions, batch_size, num_candidates = self.flatten_captions(captions)

        device = next(self.projection.parameters()).device

        with torch.no_grad():
            raw_embeddings = self.encoder.encode(
                flat_captions,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        raw_embeddings = raw_embeddings.to(device)
        t = self.projection(raw_embeddings)
        t = F.normalize(t, p=2, dim=-1)
        t = t.view(batch_size, num_candidates, self.config.shared_dim)

        if self.config.debug:
            print(
                "[CaptionEncoder] "
                f"batch={batch_size} candidates={num_candidates} "
                f"captions_encoded={len(flat_captions)} device={t.device}"
            )
            print("[CaptionEncoder] captions -> T")
            preview = t[0, 0, :6].detach().cpu().tolist()
            preview = [round(value, 4) for value in preview]
            print(f"[CaptionEncoder] first caption vector preview={preview}")

        return t
