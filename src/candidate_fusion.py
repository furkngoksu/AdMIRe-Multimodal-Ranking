from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCandidateFusion(nn.Module):
    """Fuse image and caption representations into candidate representations."""

    def __init__(self, shared_dim: int = 512) -> None:
        super().__init__()

        self.shared_dim = shared_dim
        self.gate = nn.Linear(shared_dim * 2, shared_dim)

    def forward(
        self,
        image_repr: torch.Tensor,
        caption_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_repr: [B, K, D]
            caption_repr: [B, K, D]

        Returns:
            candidate_repr: [B, K, D]
        """
        if image_repr.dim() != 3 or caption_repr.dim() != 3:
            raise ValueError("Expected image_repr and caption_repr with shape [B, K, D].")

        if image_repr.shape != caption_repr.shape:
            raise ValueError("image_repr and caption_repr must have the same shape.")

        if image_repr.size(-1) != self.shared_dim:
            raise ValueError("Representation dimension must match shared_dim.")

        gate = torch.sigmoid(
            self.gate(torch.cat([image_repr, caption_repr], dim=-1))
        )
        candidate_repr = gate * image_repr + (1.0 - gate) * caption_repr

        return F.normalize(candidate_repr, p=2, dim=-1)
