from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineRanker(nn.Module):
    """Compute cosine scores between query and candidate representations."""

    def __init__(self, logit_scale: float = 20.0, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            # CLIP-style initialization: ln(1/0.07) ~ 2.659
            # This makes exp(logit_scale) ~ 14.3 initially.
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)
        else:
            self.register_buffer("logit_scale", torch.tensor(logit_scale))

    def forward(
        self,
        query_repr: torch.Tensor,
        candidate_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_repr: [B, D]
            candidate_repr: [B, K, D]

        Returns:
            scores: [B, K]
        """
        if query_repr.dim() != 2 or candidate_repr.dim() != 3:
            raise ValueError("Expected query_repr [B, D] and candidate_repr [B, K, D].")

        if query_repr.size(0) != candidate_repr.size(0):
            raise ValueError("Batch sizes must match.")

        if query_repr.size(-1) != candidate_repr.size(-1):
            raise ValueError("Embedding dimensions must match.")

        query_repr = F.normalize(query_repr, p=2, dim=-1)
        candidate_repr = F.normalize(candidate_repr, p=2, dim=-1)

        # [B, 1, D] * [B, K, D] -> [B, K, D] -> [B, K]
        cosine_sim = torch.sum(
            query_repr.unsqueeze(1) * candidate_repr,
            dim=-1,
        )

        if self.logit_scale.requires_grad:
            return cosine_sim * self.logit_scale.exp()
        return cosine_sim * self.logit_scale


def scores_to_ranked_indices(scores: torch.Tensor) -> list[list[int]]:
    """Convert scores [B, K] to descending candidate index rankings."""
    if scores.dim() != 2:
        raise ValueError("scores must have shape [B, K].")

    return scores.argsort(dim=-1, descending=True).cpu().tolist()


def scores_to_image_order(
    scores: torch.Tensor,
    candidate_names: list[list[str]],
) -> list[list[str]]:
    """Convert scores into ranked image filename order."""
    ranked_indices = scores_to_ranked_indices(scores)

    if len(ranked_indices) != len(candidate_names):
        raise ValueError("Batch sizes must match.")

    predicted_orders: list[list[str]] = []
    for sample_indices, sample_names in zip(ranked_indices, candidate_names):
        predicted_orders.append([sample_names[index] for index in sample_indices])

    return predicted_orders


def build_prediction_records(
    sample_ids: list[str],
    candidate_names: list[list[str]],
    scores: torch.Tensor,
) -> list[dict[str, Any]]:
    """Build inspectable prediction records from model scores."""
    predicted_orders = scores_to_image_order(
        scores=scores,
        candidate_names=candidate_names,
    )
    score_lists = scores.detach().cpu().tolist()

    records: list[dict[str, Any]] = []
    for sample_id, predicted_order, sample_scores in zip(
        sample_ids,
        predicted_orders,
        score_lists,
    ):
        records.append(
            {
                "sample_id": sample_id,
                "predicted_order": predicted_order,
                "scores": sample_scores,
            }
        )

    return records
