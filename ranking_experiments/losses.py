from __future__ import annotations

import torch
import torch.nn.functional as F


def listwise_ranking_loss(
    scores: torch.Tensor,
    gold_relevance: torch.Tensor,
) -> torch.Tensor:
    """
    Listwise ranking loss for one query and five candidates.

    Args:
        scores: [B, K] model scores.
        gold_relevance: [B, K] per-candidate relevance values.

    Returns:
        Scalar loss.
    """
    if scores.shape != gold_relevance.shape:
        raise ValueError("scores and gold_relevance must have the same shape.")

    target_distribution = F.softmax(gold_relevance.float(), dim=-1)
    log_scores = F.log_softmax(scores, dim=-1)

    return -(target_distribution * log_scores).sum(dim=-1).mean()
