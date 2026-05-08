from __future__ import annotations

import torch


def top1_accuracy(
    scores: torch.Tensor,
    gold_order_indices: torch.Tensor,
) -> float:
    """Top-1 image accuracy for ranked candidate scores."""
    predicted_top = scores.argmax(dim=-1)
    gold_top = gold_order_indices[:, 0]

    return (predicted_top == gold_top).float().mean().item()


def ndcg_at_k(
    scores: torch.Tensor,
    gold_relevance: torch.Tensor,
    k: int = 5,
) -> float:
    """Mean nDCG@k for candidate ranking scores."""
    if scores.shape != gold_relevance.shape:
        raise ValueError("scores and gold_relevance must have the same shape.")

    k = min(k, scores.size(1))

    ranked_indices = scores.argsort(dim=-1, descending=True)[:, :k]
    ranked_relevance = gold_relevance.gather(1, ranked_indices)

    ideal_relevance = gold_relevance.sort(dim=-1, descending=True).values[:, :k]

    discounts = torch.log2(
        torch.arange(k, device=scores.device, dtype=torch.float) + 2.0
    )

    dcg = ((2.0**ranked_relevance - 1.0) / discounts).sum(dim=-1)
    ideal_dcg = ((2.0**ideal_relevance - 1.0) / discounts).sum(dim=-1)

    return (dcg / ideal_dcg.clamp(min=1e-9)).mean().item()
