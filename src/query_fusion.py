from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryFusion(nn.Module):
    """Inject the idiomaticity signal into the query representation."""

    def __init__(self, shared_dim: int = 512) -> None:
        super().__init__()

        self.shared_dim = shared_dim
        self.m_projection = nn.Linear(1, shared_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        query_repr: torch.Tensor,
        idiom_signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_repr: [B, D]
            idiom_signal: [B, 1]

        Returns:
            fused_query: [B, D]
        """
        if query_repr.dim() != 2:
            raise ValueError("query_repr must have shape [B, D].")

        if query_repr.size(-1) != self.shared_dim:
            raise ValueError("query_repr last dimension must match shared_dim.")

        if idiom_signal.dim() != 2 or idiom_signal.size(-1) != 1:
            raise ValueError("idiom_signal must have shape [B, 1].")

        if idiom_signal.size(0) != query_repr.size(0):
            raise ValueError("Batch sizes must match.")

        m_repr = self.m_projection(idiom_signal.float())
        fused_query = query_repr + self.alpha * m_repr

        return F.normalize(fused_query, p=2, dim=-1)
