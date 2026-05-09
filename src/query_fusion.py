from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import QueryFusionConfig


class QueryFusion(nn.Module):
    """Enrich Q with an idiomaticity probability before candidate matching."""

    def __init__(self, config: QueryFusionConfig = QueryFusionConfig()) -> None:
        super().__init__()

        self.config = config

        if config.m_embedding_dim > 0:
            self.m_embedding = nn.Sequential(
                nn.Linear(1, config.m_embedding_dim),
                nn.Tanh(),
            )
            self.enrichment = nn.Linear(
                config.shared_dim + config.m_embedding_dim,
                config.shared_dim,
            )
            self.reset_enrichment_parameters()
        else:
            self.m_embedding = None
            self.enrichment = None

    def reset_enrichment_parameters(self) -> None:
        """Start close to Q while leaving a small trainable M path."""
        if self.enrichment is None:
            return

        with torch.no_grad():
            self.enrichment.weight.zero_()
            self.enrichment.bias.zero_()
            self.enrichment.weight[:, : self.config.shared_dim].copy_(
                torch.eye(self.config.shared_dim)
            )
            self.enrichment.weight[:, self.config.shared_dim :].normal_(
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        query_repr: torch.Tensor,
        idiom_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_repr: [B, D]
            idiom_prob: [B, 1], idiomatic probability in [0, 1].

        Returns:
            enriched_query: [B, D]
        """
        if query_repr.dim() != 2:
            raise ValueError("query_repr must have shape [B, D].")

        if query_repr.size(-1) != self.config.shared_dim:
            raise ValueError("query_repr last dimension must match shared_dim.")

        if self.config.m_embedding_dim == 0:
            return F.normalize(query_repr, p=2, dim=-1)

        m_embedding = self.m_embedding
        enrichment = self.enrichment
        if m_embedding is None or enrichment is None:
            raise RuntimeError("QueryFusion M branch is not initialized.")

        if idiom_prob.dim() == 1:
            idiom_prob = idiom_prob.unsqueeze(-1)

        if idiom_prob.dim() != 2 or idiom_prob.size(-1) != 1:
            raise ValueError("idiom_prob must have shape [B, 1].")

        if idiom_prob.size(0) != query_repr.size(0):
            raise ValueError("Batch sizes must match.")

        idiom_input = 2.0 * idiom_prob.float() - 1.0
        m_emb = m_embedding(idiom_input)

        enriched_query = enrichment(
            torch.cat([query_repr, m_emb], dim=-1)
        )

        return F.normalize(enriched_query, p=2, dim=-1)
