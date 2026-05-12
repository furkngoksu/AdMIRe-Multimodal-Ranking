from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import QueryFusionConfig


class BaseQueryFusion(nn.Module):
    """Base class for Query-Idiom Fusion."""
    def __init__(self, config: QueryFusionConfig):
        super().__init__()
        self.config = config

    def forward(self, query_repr: torch.Tensor, idiom_prob: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConcatQueryFusion(BaseQueryFusion):
    """Enrich Q by projecting M to a vector and concatenating."""
    def __init__(self, config: QueryFusionConfig) -> None:
        super().__init__(config)
        if config.m_embedding_dim > 0:
            self.m_embedding = nn.Sequential(
                nn.Linear(1, config.m_embedding_dim),
                nn.Tanh(),
            )
            self.enrichment = nn.Linear(
                config.shared_dim + config.m_embedding_dim,
                config.shared_dim,
            )
            self._reset_enrichment_parameters()
        else:
            self.m_embedding = None
            self.enrichment = None

    def _reset_enrichment_parameters(self) -> None:
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

    def forward(self, query_repr: torch.Tensor, idiom_prob: torch.Tensor) -> torch.Tensor:
        if self.config.m_embedding_dim == 0:
            return F.normalize(query_repr, p=2, dim=-1)
        
        idiom_input = 2.0 * idiom_prob.float() - 1.0
        m_emb = self.m_embedding(idiom_input)
        enriched_query = self.enrichment(torch.cat([query_repr, m_emb], dim=-1))
        return F.normalize(enriched_query, p=2, dim=-1)


class GatedQueryFusion(BaseQueryFusion):
    """Gate Q dimensions using a vector learned from M probability/embedding."""
    def __init__(self, config: QueryFusionConfig) -> None:
        super().__init__(config)
        self.input_dim = config.m_embedding_dim if config.m_embedding_dim > 0 else 1
        
        if config.m_embedding_dim > 0:
            self.m_embedding = nn.Sequential(
                nn.Linear(1, config.m_embedding_dim),
                nn.Tanh(),
            )
        else:
            self.m_embedding = None

        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, config.shared_dim),
            nn.Sigmoid()
        )
        
        # Initialize gate to be mostly open (high bias)
        with torch.no_grad():
            self.gate[0].weight.zero_()
            self.gate[0].bias.fill_(2.0)

    def forward(self, query_repr: torch.Tensor, idiom_prob: torch.Tensor) -> torch.Tensor:
        idiom_input = 2.0 * idiom_prob.float() - 1.0
        
        if self.m_embedding is not None:
            m_feat = self.m_embedding(idiom_input)
        else:
            m_feat = idiom_input
            
        gate_val = self.gate(m_feat)
        enriched_query = query_repr * gate_val
        return F.normalize(enriched_query, p=2, dim=-1)


class FiLMScaleQueryFusion(BaseQueryFusion):
    """Scale Q features using M score (Scale-only FiLM)."""
    def __init__(self, config: QueryFusionConfig) -> None:
        super().__init__(config)
        self.input_dim = config.m_embedding_dim if config.m_embedding_dim > 0 else 1
        
        if config.m_embedding_dim > 0:
            self.m_embedding = nn.Sequential(
                nn.Linear(1, config.m_embedding_dim),
                nn.Tanh(),
            )
        else:
            self.m_embedding = None

        self.gamma_net = nn.Linear(self.input_dim, config.shared_dim)
        
        with torch.no_grad():
            self.gamma_net.weight.zero_()
            self.gamma_net.bias.zero_()

    def forward(self, query_repr: torch.Tensor, idiom_prob: torch.Tensor) -> torch.Tensor:
        if self.config.m_embedding_dim == 0:
            return F.normalize(query_repr, p=2, dim=-1)

        idiom_input = 2.0 * idiom_prob.float() - 1.0
        m_feat = self.m_embedding(idiom_input)

        gamma = self.gamma_net(m_feat)
        enriched_query = (1.0 + gamma) * query_repr
        return F.normalize(enriched_query, p=2, dim=-1)


class FiLMFullQueryFusion(BaseQueryFusion):
    """Scale and shift Q features using M score (Full FiLM)."""
    def __init__(self, config: QueryFusionConfig) -> None:
        super().__init__(config)
        self.input_dim = config.m_embedding_dim if config.m_embedding_dim > 0 else 1
        
        if config.m_embedding_dim > 0:
            self.m_embedding = nn.Sequential(
                nn.Linear(1, config.m_embedding_dim),
                nn.Tanh(),
            )
        else:
            self.m_embedding = None

        self.gamma_net = nn.Linear(self.input_dim, config.shared_dim)
        self.beta_net = nn.Linear(self.input_dim, config.shared_dim)
        
        with torch.no_grad():
            self.gamma_net.weight.zero_()
            self.gamma_net.bias.zero_()
            self.beta_net.weight.zero_()
            self.beta_net.bias.zero_()

    def forward(self, query_repr: torch.Tensor, idiom_prob: torch.Tensor) -> torch.Tensor:
        idiom_input = 2.0 * idiom_prob.float() - 1.0
        
        if self.m_embedding is not None:
            m_feat = self.m_embedding(idiom_input)
        else:
            m_feat = idiom_input

        gamma = self.gamma_net(m_feat)
        beta = self.beta_net(m_feat)
        
        enriched_query = (1.0 + gamma) * query_repr + beta
        return F.normalize(enriched_query, p=2, dim=-1)


def build_query_fusion(config: QueryFusionConfig = QueryFusionConfig()) -> BaseQueryFusion:
    """Factory function to build the correct QueryFusion module."""
    if config.fusion_type == "concat":
        return ConcatQueryFusion(config)
    elif config.fusion_type == "gated":
        return GatedQueryFusion(config)
    elif config.fusion_type == "film":
        return FiLMScaleQueryFusion(config)
    elif config.fusion_type == "film_full":
        return FiLMFullQueryFusion(config)
    else:
        raise ValueError(f"Unknown fusion_type: {config.fusion_type}")
