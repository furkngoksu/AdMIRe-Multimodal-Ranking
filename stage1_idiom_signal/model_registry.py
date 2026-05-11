from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

def mark_compound(sentence: str, compound: str) -> str:
    """Helper to add <mwe> tags around the compound in a sentence."""
    if not compound: return sentence
    match = re.search(re.escape(compound), sentence, flags=re.IGNORECASE)
    if not match: return sentence
    s, e = match.span()
    return f"{sentence[:s]}<mwe>{sentence[s:e]}</mwe>{sentence[e:]}"

class BaseMModelEngine:
    """Base class for all M-Signal model inference characteristics."""
    def prepare_texts(self, lang: str, sent: str, comp: str, loader: Any) -> List[str]:
        raise NotImplementedError
    
    def get_features(self, pooled_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def build_classifier(self, input_dim: int, params: Dict[str, Any]) -> nn.Module:
        """Default MLP with BatchNorm and GELU."""
        return nn.Sequential(
            nn.Linear(input_dim, params["hidden_1"]), 
            nn.BatchNorm1d(params["hidden_1"]), 
            nn.GELU(), 
            nn.Dropout(params["dropout_1"]),
            nn.Linear(params["hidden_1"], params["hidden_2"]), 
            nn.BatchNorm1d(params["hidden_2"]), 
            nn.GELU(), 
            nn.Dropout(params["dropout_2"]),
            nn.Linear(params["hidden_2"], 1),
        )

class BaselineEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        return [sent]
    def get_features(self, pooled_vectors):
        return pooled_vectors[:, 0]

class MarkedMLPEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        marked = mark_compound(sent, comp)
        return [f"[LANG={lang}] {marked} [MWE] {comp}"]
    def get_features(self, pooled_vectors):
        return pooled_vectors[:, 0]

class MarkedSimpleEngine(MarkedMLPEngine):
    def build_classifier(self, input_dim: int, params: Dict[str, Any]) -> nn.Module:
        """Simple ReLU architecture used by marked_simple."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), 
            nn.ReLU(), 
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(input_dim // 2, 1),
        )

class FusionMLPEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        marked = mark_compound(sent, comp)
        rewrites = loader.get_rewrites(sent)
        return [
            f"[LANG={lang}] {marked} [MWE] {comp}", # Hybrid marked (h)
            sent, # Raw target (t)
            rewrites["lex_sub"],
            rewrites["comp_drop"],
            rewrites["paraphrase"],
            rewrites["literalize"]
        ]
    def get_features(self, pooled_vectors):
        h = pooled_vectors[:, 0] 
        t = pooled_vectors[:, 1]
        diffs = [torch.abs(t - pooled_vectors[:, i]) for i in range(2, 6)]
        return torch.cat([h] + diffs, dim=-1)

class PerturbationMLPEngine(BaseMModelEngine):
    def prepare_texts(self, lang, sent, comp, loader):
        rewrites = loader.get_rewrites(sent)
        return [
            sent, # target (t)
            rewrites["lex_sub"],
            rewrites["comp_drop"],
            rewrites["paraphrase"],
            rewrites["literalize"]
        ]
    def get_features(self, pooled_vectors):
        t = pooled_vectors[:, 0]
        # Features: concatenated abs diffs between t and the 4 rewrites
        diffs = [torch.abs(t - pooled_vectors[:, i]) for i in range(1, 5)]
        return torch.cat(diffs, dim=-1)

# Registry to map model names to their engines
MODEL_REGISTRY: Dict[str, BaseMModelEngine] = {
    "baseline": BaselineEngine(),
    "marked_mlp": MarkedMLPEngine(),
    "marked_simple": MarkedSimpleEngine(),
    "fusion_mlp": FusionMLPEngine(),
    "perturbation_mlp": PerturbationMLPEngine(),
}
