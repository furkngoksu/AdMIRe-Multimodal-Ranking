from __future__ import annotations

import torch
import torch.nn as nn


class IdiomSignal(nn.Module):
    """Compute the idiomaticity signal M for each sentence-compound pair."""

    def forward(
        self,
        sentences: list[str],
        compounds: list[str],
    ) -> torch.Tensor:
        """
        Args:
            sentences: batch of context sentences
            compounds: batch of target compounds

        Returns:
            idiom_signal: [B, 1]
        """
        raise NotImplementedError("IdiomSignal must return M with shape [B, 1].")
