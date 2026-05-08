from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_weights(
    modules: dict[str, nn.Module],
    path: str,
) -> None:
    weights = {
        name: module.state_dict()
        for name, module in modules.items()
    }

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    torch.save(weights, path_obj)


def load_weights(
    modules: dict[str, nn.Module],
    path: str,
    device: torch.device,
) -> None:
    weights = torch.load(path, map_location=device)

    for name, module in modules.items():
        module.load_state_dict(weights[name])
