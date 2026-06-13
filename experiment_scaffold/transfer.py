"""Backbone checkpoint transfer + freezing for the age->VWM sequential cells.

Save an age-pretrained GCN backbone, reload it into a fresh target model
(``strict=True`` is safe: identity and glm_diagonal carriers are both 400-dim,
so the backbone input dim is identical across all 11 cells), and freeze it for
the frozen-feature modes (B2, B4).
"""
from __future__ import annotations

import torch


def save_backbone(backbone, path) -> None:
    """Persist just the backbone ``state_dict`` to ``path``."""
    torch.save(backbone.state_dict(), str(path))


def load_backbone(backbone, path, strict: bool = True):
    """Load a saved backbone ``state_dict`` into ``backbone`` (returns it)."""
    backbone.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True), strict=strict)
    return backbone


def freeze(module):
    """Set ``requires_grad=False`` on every parameter (frozen-feature mode)."""
    return set_trainable(module, False)


def set_trainable(module, flag: bool):
    """Set ``requires_grad=flag`` on every parameter of ``module`` (returns it)."""
    for p in module.parameters():
        p.requires_grad_(flag)
    return module
