"""
Two heads map CLIP 512-d features → d-dim embeddings:
    EuclideanHead:  u ∈ R^d
    PoincareHead:   v ∈ R^d_c 
"""

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import geoopt


class MLP(nn.Module):
    def __init__(self, d_in: int = 512, d_hidden: int = 256, d_out: int = 8, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EuclideanHead(nn.Module):
    def __init__(self, d_in: int = 512, d_hidden: int = 256, d_out: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mlp = MLP(d_in, d_hidden, d_out, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class PoincareHead(nn.Module):
    MAX_TANGENT_NORM = 15.0  # prevents tanh saturation at the ball boundary

    def __init__(
        self,
        d_in: int = 512,
        d_hidden: int = 256,
        d_out: int = 8,
        dropout: float = 0.1,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.mlp = MLP(d_in, d_hidden, d_out, dropout)
        self.ball = geoopt.PoincareBall(c=curvature)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = self.mlp(z)
        norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(norm, max=self.MAX_TANGENT_NORM) / norm
        u = u * scale
        return self.ball.expmap0(u)


class ClassifierLayer(nn.Module):

    def __init__(
        self,
        num_classes: int = 27,
        dim: int = 8,
        geometry: Literal["euclidean", "hyperbolic"] = "euclidean",
        curvature: float = 1.0,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.geometry = geometry
        init = torch.randn(num_classes, dim) * init_scale
        if geometry == "euclidean":
            self.prototypes = nn.Parameter(init)
            self.ball = None
        elif geometry == "hyperbolic":
            self.ball = geoopt.PoincareBall(c=curvature)
            self.prototypes = geoopt.ManifoldParameter(init, manifold=self.ball)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B, d); prototypes: (K, d)
        if self.geometry == "euclidean":
            # ‖v - p‖₂ via broadcasting
            dists = torch.cdist(v, self.prototypes)  # (B, K)
        else:
            # ball.dist supports broadcasting; expand to (B, K, d)
            v_exp = v.unsqueeze(1)              # (B, 1, d)
            p_exp = self.prototypes.unsqueeze(0)  # (1, K, d)
            dists = self.ball.dist(v_exp, p_exp)  # (B, K)
        return -dists  # logits
