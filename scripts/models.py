"""
Two heads map CLIP 512-d features → d-dim embeddings:
    EuclideanHead:  u ∈ R^d
    PoincareHead:   v ∈ R^d_c 
"""

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn

try:
    import geoopt
except ModuleNotFoundError:
    geoopt = None


BALL_EPS = 1e-5


def project_to_poincare_ball(
    x: torch.Tensor,
    curvature: float = 1.0,
    eps: float = BALL_EPS,
) -> torch.Tensor:
    if curvature <= 0:
        raise ValueError("curvature must be positive for Poincare geometry")
    max_norm = (1.0 - eps) / (curvature ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def poincare_expmap0(
    u: torch.Tensor,
    curvature: float = 1.0,
    eps: float = BALL_EPS,
) -> torch.Tensor:
    sqrt_c = torch.as_tensor(curvature, device=u.device, dtype=u.dtype).sqrt()
    norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    mapped = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)
    return project_to_poincare_ball(mapped, curvature=curvature, eps=eps)


def poincare_logmap0(
    x: torch.Tensor,
    curvature: float = 1.0,
    eps: float = BALL_EPS,
) -> torch.Tensor:
    x = project_to_poincare_ball(x, curvature=curvature, eps=eps)
    sqrt_c = torch.as_tensor(curvature, device=x.device, dtype=x.dtype).sqrt()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scaled_norm = (sqrt_c * norm).clamp_max(1.0 - eps)
    return torch.atanh(scaled_norm) * x / (sqrt_c * norm)


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float = 1.0,
    eps: float = BALL_EPS,
) -> torch.Tensor:
    x = project_to_poincare_ball(x, curvature=curvature, eps=eps)
    y = project_to_poincare_ball(y, curvature=curvature, eps=eps)
    x_sq = x.pow(2).sum(dim=-1)
    y_sq = y.pow(2).sum(dim=-1)
    diff_sq = (x - y).pow(2).sum(dim=-1)
    denom = (1.0 - curvature * x_sq).clamp_min(1e-12)
    denom = denom * (1.0 - curvature * y_sq).clamp_min(1e-12)
    arg = 1.0 + 2.0 * curvature * diff_sq / denom
    sqrt_c = torch.as_tensor(curvature, device=x.device, dtype=x.dtype).sqrt()
    return torch.acosh(arg.clamp_min(1.0 + eps)) / sqrt_c


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
        self.curvature = curvature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = self.mlp(z)
        norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(norm, max=self.MAX_TANGENT_NORM) / norm
        u = u * scale
        return poincare_expmap0(u, curvature=self.curvature)


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
        self.curvature = curvature
        init = torch.randn(num_classes, dim) * init_scale
        if geometry == "euclidean":
            self.prototypes = nn.Parameter(init)
        elif geometry == "hyperbolic":
            init = project_to_poincare_ball(init, curvature=curvature)
            if geoopt is None:
                self.prototypes = nn.Parameter(init)
            else:
                ball = geoopt.PoincareBall(c=curvature)
                self.prototypes = geoopt.ManifoldParameter(init, manifold=ball)
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B, d); prototypes: (K, d)
        if self.geometry == "euclidean":
            dists = torch.cdist(v, self.prototypes)  # (B, K)
        else:
            v_exp = v.unsqueeze(1)  # (B, 1, d)
            p_exp = self.prototypes.unsqueeze(0)  # (1, K, d)
            dists = poincare_distance(v_exp, p_exp, curvature=self.curvature)
        return -dists  # logits
