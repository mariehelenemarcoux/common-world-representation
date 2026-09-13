import torch
import torch.nn as nn
from typing import Tuple

class HornTorusProjection(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.to_toroidal = nn.Linear(embed_dim, 3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.tanh(self.to_toroidal(x))
        r, theta, phi = coords[..., 0], coords[..., 1], coords[..., 2]
        rho = torch.abs(r)
        
        toroidal_embedding = torch.cat([
            (1 + torch.cos(theta)) * torch.cos(phi).unsqueeze(-1),
            (1 + torch.cos(theta)) * torch.sin(phi).unsqueeze(-1),
            torch.sin(theta).unsqueeze(-1)
        ], dim=-1)
        
        return toroidal_embedding, rho
