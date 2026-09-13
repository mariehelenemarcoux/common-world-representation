import torch
import torch.nn as nn
from typing import Tuple

class ConatusEthicalDensity(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.density_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ethical_mass = self.density_estimator(x)
        densified_representation = x * ethical_mass
        return densified_representation, ethical_mass
