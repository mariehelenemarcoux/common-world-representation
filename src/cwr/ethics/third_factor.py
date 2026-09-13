import torch
import torch.nn as nn
from typing import Dict

class DabrowskiKohlbergGating(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.third_factor_evaluator = nn.Linear(embed_dim + 1, 2)
        
    def forward(self, action_embedding: torch.Tensor, ethical_mass: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([action_embedding, ethical_mass], dim=-1)
        logits = self.third_factor_evaluator(combined)
        probs = torch.softmax(logits, dim=-1)
        
        level_1_2_utility = probs[..., 0]
        level_4_integrity = probs[..., 1]
        third_factor_active = level_4_integrity > level_1_2_utility
        
        return {
            "level_1_2_score": level_1_2_utility,
            "level_4_score": level_4_integrity,
            "third_factor_active": third_factor_active
        }
