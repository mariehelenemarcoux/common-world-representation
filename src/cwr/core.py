import torch
import torch.nn as nn
from typing import Dict
from .topology.horn_torus import HornTorusProjection
from .topology.harmonic_resonance import HarmonicResonanceModule
from .ethics.conatus_density import ConatusEthicalDensity
from .ethics.third_factor import DabrowskiKohlbergGating

class MetaConsciousCore(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.torus_proj = HornTorusProjection(embed_dim)
        self.harmonic_res = HarmonicResonanceModule(embed_dim)
        self.conatus_density = ConatusEthicalDensity(embed_dim)
        self.gating = DabrowskiKohlbergGating(embed_dim)
        
    def forward(self, input_representations: torch.Tensor) -> Dict[str, torch.Tensor]:
        tuned_x = self.harmonic_res(input_representations)
        toroidal_state, rho_singularity = self.torus_proj(tuned_x)
        densified_x, ethical_mass = self.conatus_density(tuned_x)
        governance = self.gating(densified_x, ethical_mass)
        
        aligned_output = torch.where(
            governance["third_factor_active"].unsqueeze(-1),
            densified_x,
            densified_x * 0.1
        )
        
        return {
            "aligned_output": aligned_output,
            "ethical_mass": ethical_mass,
            "singularity_distance": rho_singularity,
            "level_4_integrity": governance["level_4_score"],
            "third_factor_active": governance["third_factor_active"]
        }
