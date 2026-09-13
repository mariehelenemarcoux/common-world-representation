import torch
import torch.nn as nn

class HarmonicResonanceModule(nn.Module):
    def __init__(self, embed_dim: int, base_freq: float = 963.0):
        super().__init__()
        self.base_freq = base_freq
        self.freq_weights = nn.Parameter(torch.randn(embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        harmonic_signal = torch.sin(2 * torch.pi * self.base_freq * self.freq_weights)
        return x * (1.0 + 0.05 * harmonic_signal)
