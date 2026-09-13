import torch
import torch.nn as nn
import numpy as np


class HarmonicResonance(nn.Module):
    """
    Module de résonance harmonique fondé sur l'ancrage fréquentiel à 963 Hz.
    Applique une modulation sinusoïdale aux représentations vectorielles
    pour stabiliser l'alignement fréquentiel du système.
    """

    def __init__(self, frequency: float = 963.0):
        super().__init__()
        self.frequency = frequency
        # Facteur d'échelle dérivé de la fréquence harmonique
        self.scale_factor = float(np.pi * (frequency / 1000.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applique la modulation harmonique au tenseur d'entrée.

        Args:
            x (torch.Tensor): Tenseur d'entrée.

        Returns:
            torch.Tensor: Tenseur modulé par la fréquence de résonance.
        """
        resonance_mask = torch.sin(x * self.scale_factor) + 1.0
        return x * (0.5 * resonance_mask)
