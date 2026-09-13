import torch
import torch.nn as nn


class DabrowskiKohlbergGating(nn.Module):
    """
    Filtrage fondé sur la désintégration positive (Dąbrowski - Niveau IV)
    et les stades de jugement moral (Kohlberg).
    Gouverne les représentations selon le niveau de maturité éthique.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, ethical_mass: torch.Tensor) -> torch.Tensor:
        """
        Module l'entrée par la masse éthique calculée.

        Args:
            x (torch.Tensor): Empreinte vectorielle.
            ethical_mass (torch.Tensor): Masse conative (0 à 1).

        Returns:
            torch.Tensor: Empreinte filtrée par le Troisième Facteur.
        """
        gate = self.activation(self.gate_layer(x)) * ethical_mass
        return x * gate
