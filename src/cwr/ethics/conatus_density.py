import torch
import torch.nn as nn


class ConatusEthicalDensity(nn.Module):
    """
    Module d'évaluation de la densité éthique fondé sur le principe du Conatus (Spinoza).
    Mesure l'effort d'auto-conservation et d'intégrité systémique au sein des représentations vectorielles.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Reseau d'estimation de la masse conative
        self.density_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule la masse éthique globale (valeur entre 0 et 1).

        Args:
            x (torch.Tensor): Empreinte vectorielle issue de la projection harmonique.

        Returns:
            torch.Tensor: Masse éthique scalaire ou par batch.
        """
        return self.density_network(x)
