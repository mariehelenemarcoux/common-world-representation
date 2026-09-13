import torch
import torch.nn as nn


class HornTorusProjection(nn.Module):
    """
    Projection topologique sur la géométrie du Horn Torus.
    Projette un espace vectoriel d'entrée vers une variété toroïdale
    et calcule la distance à la singularité centrale.
    """

    def __init__(self, input_dim: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Couches de projection spatiale
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant pour la projection topologique.

        Args:
            x (torch.Tensor): Tenseur d'entrée (batch_size, input_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - torus_embedding: Empreinte projetée sur le torus.
                - singularity_dist: Distance scalaire par rapport au centre du torus.
        """
        torus_embedding = self.activation(self.projection(x))

        # Calcul de la distance à la singularité centrale (norme L2)
        singularity_dist = torch.norm(torus_embedding, dim=-1, keepdim=True)

        return torus_embedding, singularity_dist
