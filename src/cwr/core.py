import torch
import torch.nn as nn
from cwr.topology.horn_torus import HornTorusProjection
from cwr.topology.harmonic_resonance import HarmonicResonance
from cwr.ethics.conatus_density import ConatusEthicalDensity
from cwr.ethics.third_factor import DabrowskiKohlbergGating


class MetaConsciousCore(nn.Module):
    """
    Noyau d'intégration éthique et topologique.
    Synthétise la projection spatiale (Horn Torus), le filtrage fréquentiel (963 Hz),
    et les mécanismes de maturité éthique (Spinoza / Dąbrowski / Kohlberg).
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        base_frequency: float = 963.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Projections topologiques et résonance
        self.horn_torus = HornTorusProjection(input_dim=input_dim, embedding_dim=hidden_dim)
        self.harmonic_resonance = HarmonicResonance(frequency=base_frequency)

        # Composants d'évaluation éthique
        self.conatus_density = ConatusEthicalDensity(hidden_dim=hidden_dim)
        self.ethics_gating = DabrowskiKohlbergGating(hidden_dim=hidden_dim)

        # Projection de sortie alignée
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Passe avant du modèle.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (batch_size, input_dim).
            
        Returns:
            dict: Dictionnaire contenant les tenseurs de sortie alignés,
                  la masse éthique calculée et les métriques topologiques.
        """
        # 1. Projection sur la topologie du Horn Torus
        torus_embedding, singularity_dist = self.horn_torus(x)

        # 2. Application de l'ancrage fréquentiel (963 Hz)
        resonant_embedding = self.harmonic_resonance(torus_embedding)

        # 3. Calcul de la densité conative (Spinoza)
        ethical_mass = self.conatus_density(resonant_embedding)

        # 4. Filtrage par le Troisième Facteur (Dąbrowski Level IV / Kohlberg)
        gated_embedding = self.ethics_gating(resonant_embedding, ethical_mass)

        # 5. Production du tenseur de sortie final
        aligned_output = self.output_layer(gated_embedding)

        return {
            "aligned_output": aligned_output,
            "ethical_mass": ethical_mass,
            "singularity_distance": singularity_dist,
            "gated_embedding": gated_embedding,
        }

    def compute_loss(
        self,
        outputs: dict,
        target: torch.Tensor,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> torch.Tensor:
        """
        Calcule la perte combinée éthique et topologique pour l'entraînement d'agents.

        Args:
            outputs (dict): Le dictionnaire retourné par forward().
            target (torch.Tensor): Le tenseur cible attendu.
            alpha (float): Poids pour l'erreur de reconstruction/alignement.
            beta (float): Poids pour la pénalité d'invariance éthique.

        Returns:
            torch.Tensor: La valeur scalaire de perte pour la rétropropagation.
        """
        # Perte de reconstruction spatiale / tâche
        task_loss = nn.functional.mse_loss(outputs["aligned_output"], target)

        # Pénalité éthique : maximiser la masse conative et minimiser la déviation topologique
        ethical_penalty = (1.0 - outputs["ethical_mass"].mean()) + outputs["singularity_distance"].mean()

        # Perte globale combinée
        total_loss = (alpha * task_loss) + (beta * ethical_penalty)
        return total_loss
