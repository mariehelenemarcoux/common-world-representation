import torch
from cwr.core import MetaConsciousCore

def run_agent_training_loop():
    # Initialisation du noyau éthique-topologique
    model = MetaConsciousCore(input_dim=64, hidden_dim=128, output_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Simulation d'une donnée d'entrée et d'une cible
    dummy_input = torch.randn(8, 64)
    dummy_target = torch.randn(8, 64)

    # Étape d'entraînement
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = model.compute_loss(outputs, dummy_target)
    loss.backward()
    optimizer.step()

    print(f"Cycle réussi ! Perte d'entraînement calculée : {loss.item():.4f}")

if __name__ == "__main__":
    run_agent_training_loop()
