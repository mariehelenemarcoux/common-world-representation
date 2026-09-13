import torch
from cwr.core import MetaConsciousCore

def test_meta_conscious_core_forward():
    embed_dim = 32
    batch_size = 2
    model = MetaConsciousCore(embed_dim=embed_dim)
    dummy_input = torch.randn(batch_size, embed_dim)
    output = model(dummy_input)
    
    assert "aligned_output" in output
    assert "ethical_mass" in output
    assert output["aligned_output"].shape == (batch_size, embed_dim)
