import torch
from models import BabyAutoEncoder


def test():
    n_in = n_out = 64
    n_latent = 4

    model = BabyAutoEncoder(n_in, n_latent, n_out)

    x = torch.rand(1, n_in)

    out = model(x)
    assert(out.shape == (1, n_out))
