import torch
from torch import nn 


class BabyAutoEncoder(nn.Module):
    """Pretty much the simplest autoencoder you can imagine."""

    def __init__(self, n_in, n_latent, n_out):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_in, n_latent),
            nn.ReLU(inplace=True),
            nn.Linear(n_latent, n_out)
        )
    
    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, x):
        ...  # TODO: reconstruction loss
