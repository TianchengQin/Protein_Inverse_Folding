import torch
import torch.nn as nn
from models.common import ResidualBlock2d


class Heteroencoder(nn.Module):
    def __init__(self, latent_dim=256,dropout_rate=0.2):
        super(Heteroencoder, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock2d(1, 32, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),
            ResidualBlock2d(32, 64, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),
            ResidualBlock2d(64, 128, dropout_rate=dropout_rate),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, latent_dim)
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.tanh = nn.Tanh()
    def forward(self, x):

        feat = self.encoder(x)
        B = x.size(0)
        feat_flat = feat.view(B, -1)
        latent= self.fc_layers(feat_flat)

        latent = self.latent_norm(latent)
        return latent
