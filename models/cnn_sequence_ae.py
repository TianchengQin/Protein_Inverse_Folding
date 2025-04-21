import torch
import torch.nn as nn
from models.common import ResidualBlock2d


class SequenceAutoencoder(nn.Module):

    def __init__(self, latent_dim=256,dropout_rate = 0.2):
        super(SequenceAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            ResidualBlock2d(1, 32, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),
            ResidualBlock2d(32, 64, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),
            ResidualBlock2d(64, 128, dropout_rate=dropout_rate),
            nn.MaxPool2d(2)
        )


        self.fc_enc = nn.Linear(128 * 8 * 2, latent_dim)

        self.latent_norm = nn.LayerNorm(latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 128 * 8 * 3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),
            # use a sigmoid to output a number between 0, 1
            nn.Sigmoid()
        )
        self.tanh = nn.Tanh()
    def forward(self, x):

        enc = self.encoder(x)

        B = x.size(0)
        enc_flat = enc.view(B, -1)
        latent = self.fc_enc(enc_flat)
        latent = self.latent_norm(latent)

        dec_input = self.fc_dec(latent)
        dec_input = dec_input.view(B, 128, 8, 3)
        x_recon = self.decoder(dec_input)
        return x_recon, latent