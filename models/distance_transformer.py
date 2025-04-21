import torch
import torch.nn as nn
from models.common import PositionalEncoding


class MultiScaleDistanceTransformer(nn.Module):
    def __init__(self, max_len, num_aa, latent_dim=128, d_model=256, dropout_rate=0.2):
        super(MultiScaleDistanceTransformer, self).__init__()
        # multi-scale convolution branches
        self.branch3 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.branch7 = nn.Conv2d(1, 32, kernel_size=7, padding=3)

        # batchNorm, activation, dropout block
        self.ms_seq = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        # project to latent space
        self.conv_proj = nn.Conv2d(96, latent_dim, kernel_size=1)

        # first convolution block using conv + batch norm, activation, dropout
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(latent_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )

        # residual connection
        self.residual1 = nn.Conv2d(latent_dim, d_model, kernel_size=1)

        # second convolution block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )

        # adaptive pooling to (max_len, 1)
        self.pool = nn.AdaptiveAvgPool2d((max_len, 1))

        # Positional encoding and transformer encoder
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout=dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.dropout_transformer = nn.Dropout(dropout_rate)

        # Final classification layer
        self.fc = nn.Linear(d_model, num_aa)

    def forward(self, dist_input):
        # compute multi-scale features
        b3 = self.branch3(dist_input)
        b5 = self.branch5(dist_input)
        b7 = self.branch7(dist_input)

        # concat multi-scale features
        x = torch.cat([b3, b5, b7], dim=1)
        x = self.ms_seq(x)

        # project features
        x = self.conv_proj(x)

        # convolution block with residual connection
        x_block = self.conv_block1(x)
        res = self.residual1(x)
        x = x_block + res
        x = self.conv_block2(x)

        # pooling and reshape for transformer input
        x = self.pool(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)

        # add positional encoding and pass through transformer
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = self.dropout_transformer(x)

        # classification layer
        logits = self.fc(x)
        return logits
