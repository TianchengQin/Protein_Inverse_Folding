import torch.nn as nn
from models.common import ResidualBlock2d


class ProteinCNNLSTM(nn.Module):
    def __init__(self, max_len=64, num_aa=20, latent_dim=128, dropout_rate=0.3):
        super(ProteinCNNLSTM, self).__init__()
        self.max_len = max_len
        self.num_aa = num_aa
        self.latent_dim = latent_dim

        # encoder: series of ResidualBlock2d with max pooling
        self.encoder = nn.Sequential(
            ResidualBlock2d(1, 32, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),
            ResidualBlock2d(32, 64, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),
            ResidualBlock2d(64, 128, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # project flattened features to latent space
        self.fc_latent = nn.Linear(128 * 8 * 8, latent_dim)

        # decoder: project latent features to sequence representation
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, max_len * 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        # LSTM decoder: 4 layers
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=4, batch_first=True, dropout=dropout_rate)
        # final classifier: maps LSTM output to amino acid logits
        self.classifier = nn.Linear(256, num_aa)

    def encode(self, x):
        # Encode input with CNN blocks and flatten
        x = self.encoder(x)
        latent = self.fc_latent(x)
        return latent

    def decode(self, h):
        B = h.size(0)
        # Project latent vector and reshape to sequence format
        h = self.fc_decode(h)
        h = h.view(B, self.max_len, 256)
        lstm_out, _ = self.lstm(h)
        logits = self.classifier(lstm_out)
        return logits

    def forward(self, dist_input):
        # encode then decode
        latent = self.encode(dist_input)
        logits = self.decode(latent)
        return logits

