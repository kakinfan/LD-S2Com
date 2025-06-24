import torch.nn as nn

class SemanticDecoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 512, 8, 8)
        x = self.decoder(x)
        return x