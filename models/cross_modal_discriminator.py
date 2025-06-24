# 包含深度模型定义（判别器）

import torch
import torch.nn as nn

class CrossModalDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        return self.fc(torch.cat([z1, z2], dim=1))