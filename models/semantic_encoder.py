# 包含深度模型定义（编码器）

import torch.nn as nn
import torchvision.models as models

class SemanticEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉fc
        self.fc = nn.Linear(resnet.fc.in_features, latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z