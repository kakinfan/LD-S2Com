# 扩散过程模块

import torch
import torch.nn as nn

class SimpleDiffusion(nn.Module):
    def __init__(self, latent_dim, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1. - self.betas, dim=0))
    
    def q_sample(self, z, t, noise=None):
        # 正向扩散：
        # z: [batch, latent_dim]
        # t: [batch]
        if noise is None:
            noise = torch.randn_like(z)
        # 变成 [batch, 1]，以便广播
        sqrt_alpha_cumprod = self.alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]).unsqueeze(1)
        return sqrt_alpha_cumprod * z + sqrt_one_minus_alpha_cumprod * noise

    def p_sample(self, z_t, t, model):
        # 反向扩散：用模型预测噪声，然后还原
        pred_noise = model(z_t, t)
        sqrt_recip_alpha_cumprod = (1. / self.alphas_cumprod[t]) ** 0.5
        sqrt_recipm1_alpha_cumprod = (1. / self.alphas_cumprod[t] - 1) ** 0.5
        z_0 = sqrt_recip_alpha_cumprod * z_t - sqrt_recipm1_alpha_cumprod * pred_noise
        return z_0