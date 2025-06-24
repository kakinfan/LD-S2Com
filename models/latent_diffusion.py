import torch
import torch.nn as nn

class LatentDiffusion(nn.Module):
    def __init__(self, encoder, decoder, diffusion, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        z_noisy, noise = self.diffusion.q_sample(z)
        z_denoised = self.diffusion.p_sample(z_noisy)
        x_recon = self.decoder(z_denoised)
        return x_recon, noise