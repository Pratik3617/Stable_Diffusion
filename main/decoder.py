import torch
from torch import nn

from common.vae_attention_block import VAE_Attention_Block
from common.vae_residual_block import VAE_Residual_Block


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_Residual_Block(512, 512),

            VAE_Attention_Block(512),

            VAE_Residual_Block(512, 512),
            VAE_Residual_Block(512, 512),
            VAE_Residual_Block(512, 512),

            # (batch_Size, 512, height/8, width/8) -> (batch_Size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),

           # (batch_Size, 512, height/8, width/8) -> (batch_Size, 512, height/4, width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_Residual_Block(512, 512),
            VAE_Residual_Block(512, 512),
            VAE_Residual_Block(512, 512),

            # (batch_Size, 512, height/4, width/4) -> (batch_Size, 512, height/2, width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_Residual_Block(512, 256),
            VAE_Residual_Block(256, 256),
            VAE_Residual_Block(256, 256),

            # (batch_Size, 256, height/2, width/2) -> (batch_Size, 256, height, width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_Residual_Block(256, 128),
            VAE_Residual_Block(128, 128),
            VAE_Residual_Block(128, 128),

            nn.GroupNorm(32, 128),

            nn.silu(),

            # (batch_Size, 128, height, width) -> (batch_Size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, height/8, width/8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # (batch_size, 3, height, width)
        return x
    