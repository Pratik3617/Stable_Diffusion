import torch
from torch import nn
from torch.nn import functional as F

class VAE_Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, Kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, Kernel_size=3, padding=1)

        if in_channels == out_channels:
            # no change if channel dims match
            self.residual_layer = nn.Identity()
        else:
            # project input to correct shape so it can be added later
            self.residual_layer = nn.Conv2d(in_channels, out_channels, Kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, Height, width)
        
        residue = x # save input for skip connection

        # main branch
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        # if the in_channels == out_channels then we could have return this "x + residue"
        return x + self.residual_layer(residue) # skip connection