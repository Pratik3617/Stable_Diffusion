import torch
from torch import nn
from attention import SelfAttention

class VAE_Attention_Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, Features, height, width)

        residue = x

        n, c, h, w = x.shape

        # (batch_size, Features, height, width) -> (batch_size, Features, height * width)
        x = x.view(n, c, h*w)

        # (batch_size, Features, height * width) -> (batch_size, height * width, Features)
        x = x.transpose(-1, -2)

        # (batch_size, height * width, Features) -> (batch_size, height * width, Features)
        x = self.attention(x) # query, key and value are same

        # (batch_size, height * width, Features) -> (batch_size, Features, height * width)
        x = x.transpose(-1, -2)

        # (batch_size, Features, height * width) -> (batch_size, Features, height, width)
        x = x.view(n, c, h, w)

        x += residue

        return x
