import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_Attention_Block, VAE_Residual_Block
 
class VAE_Encoder(nn.Sequential):
    """
        We are decreasing the size of the image while increasing the number of features
        i.e the number of pixels will decrease while each pixel preserves more features of the image
    """
    super().__init__(
 
        # (batch_size, channel, height, width) -> (batch_size, 128, height, width)
        nn.Conv2d(3, 128, kernel_size = 3, padding = 1),
 
        # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
        VAE_Residual_Block(128, 128),
 
        # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
        VAE_Residual_Block(128, 128),
 
        # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
        nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),
 
        # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
        VAE_Residual_Block(128, 256),
 
        # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
        VAE_Residual_Block(256, 256),
 
        # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
        nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),
 
        # (batch_size, 128, height/4, width/4) -> (batch_size, 512, height/4, width/4)
        VAE_Residual_Block(256, 512),
 
        # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
        VAE_Residual_Block(512, 512),
 
        # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
        nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0),
 
        # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
        VAE_Residual_Block(512, 512),
 
        # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
        VAE_Residual_Block(512, 512),
 
        # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
        VAE_Residual_Block(512, 512),
 
        # run self attention to each pixels
        # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
        VAE_Attention_Block(512),
 
        # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
        VAE_Residual_Block(512, 512),
 
        # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
        nn.GroupNorm(32, 512),  # Normalization
 
        # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
        nn.SiLU(), # activation
 
        # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
        nn.Conv2d(512, 8, kernel_size = 3, padding = 1),
 
        # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
        nn.Conv2d(8, 8, kernel_size =1, padding = 0),
    )
 
    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, channel, Height, Width)
        # noise : (batch_size, Out_channels, Height/8, width/8)
 
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # (padding_left, padding_top, padding_right, padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
 
        # (batch_size, 8, height/8, width/8) -> two tensors of shape (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
       
        # performing clmp i.e if the variance is too large or small make it to the given range
        log_variance = torch.clamp(log_variance, -30, 20)
 
        variance = log_variance.exp()
 
        stdev = variance.sqrt()
 
        # how to sample from N(0,1) distribution
        # Z = N(0,1) -> N(mean, variance)?
        # X = mean + stdev * Z
 
        x = mean + stdev * noise
 
        # scale the output by a constant
        x *= 0.18215
 
        return x