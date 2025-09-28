import torch 
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # respresenting matrices W_k, W_q and W_v using one Linear Layer instead of representing them seprately
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # W_o matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads

        # dimension of each head
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch_size, Seq_Len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, Dim*3) -> 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunks(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, H, dim/H)   H -> no. of heads
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # mask where the upper triangle is made up of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, Dim/H) -> (batch_size, H, seq_len, Dim/H)
        output = weight @ v

        # (batch_size, H, seq_len, Dim/H) -> (batch_size, seq_len, H, Dim/H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output