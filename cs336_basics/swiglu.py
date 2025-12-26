import torch
from torch import nn
from cs336_basics.linear import Linear
from einops import einsum
from jaxtyping import Float
import numpy as np


def silu(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return x / (1 + torch.exp(-x))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Construct a FFN.
        
        :param self: Description
        :param d_model: Hidden dimension of the model.
        :type d_model: int
        """
        super().__init__()
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.w1 = Linear(
            in_features=self.d_model,
            out_features=self.d_ff
        )
        self.w2 = Linear(
            in_features=self.d_ff,
            out_features=self.d_model
        )
        self.w3 = Linear(
            in_features=self.d_model,
            out_features=self.d_ff
        )
    

    def forward(self, in_features: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
        temp1 = silu(self.w1.forward(in_features))
        temp2 = temp1 * self.w3.forward(in_features)
        return self.w2.forward(temp2)
