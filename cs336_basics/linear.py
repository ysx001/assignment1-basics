import torch
from torch import nn
import numpy as np
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation model.
        
        Params:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        w_sigma = np.sqrt(2 / in_features + out_features)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features),
                std=w_sigma,
                a=-3 * w_sigma,
                b=3 * w_sigma)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation of the input"""
        return einsum(x, self.weight, 
                      "... d_in, d_out d_in -> ... d_out")
    
