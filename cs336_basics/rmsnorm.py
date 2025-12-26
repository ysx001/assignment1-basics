import torch
from torch import nn
from jaxtyping import Float
from einops import reduce, repeat
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device=None, dtype: torch.dtype=None):
        """
        Construct the RMSNorm module.
        
        :param d_model: Hidden dimension of the model
        :type d_model: int
        :param eps: Epsilon value for numerical stablity
        :type eps: float
        :param device: Device to store the parameters on
        :param dtype: Data type of the parameters
        """
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.gain = nn.Parameter(
            torch.Tensor(np.ones(d_model))
        )
    
    def forward(self, x: Float[torch.Tensor, "b n d"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        b, n, _ = x.shape
        rms = np.sqrt(
            reduce(np.square(x), "b n d -> b n", "mean") + self.eps
        )
        rms = repeat(rms, "b n -> b n d", d=self.d_model)
        gain = repeat(self.gain, "d -> b n d", b=b, n=n)
        result = x / rms * gain
        return result.to(in_dtype)