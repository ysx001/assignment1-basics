import torch
from torch import nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
import math

class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
        ):
        """
        Construct the RoPE module and create buffers if needed
        
        :param theta: theta value for the RoPE
        :type theta: float
        :param d_k: dimension of query and key vectors
        :type d_k: int
        :param max_seq_len: maximum sequence length that will be inputted
        :type max_seq_len: int
        :param device: device to store the buffer on
        """
        super().__init__()
        # 1. Create indices: [0, 2, 4, ..., d_k-2]
        # 2. Divide by d_k: [0/d, 2/d, ...]
        # This represents the "(2k-2)/d" exponent
        exponent = torch.arange(0, d_k, 2) / d_k
        # 3. Exponentiate: 1.0 / (base ** exponent)
        # This creates the vector of theta_i values
        # x^y = exp(y ln(x)) for numerical stablity
        inv_freq = torch.pow(theta, -exponent)
        # inv_freq = torch.exp(-exponent * math.log(theta)) # vector of size d_k / 2

        # positions: [0, 1, 2, ..., max_seq_len]
        i = torch.arange(max_seq_len) # vector of size L

        angles = einsum(i, inv_freq, "L, dk_2 -> L dk_2")  # matrix of shape (L, dk_2)
        # angles = i * inv_freq
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
    
    def forward(
            self,
            x: Float[torch.Tensor, "... seq_len dk"],
            token_positions: Int[torch.Tensor, "... seq_len"]
        ) -> Float[torch.Tensor, "... seq_len dk"]:
        """
        Process an input tensor of shape (..., seq_len, d_k)
        and return a tensor of the same shape.
        
        :param self: Description
        :param x: Input tensor.
        :type x: torch.Tensor
        :param token_positions: Specify the token positions of x along the sequence dimension.
        :type token_positions: torch.Tensor
        :return: A output tensor of the same shape as the input tensor
        :rtype: Tensor
        """
        cos_i = self.cos[token_positions.long()]
        sin_i = self.sin[token_positions.long()]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rot_even = x_even * cos_i - x_odd * sin_i
        rot_odd = x_even * sin_i + x_odd * cos_i

        stacked_rot = rearrange([rot_even, rot_odd], "s ... -> ... s") # (2, ..., seq_len, d_k // 2) -> (..., seq_len, d_k // 2, 2)
        return rearrange(stacked_rot, "... d1 d2 -> ... (d1 d2)")