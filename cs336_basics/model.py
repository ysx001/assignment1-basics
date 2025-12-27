import torch
from torch import nn
from jaxtyping import Float, Bool, Int
from einops import einsum, reduce, repeat, rearrange
import math
import numpy as np

from .nn_utils import softmax


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    return einsum(
        softmax(scores, dim=-1),
        V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )


def silu(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return x / (1 + torch.exp(-x))

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
        self.weight = nn.Parameter(
            torch.Tensor(np.ones(d_model))
        )
    
    def forward(self, x: Float[torch.Tensor, "b n d"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        b, n, _ = x.shape
        rms = torch.sqrt(
            reduce(torch.square(x), "b n d -> b n", "mean") + self.eps
        )
        rms = repeat(rms, "b n -> b n d", d=self.d_model)
        gain = repeat(self.weight, "d -> b n d", b=b, n=n)
        result = x / rms * gain
        return result.to(in_dtype)

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


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module.
        
        :param num_embeddings: int Size of the vocabulary
        :param embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        :param device: torch.device | None = None Device to store the parameters on
        :param dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.num_embeddings, self.embedding_dim),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation of the input"""
        # token_ids has shape (batch_size, sequence_length)
        # embeddings had shape (vocab_size, d_model)
        return self.embeddings[token_ids.long()]


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


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Construct causal multi-head self-attention.

        :param self: Description
        :param d_model: Dimensionality of the Transformer block inputs.
        :type d_model: int
        :param num_heads: Number of heads to use in multi-head self-attention.
        :type num_heads: int
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads
        self.q_proj = Linear(
            in_features=self.d_model,
            out_features=self.num_heads * self.d_k, # d_model
        )
        self.k_proj = Linear(
            in_features=self.d_model,
            out_features=self.num_heads * self.d_k, # d_model
        )
        self.v_proj = Linear(
            in_features=self.d_model,
            out_features=self.num_heads * self.d_v, # d_model

        )
        self.output_proj = Linear(
            in_features=self.num_heads * self.d_v, # d_model
            out_features=self.d_model,
        )
    
    def forward(
            self,
            x: Float[torch.Tensor,  " ... seq_len d_model"],
            rope: RoPE = None,
            token_positions: Int[torch.Tensor, "... seq_len"] = None
        ):
        seq_len = x.shape[-2]
        q = self.q_proj(x) # ... seq_len d_model, ... d_model (n d_k) -> ... seq_len (n d_k)
        k = self.k_proj(x) # ... seq_len d_model, ... d_model (n d_k) -> ... seq_len (n d_k)
        v = self.v_proj(x) # ... seq_len d_model, ... d_model (n d_v) -> ... seq_len (n d_v)

        q = rearrange(q, "... seq_len (n d_k) -> ... n seq_len d_k", n=self.num_heads)
        k = rearrange(k, "... seq_len (n d_k) -> ... n seq_len d_k", n=self.num_heads)
        v = rearrange(v, "... seq_len (n d_v) -> ... n seq_len d_v", n=self.num_heads)

        mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        if rope is not None and token_positions is not None:
            q = rope.forward(q, token_positions=token_positions)
            k = rope.forward(k, token_positions=token_positions)

        multi_head = scaled_dot_product_attention(q, k, v, mask=mask) # ... n seq_len d_v
        multi_head = rearrange(multi_head, "... n seq_len d_v -> ... seq_len (n d_v)")
        return self.output_proj(multi_head)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int):
        """
        Construct a pre-norm TranformerBlock.

        :param self: Description
        :param d_model: Dimensionality of the Transformer block inputs.
        :type d_model: int
        :param num_heads: Number of heads to use in multi-head self-attention.
        :type num_heads: int
        :param d_ff: Dimensionality of the position-wise feed-forward inner layer.
        :type d_ff: int
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model=self.d_model)
        self.ln2 = RMSNorm(d_model=self.d_model)
        self.attn = CausalMultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads
        )
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff)
    
    def forward(self,
                x: Float[torch.Tensor, "batch_size seq_len d_model"],
                rope: RoPE | None = None,
                token_positions: Int[torch.Tensor, "... seq_len"] | None = None
                ):
        x = x + self.attn(self.ln1(x), rope, token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

