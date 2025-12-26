import torch
from jaxtyping import Float, Bool
from einops import einsum
import math

def softmax(v: torch.Tensor, dim: int):
    max_v = torch.max(v, dim=dim, keepdim=True).values
    v_exp = torch.exp(v - max_v)
    return  v_exp / torch.sum(v_exp, dim=dim, keepdim=True)

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

