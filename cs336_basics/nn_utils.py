import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
import math
from typing import Iterable


def softmax(v: torch.Tensor, dim: int):
    max_v = torch.max(v, dim=dim, keepdim=True).values
    v_exp = torch.exp(v - max_v)
    return  v_exp / torch.sum(v_exp, dim=dim, keepdim=True)


def cross_entropy(
        logits: Float[torch.Tensor, "batch_size ... vocab_size"],
        targets: Float[torch.Tensor, "batch_size ..."]
    ) -> Float[torch.Tensor, "1"]:

    ##### log softmax  #####
    # max along vocab size
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits_exp = torch.exp(logits - max_logits)
    # log sum exp trick
    log_sum_exp = torch.log(torch.sum(logits_exp, dim=-1, keepdim=True))
    # log(a/b) = log a - log b
    log_softmax = logits - max_logits - log_sum_exp
    ##### log softmax  #####

    vocab_size = logits.shape[-1]
    mask = F.one_hot(targets, num_classes=vocab_size).float() # batch_size vocab_size
    ret = einsum(log_softmax, mask, 'batch_size ... vocab_size, batch_size ... vocab_size -> batch_size ...')
    return -ret.mean()


def lr_cosine_schedule(
        t: int,
        warmup_step: int,
        lr_max: float,
        lr_min: float,
        horizon: int,
    ) -> float:
    if t < warmup_step:
        return (t / warmup_step) * lr_max
    elif t >= warmup_step and t <= horizon:
        return lr_min + (1/2) * (1 + math.cos((t - warmup_step) / (horizon - warmup_step) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
    
def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter],
        max_l2_norm: float,
        eps: float = 1e-6
    ):
    # given the gradient for all parameters
    grads = [param.grad for param in parameters if param.grad is not None]

    # we compute its l2 norm
    l2_norm = 0
    for grad in grads:
        l2_norm += (grad**2).sum()
    l2_norm = torch.sqrt(l2_norm)

    if l2_norm >= max_l2_norm:
        factor = max_l2_norm / (l2_norm + eps)
        for grad in grads:
            grad *= factor