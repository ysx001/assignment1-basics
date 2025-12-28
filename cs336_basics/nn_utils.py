import torch
from jaxtyping import Float
import torch.nn.functional as F
from einops import einsum

def softmax(v: torch.Tensor, dim: int):
    max_v = torch.max(v, dim=dim, keepdim=True).values
    v_exp = torch.exp(v - max_v)
    return  v_exp / torch.sum(v_exp, dim=dim, keepdim=True)


def cross_entropy(
        logits: Float[torch.Tensor, "batch_size ... vocab_size"],
        targets: Float[torch.Tensor, "batch_size ..."]
    ) -> Float[torch.Tensor, "1"]:
    # max along vocab size
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits_exp = torch.exp(logits - max_logits)
    # log sum exp trick
    log_sum_exp = torch.log(torch.sum(logits_exp, dim=-1, keepdim=True))
    # log(a/b) = log a - log b
    log_softmax = logits - max_logits - log_sum_exp
    vocab_size = logits.shape[-1]
    mask = F.one_hot(targets, num_classes=vocab_size).float() # batch_size vocab_size
    ret = einsum(log_softmax, mask, 'batch_size ... vocab_size, batch_size ... vocab_size -> batch_size ...')
    return -ret.mean()