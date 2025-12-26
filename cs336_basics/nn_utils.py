import torch

def softmax(v: torch.Tensor, dim: int):
    max_v = torch.max(v, dim=dim, keepdim=True).values
    v_exp = torch.exp(v - max_v)
    return  v_exp / torch.sum(v_exp, dim=dim, keepdim=True)
