from collections.abc import Callable, Iterable
from typing import Optional

import torch
from torch import nn
import math



class SGD(torch.optim.Optimizer):
    def __init__(self, params: list[nn.Parameter], lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    

    def step(self, closure: Optional[Callable] = None):
        """
        Make one update of the parameters.
        During the training loop, this will be called after the backward pass.
        So it will have access to the gradients on the last batch.
        This method should iterate through each parameter tensor p and modify them in place.
        i.e. setting p.data, which holds the tensor associated with that parameter based
        on the gradient p.grad (if it exisits), the tensor representing the gradient
        of the loss with respect to that parameter.
        
        :param self: Description
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]    # get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]   # get state associated with p.
                t = state.get("t", 0)   # get iteration number from the state, or initial value.
                grad = p.grad.data      # get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad      # update weight tensor in place.
                state["t"] = t + 1      # increment iteration number.
        return loss



class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: list[nn.Parameter],
            lr: float,
            weight_decay: float,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8
    ):
        """
        Construct an AdamW optimizer.
        AdamW proposes a mofification to Adam that improves regularization
        by adding weight decay (at each iteration, we pull the parameters
        towards 0), in a way that is decoupled from the gradient update.
        
        :param self: Description
        :param params: Description
        :type params: list[nn.Parameter]
        :param weight_decay: weight decay rate.
        :type weight_decay: float
        :param beta_1: Contol the updates to the first moment estimates
        :type beta_1: float
        :param beta_2: Contol the updates to the second moment estimates
        :type beta_2: float
        :param eps: Improve numerical stability in case we get etremely small values in v.
        :type eps: float
        """
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m = state.get("m", 0)                       # get the first moment from state, or initial value
                m = beta_1 * m + (1 - beta_1) * grad        # update the first momemt estimate
                v = state.get("v", 0)                       # get the second moment from state, or initial value
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)   # update the second moment estimate
                t = state.get("t", 1)                       # get the current step, or initial value
                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)  # update the parameters
                p.data -= lr * weight_decay * p.data        # apply weight decay
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
        
# if __name__ == "__main__":
#     weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
#     opt = SGD([weights], lr=1e3)

#     for t in range(100):
#         opt.zero_grad()     # reset the gradients for all learnable parameters.
#         loss = (weights**2).mean()  # compute a scalar loss value.
#         print(loss.cpu().item())
#         loss.backward()     # run backward pass, which computes gradients.
#         opt.step()          # run optimizer step.
