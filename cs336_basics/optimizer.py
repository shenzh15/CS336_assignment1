from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["t"] += 1
                t = state["t"]
                grad = p.grad.data
                m = state["m"]
                v = state["v"]
                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                lr_t = lr * math.sqrt(1 - betas[1]**(t)) / (1 - betas[0]**(t))
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                p.data.add_(p.data, alpha=-weight_decay*lr)
        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    Args:
        it: Current iteration number (t)
        max_learning_rate: Maximum learning rate (lr_max)
        min_learning_rate: Minimum learning rate (lr_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Total number of iterations for cosine cycle (T_c)
    Returns:
        Learning rate for the current iteration
    """
    # Warm-up phase: If t < Tw, then lr_t = (t/Tw) * lr_max
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    # Cosine annealing phase: If Tw ≤ t ≤ Tc
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    # Post-annealing phase: If t > Tc, then lr_t = lr_min
    else:
        return min_learning_rate

def clip_grad_norm(parameters, max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Clip gradients by L2 norm.
    Args:
        parameters: Iterable of parameters whose gradients will be clipped
        max_l2_norm: Maximum L2 norm of gradients
        eps: Small epsilon value to avoid division by zero (PyTorch default: 1e-6)
    """
    # Calculate total L2 norm in a single pass
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += (p.grad.data ** 2).sum().item()
    if total_norm == 0.0:
        return
    total_norm = math.sqrt(total_norm)
    # Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + eps)
    # Only clip if current norm exceeds max_l2_norm
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
