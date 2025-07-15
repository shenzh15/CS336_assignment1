import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_stable = logits - logits_max
    logsumexp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
    target_logits = torch.gather(logits_stable, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = logsumexp - target_logits
    return loss.mean()
