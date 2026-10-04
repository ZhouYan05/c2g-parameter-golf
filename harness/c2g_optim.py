"""Optimizers: AdamW baseline and Muon (Newton-Schulz orthogonalized momentum).

Muon follows the modded-nanoGPT implementation referenced by the challenge:
https://github.com/KellerJordan/Muon
"""
import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = NS_COEFFS
    X = G.to(torch.float32)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum SGD with per-matrix orthogonalized updates (2D params only)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5 if p.ndim == 2 else 1.0
                p.add_(upd, alpha=-lr * scale)


def build_optimizers(model, lr, opt, weight_decay=0.0, muon_lr=None):
    """Split params into matrix (>=2D) and vector/scalar groups."""
    matrices = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    others = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    seen = set()
    uniq_matrices = []
    for p in matrices:
        if id(p) not in seen:
            seen.add(id(p))
            uniq_matrices.append(p)
    optimizers = []
    names = []
    if opt == "muon":
        optimizers.append(Muon(uniq_matrices, lr=muon_lr if muon_lr else 20 * lr))
        names.append("muon")
        optimizers.append(torch.optim.AdamW(others, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay))
        names.append("adamw_scalars")
    else:
        optimizers.append(torch.optim.AdamW(uniq_matrices + others, lr=lr, betas=(0.9, 0.95),
                                           eps=1e-8, weight_decay=weight_decay))
        names.append("adamw")
    return optimizers, names


def lr_at(step, total, base_lr, warmup, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    import math
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
