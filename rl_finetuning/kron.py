"""HybridKronAdamW: PSGD-Kron for 2D non-embedding params, AdamW for the rest.

Mirrors HybridSOAPAdamW's classification scheme so the comparison vs SOAP is
clean. Uses heavyball.PSGDKron under the hood.

For FSDP2 compatibility we swap DTensor params for their local shards on each
rank (same shim as soap.py). AdamW already handles DTensor natively in modern
PyTorch, but the sub-optimizer is built with local tensors here so the param
groups stay consistent across both branches.
"""

import torch
import torch.optim as optim
import heavyball

_EMBED_DIM_THRESHOLD = 100_000


def _to_local(p):
    return p.to_local() if hasattr(p, "to_local") else p


class HybridKronAdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 5e-5,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        # Kron-specific knobs:
        max_size_triangular: int = 2048,
        min_ndim_triangular: int = 2,
        precond_lr: float = 0.1,
        merge_dims: bool = False,
        preconditioner_update_probability=None,
        adamw_betas=(0.9, 0.999),
    ):
        all_params = list(params)
        # Classify by FULL shape — DTensor.shape is the unsharded logical shape.
        adamw_full, kron_full = [], []
        _shape_counts = {}
        for p in all_params:
            key = (p.ndim, tuple(p.shape))
            _shape_counts[key] = _shape_counts.get(key, 0) + 1
            if p.ndim != 2 or max(p.shape) > _EMBED_DIM_THRESHOLD:
                adamw_full.append(p)
            else:
                kron_full.append(p)
        print(f"[HybridKronAdamW] param shape histogram: {sorted(_shape_counts.items())[:20]} (total={len(all_params)})")

        self._adamw = optim.AdamW(
            adamw_full, lr=lr, betas=adamw_betas, weight_decay=weight_decay, eps=eps
        )
        kron_kwargs = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            precond_lr=precond_lr,
            merge_dims=merge_dims,
            multi_tensor=False,
            compile_step=False,
        )
        if preconditioner_update_probability is not None:
            kron_kwargs["preconditioner_update_probability"] = preconditioner_update_probability
        self._kron = heavyball.PSGDKron(kron_full, **kron_kwargs)

        super().__init__(
            [{"params": adamw_full, "kind": "adamw"}, {"params": kron_full, "kind": "kron"}],
            {"lr": lr, "weight_decay": weight_decay},
        )

        n_a = sum(p.numel() for p in adamw_full)
        n_k = sum(p.numel() for p in kron_full)
        print(f"[HybridKronAdamW] AdamW params: {len(adamw_full)} ({n_a/1e6:.1f}M), Kron params: {len(kron_full)} ({n_k/1e6:.1f}M)")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # AdamW: works on DTensors natively.
        self._adamw.step()

        # PSGDKron: doesn't speak DTensor (no sharding strategy for logsumexp etc).
        # Swap DTensor params → local shards on each rank for the duration of
        # kron.step(), migrating optimizer state alongside, then swap back.
        swaps = []
        for group in self._kron.param_groups:
            params = group["params"]
            for i, p in enumerate(params):
                if not hasattr(p, "to_local"):
                    continue
                loc = p.to_local()
                if p.grad is not None and hasattr(p.grad, "to_local"):
                    loc.grad = p.grad.to_local()
                else:
                    loc.grad = p.grad
                if p in self._kron.state and loc not in self._kron.state:
                    self._kron.state[loc] = self._kron.state.pop(p)
                params[i] = loc
                swaps.append((params, i, p, loc))

        try:
            self._kron.step()
        finally:
            for params, i, p, loc in swaps:
                params[i] = p
                if loc in self._kron.state and p not in self._kron.state:
                    self._kron.state[p] = self._kron.state.pop(loc)
                loc.grad = None
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._adamw.zero_grad(set_to_none=set_to_none)
        self._kron.zero_grad(set_to_none=set_to_none)
