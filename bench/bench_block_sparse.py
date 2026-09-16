"""Block-sparse attention speedup on MI300X.

Baseline is the *dense* kernel on the same shapes. Two things this measures honestly:

  * Block sparsity only pays off once the KV loop dominates. At seqlen 2048 the kernel is
    launch/occupancy-bound and sparsity is a wash; at 8192 it tracks the density.
  * It is the wrong tool for plain causal masking. The dense kernel already skips blocks
    via its own causal/window block planning, and does so with autotuned tile sizes,
    whereas the sparse path pins tiles to the sparsity granularity. Use block sparsity
    for patterns the dense path cannot express, not to reimplement causal.

Usage: python3 -m bench.bench_block_sparse
"""

import torch
import triton

from flex_attention import dense_to_block_sparse, flash_attn_func

BATCH, NHEADS, HEAD_DIM, BLOCK = 2, 8, 64, 64
DTYPE = torch.bfloat16


def build_mask(seqlen, keep_every, device):
    """A strided block pattern the dense causal/window planner cannot express."""
    nblk = seqlen // BLOCK
    qb = torch.arange(nblk, device=device).view(1, 1, nblk, 1)
    kb = torch.arange(nblk, device=device).view(1, 1, 1, nblk)
    keep = (((qb * 7 + kb) % keep_every) == 0).expand(BATCH, NHEADS, nblk, nblk)
    return dense_to_block_sparse(keep, torch.zeros_like(keep), (BLOCK, BLOCK)), keep.float().mean().item()


def main():
    device = "cuda"
    print(f"{'seqlen':>7s} {'density':>8s} {'dense':>10s} {'sparse':>10s} {'speedup':>8s}  (fwd+bwd)")
    print("-" * 58)
    for seqlen in (2048, 8192):
        for keep_every in (10, 4):
            mk = lambda: (  # noqa: E731
                torch.randn(BATCH, seqlen, NHEADS, HEAD_DIM, dtype=DTYPE, device=device) * 0.3
            ).requires_grad_()
            q, k, v = mk(), mk(), mk()
            scale = HEAD_DIM**-0.5
            bs, density = build_mask(seqlen, keep_every, device)
            grad = torch.ones(BATCH, seqlen, NHEADS, HEAD_DIM, dtype=DTYPE, device=device)

            dense_ms = triton.testing.do_bench(
                lambda: flash_attn_func(q, k, v, softmax_scale=scale).backward(grad),
                warmup=10, rep=30,
            )
            sparse_ms = triton.testing.do_bench(
                lambda: flash_attn_func(
                    q, k, v, softmax_scale=scale, block_sparse_tensors=bs
                ).backward(grad),
                warmup=10, rep=30,
            )
            print(
                f"{seqlen:7d} {density:7.0%} {dense_ms:9.3f}m {sparse_ms:9.3f}m "
                f"{dense_ms / sparse_ms:7.2f}x"
            )


if __name__ == "__main__":
    main()
