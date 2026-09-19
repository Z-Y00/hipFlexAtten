"""Sweep the block-sparse kernel knobs the autotuner never gets to see.

Block-sparse pins the kernel tiles to the sparsity granularity, so it bypasses the
autotuner entirely -- which also freezes waves_per_eu / num_warps / num_stages /
PRE_LOAD_V / matrix_instr_nonkdim at whatever the launcher hardcodes. Those are not
forced by the granularity, so they are free to tune. This sweeps them against the
current defaults to see whether any headroom exists.

Pin to an idle GPU (rocm-smi --showuse); a busy neighbour swamps the differences.

Usage: HIP_VISIBLE_DEVICES=5 python3 -m bench.sweep_sparse_knobs
"""

import itertools

import torch

import flex_attention._vendor.aiter_flash_attn.bwd as bwd_mod
import flex_attention._vendor.aiter_flash_attn.fwd_prefill as fwd_mod
from flex_attention import dense_to_block_sparse, flash_attn_func

BATCH, NHEADS, HEAD_DIM, BLOCK = 2, 8, 64, 64
DTYPE = torch.bfloat16


def median_ms(fn, warmup=15, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def build(seqlen, device, keep_every=4):
    nblk = seqlen // BLOCK
    qb = torch.arange(nblk, device=device).view(1, 1, nblk, 1)
    kb = torch.arange(nblk, device=device).view(1, 1, 1, nblk)
    keep = (((qb * 7 + kb) % keep_every) == 0).expand(BATCH, NHEADS, nblk, nblk)
    return dense_to_block_sparse(keep, torch.zeros_like(keep), (BLOCK, BLOCK))


def sweep(seqlen=8192):
    device = "cuda"
    torch.manual_seed(0)
    mk = lambda g: (  # noqa: E731
        torch.randn(BATCH, seqlen, NHEADS, HEAD_DIM, dtype=DTYPE, device=device) * 0.3
    ).requires_grad_(g)
    bs = build(seqlen, device)

    print(f"--- forward, seqlen={seqlen} ---")
    q, k, v = mk(False), mk(False), mk(False)
    base = dict(fwd_mod.SPARSE_FWD_KNOBS)
    results = []
    for waves, warps, stages, preload in itertools.product((1, 2, 3), (4, 8), (1, 2), (False, True)):
        cand = dict(waves_per_eu=waves, PRE_LOAD_V=preload, num_stages=stages, num_warps=warps)
        fwd_mod.SPARSE_FWD_KNOBS.clear(); fwd_mod.SPARSE_FWD_KNOBS.update(cand)
        try:
            with torch.no_grad():
                ms = median_ms(lambda: flash_attn_func(q, k, v, block_sparse_tensors=bs))
        except Exception as exc:
            ms = float("inf"); cand["err"] = type(exc).__name__
        results.append((ms, cand))
    fwd_mod.SPARSE_FWD_KNOBS.clear(); fwd_mod.SPARSE_FWD_KNOBS.update(base)
    results.sort(key=lambda r: r[0])
    cur = [r for r in results if all(r[1].get(kk) == vv for kk, vv in base.items())][0]
    for ms, c in results[:6]:
        tag = "  <-- current default" if c == cur[1] else ""
        print(f"  {ms:8.4f}ms  {c}{tag}")
    print(f"  current default: {cur[0]:.4f}ms   best: {results[0][0]:.4f}ms "
          f"({cur[0] / results[0][0]:.3f}x)")

    print(f"\n--- backward, seqlen={seqlen} ---")
    q, k, v = mk(True), mk(True), mk(True)
    grad = torch.ones(BATCH, seqlen, NHEADS, HEAD_DIM, dtype=DTYPE, device=device)
    base_b = dict(bwd_mod.SPARSE_BWD_KNOBS)
    results = []
    for waves, warps, stages, nonkdim in itertools.product((1, 2), (4, 8), (1, 2), (None, 16)):
        cand = dict(BLK_SLICE_FACTOR=1, waves_per_eu=waves, num_stages=stages, num_warps=warps)
        if nonkdim:
            cand["matrix_instr_nonkdim"] = nonkdim
        bwd_mod.SPARSE_BWD_KNOBS.clear(); bwd_mod.SPARSE_BWD_KNOBS.update(cand)

        def step():
            for t in (q, k, v):
                t.grad = None
            flash_attn_func(q, k, v, block_sparse_tensors=bs).backward(grad)

        try:
            ms = median_ms(step, warmup=8, reps=15)
        except Exception as exc:
            ms = float("inf"); cand["err"] = type(exc).__name__
        results.append((ms, cand))
    bwd_mod.SPARSE_BWD_KNOBS.clear(); bwd_mod.SPARSE_BWD_KNOBS.update(base_b)
    results.sort(key=lambda r: r[0])
    cur = [r for r in results if r[1] == base_b][0]
    for ms, c in results[:6]:
        tag = "  <-- current default" if c == cur[1] else ""
        print(f"  {ms:8.4f}ms  {c}{tag}")
    print(f"  current default: {cur[0]:.4f}ms   best: {results[0][0]:.4f}ms "
          f"({cur[0] / results[0][0]:.3f}x)")


if __name__ == "__main__":
    sweep(8192)
