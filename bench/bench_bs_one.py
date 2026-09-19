"""Single-metric block-sparse forward timing, for A/B alternation on a shared GPU.

Prints one number (milliseconds) so callers can alternate checkouts process-by-process
and compare medians. Pick the case with `CASE=full|masked`, the length with `SEQ=8192`.

PIN THIS TO AN IDLE GPU. On a shared node, a neighbour saturating the default device
inflates timings by ~40% and swamps the effect being measured -- enough to invert the
apparent sign of a real 20% win. Check `rocm-smi --showuse` and pick a 0% device.

Usage: HIP_VISIBLE_DEVICES=5 SEQ=8192 CASE=full python3 -m bench.bench_bs_one
"""

import os

import torch
import triton
import triton.language as tl

from flex_attention import dense_to_block_sparse, flash_attn_func

BATCH, NHEADS, HEAD_DIM, BLOCK = 2, 8, 64, 64
DTYPE = torch.bfloat16


@triton.jit
def _band_mask_mod(b, h, q_idx, kv_idx):
    d = q_idx - kv_idx
    return (d < 96) & (d > -96)


def main():
    seqlen = int(os.environ.get("SEQ", 8192))
    case = os.environ.get("CASE", "full")
    device = "cuda"
    torch.manual_seed(0)
    mk = lambda: (  # noqa: E731
        torch.randn(BATCH, seqlen, NHEADS, HEAD_DIM, dtype=DTYPE, device=device) * 0.3
    )
    q, k, v = mk(), mk(), mk()
    scale = HEAD_DIM**-0.5
    nblk = seqlen // BLOCK
    qb = torch.arange(nblk, device=device).view(1, 1, nblk, 1)
    kb = torch.arange(nblk, device=device).view(1, 1, 1, nblk)

    if case == "full":
        keep = (((qb * 7 + kb) % 4) == 0).expand(BATCH, NHEADS, nblk, nblk)
        bs = dense_to_block_sparse(keep, torch.zeros_like(keep), (BLOCK, BLOCK))
        mod = None
    else:
        touched = ((qb - kb).abs() <= 1).expand(BATCH, NHEADS, nblk, nblk)
        bs = dense_to_block_sparse(torch.zeros_like(touched), touched, (BLOCK, BLOCK))
        mod = _band_mask_mod

    with torch.no_grad():
        ms = triton.testing.do_bench(
            lambda: flash_attn_func(
                q, k, v, softmax_scale=scale, mask_mod=mod, block_sparse_tensors=bs
            ),
            warmup=50, rep=200,
        )
    print(f"{ms:.5f}")


if __name__ == "__main__":
    main()
