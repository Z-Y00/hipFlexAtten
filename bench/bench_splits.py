"""Split-KV (num_splits) occupancy benefit on MI300X.

Splitting only helps when the natural grid -- batch * nheads * ceil(seqlen_q/BLOCK_M) --
is too small to fill the GPU's 304 CUs. That is the short-query / long-context shape
(decode, prefix reuse). With a large grid it is pure overhead, which the wide grid row
below is included to show.

Usage: python3 -m bench.bench_splits
"""

import torch
import triton

from flex_attention import flash_attn_func

DTYPE = torch.bfloat16
HEAD_DIM = 128

# (batch, seqlen_q, seqlen_k, nheads, label)
CASES = [
    (1, 1, 8192, 8, "decode: 1 query, 8K ctx"),
    (1, 1, 32768, 8, "decode: 1 query, 32K ctx"),
    (1, 128, 16384, 8, "short prefix: 128q, 16K ctx"),
    (8, 4096, 4096, 32, "wide grid (splitting should not help)"),
]


def main():
    device = "cuda"
    print(f"{'case':38s} {'grid':>7s} " + " ".join(f"{f'split={s}':>10s}" for s in (1, 2, 4, 8, 16)))
    print("-" * 100)
    for batch, sq, sk, nheads, label in CASES:
        q = torch.randn(batch, sq, nheads, HEAD_DIM, dtype=DTYPE, device=device) * 0.3
        k = torch.randn(batch, sk, nheads, HEAD_DIM, dtype=DTYPE, device=device) * 0.3
        v = torch.randn(batch, sk, nheads, HEAD_DIM, dtype=DTYPE, device=device) * 0.3
        scale = HEAD_DIM**-0.5
        grid = batch * nheads * max(1, (sq + 127) // 128)
        times = []
        with torch.no_grad():
            for s in (1, 2, 4, 8, 16):
                ms = triton.testing.do_bench(
                    lambda s=s: flash_attn_func(q, k, v, softmax_scale=scale, num_splits=s),
                    warmup=20, rep=50,
                )
                times.append(ms)
        best = min(times)
        cells = " ".join(f"{t:9.3f}m" + ("*" if t == best else " ") for t in times)
        print(f"{label:38s} {grid:7d} {cells}")
    print("\n* = fastest for that shape")


if __name__ == "__main__":
    main()
