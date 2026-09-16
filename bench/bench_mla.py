"""MLA / large-head-dim throughput on MI300X.

No AITER baseline column here: upstream AITER has no MLA path to compare against (its
tuned configs assume head_dim <= 256 and simply run out of LDS beyond that), so this
measures our own shapes against each other. The `blk` column is the sequence-block size
the LDS cap allows -- see flex_attention/_vendor/aiter_flash_attn/utils.py.

Usage: python3 -m bench.bench_mla
"""

import torch
import triton

from flex_attention import flash_attn_func
from flex_attention._vendor.aiter_flash_attn.utils import max_block_for_lds

SHAPES = [
    (128, 128, "baseline 128/128"),
    (192, 128, "MLA 192/128 (DeepSeek)"),
    (256, 256, "256/256"),
    (64, 512, "MLA absorbed 64/512"),
    (512, 512, "MLA absorbed 512/512"),
    (576, 512, "MLA full 576/512"),
]

BATCH, SEQLEN, NHEADS = 1, 4096, 8
DTYPE = torch.bfloat16


def main():
    print(f"{'shape':>11s}  {'label':26s} {'fwd(ms)':>9s} {'fwd+bwd(ms)':>12s} {'blk':>5s}")
    print("-" * 72)
    for hd_qk, hd_v, label in SHAPES:
        mk = lambda d: (  # noqa: E731
            torch.randn(BATCH, SEQLEN, NHEADS, d, dtype=DTYPE, device="cuda") * 0.1
        ).requires_grad_()
        q, k, v = mk(hd_qk), mk(hd_qk), mk(hd_v)
        scale = hd_qk**-0.5

        fwd_ms = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, softmax_scale=scale), warmup=10, rep=40
        )
        do = torch.randn_like(flash_attn_func(q, k, v, softmax_scale=scale))

        def fwd_bwd():
            flash_attn_func(q, k, v, softmax_scale=scale).backward(do)

        fwd_bwd_ms = triton.testing.do_bench(fwd_bwd, warmup=10, rep=40)
        padded = 1 << (hd_qk - 1).bit_length()
        blk = max_block_for_lds(padded, torch.finfo(DTYPE).bits // 8)
        print(
            f"{f'{hd_qk}/{hd_v}':>11s}  {label:26s} {fwd_ms:9.3f} {fwd_bwd_ms:12.3f} {blk:5d}"
        )


if __name__ == "__main__":
    main()
