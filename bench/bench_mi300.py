"""Benchmark flex_attention (AITER fwd + AITER bwd) against AITER's own fwd+bwd
(bench/_aiter_ref, an unmodified benchmark-only copy), on MI300/CDNA3.

The AITER-side backward uses the same mode selection as flex_attention/interface.py
(causal -> "split", non-causal -> "fused"), since AITER's "fused"/"fused_atomic" causal
backward is broken at the pinned commit -- see
flex_attention/_vendor/aiter_flash_attn/NOTICE.md. Comparing against anything else
for causal shapes wouldn't be apples-to-apples.

Usage: python3 -m bench.bench_mi300
"""

import torch
import triton

from bench._aiter_ref import interface_v3 as aiter_ref
from bench._aiter_ref.bwd import attention_backward_triton_impl as aiter_bwd_impl
from flex_attention.interface import _FlashAttnFunc

SHAPES = [
    # (batch, seqlen, nheads_q, nheads_k, head_dim, causal)
    (4, 2048, 32, 32, 128, False),
    (4, 2048, 32, 32, 128, True),
    (4, 2048, 32, 8, 128, False),
    (4, 2048, 32, 8, 128, True),
    (2, 8192, 32, 8, 128, False),
    (2, 8192, 32, 8, 128, True),
    (2, 8192, 32, 8, 64, True),
]

DTYPE = torch.bfloat16
DEVICE = "cuda"


def flops(batch, seqlen, nheads_q, head_dim, causal):
    # 2 matmuls (QK^T and PV), each 2*seqlen_q*seqlen_k*head_dim FLOPs per head.
    f = 4 * batch * nheads_q * seqlen * seqlen * head_dim
    return f / 2 if causal else f


def make_qkv(batch, seqlen, nheads_q, nheads_k, head_dim):
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, nheads_q, head_dim, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads_k, head_dim, dtype=DTYPE, device=DEVICE, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads_k, head_dim, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return q, k, v


def bench_ours(q, k, v, causal):
    scale = q.shape[-1] ** -0.5

    def fwd():
        return _FlashAttnFunc.apply(q, k, v, scale, causal, (None, None), False, False, None, None, None, None)

    fwd_ms = triton.testing.do_bench(fwd, warmup=25, rep=100)

    o, _ = fwd()
    do = torch.randn_like(o)

    def fwd_bwd():
        o, _ = _FlashAttnFunc.apply(q, k, v, scale, causal, (None, None), False, False, None, None, None, None)
        o.backward(do)

    fwd_bwd_ms = triton.testing.do_bench(fwd_bwd, warmup=25, rep=100)
    return fwd_ms, fwd_bwd_ms


def _aiter_fwd_kwargs(q, k, v, causal, scale):
    return dict(
        q=q,
        k=k,
        v=v,
        k_new=None,
        v_new=None,
        qv=None,
        out=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        page_table=None,
        kv_batch_idx=None,
        leftpad_k=None,
        rotary_cos=None,
        rotary_sin=None,
        seqlens_rotary=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
    )


def bench_aiter(q, k, v, causal):
    scale = q.shape[-1] ** -0.5

    def fwd():
        return aiter_ref.fwd(**_aiter_fwd_kwargs(q, k, v, causal, scale))

    fwd_ms = triton.testing.do_bench(fwd, warmup=25, rep=100)

    out, lse, _, _ = fwd()
    do = torch.randn_like(out)
    batch, seqlen, nheads_q, _ = q.shape
    mode = "split" if causal else "fused"

    def fwd_bwd():
        out, lse, _, _ = aiter_ref.fwd(**_aiter_fwd_kwargs(q, k, v, causal, scale))
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        delta = torch.zeros(batch, nheads_q, seqlen, dtype=torch.float32, device=q.device)
        aiter_bwd_impl(
            do=do,
            q=q,
            k=k,
            v=v,
            o=out,
            softmax_lse=lse,
            dq=dq,
            dk=dk,
            dv=dv,
            delta=delta,
            sm_scale=scale,
            alibi_slopes=None,
            causal=causal,
            layout="bshd",
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            use_exp2=True,
            mode=mode,
        )

    fwd_bwd_ms = triton.testing.do_bench(fwd_bwd, warmup=25, rep=100)
    return fwd_ms, fwd_bwd_ms


def main():
    header = (
        f"{'shape (b,s,hq,hk,d,causal)':38s} "
        f"{'ours fwd(ms)':>13s} {'aiter fwd(ms)':>13s} {'fwd speedup':>12s} "
        f"{'ours fwd+bwd(ms)':>17s} {'aiter fwd+bwd(ms)':>18s} {'fwd+bwd speedup':>16s}"
    )
    print(header)
    print("-" * len(header))
    for batch, seqlen, nheads_q, nheads_k, head_dim, causal in SHAPES:
        q, k, v = make_qkv(batch, seqlen, nheads_q, nheads_k, head_dim)
        ours_fwd, ours_fwd_bwd = bench_ours(q, k, v, causal)
        aiter_fwd, aiter_fwd_bwd = bench_aiter(q, k, v, causal)
        shape_str = f"({batch},{seqlen},{nheads_q},{nheads_k},{head_dim},{causal})"
        print(
            f"{shape_str:38s} "
            f"{ours_fwd:13.3f} {aiter_fwd:13.3f} {aiter_fwd / ours_fwd:11.2f}x "
            f"{ours_fwd_bwd:17.3f} {aiter_fwd_bwd:18.3f} {aiter_fwd_bwd / ours_fwd_bwd:15.2f}x"
        )


if __name__ == "__main__":
    main()
