import pytest
import torch
import triton
import triton.language as tl

from flex_attention import (
    causal_mask_mod,
    flash_attn_func,
    flash_attn_varlen_func,
    identity_score_mod_bwd,
    make_sliding_window_mask_mod,
)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SLOPE = -0.1


@triton.jit
def linear_penalty_score_mod(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx).to(tl.float32) * -0.1


def ref_attention(q, k, v, score_mod_fn=None, mask_mod_fn=None, softmax_scale=None):
    """Eager reference. score_mod_fn/mask_mod_fn take (q_idx, kv_idx) index tensors."""
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_k, _ = v.shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    group = nheads_q // nheads_k

    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(group, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(group, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * softmax_scale

    q_idx = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
    kv_idx = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
    if score_mod_fn is not None:
        scores = score_mod_fn(scores, q_idx, kv_idx)
    if mask_mod_fn is not None:
        scores = scores.masked_fill(~mask_mod_fn(q_idx, kv_idx), float("-inf"))

    p = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2).to(q.dtype)


def _rand_qkv(batch, seqlen, nheads_q, nheads_k, head_dim, dtype, device):
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, nheads_q, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seqlen, nheads_k, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seqlen, nheads_k, head_dim, dtype=dtype, device=device) * 0.1
    return [t.requires_grad_() for t in (q, k, v)]


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa_ratio", [1, 4])
@pytest.mark.parametrize(
    "use_score_mod,use_mask_mod",
    [(True, False), (False, True), (True, True)],
    ids=["score_mod", "mask_mod", "both"],
)
def test_mods_forward_backward(dtype, gqa_ratio, use_score_mod, use_mask_mod):
    device = "cuda"
    batch, seqlen, head_dim, nheads_k = 2, 384, 64, 4
    nheads_q = nheads_k * gqa_ratio
    q, k, v = _rand_qkv(batch, seqlen, nheads_q, nheads_k, head_dim, dtype, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]

    out = flash_attn_func(
        q,
        k,
        v,
        score_mod=linear_penalty_score_mod if use_score_mod else None,
        score_mod_bwd=identity_score_mod_bwd if use_score_mod else None,
        mask_mod=causal_mask_mod if use_mask_mod else None,
    )
    ref_out = ref_attention(
        qr,
        kr,
        vr,
        score_mod_fn=(lambda s, qi, ki: s + (qi - ki).float() * SLOPE) if use_score_mod else None,
        mask_mod_fn=(lambda qi, ki: ki <= qi) if use_mask_mod else None,
    )

    atol = 2e-2 if dtype is torch.float16 else 3e-2
    torch.testing.assert_close(out.float(), ref_out.float(), atol=atol, rtol=atol)

    do = torch.randn_like(out)
    out.backward(do)
    ref_out.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=atol * 5, rtol=atol * 5, msg=lambda m: f"{name}: {m}"
        )


@needs_gpu
def test_mask_mod_matches_causal_flag():
    """A causal mask_mod must agree with the kernel's own causal=True path (square case)."""
    device = "cuda"
    q, k, v = _rand_qkv(2, 256, 4, 4, 64, torch.float16, device)
    with torch.no_grad():
        out_flag = flash_attn_func(q, k, v, causal=True)
        out_mod = flash_attn_func(q, k, v, mask_mod=causal_mask_mod)
    torch.testing.assert_close(out_flag.float(), out_mod.float(), atol=2e-3, rtol=2e-3)


@needs_gpu
def test_sliding_window_mask_mod_matches_window_flag():
    device = "cuda"
    q, k, v = _rand_qkv(2, 256, 4, 4, 64, torch.float16, device)
    window_mod = make_sliding_window_mask_mod(64, 8)
    with torch.no_grad():
        out_flag = flash_attn_func(q, k, v, window_size=(64, 8))
        out_mod = flash_attn_func(q, k, v, mask_mod=window_mod)
    torch.testing.assert_close(out_flag.float(), out_mod.float(), atol=2e-3, rtol=2e-3)


@needs_gpu
def test_varlen_mask_mod_forward():
    device = "cuda"
    dtype = torch.float16
    nheads, head_dim = 4, 64
    seqlens = [37, 111, 5]
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total = cu[-1].item()
    torch.manual_seed(0)
    q = torch.randn(total, nheads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(total, nheads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(total, nheads, head_dim, dtype=dtype, device=device) * 0.1

    out = flash_attn_varlen_func(q, k, v, cu, cu, max(seqlens), max(seqlens), mask_mod=causal_mask_mod)

    # Per-sequence eager reference; mask_mod indices are sequence-local.
    for b, _ in enumerate(seqlens):
        s, e = cu[b].item(), cu[b + 1].item()
        ref = ref_attention(
            q[s:e].unsqueeze(0), k[s:e].unsqueeze(0), v[s:e].unsqueeze(0),
            mask_mod_fn=lambda qi, ki: ki <= qi,
        ).squeeze(0)
        torch.testing.assert_close(out[s:e].float(), ref.float(), atol=2e-2, rtol=2e-2)


@needs_gpu
def test_score_mod_without_bwd_raises_when_grad_required():
    device = "cuda"
    q, k, v = _rand_qkv(1, 64, 2, 2, 32, torch.float16, device)
    with pytest.raises(ValueError, match="score_mod_bwd"):
        flash_attn_func(q, k, v, score_mod=linear_penalty_score_mod)


@needs_gpu
def test_score_mod_without_bwd_allowed_under_no_grad():
    device = "cuda"
    q, k, v = _rand_qkv(1, 64, 2, 2, 32, torch.float16, device)
    with torch.no_grad():
        flash_attn_func(q, k, v, score_mod=linear_penalty_score_mod)


@needs_gpu
@pytest.mark.parametrize("window_size", [(None, None), (128, 0)])
def test_mod_composes_with_causal_flag(window_size):
    """Regression: score_mod/mask_mod combined with causal (and window) in the backward.

    This used to be rejected because the causal backward routed to AITER's uninstrumented
    split kernels. The backward now always uses the instrumented fused kernels.
    """
    device = "cuda"
    batch, seqlen, head_dim, nheads = 2, 256, 64, 4
    q, k, v = _rand_qkv(batch, seqlen, nheads, nheads, head_dim, torch.float16, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]

    out = flash_attn_func(
        q, k, v, causal=True, window_size=window_size,
        score_mod=linear_penalty_score_mod, score_mod_bwd=identity_score_mod_bwd,
    )
    left = window_size[0]
    ref = ref_attention(
        qr, kr, vr,
        score_mod_fn=lambda s, qi, ki: s + (qi - ki).float() * SLOPE,
        mask_mod_fn=(
            (lambda qi, ki: (ki <= qi) & (ki >= qi - left)) if left is not None
            else (lambda qi, ki: ki <= qi)
        ),
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1e-1, rtol=1e-1,
            msg=lambda m, n=name: f"{n}: {m}",
        )
