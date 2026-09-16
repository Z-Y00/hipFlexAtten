"""MLA / large-head-dim shapes (Phase 4).

Upstream FA-4 needs dedicated kernels for these on NVIDIA (a Blackwell 2-CTA kernel for
head_dim=256, split dQ/dK GEMM kernels for the MLA backward) because WGMMA tiling and
Blackwell's cluster launch force it. On CDNA3 there is no such constraint: the only
blocker was CDNA3's 64 KiB LDS budget, which the vendored kernels now handle by capping
the sequence-block size when the padded head dim is large. So all of these run through
the same single kernel pair as ordinary attention.
"""

import pytest
import torch

from flex_attention import flash_attn_func, flash_attn_varlen_func

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# (head_dim_qk, head_dim_v, id)
MLA_SHAPES = [
    (192, 128, "deepseek-192-128"),
    (256, 256, "hd256"),
    (64, 512, "mla-absorbed-64-512"),
    (512, 512, "mla-absorbed-512-512"),
    (576, 512, "mla-full-576-512"),
]


def ref_attention(q, k, v, softmax_scale, causal=False):
    batch, seqlen_q, nheads_q, _ = q.shape
    _, seqlen_k, nheads_k, _ = v.shape
    group = nheads_q // nheads_k
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(group, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(group, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * softmax_scale
    if causal:
        row = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
        col = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
        scores = scores.masked_fill(col > row + (seqlen_k - seqlen_q), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2).to(q.dtype)


def _qkv(batch, seqlen, nheads_q, nheads_k, hd_qk, hd_v, dtype, device):
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, nheads_q, hd_qk, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seqlen, nheads_k, hd_qk, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seqlen, nheads_k, hd_v, dtype=dtype, device=device) * 0.1
    return [t.requires_grad_() for t in (q, k, v)]


@needs_gpu
@pytest.mark.parametrize("hd_qk,hd_v,_id", MLA_SHAPES, ids=[s[2] for s in MLA_SHAPES])
@pytest.mark.parametrize("causal", [False, True])
def test_mla_shapes_forward_backward(hd_qk, hd_v, _id, causal):
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen, nheads = 1, 256, 2
    q, k, v = _qkv(batch, seqlen, nheads, nheads, hd_qk, hd_v, dtype, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    scale = hd_qk**-0.5

    out = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    ref = ref_attention(qr, kr, vr, scale, causal=causal)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1.5e-1, rtol=1.5e-1,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@needs_gpu
@pytest.mark.parametrize("hd_qk,hd_v,_id", MLA_SHAPES, ids=[s[2] for s in MLA_SHAPES])
def test_mla_shapes_gqa(hd_qk, hd_v, _id):
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen, nheads_k, gqa = 1, 256, 2, 4
    q, k, v = _qkv(batch, seqlen, nheads_k * gqa, nheads_k, hd_qk, hd_v, dtype, device)
    scale = hd_qk**-0.5
    with torch.no_grad():
        out = flash_attn_func(q, k, v, softmax_scale=scale)
        ref = ref_attention(q, k, v, scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@needs_gpu
@pytest.mark.parametrize("hd_qk,hd_v,_id", MLA_SHAPES, ids=[s[2] for s in MLA_SHAPES])
def test_mla_shapes_varlen(hd_qk, hd_v, _id):
    device = "cuda"
    dtype = torch.bfloat16
    nheads = 2
    seqlens = [29, 130]
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total = cu[-1].item()
    torch.manual_seed(0)
    q = torch.randn(total, nheads, hd_qk, dtype=dtype, device=device) * 0.1
    k = torch.randn(total, nheads, hd_qk, dtype=dtype, device=device) * 0.1
    v = torch.randn(total, nheads, hd_v, dtype=dtype, device=device) * 0.1
    scale = hd_qk**-0.5

    out = flash_attn_varlen_func(q, k, v, cu, cu, max(seqlens), max(seqlens), softmax_scale=scale)
    for b in range(len(seqlens)):
        s, e = cu[b].item(), cu[b + 1].item()
        ref = ref_attention(
            q[s:e].unsqueeze(0), k[s:e].unsqueeze(0), v[s:e].unsqueeze(0), scale
        ).squeeze(0)
        torch.testing.assert_close(out[s:e].float(), ref.float(), atol=3e-2, rtol=3e-2)


@needs_gpu
def test_head_dim_too_large_raises_cleanly():
    """Beyond what LDS can hold we should give a clear error, not a raw Triton OOM."""
    device = "cuda"
    # padded to 4096; even a 16-row block needs 128 KiB > 64 KiB.
    q, k, v = _qkv(1, 64, 1, 1, 4096, 4096, torch.bfloat16, device)
    with pytest.raises((ValueError, RuntimeError)):
        with torch.no_grad():
            flash_attn_func(q, k, v)


@needs_gpu
@pytest.mark.parametrize("flag", ["causal", "deterministic"])
def test_asymmetric_head_dim_backward_supported(flag):
    """Regression: causal/deterministic with head_dim_qk != head_dim_v.

    This used to route to AITER's split backward, which carries a single head dim and
    read past the end of V (a latent fault that only tripped under some allocator
    layouts). The backward now always uses the fused kernels, which carry separate QK/V
    dims, so this must produce correct gradients.
    """
    device = "cuda"
    q, k, v = _qkv(1, 128, 2, 2, 192, 128, torch.bfloat16, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    scale = 192**-0.5
    out = flash_attn_func(q, k, v, softmax_scale=scale, **{flag: True})
    ref = ref_attention(qr, kr, vr, scale, causal=(flag == "causal"))
    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1.5e-1, rtol=1.5e-1,
            msg=lambda m, n=name: f"{n}: {m}",
        )
