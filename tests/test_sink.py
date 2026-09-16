"""learnable_sink (Phase 2).

An attention sink is one extra per-head logit competing in the softmax with no value
vector behind it, so it only enters the denominator: it lets a row attend to "nothing"
and thereby shrink its output. gpt-oss trains with this.

The forward folds the sink into l_i in the epilogue; the backward needs no kernel change
because the LSE written out is already sink-inclusive, so dq/dk/dv recover a correctly
normalized p. Only dsink itself is computed separately, analytically.
"""

import pytest
import torch

from flex_attention import flash_attn_func

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def ref_attention(q, k, v, sink=None, causal=False, softmax_scale=None):
    """Reference with the sink as an explicit extra logit carrying a zero value vector."""
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_k, head_dim_v = v.shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    group = nheads_q // nheads_k
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(group, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(group, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * softmax_scale
    if causal:
        row = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
        col = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
        scores = scores.masked_fill(col > row + (seqlen_k - seqlen_q), float("-inf"))
    if sink is not None:
        # append one column of sink logits with a zero-valued "value" row
        sink_col = sink.float().view(1, nheads_q, 1, 1).expand(batch, nheads_q, seqlen_q, 1)
        scores = torch.cat([scores, sink_col], dim=-1)
        vf = torch.cat([vf, vf.new_zeros(batch, nheads_q, 1, head_dim_v)], dim=2)
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2).to(q.dtype)


def _qkv(batch, seqlen, nheads_q, nheads_k, head_dim, dtype, device):
    torch.manual_seed(0)
    mk = lambda h: (torch.randn(batch, seqlen, h, head_dim, dtype=dtype, device=device) * 0.3)
    return [t.requires_grad_() for t in (mk(nheads_q), mk(nheads_k), mk(nheads_k))]


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gqa_ratio", [1, 4])
def test_sink_forward_backward(dtype, causal, gqa_ratio):
    device = "cuda"
    batch, seqlen, head_dim, nheads_k = 2, 256, 64, 4
    nheads_q = nheads_k * gqa_ratio
    q, k, v = _qkv(batch, seqlen, nheads_q, nheads_k, head_dim, dtype, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    torch.manual_seed(7)
    sink = (torch.randn(nheads_q, dtype=torch.float32, device=device)).requires_grad_()
    sink_r = sink.detach().clone().requires_grad_()

    out = flash_attn_func(q, k, v, learnable_sink=sink, causal=causal)
    ref = ref_attention(qr, kr, vr, sink=sink_r, causal=causal)

    atol = 2e-2 if dtype is torch.float16 else 3e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=atol)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv"), (sink, sink_r, "dsink")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1e-1, rtol=1e-1,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@needs_gpu
def test_sink_shrinks_output():
    """A large sink logit wins most of the softmax mass, driving the output toward zero."""
    device = "cuda"
    q, k, v = _qkv(1, 128, 4, 4, 64, torch.float32, device)
    with torch.no_grad():
        no_sink = flash_attn_func(q, k, v)
        big_sink = flash_attn_func(
            q, k, v, learnable_sink=torch.full((4,), 30.0, device=device)
        )
        tiny_sink = flash_attn_func(
            q, k, v, learnable_sink=torch.full((4,), -30.0, device=device)
        )
    assert big_sink.abs().max() < no_sink.abs().max() * 0.05, "large sink did not absorb mass"
    torch.testing.assert_close(tiny_sink.float(), no_sink.float(), atol=1e-3, rtol=1e-3)


@needs_gpu
def test_sink_validation():
    device = "cuda"
    q, k, v = _qkv(1, 64, 2, 2, 32, torch.float16, device)
    with pytest.raises(ValueError, match="1-D"):
        flash_attn_func(q, k, v, learnable_sink=torch.zeros(2, 2, device=device))
    with pytest.raises(ValueError, match="floating point"):
        flash_attn_func(q, k, v, learnable_sink=torch.zeros(2, dtype=torch.int32, device=device))
