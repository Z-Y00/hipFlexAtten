"""Split-KV forward (``num_splits > 1``).

Splitting slices the KV loop across programs and reduces the partials afterwards. It
changes only *how* the forward is evaluated: the output and LSE must match the unsplit
path, and the backward is unaffected (it re-derives everything from o and lse).
"""

import pytest
import torch

from flex_attention import flash_attn_func, flash_attn_varlen_func

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _qkv(batch, seqlen, nheads, head_dim, dtype, device):
    torch.manual_seed(0)
    mk = lambda: torch.randn(batch, seqlen, nheads, head_dim, dtype=dtype, device=device) * 0.3  # noqa: E731
    return mk(), mk(), mk()


@needs_gpu
@pytest.mark.parametrize("num_splits", [2, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", [512, 2048])
def test_splits_match_unsplit(num_splits, causal, seqlen):
    device = "cuda"
    batch, nheads, head_dim = 1, 4, 64
    q, k, v = _qkv(batch, seqlen, nheads, head_dim, torch.bfloat16, device)
    scale = head_dim**-0.5
    with torch.no_grad():
        ref, ref_lse = flash_attn_func(
            q, k, v, softmax_scale=scale, causal=causal, return_lse=True
        )
        out, lse = flash_attn_func(
            q, k, v, softmax_scale=scale, causal=causal, num_splits=num_splits, return_lse=True
        )
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)


@needs_gpu
def test_splits_more_than_kv_blocks():
    """More splits than there are KV blocks: the extra splits own nothing and must
    report -inf LSE so the combine ignores them rather than folding in a phantom term."""
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 64, 2, 64
    q, k, v = _qkv(batch, seqlen, nheads, head_dim, torch.bfloat16, device)
    scale = head_dim**-0.5
    with torch.no_grad():
        ref = flash_attn_func(q, k, v, softmax_scale=scale)
        out = flash_attn_func(q, k, v, softmax_scale=scale, num_splits=16)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-3, rtol=3e-3)


@needs_gpu
def test_splits_with_backward_and_sink():
    """Splits compose with the rest of the feature set and leave gradients unchanged."""
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 512, 4, 64
    q, k, v = [t.requires_grad_() for t in _qkv(batch, seqlen, nheads, head_dim, torch.bfloat16, device)]
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    sink = torch.randn(nheads, dtype=torch.float32, device=device).requires_grad_()
    sink_r = sink.detach().clone().requires_grad_()
    scale = head_dim**-0.5

    out = flash_attn_func(q, k, v, softmax_scale=scale, causal=True, learnable_sink=sink, num_splits=4)
    ref = flash_attn_func(qr, kr, vr, softmax_scale=scale, causal=True, learnable_sink=sink_r)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-3, rtol=3e-3)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv"), (sink, sink_r, "dsink")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1e-2, rtol=1e-2,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@needs_gpu
def test_splits_validation():
    device = "cuda"
    q, k, v = _qkv(1, 64, 2, 32, torch.bfloat16, device)
    with pytest.raises(ValueError, match="num_splits"):
        flash_attn_func(q, k, v, num_splits=0)

    nheads, head_dim, seqlen = 2, 32, 64
    cu = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    t = torch.randn(seqlen, nheads, head_dim, dtype=torch.bfloat16, device=device)
    with pytest.raises(NotImplementedError, match="varlen"):
        flash_attn_varlen_func(t, t, t, cu, cu, seqlen, seqlen, num_splits=4)
