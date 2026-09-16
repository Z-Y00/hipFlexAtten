"""softcap (Phase 2).

Implemented as a score_mod pair, mirroring upstream FA-4 (which also builds softcap from
create_softcap_scoremod / _bwd rather than a dedicated kernel flag).
"""

import pytest
import torch

from flex_attention import causal_mask_mod, flash_attn_func, make_softcap_score_mod

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def ref_attention(q, k, v, softcap, causal=False, window_size=(None, None), softmax_scale=None):
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_k, _ = v.shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    group = nheads_q // nheads_k
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(group, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(group, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * softmax_scale
    if softcap > 0:  # pre-mask, as upstream does
        scores = torch.tanh(scores / softcap) * softcap
    row = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
    col = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
    off = seqlen_k - seqlen_q
    if causal:
        scores = scores.masked_fill(col > row + off, float("-inf"))
    if window_size[0] is not None:
        scores = scores.masked_fill(col < row + off - window_size[0], float("-inf"))
    if window_size[1] is not None:
        scores = scores.masked_fill(col > row + off + window_size[1], float("-inf"))
    p = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2).to(q.dtype)


def _qkv(batch, seqlen, nheads_q, nheads_k, head_dim, dtype, device, scale=1.0):
    torch.manual_seed(0)
    mk = lambda h: (torch.randn(batch, seqlen, h, head_dim, dtype=dtype, device=device) * scale)
    return [t.requires_grad_() for t in (mk(nheads_q), mk(nheads_k), mk(nheads_k))]


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcap", [1.0, 20.0])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gqa_ratio", [1, 4])
def test_softcap_forward_backward(dtype, softcap, causal, gqa_ratio):
    device = "cuda"
    batch, seqlen, head_dim, nheads_k = 2, 256, 64, 4
    # large inputs so the scores actually reach into tanh's saturating region
    q, k, v = _qkv(batch, seqlen, nheads_k * gqa_ratio, nheads_k, head_dim, dtype, device, scale=1.0)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]

    out = flash_attn_func(q, k, v, softcap=softcap, causal=causal)
    ref = ref_attention(qr, kr, vr, softcap, causal=causal)

    atol = 2e-2 if dtype is torch.float16 else 3e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=atol)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1e-1, rtol=1e-1,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@needs_gpu
def test_softcap_actually_saturates():
    """Guard against a no-op implementation: with a small cap the output must differ
    substantially from uncapped attention."""
    device = "cuda"
    q, k, v = _qkv(1, 128, 4, 4, 64, torch.float32, device, scale=3.0)
    with torch.no_grad():
        capped = flash_attn_func(q, k, v, softcap=0.5)
        plain = flash_attn_func(q, k, v)
    assert (capped.float() - plain.float()).abs().max() > 1e-2


@needs_gpu
def test_softcap_does_not_unmask():
    """Regression: tanh(-inf) = -1, so a softcap applied after masking would turn masked
    positions back on. Causal softcap must still be strictly causal."""
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 128, 2, 64
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, nheads, head_dim, dtype=torch.float32, device=device)
    k = torch.randn(batch, seqlen, nheads, head_dim, dtype=torch.float32, device=device)
    # Only the last key position is non-zero; under a causal mask query 0 must not see it.
    v = torch.zeros(batch, seqlen, nheads, head_dim, dtype=torch.float32, device=device)
    v[:, -1] = 1.0
    with torch.no_grad():
        out = flash_attn_func(q, k, v, softcap=1.0, causal=True)
    assert out[0, 0].abs().max() < 1e-5, "masked position leaked through softcap"


@needs_gpu
def test_softcap_rejects_score_mod_and_negative():
    device = "cuda"
    q, k, v = _qkv(1, 64, 2, 2, 32, torch.float16, device)
    with pytest.raises(ValueError, match="cannot be used together"):
        flash_attn_func(q, k, v, softcap=1.0, score_mod=causal_mask_mod)
    with pytest.raises(ValueError, match="non-negative"):
        flash_attn_func(q, k, v, softcap=-1.0)
    with pytest.raises(ValueError, match="positive"):
        make_softcap_score_mod(0.0)


@needs_gpu
def test_multiple_softcap_values_in_one_process():
    """Regression: parameterized mods are generated as source with the constant baked in.

    Triton keys compiled kernels partly on JIT function source, so two closures built
    from the same `def` with different captured constants used to hash identically --
    the second value silently reused the first's compiled kernel and returned wrong
    output. Exercise several values in one process and check each against its own
    reference.
    """
    device = "cuda"
    q, k, v = _qkv(1, 128, 4, 4, 64, torch.float32, device, scale=2.0)
    caps = (0.5, 1.0, 5.0, 20.0)
    outs = {}
    for softcap in caps:
        with torch.no_grad():
            outs[softcap] = flash_attn_func(q, k, v, softcap=softcap)
        ref = ref_attention(q, k, v, softcap)
        # Tolerance is loose because tl.dot on CDNA3 uses reduced precision even for
        # fp32 inputs; the decisive collision check is the pairwise inequality below.
        torch.testing.assert_close(
            outs[softcap].float(), ref.float(), atol=2e-2, rtol=2e-2,
            msg=lambda m, s=softcap: f"softcap={s}: {m}",
        )
    # Under the collision bug every value after the first reused the first's kernel and
    # returned bitwise-identical output. Distinct caps must give distinct results.
    for i, a in enumerate(caps):
        for b in caps[i + 1:]:
            assert not torch.equal(outs[a], outs[b]), (
                f"softcap={a} and softcap={b} produced identical output -- "
                f"generated mods collided in triton's cache"
            )


@needs_gpu
def test_multiple_window_mask_mods_in_one_process():
    """Same regression for make_sliding_window_mask_mod."""
    from flex_attention import make_sliding_window_mask_mod

    device = "cuda"
    q, k, v = _qkv(1, 128, 4, 4, 64, torch.float32, device)
    for left in (16, 32, 64):
        mod = make_sliding_window_mask_mod(left, 0)
        with torch.no_grad():
            out = flash_attn_func(q, k, v, mask_mod=mod)
            ref = flash_attn_func(q, k, v, causal=True, window_size=(left, 0))
        torch.testing.assert_close(
            out.float(), ref.float(), atol=2e-3, rtol=2e-3,
            msg=lambda m, l=left: f"window_left={l}: {m}",
        )
