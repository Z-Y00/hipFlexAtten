"""Block-sparse attention (Phase 5).

The kernel walks an explicit per-(batch, head, q_block) list of KV blocks instead of a
contiguous range. Fully-unmasked blocks skip masking entirely; partial blocks apply
mask_mod. Empty blocks are never visited.

Unlike upstream FA-4 this needs no dq_write_order/semaphore metadata: AITER's fused
backward computes dQ in a separate per-Q-block phase that stores its tile directly, so
there is no cross-program accumulation to order.
"""

import pytest
import torch
import triton
import triton.language as tl

from flex_attention import (
    causal_mask_mod,
    create_block_sparse_from_mask_mod,
    dense_to_block_sparse,
    flash_attn_func,
)
from flex_attention.block_sparse import BlockCategory

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

BLOCK = 64


@triton.jit
def _banded_mask_mod(b, h, q_idx, kv_idx):
    # keep |q - kv| < 96: a band that leaves whole blocks empty
    d = q_idx - kv_idx
    return (d < 96) & (d > -96)


def _banded_ref_fn(b, h, q_idx, kv_idx):
    d = q_idx - kv_idx
    return (d < 96) & (d > -96)


def ref_attention(q, k, v, mask_fn, softmax_scale=None):
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_k, _ = v.shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    group = nheads_q // nheads_k
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(group, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(group, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * softmax_scale
    qi = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
    ki = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
    keep = mask_fn(0, 0, qi, ki)
    scores = scores.masked_fill(~keep, float("-inf"))
    p = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2).to(q.dtype)


def _qkv(batch, seqlen, nheads_q, nheads_k, head_dim, dtype, device):
    torch.manual_seed(0)
    mk = lambda h: (torch.randn(batch, seqlen, h, head_dim, dtype=dtype, device=device) * 0.3)
    return [t.requires_grad_() for t in (mk(nheads_q), mk(nheads_k), mk(nheads_k))]


@needs_gpu
def test_block_sparse_causal_matches_dense_kernel():
    """A causal block mask must reproduce the dense causal kernel exactly."""
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 256, 2, 64
    q, k, v = _qkv(batch, seqlen, nheads, nheads, head_dim, torch.bfloat16, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    scale = head_dim**-0.5
    bs = create_block_sparse_from_mask_mod(
        lambda b, h, qi, ki: ki <= qi, batch, nheads, seqlen, seqlen,
        block_size=(BLOCK, BLOCK), device=device,
    )
    out = flash_attn_func(
        q, k, v, softmax_scale=scale, mask_mod=causal_mask_mod, block_sparse_tensors=bs
    )
    ref = flash_attn_func(qr, kr, vr, softmax_scale=scale, causal=True)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1e-2, rtol=1e-2,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@needs_gpu
@pytest.mark.parametrize("gqa_ratio", [1, 4])
def test_block_sparse_banded_forward_backward(gqa_ratio):
    """A band mask leaves whole blocks empty, so sparsity does real work here."""
    device = "cuda"
    batch, seqlen, nheads_k, head_dim = 2, 256, 2, 64
    nheads_q = nheads_k * gqa_ratio
    q, k, v = _qkv(batch, seqlen, nheads_q, nheads_k, head_dim, torch.bfloat16, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    scale = head_dim**-0.5
    bs = create_block_sparse_from_mask_mod(
        _banded_ref_fn, batch, nheads_q, seqlen, seqlen,
        block_size=(BLOCK, BLOCK), device=device,
    )
    # the band must actually leave blocks out, else the test proves nothing
    visited = (bs.mask_block_cnt + bs.full_block_cnt).float().mean().item()
    assert visited < seqlen / BLOCK, "band mask did not skip any blocks"

    out = flash_attn_func(
        q, k, v, softmax_scale=scale, mask_mod=_banded_mask_mod, block_sparse_tensors=bs
    )
    ref = ref_attention(qr, kr, vr, _banded_ref_fn, scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1e-1, rtol=1e-1,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@needs_gpu
def test_block_sparse_empty_rows():
    """A q block with no visible KV blocks must produce zeros, not NaN."""
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 256, 1, 64
    q, k, v = _qkv(batch, seqlen, nheads, nheads, head_dim, torch.bfloat16, device)
    nq = nkv = seqlen // BLOCK
    full = torch.zeros(batch, nheads, nq, nkv, dtype=torch.bool, device=device)
    partial = torch.zeros_like(full)
    full[:, :, 1:, 0] = True  # q block 0 sees nothing at all
    bs = dense_to_block_sparse(full, partial, (BLOCK, BLOCK))
    with torch.no_grad():
        out = flash_attn_func(q, k, v, block_sparse_tensors=bs)
    assert torch.isfinite(out).all(), "empty q block produced non-finite output"
    assert out[:, :BLOCK].abs().max() == 0, "empty q block should output zeros"
    assert out[:, BLOCK:].abs().max() > 0, "non-empty q blocks should be populated"


@needs_gpu
def test_block_sparse_varlen_matches_dense_varlen():
    """Sparse block indices are sequence-local under varlen, exactly like mask_mod's."""
    from flex_attention import create_block_sparse_varlen, flash_attn_varlen_func

    device = "cuda"
    nheads, head_dim = 2, 64
    seqlens = [128, 192, 64]  # deliberately ragged, none a multiple of the other
    cu = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    total, max_s = cu[-1].item(), max(seqlens)
    torch.manual_seed(0)
    mk = lambda: (  # noqa: E731
        torch.randn(total, nheads, head_dim, dtype=torch.bfloat16, device=device) * 0.3
    ).requires_grad_()
    q, k, v = mk(), mk(), mk()
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    scale = head_dim**-0.5

    bs = create_block_sparse_varlen(
        lambda b, h, qi, ki: ki <= qi, cu, cu, nheads, block_size=(BLOCK, BLOCK)
    )
    # the shortest sequence must have fewer populated q blocks than the longest
    assert bs.mask_block_cnt[2, 0].sum() < bs.mask_block_cnt[1, 0].sum()

    out = flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, softmax_scale=scale,
        mask_mod=causal_mask_mod, block_sparse_tensors=bs,
    )
    ref = flash_attn_varlen_func(qr, kr, vr, cu, cu, max_s, max_s, softmax_scale=scale, causal=True)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    do = torch.randn_like(out)
    out.backward(do)
    ref.backward(do)
    for got, want, name in [(q, qr, "dq"), (k, kr, "dk"), (v, vr, "dv")]:
        torch.testing.assert_close(
            got.grad.float(), want.grad.float(), atol=1e-2, rtol=1e-2,
            msg=lambda m, n=name: f"{n}: {m}",
        )


# -- BlockCategory classification (pure host-side, no GPU needed) --


def test_classify_causal_as_banded_not_sparse():
    """A causal mask's diagonal blocks are a translation-invariant band, so they should
    classify as CAUSAL rather than the fully-general PARTIAL_SPARSE bucket."""
    bs = create_block_sparse_from_mask_mod(
        lambda b, h, qi, ki: ki <= qi, batch=1, nheads=1, seqlen_q=256, seqlen_k=256,
        block_size=(64, 64), device="cpu",
    )
    assert bs.causal is not None
    assert bs.partial_dense is None
    assert bs.partial_sparse is None
    # lower-triangular block coverage for a 4x4 block grid: 4+3+2+1 visited blocks
    assert (bs.mask_block_cnt + bs.full_block_cnt).sum() == 10


def test_classify_dense_vs_sparse_partial_blocks():
    """A partial block that's mostly kept classifies PARTIAL_DENSE; mostly masked
    classifies PARTIAL_SPARSE -- neither is a diagonal band, so CAUSAL must stay empty."""
    torch.manual_seed(0)
    mostly_kept = torch.rand(64, 64) > 0.3  # ~70% kept
    bs_dense = create_block_sparse_from_mask_mod(
        lambda b, h, qi, ki: mostly_kept[qi % 64, ki % 64], batch=1, nheads=1,
        seqlen_q=64, seqlen_k=64, block_size=(64, 64), device="cpu",
    )
    assert bs_dense.causal is None
    assert bs_dense.partial_dense is not None
    assert bs_dense.partial_sparse is None

    torch.manual_seed(1)
    mostly_masked = torch.rand(64, 64) > 0.85  # ~15% kept
    bs_sparse = create_block_sparse_from_mask_mod(
        lambda b, h, qi, ki: mostly_masked[qi % 64, ki % 64], batch=1, nheads=1,
        seqlen_q=64, seqlen_k=64, block_size=(64, 64), device="cpu",
    )
    assert bs_sparse.causal is None
    assert bs_sparse.partial_dense is None
    assert bs_sparse.partial_sparse is not None


def test_classify_full_and_empty_blocks():
    """A block that's entirely kept is FULL; a block with nothing kept is EMPTY (never
    materialized in any category's index list)."""
    full = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    partial = torch.zeros_like(full)
    full[:, :, 0, 0] = True  # block (0,0): FULL
    # block (0,1), (1,0), (1,1) left False in both -> EMPTY
    bs = dense_to_block_sparse(full, partial, (64, 64))
    assert bs.full_block_cnt.sum().item() == 1
    assert bs.mask_block_cnt.sum().item() == 0
    assert bs.causal is None and bs.partial_dense is None and bs.partial_sparse is None


def test_backward_plan_drops_full_fast_path():
    """The backward's combined/transposed plans fold FULL into the masked list and carry
    no FULL fast path -- see BlockPlan.combined's docstring for why."""
    from flex_attention.block_sparse import backward_block_sparse

    bs = create_block_sparse_from_mask_mod(
        lambda b, h, qi, ki: ki <= qi, batch=1, nheads=1, seqlen_q=256, seqlen_k=256,
        block_size=(64, 64), device="cpu",
    )
    dq_plan, dkdv_plan = backward_block_sparse(bs, num_kv_blocks=4)
    assert dq_plan.full is None and dkdv_plan.full is None
    # every originally-visited (full or masked) block must survive the merge
    assert dq_plan.mask_block_cnt.sum().item() == (bs.full_block_cnt + bs.mask_block_cnt).sum().item()
