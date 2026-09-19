"""Profile-guided selection of a block-sparse implementation.

The optimal block size is not predictable from the pattern alone -- measured on MI300X
at seqlen 4096, a plain causal mask prefers 128x128 while a strided block pattern
prefers 64x64, each by a wide enough margin (1.3-1.5x) that guessing costs real time.
These tests check the mechanism, not the specific winner, since that is hardware- and
shape-dependent by design.
"""

import pytest
import torch
import triton
import triton.language as tl

from flex_attention import flash_attn_func, tune_block_plan
from flex_attention.tuning import _is_pure_causal
from flex_attention.block_sparse import dense_to_block_sparse

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

BLOCK = 64


def _strided(b, h, q_idx, kv_idx):
    return ((((q_idx // BLOCK) * 5 + (kv_idx // BLOCK)) % 3) == 0) & (kv_idx <= q_idx)


@triton.jit
def _strided_mask_mod(b, h, q_idx, kv_idx):
    # the @triton.jit twin of _strided; causal is left to the causal= flag
    return (((q_idx // 64) * 5 + (kv_idx // 64)) % 3) == 0


def test_is_pure_causal_detects_an_exact_causal_plan():
    """A plan covering exactly the causal blocks is dense-equivalent; a sparser one is not."""
    full = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    causal_plan = dense_to_block_sparse(full, torch.zeros_like(full), (64, 64)).with_causal(256, 256)
    assert _is_pure_causal(causal_plan, 256, 256)

    holed = causal_plan.full.to_dense(4).clone()
    holed[:, :, 3, 0] = False  # drop one strictly-below-diagonal block
    not_causal = dense_to_block_sparse(holed, causal_plan.masked.to_dense(4), (64, 64))
    assert not _is_pure_causal(not_causal, 256, 256)


@needs_gpu
def test_tune_rejects_a_block_size_too_coarse_for_the_mask():
    """_strided varies inside a 128x128 tile, so that granularity cannot express it.

    Without a mask_mod the kernel would keep those partial blocks whole and silently
    attend excluded positions, so the candidate must be rejected rather than timed.
    """
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 512, 2, 64
    torch.manual_seed(0)
    mk = lambda: torch.randn(  # noqa: E731
        batch, seqlen, nheads, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.3
    q, k, v = mk(), mk(), mk()

    result = tune_block_plan(
        _strided, q, k, v, causal=True,
        block_sizes=((64, 64), (128, 128)), warmup=3, reps=5,
    )
    assert result.best.ok and result.best.plan is not None
    assert result.best.block_size == (64, 64)
    coarse = [c for c in result.candidates if c.name.endswith("128x128")][0]
    assert not coarse.ok and "too coarse" in coarse.error
    # a strided pattern is not expressible by the causal flag, so no dense candidate
    assert not any("dense" in c.name for c in result.candidates)


@needs_gpu
def test_tune_accepts_coarse_block_sizes_when_a_mask_mod_resolves_them():
    """With a mask_mod supplied the kernel can resolve partial blocks, so every block
    size is admissible again -- and all of them must agree numerically."""
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 512, 2, 64
    torch.manual_seed(0)
    mk = lambda: torch.randn(  # noqa: E731
        batch, seqlen, nheads, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.3
    q, k, v = mk(), mk(), mk()

    result = tune_block_plan(
        _strided, q, k, v, causal=True, mask_mod=_strided_mask_mod,
        block_sizes=((64, 64), (128, 128)), warmup=3, reps=5,
    )
    usable = [c for c in result.candidates if c.ok]
    assert len(usable) == 2, [c.error for c in result.candidates]

    outs = [
        flash_attn_func(
            q, k, v, causal=True, mask_mod=_strided_mask_mod, block_sparse_tensors=c.plan
        )
        for c in usable
    ]
    for other in outs[1:]:
        torch.testing.assert_close(outs[0].float(), other.float(), atol=2e-2, rtol=2e-2)


@needs_gpu
def test_tune_offers_dense_only_when_the_mask_is_exactly_causal():
    device = "cuda"
    batch, seqlen, nheads, head_dim = 1, 512, 2, 64
    torch.manual_seed(0)
    mk = lambda: torch.randn(  # noqa: E731
        batch, seqlen, nheads, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.3
    q, k, v = mk(), mk(), mk()
    result = tune_block_plan(
        lambda b, h, qi, ki: ki <= qi, q, k, v, causal=True,
        block_sizes=((64, 64),), warmup=3, reps=5,
    )
    assert any("dense" in c.name for c in result.candidates)


@needs_gpu
def test_tune_skips_block_sizes_that_do_not_divide_the_sequence():
    """A generic block-size sweep must not raise on a size that cannot apply."""
    device = "cuda"
    torch.manual_seed(0)
    mk = lambda: torch.randn(1, 192, 2, 64, dtype=torch.bfloat16, device=device) * 0.3  # noqa: E731
    q, k, v = mk(), mk(), mk()
    result = tune_block_plan(
        lambda b, h, qi, ki: ki <= qi, q, k, v, causal=True,
        block_sizes=((64, 64), (128, 128)), warmup=2, reps=3,
    )
    skipped = [c for c in result.candidates if c.error == "seqlen not divisible"]
    assert skipped and result.best.ok  # 128 cannot divide 192; 64 still wins through


@needs_gpu
def test_tune_refuses_a_mask_nothing_can_resolve():
    """A causal mask_fn passed without causal=True and without a mask_mod leaves the
    diagonal partial blocks unresolvable, so every candidate must be rejected rather
    than quietly returning a plan that attends masked-out positions."""
    device = "cuda"
    torch.manual_seed(0)
    mk = lambda: torch.randn(1, 256, 2, 64, dtype=torch.bfloat16, device=device) * 0.3  # noqa: E731
    q, k, v = mk(), mk(), mk()
    with pytest.raises(RuntimeError, match="every candidate failed"):
        tune_block_plan(
            lambda b, h, qi, ki: ki <= qi, q, k, v,  # no causal=, no mask_mod=
            block_sizes=((64, 64),), warmup=2, reps=3,
        )
