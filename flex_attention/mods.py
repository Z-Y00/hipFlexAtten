"""Ready-made score_mod / mask_mod callables for flash_attn_func / flash_attn_varlen_func.

All of these are ``@triton.jit`` functions inlined into the attention kernel at compile
time. Write your own the same way:

    @triton.jit
    def my_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx        # True = keep

    @triton.jit
    def my_score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_expr

    @triton.jit
    def my_score_mod_bwd(dscore, score, b, h, q_idx, kv_idx):
        return dscore * d_my_score_mod_d_score   # the VJP, written by hand

``q_idx`` / ``kv_idx`` are index tiles that broadcast against ``score``; ``b`` and ``h``
are scalars. Under varlen, indices are sequence-local, so these mods work unchanged for
both dense and varlen.
"""

import triton
import triton.language as tl

__all__ = ["causal_mask_mod", "identity_score_mod_bwd", "make_sliding_window_mask_mod"]


@triton.jit
def causal_mask_mod(b, h, q_idx, kv_idx):
    """Top-left-aligned causal mask (query i attends keys 0..i).

    Use this instead of ``causal=True`` when combining causality with score_mod/mask_mod
    and you need gradients -- see flash_attn_func's docstring for why.

    NOTE: this is top-left aligned. ``causal=True`` in the kernels is *bottom-right*
    aligned (query i attends keys 0..i + (seqlen_k - seqlen_q)), which differs whenever
    seqlen_q != seqlen_k. They agree for the common square case.
    """
    return kv_idx <= q_idx


@triton.jit
def identity_score_mod_bwd(dscore, score, b, h, q_idx, kv_idx):
    """VJP for any purely *additive* score_mod (``score + f(b, h, q_idx, kv_idx)``).

    d/d(score) of (score + const) is 1, so the incoming gradient passes straight through.
    """
    return dscore


def make_sliding_window_mask_mod(window_left: int, window_right: int):
    """Build a mask_mod for a sliding window, as an alternative to ``window_size=`` when
    gradients are needed alongside score_mod/mask_mod.

    Negative bounds mean "unbounded on that side", matching the ``window_size`` convention.
    Returns a ``@triton.jit`` function; build it once and reuse it, since each distinct
    function object is a separate Triton compilation.

    NOTE: like ``causal_mask_mod``, this is top-left aligned, whereas the ``window_size=``
    kernel path is bottom-right aligned. They agree when seqlen_q == seqlen_k.
    """
    # Closure constants must be wrapped with tl.constexpr(...) to be visible inside a
    # @triton.jit body; an annotated assignment (`x: tl.constexpr = ...`) is not supported.
    left = tl.constexpr(window_left if window_left >= 0 else -1)
    right = tl.constexpr(window_right if window_right >= 0 else -1)
    # Pick the branch at build time so the kernel body has no dead comparisons.
    has_left = tl.constexpr(window_left >= 0)
    has_right = tl.constexpr(window_right >= 0)

    @triton.jit
    def sliding_window_mask_mod(b, h, q_idx, kv_idx):
        rel = kv_idx - q_idx
        if has_left and has_right:
            keep = (rel >= -left) & (rel <= right)
        elif has_left:
            keep = rel >= -left
        elif has_right:
            keep = rel <= right
        else:
            keep = rel >= (q_idx - q_idx - 1)  # always true, shaped like rel
        return keep

    return sliding_window_mask_mod
