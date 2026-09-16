"""Public API mirroring flash_attn.cute.interface's flash_attn_func / flash_attn_varlen_func,
backed by a Triton/AMD (CDNA3/MI300) implementation: AITER's own fwd_prefill + bwd
kernels for both passes (see flex_attention/_vendor/aiter_flash_attn/NOTICE.md for
provenance -- an earlier version used a different, Primus-Turbo-derived backward, but
that recomputed QK^T twice per tile pair and benchmarked ~1.3-1.8x slower; see the
NOTICE for details). See /home/lorri/.claude/plans/piped-painting-nebula.md for the
overall port plan.

Phase 1 scope: causal, varlen, GQA/MQA, sliding window, return_lse, forward+backward.
Everything else in flash_attn.cute's signature (softcap, learnable_sink, score_mod,
mask_mod, qv/gather_kv_indices, block_sparse_tensors) is explicitly NotImplementedError
until its own phase lands.
"""

from typing import Callable, Optional, Tuple

import torch

from flex_attention._vendor.aiter_flash_attn.bwd import attention_backward_triton_impl
from flex_attention._vendor.aiter_flash_attn.fwd_prefill import (
    attention_forward_prefill_triton_impl,
)

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]


def _resolve_window(window_size: Tuple[Optional[int], Optional[int]]) -> Tuple[int, int]:
    left, right = window_size
    return (-1 if left is None else left, -1 if right is None else right)


def _check_unsupported(
    qv,
    gather_kv_indices,
    learnable_sink,
    softcap,
    score_mod,
    score_mod_bwd,
    mask_mod,
    aux_tensors,
    aux_scalars,
    block_sparse_tensors,
    block_sparse_tensors_bwd,
):
    if qv is not None:
        raise NotImplementedError("qv-packed input is not yet supported by the Triton/AMD backend")
    if gather_kv_indices is not None:
        raise NotImplementedError(
            "gather_kv_indices (top-k sparse KV / MLA) is not yet supported by the Triton/AMD backend"
        )
    if learnable_sink is not None:
        raise NotImplementedError("learnable_sink is not yet wired up in the Triton/AMD backend (Phase 2)")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not yet supported by the Triton/AMD backend (Phase 2)")
    if score_mod is not None or score_mod_bwd is not None:
        raise NotImplementedError("score_mod is not yet supported by the Triton/AMD backend (Phase 3)")
    if mask_mod is not None:
        raise NotImplementedError("mask_mod is not yet supported by the Triton/AMD backend (Phase 3)")
    if aux_tensors is not None or aux_scalars is not None:
        raise NotImplementedError("aux_tensors/aux_scalars are only used by score_mod/mask_mod, unsupported for now")
    if block_sparse_tensors is not None or block_sparse_tensors_bwd is not None:
        raise NotImplementedError("block_sparse_tensors is not yet supported by the Triton/AMD backend (Phase 5)")


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[Optional[int], Optional[int]],
        deterministic: bool,
        return_lse: bool,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
    ):
        is_varlen = cu_seqlens_q is not None
        layout = "thd" if is_varlen else "bshd"
        window_size_left, window_size_right = _resolve_window(window_size)
        head_dim = q.shape[-1]
        head_dim_v = v.shape[-1]
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        if is_varlen:
            total_seqlen_q, nheads_q, _ = q.shape
            o = torch.empty(total_seqlen_q, nheads_q, head_dim_v, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(nheads_q, total_seqlen_q, dtype=torch.float32, device=q.device)
        else:
            batch, seqlen_q, nheads_q, _ = q.shape
            o = torch.empty(batch, seqlen_q, nheads_q, head_dim_v, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(batch, nheads_q, seqlen_q, dtype=torch.float32, device=q.device)

        attention_forward_prefill_triton_impl(
            q,
            k,
            v,
            o,
            softmax_lse,
            None,  # sd_mask
            softmax_scale,
            None,  # alibi_slopes
            causal,
            window_size_left,
            window_size_right,
            None,  # bias
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q if max_seqlen_q is not None else 0,
            max_seqlen_k if max_seqlen_k is not None else 0,
            0.0,  # dropout_p
            None,  # philox_seed
            None,  # philox_offset
            False,  # return_scores
            True,  # use_exp2
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
        )

        ctx.save_for_backward(q, k, v, o, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.layout = layout
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.return_lse = return_lse
        ctx.deterministic = deterministic
        return (o, softmax_lse) if return_lse else (o, None)

    @staticmethod
    def backward(ctx, do, _dlse_unused):
        q, k, v, o, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        is_varlen = ctx.layout == "thd"
        if is_varlen:
            total_seqlen_q, nheads_q, _ = q.shape
            delta = torch.zeros(nheads_q, total_seqlen_q, dtype=torch.float32, device=q.device)
        else:
            batch, seqlen_q, nheads_q, _ = q.shape
            delta = torch.zeros(batch, nheads_q, seqlen_q, dtype=torch.float32, device=q.device)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # AITER's "fused" causal backward (this project's default -- see the docstring on
        # flash_attn_func) produces a wrong dV at this pinned commit (fedccf0, confirmed by
        # a minimal repro bypassing this wrapper entirely: fused causal dV differs from a
        # plain-PyTorch reference by O(1) absolute error, "fused_atomic" causal crashes with
        # a missing-argument TypeError from inside bwd.py itself, and "split" causal matches
        # the reference to fp16/bf16 precision). So causal always uses "split" regardless of
        # `deterministic`, until this is fixed upstream or independently root-caused; only
        # non-causal gets the deterministic/fused choice.
        mode = "split" if (ctx.causal or ctx.deterministic) else "fused"
        has_window = ctx.window_size_left != -1 or ctx.window_size_right != -1
        if mode == "split" and has_window:
            # AITER's "split" backward doesn't support window_size at all, and "fused" -
            # the only mode that does - has a broken causal dV (see the comment above).
            # There's no working backward for causal+window at this pinned aiter commit.
            raise NotImplementedError(
                "causal + window_size backward is not supported: AITER's 'fused' mode "
                "(the only mode with window_size support) has a broken causal dV at the "
                "pinned commit, and 'split' mode doesn't support window_size at all. "
                "See flex_attention/_vendor/aiter_flash_attn/NOTICE.md."
            )

        attention_backward_triton_impl(
            do=do.contiguous(),
            q=q,
            k=k,
            v=v,
            o=o,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            delta=delta,
            sm_scale=ctx.softmax_scale,
            alibi_slopes=None,
            causal=ctx.causal,
            layout=ctx.layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q if ctx.max_seqlen_q is not None else q.shape[1],
            max_seqlen_k=ctx.max_seqlen_k if ctx.max_seqlen_k is not None else k.shape[1],
            use_exp2=True,
            mode=mode,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse: bool = False,
):
    """Dense (bshd) flash attention. q/k/v: (batch, seqlen, nheads, head_dim).

    ``pack_gqa`` is accepted for signature compatibility with flash_attn.cute but is a
    no-op here: GQA/MQA is handled by plain head-index broadcast in both the forward and
    backward Triton kernels, which needs no explicit "packing" step on AMD.
    ``deterministic=False`` (default) uses AITER's ``bwd.py`` "fused" backward mode
    (fastest: single pass per Q/K tile pair, ``tl.atomic_add`` into dQ across K-blocks,
    so dQ's accumulation order -- and thus its exact floating-point value -- can vary
    run to run). ``deterministic=True`` uses its "split" mode instead (no atomics, each
    kernel owns its own output tile, ~1.3-1.8x slower; see the backend's NOTICE.md for
    why we benchmarked and chose this tradeoff rather than always using the slower one).
    Note AITER's "split" mode does not support ``window_size`` in the backward pass; that
    combination raises NotImplementedError from within bwd.py itself.
    Also note: when ``causal=True``, "split" mode is always used regardless of
    ``deterministic`` -- AITER's "fused"/"fused_atomic" causal backward are broken at the
    pinned commit (wrong dV / a crash respectively; see the backend's NOTICE.md).
    ``num_splits`` (split-KV forward) is not yet wired up; only num_splits=1 is supported.
    """
    _check_unsupported(
        qv,
        gather_kv_indices,
        learnable_sink,
        softcap,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
    )
    if num_splits != 1:
        raise NotImplementedError("num_splits != 1 is not yet supported by the Triton/AMD backend")

    o, lse = _FlashAttnFunc.apply(
        q, k, v, softmax_scale, causal, window_size, deterministic, return_lse, None, None, None, None
    )
    return (o, lse) if return_lse else o


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse: bool = False,
):
    """Varlen (thd) flash attention. q/k/v: (total_seqlen, nheads, head_dim);
    cu_seqlens_{q,k} are int32 [batch + 1] cumulative sequence-length offsets.
    See flash_attn_func for the pack_gqa/deterministic/num_splits notes.
    """
    _check_unsupported(
        qv,
        gather_kv_indices,
        learnable_sink,
        softcap,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
    )
    if num_splits != 1:
        raise NotImplementedError("num_splits != 1 is not yet supported by the Triton/AMD backend")

    o, lse = _FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        deterministic,
        return_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    return (o, lse) if return_lse else o
