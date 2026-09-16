"""Public API mirroring flash_attn.cute.interface's flash_attn_func / flash_attn_varlen_func,
backed by a Triton/AMD (CDNA3/MI300) implementation: AITER's own fwd_prefill + bwd
kernels for both passes (see flex_attention/_vendor/aiter_flash_attn/NOTICE.md for
provenance -- an earlier version used a different, Primus-Turbo-derived backward, but
that recomputed QK^T twice per tile pair and benchmarked ~1.3-1.8x slower; see the
NOTICE for details). See /home/lorri/.claude/plans/piped-painting-nebula.md for the
overall port plan.

Supported: causal, varlen, GQA/MQA, sliding window, return_lse, forward+backward
(Phase 1); score_mod / mask_mod / score_mod_bwd (Phase 3); MLA and other large or
asymmetric head dims up to 576/512 (Phase 4). Still unsupported, each raising a clear
NotImplementedError: softcap and learnable_sink (Phase 2), qv / gather_kv_indices
(top-k sparse KV), and block_sparse_tensors (Phase 5).

All supported features compose freely (causal with window_size, score_mod and MLA head
dims together, and so on). Earlier revisions rejected several such combinations; those
restrictions all traced back to a single upstream defect -- AITER's tuned backward
configs set matrix_instr_nonkdim=16, which miscompiles the accumulating tl.dot in the
causal diagonal sweep -- which is now patched out in the vendored kernel. See
_vendor/aiter_flash_attn/NOTICE.md.
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
    causal,
    window_size,
    requires_grad,
    head_dim_qk,
    head_dim_v,
    deterministic,
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
    if aux_tensors is not None or aux_scalars is not None:
        raise NotImplementedError("aux_tensors/aux_scalars are only used by score_mod/mask_mod, unsupported for now")
    if block_sparse_tensors is not None or block_sparse_tensors_bwd is not None:
        raise NotImplementedError("block_sparse_tensors is not yet supported by the Triton/AMD backend (Phase 5)")

    # NOTE: causal + window_size, causal + score_mod/mask_mod, and asymmetric head dims
    # with causal/deterministic were all rejected here previously. They now work: the
    # backward always uses AITER's "fused" kernels, which carry separate QK/V head dims,
    # support window_size, are where score_mod/mask_mod are instrumented, and are
    # correct for causal now that the matrix_instr_nonkdim miscompile is patched out
    # (see _sanitize_nonkdim in bwd.py).
    if score_mod_bwd is not None and score_mod is None:
        raise ValueError("score_mod_bwd was given without score_mod")
    if score_mod is not None and score_mod_bwd is None and requires_grad:
        raise ValueError(
            "score_mod requires score_mod_bwd for the backward pass: score_mod is inlined "
            "into the Triton kernel and is not auto-differentiated, so you must supply its "
            "VJP explicitly (same contract as flash_attn.cute's score_mod_bwd)."
        )


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
        score_mod=None,
        mask_mod=None,
        score_mod_bwd=None,
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
            score_mod=score_mod,
            mask_mod=mask_mod,
        )

        ctx.save_for_backward(q, k, v, o, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.score_mod = score_mod
        ctx.mask_mod = mask_mod
        ctx.score_mod_bwd = score_mod_bwd
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

        # Always "fused": it is correct for causal and non-causal alike now that the
        # matrix_instr_nonkdim miscompile is patched out (see _sanitize_nonkdim in
        # bwd.py), it is the only mode carrying separate QK/V head dims, it is the only
        # one supporting window_size, it is where score_mod/mask_mod are instrumented,
        # and it is bitwise deterministic (atomics live in the separate "fused_atomic"
        # mode, which this project never uses). AITER's "split" mode is strictly worse
        # here on every axis, so `deterministic` needs no separate path.
        mode = "fused"

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
            score_mod=ctx.score_mod,
            mask_mod=ctx.mask_mod,
            score_mod_bwd=ctx.score_mod_bwd,
        )
        # One None per non-tensor forward arg after q/k/v (15 total forward args).
        return (dq, dk, dv) + (None,) * 12


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
    ``deterministic`` is accepted for signature compatibility but is a no-op: the
    backward always uses AITER's "fused" kernels, which write each output tile from a
    single program (the atomics live in AITER's separate "fused_atomic" mode, which this
    project never uses) and were verified bitwise-identical across repeated runs. So the
    backward is always deterministic, and there is no slower alternative path to select.
    ``num_splits`` (split-KV forward) is not yet wired up; only num_splits=1 is supported.

    ``score_mod`` / ``mask_mod`` (FlexAttention-style) must be ``@triton.jit`` functions,
    inlined into the kernel at compile time:
        score_mod(score, b, h, q_idx, kv_idx) -> new_score
        mask_mod(b, h, q_idx, kv_idx) -> bool tile (True = keep)
        score_mod_bwd(dscore, score, b, h, q_idx, kv_idx) -> dscore_wrt_score_mod_input
    ``score_mod`` is NOT auto-differentiated -- if gradients are needed you must supply
    ``score_mod_bwd`` (its VJP), matching flash_attn.cute's contract. ``mask_mod`` needs
    no such hook (non-differentiable; masked positions get zero gradient).
    These compose freely with ``causal`` and ``window_size``. ``flex_attention.mods``
    also ships ready-made ``causal_mask_mod`` / ``make_sliding_window_mask_mod`` if you
    prefer to express those as mods (note those helpers are top-left aligned, whereas
    the ``causal``/``window_size`` flags are bottom-right aligned; they agree when
    seqlen_q == seqlen_k).
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
        causal,
        window_size,
        requires_grad=torch.is_grad_enabled()
        and (q.requires_grad or k.requires_grad or v.requires_grad),
        head_dim_qk=q.shape[-1],
        head_dim_v=v.shape[-1],
        deterministic=deterministic,
    )
    if num_splits != 1:
        raise NotImplementedError("num_splits != 1 is not yet supported by the Triton/AMD backend")

    o, lse = _FlashAttnFunc.apply(
        q, k, v, softmax_scale, causal, window_size, deterministic, return_lse,
        None, None, None, None, score_mod, mask_mod, score_mod_bwd,
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
    See flash_attn_func for the pack_gqa/deterministic/num_splits/score_mod notes.

    NOTE for score_mod/mask_mod under varlen: q_idx/kv_idx passed to your callables are
    *sequence-local* (0-based within each sequence), not offsets into the packed buffer,
    and ``b`` is the batch index within cu_seqlens -- so a causal mask_mod works
    unchanged across varlen and dense.
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
        causal,
        window_size,
        requires_grad=torch.is_grad_enabled()
        and (q.requires_grad or k.requires_grad or v.requires_grad),
        head_dim_qk=q.shape[-1],
        head_dim_v=v.shape[-1],
        deterministic=deterministic,
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
        score_mod,
        mask_mod,
        score_mod_bwd,
    )
    return (o, lse) if return_lse else o
