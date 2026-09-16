"""Phase 3 prototype: score_mod / mask_mod support for Triton on AMD.

Deliberately NOT wired into flex_attention.interface or AITER's fwd_prefill.py/bwd.py
yet. Per the plan (piped-painting-nebula.md), this validates the Triton-level mechanics
in a small, simple standalone kernel before touching the production, autotuned,
GQA/varlen/causal-fused AITER kernels -- so a bug here is cheap to find and fix, instead
of an obscure regression buried in a 2000-line kernel.

Convention (matches PyTorch's own FlexAttention and flash_attn.cute's mask.py):
  score_mod(score, b, h, q_idx, kv_idx) -> new_score
      score: (BLOCK_M, BLOCK_N) tile of qk*softmax_scale, *before* the online-softmax
        max/sum update. b, h: scalar int32. q_idx: (BLOCK_M, 1) int32. kv_idx:
        (1, BLOCK_N) int32. Ordinary Triton arithmetic on these broadcasts as expected.
      Must be a @triton.jit function (or None to skip).
  mask_mod(b, h, q_idx, kv_idx) -> bool tile, broadcastable to (BLOCK_M, BLOCK_N)
      True = keep, False = mask out (-inf). Must be a @triton.jit function (or None).
  score_mod_bwd(dscore, score, b, h, q_idx, kv_idx) -> dscore_before_mod
      The user-supplied VJP of score_mod: given the gradient w.r.t. score_mod's output
      (dscore) and score_mod's *input* (score, the pre-mod qk*scale -- recomputed the
      same way the forward did, not the post-mod value), returns the gradient w.r.t.
      score_mod's input. Not auto-differentiated -- same manual-VJP design as
      flash_attn.cute's score_mod_bwd parameter. mask_mod needs no such hook: it's
      non-differentiable, and this kernel re-applies it to zero out dscore at masked
      positions explicitly (rather than relying on p==0 there, since a score_mod_bwd
      formula isn't guaranteed to map a zero input gradient to zero output).

Layout: dense (batch, seqlen, nheads, head_dim), no GQA/varlen/causal-fusion/dropout
yet -- mask_mod subsumes causal (pass a causal mask_mod). Backward is a single
fused-atomic kernel (dQ accumulated via tl.atomic_add across kv-blocks, dK/dV owned
per kv-block) -- the simplest correct design, not the fastest; this is a prototype.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64

@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    sm_scale,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    nheads,
    seqlen_q,
    seqlen_k,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCORE_MOD: tl.constexpr,
    MASK_MOD: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    mask_m = offs_m < seqlen_q
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    RCP_LN2: tl.constexpr = 1.4426950408889634

    for start_n in range(0, seqlen_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen_k

        k_ptrs = (
            K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale

        if SCORE_MOD is not None:
            qk = SCORE_MOD(qk, off_b, off_h, offs_m[:, None], offs_n[None, :])
        if MASK_MOD is not None:
            keep = MASK_MOD(off_b, off_h, offs_m[:, None], offs_n[None, :])
            qk = tl.where(keep, qk, float("-inf"))
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.math.exp2((qk - m_ij_safe[:, None]) * RCP_LN2)
        p = tl.where(qk == float("-inf"), 0.0, p)

        alpha = tl.math.exp2((m_i - m_ij_safe) * RCP_LN2)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = (
            V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

        m_i = m_ij

    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.math.log2(l_i_safe) / RCP_LN2)

    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])

    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=mask_m)


def fwd(q, k, v, sm_scale, score_mod=None, mask_mod=None):
    batch, seqlen_q, nheads, head_dim = q.shape
    _, seqlen_k, _, _ = k.shape
    o = torch.empty_like(q)
    lse = torch.empty(batch, nheads, seqlen_q, dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SCORE_MOD=score_mod,
        MASK_MOD=mask_mod,
    )
    return o, lse


@triton.jit
def _bwd_preprocess(
    O, DO, Delta, stride_b, stride_m, stride_h, stride_d, nheads, seqlen_q, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    base = off_b * stride_b + off_h * stride_h
    o_ptrs = O + base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    do_ptrs = DO + base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_bh * seqlen_q + offs_m, delta, mask=mask_m)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DQ,
    DK,
    DV,
    sm_scale,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_dob,
    stride_dom,
    stride_doh,
    stride_dod,
    nheads,
    seqlen_q,
    seqlen_k,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCORE_MOD: tl.constexpr,
    MASK_MOD: tl.constexpr,
    SCORE_MOD_BWD: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < seqlen_k

    RCP_LN2: tl.constexpr = 1.4426950408889634

    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    for start_m in range(0, seqlen_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q

        q_ptrs = (
            Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        do_ptrs = (
            DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        lse = tl.load(LSE + off_bh * seqlen_q + offs_m, mask=mask_m, other=0.0)
        delta = tl.load(Delta + off_bh * seqlen_q + offs_m, mask=mask_m, other=0.0)

        qk_raw = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        qk = qk_raw
        if SCORE_MOD is not None:
            qk = SCORE_MOD(qk, off_b, off_h, offs_m[:, None], offs_n[None, :])
        keep = mask_m[:, None] & mask_n[None, :]
        if MASK_MOD is not None:
            keep = keep & MASK_MOD(off_b, off_h, offs_m[:, None], offs_n[None, :])
        qk = tl.where(keep, qk, float("-inf"))

        p = tl.math.exp2((qk - lse[:, None]) * RCP_LN2)
        p = tl.where(keep, p, 0.0)

        dv += tl.dot(tl.trans(p.to(do.dtype)), do, out_dtype=tl.float32)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta[:, None])
        if SCORE_MOD_BWD is not None:
            ds = SCORE_MOD_BWD(ds, qk_raw, off_b, off_h, offs_m[:, None], offs_n[None, :])
        ds = tl.where(keep, ds, 0.0)

        dk += tl.dot(tl.trans(ds.to(q.dtype)), q, out_dtype=tl.float32)

        dq_partial = tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32) * sm_scale
        dq_ptrs = (
            DQ + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        tl.atomic_add(dq_ptrs, dq_partial, mask=mask_m[:, None])

    dk *= sm_scale
    dk_ptrs = DK + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    dv_ptrs = DV + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    tl.store(dk_ptrs, dk, mask=mask_n[:, None])
    tl.store(dv_ptrs, dv, mask=mask_n[:, None])


def bwd(do, q, k, v, o, lse, sm_scale, score_mod=None, mask_mod=None, score_mod_bwd=None):
    batch, seqlen_q, nheads, head_dim = q.shape
    _, seqlen_k, _, _ = k.shape
    delta = torch.empty(batch * nheads, seqlen_q, dtype=torch.float32, device=q.device)

    grid_pre = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _bwd_preprocess[grid_pre](
        o.contiguous(),
        do.contiguous(),
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        nheads,
        seqlen_q,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
    )

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    do_c = do.contiguous()
    grid_bwd = (triton.cdiv(seqlen_k, BLOCK_N), batch * nheads)
    _bwd_kernel[grid_bwd](
        q,
        k,
        v,
        do_c,
        lse.reshape(batch * nheads, seqlen_q),
        delta,
        dq,
        dk,
        dv,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        do_c.stride(0),
        do_c.stride(1),
        do_c.stride(2),
        do_c.stride(3),
        nheads,
        seqlen_q,
        seqlen_k,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SCORE_MOD=score_mod,
        MASK_MOD=mask_mod,
        SCORE_MOD_BWD=score_mod_bwd,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
