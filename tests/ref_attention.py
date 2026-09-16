"""Plain PyTorch-eager reference attention: the correctness oracle for this project.

There's no NVIDIA box to diff flash_attn.cute against, and FA-4 itself can't run on
AMD, so this eager implementation (not flash_attn.cute) is ground truth throughout.
"""

import math
from typing import Optional, Tuple

import torch


def ref_attention_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q,k,v: (batch, seqlen, nheads, head_dim), nheads_k may divide nheads_q (GQA).
    Returns (out, lse) with out same dtype as q, lse float32 (batch, nheads_q, seqlen_q).
    """
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_k, head_dim_v = v.shape
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    group = nheads_q // nheads_k

    qf = q.float().transpose(1, 2)  # (b, hq, sq, d)
    kf = k.float().transpose(1, 2)  # (b, hk, sk, d)
    vf = v.float().transpose(1, 2)  # (b, hk, sk, dv)
    kf = kf.repeat_interleave(group, dim=1)
    vf = vf.repeat_interleave(group, dim=1)

    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * softmax_scale

    row = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
    col = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
    col_offset = seqlen_k - seqlen_q  # bottom-right aligned, matching the Triton kernels
    if causal:
        scores = scores.masked_fill(col > row + col_offset, float("-inf"))
    left, right = window_size
    if left is not None:
        scores = scores.masked_fill(col < row + col_offset - left, float("-inf"))
    if right is not None:
        scores = scores.masked_fill(col > row + col_offset + right, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)  # fully-masked rows (short local-attn tails)
    out = torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2).to(q.dtype)
    lse = torch.nan_to_num(lse, neginf=0.0).transpose(1, 2).contiguous()  # match kernels' 0-fill convention
    return out, lse.transpose(1, 2)


def ref_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """q,k,v: (total_seqlen, nheads, head_dim). Returns out: (total_seqlen_q, nheads_q, head_dim_v)."""
    nheads_q = q.shape[1]
    head_dim_v = v.shape[-1]
    batch = cu_seqlens_q.numel() - 1
    out = torch.empty(q.shape[0], nheads_q, head_dim_v, dtype=q.dtype, device=q.device)
    for b in range(batch):
        qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        ks, ke = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        qb = q[qs:qe].unsqueeze(0)
        kb = k[ks:ke].unsqueeze(0)
        vb = v[ks:ke].unsqueeze(0)
        ob, _ = ref_attention_dense(qb, kb, vb, causal=causal, window_size=window_size, softmax_scale=softmax_scale)
        out[qs:qe] = ob.squeeze(0)
    return out
