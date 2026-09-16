"""Standalone correctness probe for the score_mod/mask_mod prototype -- not part of the
main pytest suite (this kernel isn't wired into the public API yet). Run directly:
    python3 -m flex_attention._experimental.test_score_mod_prototype
"""

import torch
import triton
import triton.language as tl

from flex_attention._experimental.score_mod_kernel import bwd, fwd


@triton.jit
def linear_penalty_score_mod(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx).to(tl.float32) * -0.1


@triton.jit
def linear_penalty_score_mod_bwd(dscore, score, b, h, q_idx, kv_idx):
    # score_mod is additive (d/dx (x + const) == 1), so the VJP is just the identity.
    return dscore


@triton.jit
def causal_mask_mod(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def ref(q, k, v, sm_scale, use_score_mod, use_mask_mod):
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * sm_scale
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    q_idx = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
    kv_idx = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
    if use_score_mod:
        scores = scores + (q_idx - kv_idx).float() * -0.1
    if use_mask_mod:
        scores = scores.masked_fill(kv_idx > q_idx, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2).to(q.dtype)
    lse = torch.logsumexp(scores, dim=-1)  # (batch, nheads, seqlen_q), matches the kernel's LSE layout
    return out, lse


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    batch, seqlen, nheads, head_dim = 2, 256, 4, 64
    sm_scale = head_dim**-0.5

    q = (torch.randn(batch, seqlen, nheads, head_dim, dtype=dtype, device=device) * 0.1).requires_grad_()
    k = (torch.randn(batch, seqlen, nheads, head_dim, dtype=dtype, device=device) * 0.1).requires_grad_()
    v = (torch.randn(batch, seqlen, nheads, head_dim, dtype=dtype, device=device) * 0.1).requires_grad_()

    for name, score_mod, mask_mod, score_mod_bwd in [
        ("plain", None, None, None),
        ("score_mod only", linear_penalty_score_mod, None, linear_penalty_score_mod_bwd),
        ("mask_mod only", None, causal_mask_mod, None),
        ("score_mod + mask_mod", linear_penalty_score_mod, causal_mask_mod, linear_penalty_score_mod_bwd),
    ]:
        out, lse = fwd(q.detach(), k.detach(), v.detach(), sm_scale, score_mod=score_mod, mask_mod=mask_mod)
        ref_out, ref_lse = ref(q, k, v, sm_scale, score_mod is not None, mask_mod is not None)
        out_err = (out.float() - ref_out.float()).abs().max().item()
        lse_err = (lse - ref_lse).abs().max().item()
        print(f"{name:24s} fwd  out_max_err={out_err:.5f} lse_max_err={lse_err:.5f}")
        assert out_err < 1e-2, f"{name}: out mismatch"
        assert lse_err < 1e-2, f"{name}: lse mismatch"

        do = torch.randn_like(out)
        dq, dk, dv = bwd(
            do, q.detach(), k.detach(), v.detach(), out, lse, sm_scale,
            score_mod=score_mod, mask_mod=mask_mod, score_mod_bwd=score_mod_bwd,
        )
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_out, (q, k, v), do, retain_graph=False)
        dq_err = (dq.float() - ref_dq.float()).abs().max().item()
        dk_err = (dk.float() - ref_dk.float()).abs().max().item()
        dv_err = (dv.float() - ref_dv.float()).abs().max().item()
        print(f"{'':24s} bwd  dq_max_err={dq_err:.5f} dk_max_err={dk_err:.5f} dv_max_err={dv_err:.5f}")
        assert dq_err < 2e-2, f"{name}: dq mismatch"
        assert dk_err < 2e-2, f"{name}: dk mismatch"
        assert dv_err < 2e-2, f"{name}: dv mismatch"

    print("all prototype checks passed")


if __name__ == "__main__":
    main()
