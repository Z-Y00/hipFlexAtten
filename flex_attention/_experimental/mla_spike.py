"""Phase 4 design spike: do the MLA head-dim shapes work on CDNA3 as-is?

The plan flagged an open question: upstream FA-4 needs dedicated kernels for
head_dim=256 (Blackwell 2-CTA) and splits the MLA backward into separate dQ/dK GEMM
kernels, because WGMMA tiling and Blackwell's cluster launch force it. On AMD there's
no such hardware constraint -- the question is purely whether Triton can tile a large
head dim (up to 512) inside AITER's existing kernels within CDNA3's LDS/VGPR budget.

AITER's kernels already carry separate BLOCK_DMODEL_QK / BLOCK_DMODEL_V constexprs and
impose no upper bound on head dim (just next-pow2 rounding), so this measures reality
rather than guessing. Run:
    python3 -m flex_attention._experimental.mla_spike
"""

import traceback

import torch

from flex_attention import flash_attn_func

# (head_dim_qk, head_dim_v, label)
SHAPES = [
    (64, 64, "baseline square"),
    (128, 128, "baseline square (128)"),
    (192, 128, "MLA: DeepSeek 192/128"),
    (256, 256, "FA-4 dedicated-kernel shape 256/256"),
    (64, 512, "MLA absorbed: 64/512"),
    (512, 512, "MLA absorbed: 512/512"),
    (576, 512, "MLA full: 576/512 (DeepSeek-V3 rope-concat)"),
]

BATCH, SEQLEN, NHEADS = 1, 512, 4
DTYPE = torch.bfloat16


def ref_out(q, k, v, scale):
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    s = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, vf).transpose(1, 2)


def try_shape(hd_qk, hd_v, label):
    torch.manual_seed(0)
    dev = "cuda"
    q = (torch.randn(BATCH, SEQLEN, NHEADS, hd_qk, dtype=DTYPE, device=dev) * 0.1).requires_grad_()
    k = (torch.randn(BATCH, SEQLEN, NHEADS, hd_qk, dtype=DTYPE, device=dev) * 0.1).requires_grad_()
    v = (torch.randn(BATCH, SEQLEN, NHEADS, hd_v, dtype=DTYPE, device=dev) * 0.1).requires_grad_()
    scale = hd_qk**-0.5

    status = {"label": label, "shape": f"{hd_qk}/{hd_v}"}
    try:
        out = flash_attn_func(q, k, v, softmax_scale=scale)
        want = ref_out(q, k, v, scale)
        status["fwd_err"] = (out.float() - want).abs().max().item()
    except Exception as e:  # noqa: BLE001
        status["fwd_err"] = None
        status["fwd_exc"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
        return status

    try:
        do = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
        rq, rk, rv = torch.autograd.grad(want.to(DTYPE), (q, k, v), do)
        status["dq_err"] = (dq.float() - rq.float()).abs().max().item()
        status["dk_err"] = (dk.float() - rk.float()).abs().max().item()
        status["dv_err"] = (dv.float() - rv.float()).abs().max().item()
    except Exception as e:  # noqa: BLE001
        status["dq_err"] = None
        status["bwd_exc"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    return status


def main():
    print(f"{'shape':>10s}  {'label':38s} {'fwd':>10s} {'dq':>10s} {'dk':>10s} {'dv':>10s}")
    print("-" * 100)
    for hd_qk, hd_v, label in SHAPES:
        s = try_shape(hd_qk, hd_v, label)
        if s["fwd_err"] is None:
            print(f"{s['shape']:>10s}  {label:38s} FWD FAILED: {s.get('fwd_exc')}")
            continue
        if s.get("dq_err") is None:
            print(
                f"{s['shape']:>10s}  {label:38s} {s['fwd_err']:10.5f} "
                f"BWD FAILED: {s.get('bwd_exc')}"
            )
            continue
        print(
            f"{s['shape']:>10s}  {label:38s} {s['fwd_err']:10.5f} "
            f"{s['dq_err']:10.5f} {s['dk_err']:10.5f} {s['dv_err']:10.5f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
