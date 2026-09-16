import pytest
import torch

from flex_attention import flash_attn_func, flash_attn_varlen_func
from tests.ref_attention import ref_attention_dense, ref_attention_varlen

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DTYPE_TOL = {torch.float16: (2e-2, 2e-2), torch.bfloat16: (3e-2, 3e-2)}


def _rand_qkv(batch, seqlen_q, seqlen_k, nheads_q, nheads_k, head_dim, dtype, device):
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, nheads_q, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(batch, seqlen_k, nheads_k, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(batch, seqlen_k, nheads_k, head_dim, dtype=dtype, device=device) * 0.1
    return [t.requires_grad_() for t in (q, k, v)]


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gqa_ratio", [1, 4])
@pytest.mark.parametrize("window_size", [(None, None), (128, 0)])
def test_dense_forward_backward(dtype, causal, gqa_ratio, window_size):
    device = "cuda"
    batch, seqlen, head_dim = 2, 384, 64
    nheads_k = 4
    nheads_q = nheads_k * gqa_ratio
    q, k, v = _rand_qkv(batch, seqlen, seqlen, nheads_q, nheads_k, head_dim, dtype, device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]

    out, lse = flash_attn_func(q, k, v, causal=causal, window_size=window_size, return_lse=True)
    ref_out, ref_lse = ref_attention_dense(qr, kr, vr, causal=causal, window_size=window_size)

    atol, rtol = DTYPE_TOL[dtype]
    torch.testing.assert_close(out.float(), ref_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)

    do = torch.randn_like(out)
    out.backward(do)
    ref_out.backward(do)

    torch.testing.assert_close(q.grad.float(), qr.grad.float(), atol=atol * 5, rtol=rtol * 5)
    torch.testing.assert_close(k.grad.float(), kr.grad.float(), atol=atol * 5, rtol=rtol * 5)
    torch.testing.assert_close(v.grad.float(), vr.grad.float(), atol=atol * 5, rtol=rtol * 5)


@needs_gpu
def test_backward_is_deterministic():
    """The backward always uses AITER's non-atomic fused kernels, so repeated runs must
    be bitwise identical regardless of the `deterministic` flag."""
    device = "cuda"
    grads = []
    for _ in range(3):
        q, k, v = _rand_qkv(2, 512, 512, 8, 8, 64, torch.float16, device)
        out = flash_attn_func(q, k, v, causal=True)
        torch.manual_seed(1234)
        out.backward(torch.randn_like(out))
        grads.append((q.grad.clone(), k.grad.clone(), v.grad.clone()))
    for i in range(3):
        for j in (1, 2):
            assert torch.equal(grads[0][i], grads[j][i]), "backward is not deterministic"


@needs_gpu
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_forward(causal):
    device = "cuda"
    dtype = torch.float16
    nheads, head_dim = 4, 64
    seqlens_q = [37, 111, 5]
    seqlens_k = [64, 111, 20] if not causal else seqlens_q
    cu_seqlens_q = torch.tensor([0, *torch.tensor(seqlens_q).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(seqlens_k).cumsum(0).tolist()], dtype=torch.int32, device=device)
    max_seqlen_q, max_seqlen_k = max(seqlens_q), max(seqlens_k)

    torch.manual_seed(0)
    q = torch.randn(cu_seqlens_q[-1].item(), nheads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(cu_seqlens_k[-1].item(), nheads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(cu_seqlens_k[-1].item(), nheads, head_dim, dtype=dtype, device=device) * 0.1

    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=causal
    )
    ref_out = ref_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)


def test_unsupported_features_raise():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = k = v = torch.randn(1, 8, 2, 32, device=device)
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, softcap=10.0)
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, learnable_sink=torch.zeros(2, device=device))
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, block_sparse_tensors=object())
    # score_mod / mask_mod are supported as of Phase 3 -- see tests/test_score_mod.py.
