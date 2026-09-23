# hipFlexAtten

FlashAttention with a FlexAttention-style API for AMD GPUs, in Triton.

[`flash_attn/cute`](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)
(FlashAttention-4) is built entirely on NVIDIA Hopper/Blackwell primitives — TMA copies,
mbarrier warp-specialized pipelines, WGMMA/UMMA descriptors, Blackwell 2-CTA cooperative
kernels. None of that exists on CDNA, so this is a reimplementation of its *feature set*
in Triton for MI300-class hardware, not a translation of its code.

The Python API mirrors `flash_attn.cute.interface` so callers can swap backends.

Built on [ROCm/aiter](https://github.com/ROCm/aiter)'s Triton flash-attention kernels
(vendored, MIT). See [Attribution](#attribution).

## Status

Developed and tested on **MI300X (gfx942 / CDNA3)**, ROCm 7, Triton 3.5, PyTorch 2.9, and
verified on **MI355X (gfx950)**, ROCm 7.15, Triton 3.8, PyTorch 2.12. 133 tests pass on
both. RDNA is not a target.

## Install

```sh
git clone git@github.com:Z-Y00/hipFlexAtten.git
cd hipFlexAtten
pip install -e ".[dev]"
pytest tests/          # needs a ROCm GPU
```

## Usage

```python
from flex_attention import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)              # (batch, seqlen, nheads, head_dim)
out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
```

Varlen takes the packed `(total_seqlen, nheads, head_dim)` layout plus cumulative offsets:

```python
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
```

### score_mod / mask_mod

Arbitrary score and mask transforms, written as `@triton.jit` functions and inlined into
the kernel at compile time:

```python
import triton, triton.language as tl
from flex_attention import flash_attn_func, identity_score_mod_bwd

@triton.jit
def alibi_like(score, b, h, q_idx, kv_idx):
    return score - 0.1 * (q_idx - kv_idx).to(tl.float32)

out = flash_attn_func(q, k, v, score_mod=alibi_like, score_mod_bwd=identity_score_mod_bwd)
```

`score_mod` is **not** auto-differentiated. If you need gradients you must supply its
VJP as `score_mod_bwd` — the same contract as `flash_attn.cute`. Passing `score_mod`
without it raises rather than silently producing wrong gradients.
`identity_score_mod_bwd` covers any purely additive mod. `mask_mod` needs no VJP.

`flex_attention.mods` ships `causal_mask_mod`, `make_sliding_window_mask_mod(l, r)` and
`make_softcap_score_mod(cap)`.

> Note: those helpers are **top-left aligned**, whereas the `causal=` / `window_size=`
> kernel flags are **bottom-right aligned**. They agree when `seqlen_q == seqlen_k`.

### Block-sparse

```python
import triton, triton.language as tl
from flex_attention import create_block_sparse_from_mask_mod, flash_attn_func

# the block mask is built from a plain torch predicate...
bs = create_block_sparse_from_mask_mod(
    lambda b, h, q_idx, kv_idx: (q_idx - kv_idx).abs() < 1024,
    batch, nheads, seqlen_q, seqlen_k, block_size=(128, 128), device="cuda",
)

# ...and the same predicate is passed as a @triton.jit mask_mod for the partial blocks
@triton.jit
def band(b, h, q_idx, kv_idx):
    d = q_idx - kv_idx
    return (d < 1024) & (d > -1024)

out = flash_attn_func(q, k, v, mask_mod=band, block_sparse_tensors=bs)
```

The two must express the same predicate: the block mask decides which blocks are visited,
`mask_mod` masks individual positions inside the partial ones.

Empty blocks are skipped, fully-unmasked blocks skip masking entirely, and partial blocks
apply `mask_mod`. `create_block_sparse_varlen` builds the same thing for varlen (indices
are sequence-local). `causal=True` composes with `block_sparse_tensors` in both layouts —
blocks past the diagonal are dropped, straddling ones move onto the masked pass — so you
don't have to fold causality into your own mask_mod. See
[when this actually helps](#block-sparse-1).

### Picking a block size

The best block size for a sparse mask isn't predictable from the pattern alone — it's a
real tradeoff (bigger tiles are faster per-tile but coarser, so more masked-out work gets
dragged in) that flips depending on shape. `tune_block_plan` measures the candidates once
and picks the fastest, including a dense candidate when the mask happens to be exactly
causal:

```python
from flex_attention import tune_block_plan, flash_attn_func

result = tune_block_plan(my_mask_fn, q, k, v, causal=True)
print(result)                                   # per-candidate timings
out = result.apply(flash_attn_func, q, k, v, causal=True)
```

A block size that can't represent the mask exactly (a predicate that varies inside a
tile, with no `mask_mod` to resolve it) is rejected rather than silently mismeasured.

### Split-KV

`num_splits > 1` slices the KV loop across programs to raise occupancy. It changes only
*how* the forward is evaluated — output, LSE and gradients are unchanged — so it is a
pure throughput knob for short-query / long-context shapes.

## Feature support

| Feature | Status |
|---|---|
| causal, sliding window, GQA/MQA, varlen, `return_lse` | yes |
| forward + backward, fp16 / bf16 | yes |
| `softcap` | yes |
| `learnable_sink` (per-head, incl. `dsink`) | yes |
| `score_mod` / `mask_mod` / `score_mod_bwd` | yes |
| MLA & large head dims: 192/128, 256/256, 64/512, 512/512, 576/512 | yes |
| asymmetric `head_dim_qk != head_dim_v` | yes |
| block-sparse (dense and varlen) | yes |
| `num_splits > 1` | yes (dense only) |
| `qv` / `gather_kv_indices` (DeepSeek-V3.2 top-k sparse KV) | **not implemented** |
| `num_splits > 1` under varlen | **not implemented** |

Everything supported composes freely — causal with a window and a score_mod and MLA head
dims all at once is fine. Unimplemented options raise `NotImplementedError` rather than
silently ignoring the argument.

Two API notes:

- **`deterministic` is a no-op.** The backward always uses kernels that write each output
  tile from a single program (the atomic variant is never used), so it is *always*
  deterministic — verified bitwise across repeated runs. The flag is accepted for
  signature compatibility.
- **`pack_gqa` is a no-op.** GQA/MQA is handled by head-index broadcast; AMD needs no
  explicit packing step.

## Performance

MI300X, bf16. Reproduce with `python3 -m bench.bench_mi300` (and `bench_mla`,
`bench_block_sparse`, `bench_splits`).

### vs. AITER

Same forward kernel, so parity is expected; the causal backward gains come from fixing an
upstream miscompile (below).

| shape (b, s, hq, hk, d, causal) | fwd | fwd+bwd |
|---|---|---|
| (4, 2048, 32, 32, 128, False) | 1.06x | 0.99x |
| (4, 2048, 32, 8, 128, True) | 1.06x | 1.11x |
| (2, 8192, 32, 8, 128, False) | 1.06x | 1.01x |
| (2, 8192, 32, 8, 128, True) | 1.03x | 1.20x |

> The causal `fwd+bwd` baseline is AITER's *unpatched* default mode, which is numerically
> wrong at the pinned commit — so those columns compare speed only, not correctness.

Absolute forward throughput is 270–370 TFLOP/s (21–28% of the 1307 TFLOP/s bf16 dense
peak). That is the normal range for a Triton flash-attention: the peak figure is
matmul-only, and the softmax between the two GEMMs is not matmul work. A 144-point config
sweep found the tile choice already near-optimal, so further gains need a different
implementation strategy (assembly / CK / FlyDSL), not retiling.

### MLA / large head dims

B=1, S=4096, H=8:

| head_dim qk/v | fwd | fwd+bwd |
|---|---|---|
| 192/128 (DeepSeek) | 0.32 ms | 1.83 ms |
| 256/256 | 0.88 ms | 4.67 ms |
| 64/512 | 2.92 ms | 7.96 ms |
| 512/512 | 3.06 ms | 21.5 ms |
| 576/512 | 9.90 ms | 41.4 ms |

576 pads to 1024, so half of every tile is wasted and the block size drops to 32 — worth
knowing before picking that shape. Upstream FA-4 needs a dedicated Blackwell 2-CTA kernel
for head_dim 256 and split dQ/dK GEMM kernels for the MLA backward; on CDNA3 none of that
is necessary, the only constraint was the 64 KiB LDS budget.

### Block-sparse

Speedup vs. dense, fwd+bwd:

| seqlen | 10% block density | 25% block density |
|---|---|---|
| 2048 | 0.88x | 1.10x |
| 8192 | **4.87x** | **2.14x** |

**Block sparsity is the wrong tool for plain causal masking.** The dense kernel already
skips blocks via its own causal planning and keeps autotuned tile sizes, while the sparse
path pins tiles to the sparsity granularity — so it comes out *slower*. Use it for
patterns the dense path cannot express. It also needs the KV loop to dominate: at seqlen
2048 the kernel is launch-bound and sparsity is a wash.

### Split-KV

Forward, head_dim 128:

| case | grid | best | speedup |
|---|---|---|---|
| 1 query, 8K context | 8 | split=4 | 3.6x |
| 1 query, 32K context | 8 | split=16 | **12.3x** |
| 128 queries, 16K context | 8 | split=16 | 7.2x |
| b=8, s=4096, h=32 | 8192 | split=1 | — (splitting costs ~2x) |

The grid column is `batch * nheads * q_blocks` against 304 CUs; splitting helps exactly
when that is too small to fill the GPU.

## Upstream bugs found and fixed

Both are still present in ROCm/aiter and are documented in full, with repros, in
[`flex_attention/_vendor/aiter_flash_attn/NOTICE.md`](flex_attention/_vendor/aiter_flash_attn/NOTICE.md).

1. **`matrix_instr_nonkdim=16` miscompiles the causal backward.** All of AITER's tuned
   backward configs set it, forcing the 16x16x16 MFMA; the AMD Triton backend then
   miscompiles the accumulating `tl.dot(..., acc=...)` when the dot's K dimension is
   <= 16, silently dropping every loop iteration but the last from dK/dV. Only causal
   hits it, because only the causal path halves its tile for diagonal blocks. Fixing this
   is what makes causal + window, causal + score_mod, and causal + asymmetric head dims
   work at all.

2. **The backward launch grid does not cover the dQ phase.** dK/dV strides by `BLOCK_N1`
   and dQ by `BLOCK_M2` off the same program id, but the grid was sized by `BLOCK_N1`
   alone — so any config with `BLOCK_M2 < BLOCK_N1` silently computed only the leading
   rows of dQ. Every config AITER ships happens to satisfy `BLOCK_M2 >= BLOCK_N1`, so it
   never fires upstream, but it is a trap for added configs.

A methodology note, since it cost real time: a config sweep that used a **constant**
upstream gradient (`do = ones`) reported 1.33–1.39x backward "speedups" that were pure
artifact. With constant `do`, `dp - delta` nearly cancels, dQ becomes tiny, and an
absolute error threshold happily accepts configs that skip most of the dQ work. Always
sweep a backward with a random upstream gradient and a relative error check.

## Attribution

Vendored under `flex_attention/_vendor/aiter_flash_attn/` (MIT — see `LICENSE-aiter`
and `NOTICE.md` there, which records the upstream commit and every deliberate change):

- [ROCm/aiter](https://github.com/ROCm/aiter) — forward and backward Triton kernels.

`bench/_aiter_ref/` holds an unmodified copy of the same kernels, used only as a
benchmark baseline.

API shape follows [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)'s
`flash_attn/cute`. An earlier revision used [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)'s
split backward kernels; they were replaced after benchmarking and are no longer present.

## Layout

```
flex_attention/
  interface.py        public API (flash_attn_func / flash_attn_varlen_func)
  mods.py             ready-made score_mod / mask_mod builders
  block_sparse.py     block-mask metadata (BlockPlan/BlockCategory) and its backward transpose
  tuning.py           tune_block_plan: measure block-size/dense-vs-sparse candidates, pick the fastest
  split_combine.py    split-KV reduction
  _vendor/            vendored AITER kernels + NOTICE
  _experimental/      standalone prototypes kept as executable documentation
tests/                133 tests, all against a PyTorch-eager reference
bench/                MI300X/MI355X benchmarks
```

There is no NVIDIA box in the loop, and FA-4 itself cannot run on AMD, so a plain
PyTorch-eager implementation (`tests/ref_attention.py`) is ground truth throughout.
