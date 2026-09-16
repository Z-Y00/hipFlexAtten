This directory vendors code from **ROCm/aiter**
(https://github.com/ROCm/aiter), path
`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/`, at commit
`fedccf0af4219a326b82ea1157ac24aaade12f19`.

Files: `fwd_prefill.py`, `bwd.py`, `common.py`, `utils.py`.

At vendoring time the only change was rewriting the
`aiter.ops.triton._triton_kernels.flash_attn_triton_amd.*` imports to relative
imports (`.common`, `.utils`). Everything below is a deliberate change made since,
each marked in the source with a `flex_attention` comment.

Both forward (`fwd_prefill.py`) and backward (`bwd.py`) are AITER's own kernels. The
backward always runs in `bwd.py`'s `"fused"` mode. That mode writes each dQ/dK/dV tile
from a single program -- the atomics live in the separate `"fused_atomic"` mode, which
this project never uses -- so the backward is deterministic, verified bitwise-identical
across repeated runs. `deterministic=True` therefore needs no separate (slower) path.

An earlier version of this port used a vendored copy of Primus-Turbo's split dQ/dK,dV
backward kernels instead of `bwd.py`. Benchmarking on MI300 showed that design cost
~1.3-1.8x in fwd+bwd wall time, because a split design recomputes QK^T for every
(Q-block, K-block) tile pair twice -- once from dQ's perspective, once from dK/dV's --
instead of once.

## Upstream bug found and fixed here: `matrix_instr_nonkdim=16`

**Symptom.** `bwd.py`'s causal backward returned a badly wrong `dV` (O(1) absolute error
vs. a plain-PyTorch reference) and a slightly wrong `dK`, while `dQ` was correct.
Non-causal was unaffected. Reproduced with a minimal script calling
`attention_backward_triton_impl` directly, bypassing this project's wrapper.

**Root cause.** All of AITER's tuned backward configs set `matrix_instr_nonkdim=16`,
forcing the 16x16x16 MFMA. On gfx942 the AMD Triton backend then miscompiles the
accumulating `tl.dot(..., acc=...)` in `_bwd_dkdv_inner` when the dot's K dimension is
<= 16: every loop iteration except the last is silently dropped from the dK/dV
accumulators.

Only the causal path hits it, which is why the bug looked causal-specific. The causal
kernel sweeps its diagonal ("masked") blocks with `BLOCK_M1 // BLK_SLICE_FACTOR`
(32 // 2 = 16), while the non-causal path uses the full `BLOCK_M1` = 32.

**Evidence.** With `BLOCK_N1=64` and a masked block of 16, exactly the last 16 key
positions of each 64-key block had a correct `dV`; the first 48 were wrong. Sweeping the
parameters isolated the trigger precisely:

| masked block (`BLOCK_M1/BLK_SLICE_FACTOR`) | `matrix_instr_nonkdim` | correct dV |
|---|---|---|
| 32 | 16 | yes |
| 16 | 16 | **no** (only last 16 keys) |
|  8 | 16 | **no** (only last 8 keys) |
| 16 | 32 | yes |
| 16 | unset | yes |

So it is the hint, not the block size: dropping `matrix_instr_nonkdim` (letting Triton
pick the instruction) or raising it to 32 both give correct results.

**Fix.** `_sanitize_nonkdim()` removes `matrix_instr_nonkdim` from any *causal* config
whose masked sub-block (`BLOCK_M1` or `BLOCK_N2`, divided by `BLK_SLICE_FACTOR`) would
be below 32. Non-causal configs keep the hint -- they never hit the miscompile, and
sanitizing them too cost ~5% for no correctness gain. Configs whose sub-blocks stay
>= 32 keep the hint and their tuning.

This removed three feature restrictions that earlier revisions of this port had to
impose, since causal no longer needs the slower, less capable `"split"` path:
causal + `window_size`, causal + `score_mod`/`mask_mod`, and causal with
`head_dim_qk != head_dim_v` all work now. Measured on MI300X, causal fwd+bwd is
1.75-2.46x faster than routing causal through `"split"`, and roughly at parity
(0.94-1.19x) with the unpatched-but-incorrect `"fused"` kernel.

Upstream has not fixed this: as of `efc4b67` (about four hours newer than our pin) the
entire vendored directory is byte-identical. Worth re-checking on a future version bump,
and worth reporting upstream.

## Other upstream defects (not fixed here, avoided)

- **`mode="fused_atomic"` is broken under `causal=True`**: the
  `_bwd_kernel_fused_atomic_causal` launch is missing 4 positional arguments and raises
  a `TypeError` from inside `bwd.py`. We never use this mode.

- **The `"split"` backward assumes a single head dim.** `_bwd_kernel_split_*` and their
  inner helpers carry one `BLOCK_D_MODEL` / `BLOCK_D_MODEL_POW2` pair, whereas the fused
  kernels correctly carry separate `HEAD_DIM_QK` / `HEAD_DIM_V`. So the split path
  cannot express `head_dim_qk != head_dim_v`: it indexes V with the QK head dim and
  reads past the end of the tensor. Observed as a HIP `illegal memory access` on MI300X
  for head_dim 192/128 -- and it is *latent*, succeeding in isolation and only faulting
  under a different allocator layout, so it can silently corrupt instead of erroring.
  We never use this mode either, so this no longer constrains the public API.

## Other flex_attention changes

- **`score_mod` / `mask_mod` / `score_mod_bwd`** (Phase 3) are threaded through
  `attn_fwd` and both fused backward kernels as `tl.constexpr` callables, inlined at
  compile time. Passing `None` compiles them away entirely, so the default path is
  unchanged (benchmarks confirm).

- **LDS capping for MLA head dims** (Phase 4). MLA shapes overflow CDNA3's 64 KiB LDS
  with AITER's tuned block size of 128. Both wrappers compute a head-dim-aware cap
  (`max_block_for_lds` in `utils.py`) and, when it binds, bypass the autotuner to launch
  the raw JITFunction with a block size that fits. The autotuner is bypassed rather than
  filtered because Triton only calls `early_config_prune` when more than one config is
  present, which would silently miss AITER's `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0`
  mode. Ordinary head dims never take this path.

License: MIT (see LICENSE-aiter in this directory).
