This directory vendors code from **ROCm/aiter**
(https://github.com/ROCm/aiter), path
`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/`, at commit
`fedccf0af4219a326b82ea1157ac24aaade12f19`.

Files: `fwd_prefill.py`, `bwd.py`, `common.py`, `utils.py`.

Only the `aiter.ops.triton._triton_kernels.flash_attn_triton_amd.*` imports
were rewritten to relative imports (`.common`, `.utils`); no other logic was
changed at vendoring time.

Both forward (`fwd_prefill.py`) and backward (`bwd.py`) are AITER's own
kernels, used together: `deterministic=False` (default) uses `bwd.py`'s
`"fused"` mode (single pass per Q/K tile pair, `tl.atomic_add` into dQ);
`deterministic=True` uses its `"split"` mode (no atomics, ~1.3-1.8x slower).

An earlier version of this port used a vendored, hand-edited copy of
Primus-Turbo's split dQ/dK,dV backward kernels instead of `bwd.py` (for
determinism-by-construction). Benchmarking on MI300 (`bench/bench_mi300.py`)
showed that design cost ~1.3-1.8x in fwd+bwd wall time vs. AITER's own
default backward, because the split design recomputes QK^T for every
(Q-block, K-block) tile pair twice (once from dQ's perspective, once from
dK/dV's) instead of once. Switched to AITER's own bwd.py to match its
performance directly, since it's the same well-tuned source as the forward.

**Known upstream bug at this pinned commit**: `bwd.py`'s causal backward is broken in
both non-"split" modes -- confirmed with a minimal repro that calls
`attention_backward_triton_impl` directly, bypassing this project's wrapper entirely:
  - `mode="fused"` (this repo's non-causal default) produces a wrong `dV` under
    `causal=True` (O(1) absolute error vs. a plain-PyTorch reference; `dQ`/`dK` are fine).
  - `mode="fused_atomic"` under `causal=True` crashes inside `bwd.py` itself
    (`_bwd_kernel_fused_atomic_causal` call is missing 4 positional args it should be
    passing -- a bug in `bwd.py`, not a caller-side issue).
  - `mode="split"` is correct under causal (matches the reference to fp16/bf16 precision).

So `flex_attention/interface.py` always uses `mode="split"` when `causal=True`
(regardless of the `deterministic` flag), and only uses the faster `mode="fused"` for
`causal=False`, where it's been verified correct. This should be re-checked against a
newer aiter commit if/when this project updates its pinned version.

License: MIT (see LICENSE-aiter in this directory).
