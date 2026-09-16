Benchmark-only, unmodified-logic copy of **ROCm/aiter**'s
`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/` at commit
`fedccf0af4219a326b82ea1157ac24aaade12f19` (same commit as
`flex_attention/_vendor/aiter_flash_attn/`). Only the `aiter.ops.triton...`
imports were rewritten to relative imports.

This is here purely so `bench/bench_mi300.py` can compare our port against
AITER's *own* forward (should be identical - same kernel) and, more
importantly, AITER's own backward (`bwd.py`'s fused/fused_atomic/split
modes), since our backward uses a different (Primus-Turbo-derived) kernel.
Not part of the shipped `flex_attention` package.

License: MIT (see ../../flex_attention/_vendor/aiter_flash_attn/LICENSE-aiter).
