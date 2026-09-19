"""Pick a block-sparse implementation by measuring it, once, up front.

Two things about a block mask cannot be decided analytically:

* **Block size.** The kernel pins its tiles to the sparsity granularity, so a bigger
  block means fewer, fatter MFMA tiles but a coarser mask that drags in more masked-out
  work. Where the crossover lies depends on the pattern, the sequence length and the
  head dim all at once.
* **Whether to use block-sparsity at all.** If a mask happens to be exactly what the
  ``causal``/``window_size`` flags already express, the dense path computes the same
  thing *and* gets to use the autotuner, which the block-sparse path bypasses. That is
  usually the faster way to run it.

Both are cheap to settle empirically when the mask is stable and reused across steps:
time the candidates once, keep the winner. That is what :func:`tune_block_plan` does.

    result = tune_block_plan(causal_band_fn, q, k, v, causal=True)
    print(result)                      # per-candidate timings
    out = result.apply(flash_attn_func, q, k, v, causal=True)

Not every block size can express every mask. A block list is only as precise as its
granularity, so a predicate that varies inside a tile leaves that block *partial*, and
the kernel resolves partial blocks by running ``mask_mod``. Without one it would keep
the block whole and attend positions the predicate excludes -- wrong, not merely slow.
Such candidates are rejected rather than timed, so coarse block sizes only appear in
the results when a ``mask_mod`` is supplied (or when the only partial blocks are ones
straddling the causal diagonal, which the kernel already knows how to handle).

Building a plan materializes the full ``[B, H, seqlen_q, seqlen_k]`` predicate once per
candidate block size (see :func:`~flex_attention.block_sparse.create_block_sparse_from_mask_mod`),
so tune on representative shapes, not inside a training step.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from flex_attention.block_sparse import BlockPlan, create_block_sparse_from_mask_mod

__all__ = ["Candidate", "TuningResult", "tune_block_plan"]

DEFAULT_BLOCK_SIZES = ((32, 32), (64, 64), (128, 128))


@dataclass
class Candidate:
    """One implementation choice and what it measured."""

    name: str
    plan: Optional[BlockPlan]  # None means "run the dense path, no block lists"
    ms: float = float("inf")
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.ms < float("inf")

    @property
    def block_size(self):
        """The plan's block size, or None for the dense candidate."""
        return None if self.plan is None else self.plan.block_size


@dataclass
class TuningResult:
    best: Candidate
    candidates: List[Candidate]

    @property
    def plan(self) -> Optional[BlockPlan]:
        """The winning plan, or None if the dense path won."""
        return self.best.plan

    def apply(self, fn: Callable, *args, **kwargs):
        """Call ``fn`` with the winning choice wired in as ``block_sparse_tensors``."""
        return fn(*args, block_sparse_tensors=self.best.plan, **kwargs)

    def __str__(self) -> str:
        rows = [f"{'candidate':>22s} {'ms':>9s}  {'vs best':>8s}"]
        rows.append("-" * 43)
        for c in sorted(self.candidates, key=lambda c: c.ms):
            if not c.ok:
                rows.append(f"{c.name:>22s} {'n/a':>9s}  {c.error or 'failed'}")
            else:
                mark = "  <-- best" if c is self.best else ""
                rows.append(f"{c.name:>22s} {c.ms:9.4f}  {c.ms / self.best.ms:7.2f}x{mark}")
        return "\n".join(rows)


def _median_ms(fn: Callable, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _causal_block_grids(
    nq: int, nkv: int, q_bs: int, kv_bs: int, seqlen_q: int, seqlen_k: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The (fully-visible, straddling) block grids a pure causal mask induces."""
    offset = seqlen_k - seqlen_q
    qb = torch.arange(nq, device=device).view(nq, 1)
    kb = torch.arange(nkv, device=device).view(1, nkv)
    visible = (kb * kv_bs + kv_bs - 1) <= (qb * q_bs + offset)
    hidden = (kb * kv_bs) > (qb * q_bs + q_bs - 1 + offset)
    return visible, ~visible & ~hidden


def _is_pure_causal(plan: BlockPlan, seqlen_q: int, seqlen_k: int) -> bool:
    """True when ``plan`` visits exactly the blocks a plain causal mask would.

    Compared at block granularity, which is all that matters: if the block sets agree,
    the dense causal kernel and this plan schedule the same work, and the per-element
    causal predicate is identical on both paths. Cheap -- no full predicate is built.
    """
    nkv = plan.num_kv_blocks
    nq = plan.masked.index.shape[2]
    q_bs, kv_bs = plan.block_size
    visible, straddling = _causal_block_grids(
        nq, nkv, q_bs, kv_bs, seqlen_q, seqlen_k, plan.masked.index.device
    )
    full_d = (
        plan.full.to_dense(nkv) if plan.full is not None
        else torch.zeros_like(plan.masked.to_dense(nkv))
    )
    return bool((full_d == visible).all() and (plan.masked.to_dense(nkv) == straddling).all())


def _plan_is_exact(
    plan: BlockPlan, causal: bool, seqlen_q: int, seqlen_k: int, has_mask_mod: bool
) -> bool:
    """Can this plan reproduce the mask on its own, at this block size?

    A block list is only as precise as its granularity. Coarsen it and a block whose
    predicate varies inside the tile stops being all-or-nothing: it lands in the masked
    list, where the kernel resolves it by running ``mask_mod``. With no ``mask_mod`` to
    run, the kernel keeps the whole block and silently attends positions the predicate
    excludes -- so such a candidate is wrong, not merely slow, and must not be timed.

    Two ways to be exact: supply a ``mask_mod``, or have every masked block be one the
    kernel already knows how to resolve, i.e. a block straddling the causal diagonal.
    """
    if has_mask_mod:
        return True
    nkv = plan.num_kv_blocks
    masked = plan.masked.to_dense(nkv)
    if not causal:
        return not bool(masked.any())
    _, straddling = _causal_block_grids(
        plan.masked.index.shape[2], nkv, *plan.block_size,
        seqlen_q, seqlen_k, plan.masked.index.device,
    )
    return not bool((masked & ~straddling).any())


def tune_block_plan(
    mask_fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_sizes: Sequence[Tuple[int, int]] = DEFAULT_BLOCK_SIZES,
    causal: bool = False,
    consider_dense: bool = True,
    backward: bool = False,
    warmup: int = 15,
    reps: int = 25,
    **attn_kwargs,
) -> TuningResult:
    """Measure each viable block size (and the dense path) and return the fastest.

    ``mask_fn(b, h, q_idx, kv_idx) -> bool`` is the plain-PyTorch predicate, as taken by
    ``create_block_sparse_from_mask_mod``. Any extra keyword arguments -- ``mask_mod``,
    ``softmax_scale``, ``window_size`` and so on -- are forwarded to ``flash_attn_func``
    unchanged, so the candidates are timed exactly as they will be used.

    A block size that does not divide the sequence lengths is skipped rather than
    raising, so a single call can sweep a generic list. The dense candidate is offered
    only when the mask turns out to be exactly causal (``consider_dense=False`` to
    suppress it), since otherwise it would compute something different.
    """
    from flex_attention.interface import flash_attn_func

    if q.dim() != 4:
        raise ValueError("tune_block_plan expects the dense (batch, seqlen, nheads, head_dim) layout")
    batch, seqlen_q, nheads, _ = q.shape
    seqlen_k = k.shape[1]

    candidates: List[Candidate] = []
    for q_bs, kv_bs in block_sizes:
        name = f"block_sparse {q_bs}x{kv_bs}"
        if seqlen_q % q_bs or seqlen_k % kv_bs:
            candidates.append(Candidate(name, None, error="seqlen not divisible"))
            continue
        try:
            plan = create_block_sparse_from_mask_mod(
                mask_fn, batch, nheads, seqlen_q, seqlen_k,
                block_size=(q_bs, kv_bs), device=q.device,
            )
        except Exception as exc:  # a block size the kernel cannot honour (e.g. LDS)
            candidates.append(Candidate(name, None, error=type(exc).__name__))
            continue
        if not _plan_is_exact(
            plan, causal, seqlen_q, seqlen_k, attn_kwargs.get("mask_mod") is not None
        ):
            candidates.append(
                Candidate(name, None, error="too coarse for this mask (needs mask_mod)")
            )
            continue
        candidates.append(Candidate(name, plan))

    if consider_dense and causal:
        usable = [c for c in candidates if c.plan is not None]
        if usable and _is_pure_causal(usable[0].plan, seqlen_q, seqlen_k):
            candidates.append(Candidate("dense (autotuned)", None))

    grad = torch.ones_like(q) if backward else None

    def make_call(plan):
        def run():
            out = flash_attn_func(
                q, k, v, causal=causal, block_sparse_tensors=plan, **attn_kwargs
            )
            if backward:
                for t in (q, k, v):
                    t.grad = None
                out.backward(grad)
        return run

    for cand in candidates:
        if cand.error is not None and cand.plan is None and "block_sparse" in cand.name:
            continue  # already marked unusable above
        try:
            cand.ms = _median_ms(make_call(cand.plan), warmup, reps)
        except Exception as exc:  # noqa: BLE001 - a failed candidate is just not chosen
            cand.error = f"{type(exc).__name__}: {exc}"[:80]

    ok = [c for c in candidates if c.ok]
    if not ok:
        raise RuntimeError("every candidate failed:\n" + "\n".join(
            f"  {c.name}: {c.error}" for c in candidates))
    return TuningResult(best=min(ok, key=lambda c: c.ms), candidates=candidates)
