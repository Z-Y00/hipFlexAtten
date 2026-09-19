"""Block-sparse attention metadata (Phase 5), object-oriented.

A block mask partitions the (query, key) grid into blocks and classifies each one into a
:class:`BlockCategory`:

  * **FULL**           -- every position inside is unmasked; the kernel skips masking.
  * **CAUSAL**         -- masked positions form a translation-invariant diagonal band
                           (what a causal or sliding-window mask looks like at block
                           granularity): a closed-form boundary check suffices, no
                           general ``mask_mod`` evaluation needed.
  * **PARTIAL_DENSE**  -- some positions masked by an arbitrary predicate, but most of
                           the block is kept, so it is cheapest to evaluate the predicate
                           densely over the whole tile.
  * **PARTIAL_SPARSE** -- like PARTIAL_DENSE but mostly masked out; the general
                           ``mask_mod`` path is the only one that applies.
  * **EMPTY**          -- no kept positions; the block is skipped and never materialized.

Today's kernel only distinguishes two cases -- "skip masking" (FULL) and "apply
``mask_mod``" (everything else) -- so CAUSAL/PARTIAL_DENSE/PARTIAL_SPARSE currently
collapse into one combined index list at the kernel boundary (:attr:`BlockPlan.masked`,
exposed as ``mask_block_cnt``/``mask_block_idx`` for the kernel's benefit). The finer
split is still computed and exposed on :class:`BlockPlan` for introspection, tests, and
so a future kernel revision can give CAUSAL/PARTIAL_DENSE blocks cheaper treatment than
the fully general PARTIAL_SPARSE case.

The kernel consumes each category's list as a per-``(batch, head, q_block)`` list of KV
block indices, which is the same "ordered sparse" representation upstream FA-4 uses
(``mask_block_cnt`` / ``mask_block_idx``, counts plus left-packed indices).

**What this port does NOT need.** Upstream additionally carries ``dq_write_order``
metadata plus a semaphore protocol, because its backward accumulates dQ across
cooperating programs and needs a deterministic ordering. AITER's fused backward computes
dQ in a separate per-Q-block phase that stores its tile directly, so there is nothing to
order: only the index lists are required.

The backward's dK/dV phase iterates the *transpose* (per KV block, which Q blocks attend
it). :meth:`BlockPlan.transposed` derives that from the forward lists, so callers supply
only the forward direction.
"""

import enum
import weakref
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch

__all__ = [
    "BlockCategory",
    "BlockList",
    "BlockPlan",
    "BlockSparseTensors",
    "backward_block_sparse",
    "combine_block_sparse",
    "create_block_sparse_from_mask_mod",
    "create_block_sparse_varlen",
    "dense_to_block_sparse",
    "transpose_block_sparse",
]

# A partial block whose kept fraction is at or above this is classified PARTIAL_DENSE
# rather than PARTIAL_SPARSE (see BlockCategory docstring). Purely a classification
# threshold today (both categories are handled identically by the kernel); picked so a
# block is "dense" exactly when a majority of its positions survive the predicate.
_DENSE_KEEP_THRESHOLD = 0.5


class BlockCategory(enum.Enum):
    """How a (query-block, kv-block) tile should be handled by the kernel.

    See the module docstring for the full rationale behind each category.
    """

    FULL = "full"
    CAUSAL = "causal"
    PARTIAL_DENSE = "partial_dense"
    PARTIAL_SPARSE = "partial_sparse"
    EMPTY = "empty"


# Categories that need some form of masking (as opposed to FULL, which needs none, and
# EMPTY, which is never materialized). Their index lists are unioned into BlockPlan.masked
# for the kernel, since it does not yet distinguish between them.
_MASKED_CATEGORIES = (
    BlockCategory.CAUSAL,
    BlockCategory.PARTIAL_DENSE,
    BlockCategory.PARTIAL_SPARSE,
)


@dataclass(frozen=True)
class BlockList:
    """One category's ordered-sparse block indices: count + left-packed indices.

    ``count``: int32 ``[B, H, num_q_blocks]`` -- how many KV blocks this row has.
    ``index``: int32 ``[B, H, num_q_blocks, max_count]`` -- their indices, packed left;
    entries at or past ``count`` are ignored (typically zero, but not meaningful).
    ``num_cols``: the width of the grid this was built from, i.e. one past the largest
    index any entry may hold. Kept so callers can bound-check the list against a
    sequence length without a device sync (see :meth:`BlockPlan.blocks_fit_within`).
    """

    count: torch.Tensor
    index: torch.Tensor
    num_cols: int

    @classmethod
    def from_flags(cls, flags: torch.Tensor) -> "BlockList":
        """Build from a ``[B, H, num_q_blocks, num_kv_blocks]`` bool grid."""
        counts = flags.sum(dim=-1).to(torch.int32)
        max_count = int(counts.max().item()) if counts.numel() else 0
        max_count = max(max_count, 1)  # keep a non-degenerate trailing dim
        num_cols = flags.shape[-1]
        # Sort so that selected columns come first, each group in ascending column order.
        order = torch.argsort(
            (~flags).to(torch.int32) * num_cols
            + torch.arange(num_cols, device=flags.device),
            dim=-1,
            stable=True,
        )
        return cls(counts, order[..., :max_count].to(torch.int32).contiguous(), num_cols)

    def to_dense(self, num_cols: int) -> torch.Tensor:
        """Inverse of :meth:`from_flags`: expand back to a ``[..., num_cols]`` bool grid."""
        b, h, rows, max_entries = self.index.shape
        device = self.index.device
        valid = torch.arange(max_entries, device=device).view(1, 1, 1, -1) < self.count.unsqueeze(-1)
        dense = torch.zeros(b, h, rows, num_cols + 1, dtype=torch.bool, device=device)
        safe = torch.where(valid, self.index.long(), torch.full_like(self.index.long(), num_cols))
        dense.scatter_(-1, safe, True)
        return dense[..., :num_cols]

    def transpose(self, num_cols: int) -> "BlockList":
        """Invert row/column roles: per-row lists -> per-column lists."""
        dense_t = self.to_dense(num_cols).transpose(-2, -1).contiguous()
        return BlockList.from_flags(dense_t)

    @staticmethod
    def union(lists, num_cols: int) -> "BlockList":
        """Combine several BlockLists (over the same rows) into one, deduplicating."""
        present = [bl for bl in lists if bl is not None]
        if not present:
            raise ValueError("union() requires at least one non-None BlockList")
        dense = present[0].to_dense(num_cols)
        for bl in present[1:]:
            dense = dense | bl.to_dense(num_cols)
        return BlockList.from_flags(dense)


def _is_banded(keep: torch.Tensor, q_bs: int, kv_bs: int) -> torch.Tensor:
    """Classify each block as a translation-invariant diagonal band or not.

    ``keep``: ``[N, q_bs, kv_bs]`` bool, the per-position predicate within each block.
    Returns ``[N]`` bool: whether ``keep[n]`` depends only on ``q_local - kv_local``,
    i.e. is constant along every diagonal -- exactly the shape a causal or sliding-window
    mask produces at block granularity, regardless of whether the mask was literally
    written as one.
    """
    device = keep.device
    q_local = torch.arange(q_bs, device=device).view(q_bs, 1)
    kv_local = torch.arange(kv_bs, device=device).view(1, kv_bs)
    num_diags = q_bs + kv_bs - 1
    diag_id = (q_local - kv_local + (kv_bs - 1)).reshape(-1)  # [q_bs*kv_bs], in [0, num_diags)

    n = keep.shape[0]
    flat = keep.reshape(n, q_bs * kv_bs).to(torch.float32)
    diag_id_exp = diag_id.view(1, -1).expand(n, -1)
    min_buf = torch.full((n, num_diags), 2.0, device=device)
    max_buf = torch.full((n, num_diags), -1.0, device=device)
    min_buf.scatter_reduce_(1, diag_id_exp, flat, reduce="amin", include_self=True)
    max_buf.scatter_reduce_(1, diag_id_exp, flat, reduce="amax", include_self=True)
    # every diag_id in [0, num_diags) occurs at least once, so no sentinel ever survives
    return (min_buf == max_buf).all(dim=-1)


def _classify_per_block(keep: torch.Tensor, q_bs: int, kv_bs: int) -> Dict[BlockCategory, torch.Tensor]:
    """Classify each block of a dense ``[..., q_bs, kv_bs]`` predicate into categories.

    Returns one ``[...]`` bool grid per non-empty :class:`BlockCategory`.
    """
    shape = keep.shape[:-2]
    per_block = keep.reshape(-1, q_bs * kv_bs)
    any_kept = per_block.any(dim=-1).reshape(shape)
    all_kept = per_block.all(dim=-1).reshape(shape)
    partial = any_kept & ~all_kept

    banded = _is_banded(keep.reshape(-1, q_bs, kv_bs), q_bs, kv_bs).reshape(shape) & partial
    kept_frac = per_block.float().mean(dim=-1).reshape(shape)
    dense_partial = partial & ~banded & (kept_frac >= _DENSE_KEEP_THRESHOLD)
    sparse_partial = partial & ~banded & (kept_frac < _DENSE_KEEP_THRESHOLD)

    return {
        BlockCategory.FULL: all_kept,
        BlockCategory.CAUSAL: banded,
        BlockCategory.PARTIAL_DENSE: dense_partial,
        BlockCategory.PARTIAL_SPARSE: sparse_partial,
    }


@dataclass(frozen=True)
class BlockPlan:
    """A block-sparsity plan: which KV blocks each Q block attends, categorized.

    ``full``: the FULL-category list, or ``None`` if no block is fully unmasked (the
    kernel then skips its no-masking fast path entirely rather than running it with a
    zero count).
    ``masked``: the union of CAUSAL/PARTIAL_DENSE/PARTIAL_SPARSE -- everything that needs
    some form of masking. Always present (possibly all-zero-count) since the kernel
    always needs a valid tensor for this list.
    ``causal``/``partial_dense``/``partial_sparse``: the finer breakdown of ``masked``,
    each optional. Informational today; see the module docstring.
    ``block_size``: ``(q_block_size, kv_block_size)``.

    ``B`` and ``H`` (the leading dims of every list) may be 1 to broadcast across batch
    or heads.
    """

    block_size: Tuple[int, int]
    masked: BlockList
    full: Optional[BlockList] = None
    causal: Optional[BlockList] = None
    partial_dense: Optional[BlockList] = None
    partial_sparse: Optional[BlockList] = None

    # -- kernel-facing compatibility surface (today's contract; see module docstring) --

    @property
    def full_block_cnt(self) -> Optional[torch.Tensor]:
        return None if self.full is None else self.full.count

    @property
    def full_block_idx(self) -> Optional[torch.Tensor]:
        return None if self.full is None else self.full.index

    @property
    def mask_block_cnt(self) -> torch.Tensor:
        return self.masked.count

    @property
    def mask_block_idx(self) -> torch.Tensor:
        return self.masked.index

    # -- introspection --

    @property
    def num_kv_blocks(self) -> int:
        """Width of the block grid: one past the largest KV block index in any list."""
        return self.masked.num_cols

    def blocks_fit_within(self, seqlen_k: int) -> bool:
        """True when every KV block this plan can name lies wholly inside ``seqlen_k``.

        When this holds, the kernel knows a priori that no block it visits straddles the
        end of the sequence, so it can drop the per-block bounds masking it would
        otherwise need (loads and the -inf fill) -- a sparse block list, unlike a
        contiguous range, has no other way to identify a "last, partially-filled block".
        Checked against the grid width rather than the actual indices so it costs no
        device sync.
        """
        return self.num_kv_blocks * self.block_size[1] <= seqlen_k

    def category(self, category: BlockCategory) -> Optional[BlockList]:
        """The BlockList for one category, or None if that category has no members
        (FULL) or was not tracked at this granularity (CAUSAL/PARTIAL_*)."""
        return {
            BlockCategory.FULL: self.full,
            BlockCategory.CAUSAL: self.causal,
            BlockCategory.PARTIAL_DENSE: self.partial_dense,
            BlockCategory.PARTIAL_SPARSE: self.partial_sparse,
        }[category]

    # -- construction --

    @classmethod
    def from_category_flags(
        cls,
        flags: Dict[BlockCategory, torch.Tensor],
        block_size: Tuple[int, int],
    ) -> "BlockPlan":
        """Build from disjoint ``[B, H, num_q_blocks, num_kv_blocks]`` bool grids, one per
        non-EMPTY category. Categories absent from ``flags`` are treated as all-False."""
        present = [c for c in flags if c is not BlockCategory.EMPTY]
        if len(present) > 1:
            stacked = torch.stack([flags[c] for c in present], dim=0)
            if (stacked.sum(dim=0) > 1).any():
                raise ValueError("a block cannot belong to more than one category")

        full = BlockList.from_flags(flags[BlockCategory.FULL]) if BlockCategory.FULL in flags else None
        # Sub-category BlockLists are informational (unlike `full`/`masked`, which the
        # kernel reads directly), so an empty category is left as None rather than a
        # real-but-all-zero BlockList -- `category(...)` should mean "this plan has no
        # blocks of this kind", not "here is an empty list of them".
        sub = {
            c: BlockList.from_flags(flags[c])
            for c in _MASKED_CATEGORIES
            if c in flags and flags[c].any()
        }
        masked_flags_present = [flags[c] for c in _MASKED_CATEGORIES if c in flags]
        if masked_flags_present:
            masked_dense = masked_flags_present[0]
            for f in masked_flags_present[1:]:
                masked_dense = masked_dense | f
        else:
            any_flags = next(iter(flags.values()))
            masked_dense = torch.zeros_like(any_flags)
        masked = BlockList.from_flags(masked_dense)

        return cls(
            block_size=tuple(block_size),
            masked=masked,
            full=full,
            causal=sub.get(BlockCategory.CAUSAL),
            partial_dense=sub.get(BlockCategory.PARTIAL_DENSE),
            partial_sparse=sub.get(BlockCategory.PARTIAL_SPARSE),
        )

    @classmethod
    def from_full_partial(
        cls,
        full_flags: torch.Tensor,
        partial_flags: torch.Tensor,
        block_size: Tuple[int, int],
    ) -> "BlockPlan":
        """Two-bucket convenience constructor: ``partial_flags`` all become
        PARTIAL_SPARSE (no banded/dense classification, since the caller has already
        collapsed everything that isn't fully kept into one bucket)."""
        if full_flags.shape != partial_flags.shape:
            raise ValueError("full_flags and partial_flags must have the same shape")
        if (full_flags & partial_flags).any():
            raise ValueError("a block cannot be both full and partial")
        return cls.from_category_flags(
            {BlockCategory.FULL: full_flags, BlockCategory.PARTIAL_SPARSE: partial_flags},
            block_size,
        )

    @classmethod
    def from_mask_mod(
        cls,
        mask_fn: Callable,
        batch: int,
        nheads: int,
        seqlen_q: int,
        seqlen_k: int,
        block_size: Tuple[int, int] = (128, 128),
        device="cuda",
    ) -> "BlockPlan":
        """Classify a block mask by evaluating ``mask_fn`` over the full grid.

        ``mask_fn(b, h, q_idx, kv_idx) -> bool`` is a plain *PyTorch* callable operating on
        broadcast index tensors (not the ``@triton.jit`` mask_mod you pass to the kernel,
        though they should express the same predicate).

        This materializes the full ``[B, H, seqlen_q, seqlen_k]`` predicate, so it is meant
        for tests and modest sequence lengths; for long sequences, evaluate the predicate at
        block granularity and use :meth:`from_category_flags`/:meth:`from_full_partial`
        instead.
        """
        q_bs, kv_bs = block_size
        if seqlen_q % q_bs or seqlen_k % kv_bs:
            raise ValueError(
                f"seqlen_q/seqlen_k ({seqlen_q}, {seqlen_k}) must be divisible by "
                f"block_size {block_size}"
            )
        b_idx = torch.arange(batch, device=device).view(batch, 1, 1, 1)
        h_idx = torch.arange(nheads, device=device).view(1, nheads, 1, 1)
        q_idx = torch.arange(seqlen_q, device=device).view(1, 1, seqlen_q, 1)
        kv_idx = torch.arange(seqlen_k, device=device).view(1, 1, 1, seqlen_k)
        keep = mask_fn(b_idx, h_idx, q_idx, kv_idx).expand(batch, nheads, seqlen_q, seqlen_k)

        blocked = keep.reshape(batch, nheads, seqlen_q // q_bs, q_bs, seqlen_k // kv_bs, kv_bs)
        per_block = blocked.permute(0, 1, 2, 4, 3, 5)  # [B, H, nq, nkv, q_bs, kv_bs]
        return cls.from_category_flags(_classify_per_block(per_block, q_bs, kv_bs), block_size)

    @classmethod
    def from_mask_mod_varlen(
        cls,
        mask_fn: Callable,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        nheads: int,
        block_size: Tuple[int, int] = (128, 128),
    ) -> "BlockPlan":
        """Block mask for the varlen (thd) layout.

        Block indices are **sequence-local**: the kernel already offsets Q/K/V by
        ``cu_seqlens``, so a list entry ``j`` means "the j-th KV block of *this* sequence",
        exactly as ``mask_mod``'s ``kv_idx`` is sequence-local. The q-block axis is padded to
        the longest sequence; short sequences simply carry zero counts in their tail blocks
        (the kernel also returns early for q blocks past their sequence).

        ``mask_fn(b, h, q_idx, kv_idx) -> bool`` is a plain PyTorch callable, as in
        :meth:`from_mask_mod`.
        """
        q_bs, kv_bs = block_size
        device = cu_seqlens_q.device
        batch = cu_seqlens_q.numel() - 1
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
        max_q_blocks = max((s + q_bs - 1) // q_bs for s in seqlens_q)
        max_kv_blocks = max((s + kv_bs - 1) // kv_bs for s in seqlens_k)

        per_block_all = torch.zeros(
            batch, nheads, max_q_blocks, max_kv_blocks, q_bs, kv_bs, dtype=torch.bool, device=device
        )
        for b in range(batch):
            sq, sk = seqlens_q[b], seqlens_k[b]
            nq, nkv = (sq + q_bs - 1) // q_bs, (sk + kv_bs - 1) // kv_bs
            q_idx = torch.arange(nq * q_bs, device=device).view(1, nq * q_bs, 1)
            kv_idx = torch.arange(nkv * kv_bs, device=device).view(1, 1, nkv * kv_bs)
            h_idx = torch.arange(nheads, device=device).view(nheads, 1, 1)
            keep = mask_fn(b, h_idx, q_idx, kv_idx).expand(nheads, nq * q_bs, nkv * kv_bs)
            # positions past this sequence's length are never attended
            keep = keep & (q_idx < sq) & (kv_idx < sk)
            per_block = (
                keep.reshape(nheads, nq, q_bs, nkv, kv_bs).permute(0, 1, 3, 2, 4)
            )  # [H, nq, nkv, q_bs, kv_bs]
            per_block_all[b, :, :nq, :nkv] = per_block

        return cls.from_category_flags(_classify_per_block(per_block_all, q_bs, kv_bs), block_size)

    # -- derived plans (used by the backward, which does not need the fine categories) --

    def combined(self, num_kv_blocks: int) -> "BlockPlan":
        """Merge every category into a single masked list, dropping the FULL fast path.

        The backward visits far fewer blocks than the forward's split is worth, and it
        applies ``mask_mod`` to every block it visits. Running mask_mod over a fully-kept
        block is a no-op (it returns all-True), so folding FULL into the masked list is
        correct and keeps the backward to one list per direction. Skipping FULL blocks
        instead would silently drop their contribution to dQ/dK/dV.
        """
        merged = BlockList.union([self.masked, self.full], num_kv_blocks)
        return BlockPlan(block_size=self.block_size, masked=merged)

    def transposed(self, num_kv_blocks: int) -> "BlockPlan":
        """Invert the mapping: per-Q-block KV lists -> per-KV-block Q lists.

        The backward's dK/dV phase fixes a KV block and sweeps the Q blocks that attend
        it, which is exactly this transpose. Derived here rather than demanded from the
        caller.
        """
        return BlockPlan(
            block_size=self.block_size,
            masked=self.masked.transpose(num_kv_blocks),
            full=None if self.full is None else self.full.transpose(num_kv_blocks),
        )


# Deriving the backward lists costs a handful of small kernels plus a device sync (the
# .item() that sizes the packed index tensor). That is negligible once, but it ran on
# every backward call and dominated short-sequence step time. Masks are typically built
# once and reused across steps, so memoize per mask tensor.
#
# Keyed by object identity, not by value: a WeakKeyDictionary would invoke Tensor.__eq__,
# which is elementwise and raises. A weakref.finalize drops the entry when the caller
# releases the mask, so the cache cannot pin memory or alias a recycled id().
_BWD_CACHE: dict = {}


def _cache_slot(mask_idx: torch.Tensor) -> dict:
    slot = _BWD_CACHE.get(id(mask_idx))
    if slot is None:
        slot = {}
        _BWD_CACHE[id(mask_idx)] = slot
        weakref.finalize(mask_idx, _BWD_CACHE.pop, id(mask_idx), None)
    return slot


def backward_block_sparse(plan: BlockPlan, num_kv_blocks: int) -> Tuple[BlockPlan, BlockPlan]:
    """Return ``(dq_plan, dkdv_plan)`` for the backward, memoized per mask.

    ``dq_plan`` is the forward direction (per Q block -> KV blocks) with every category
    merged; ``dkdv_plan`` is its transpose (per KV block -> Q blocks).
    """
    slot = _cache_slot(plan.mask_block_idx)
    key = (id(plan.full_block_idx) if plan.full_block_idx is not None else None, num_kv_blocks)
    hit = slot.get(key)
    if hit is not None:
        return hit
    dq_plan = plan.combined(num_kv_blocks)
    dkdv_plan = dq_plan.transposed(num_kv_blocks)
    slot[key] = (dq_plan, dkdv_plan)
    return dq_plan, dkdv_plan


# -- Backward-compatible free-function API (thin wrappers over BlockPlan) --

# BlockSparseTensors was originally a NamedTuple with this exact shape; BlockPlan is its
# object-oriented successor with the same kernel-facing attributes
# (mask_block_cnt/idx, full_block_cnt/idx, block_size), so existing callers/kernels that
# only touch those attributes need no changes.
BlockSparseTensors = BlockPlan


def dense_to_block_sparse(
    full_flags: torch.Tensor,
    partial_flags: torch.Tensor,
    block_size: Tuple[int, int],
) -> BlockPlan:
    """Build a BlockPlan from two ``[B, H, num_q_blocks, num_kv_blocks]`` bool maps."""
    return BlockPlan.from_full_partial(full_flags, partial_flags, block_size)


def create_block_sparse_from_mask_mod(
    mask_fn: Callable,
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: Tuple[int, int] = (128, 128),
    device="cuda",
) -> BlockPlan:
    return BlockPlan.from_mask_mod(mask_fn, batch, nheads, seqlen_q, seqlen_k, block_size, device)


def create_block_sparse_varlen(
    mask_fn: Callable,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    nheads: int,
    block_size: Tuple[int, int] = (128, 128),
) -> BlockPlan:
    return BlockPlan.from_mask_mod_varlen(mask_fn, cu_seqlens_q, cu_seqlens_k, nheads, block_size)


def combine_block_sparse(plan: BlockPlan, num_kv_blocks: int) -> BlockPlan:
    return plan.combined(num_kv_blocks)


def transpose_block_sparse(plan: BlockPlan, num_kv_blocks: int) -> BlockPlan:
    return plan.transposed(num_kv_blocks)
