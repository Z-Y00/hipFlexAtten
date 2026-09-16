"""Block-sparse attention metadata (Phase 5).

A block mask partitions the (query, key) grid into blocks and labels each block:
  * **full**    -- every position inside is unmasked, so the kernel can skip masking
  * **partial** -- some positions are masked, so the kernel applies ``mask_mod``
  * **empty**   -- skipped entirely

The kernel consumes that as a per-``(batch, head, q_block)`` list of KV block indices,
which is the same "ordered sparse" representation upstream FA-4 uses
(``mask_block_cnt`` / ``mask_block_idx``, counts plus left-packed indices).

**What this port does NOT need.** Upstream additionally carries ``dq_write_order``
metadata plus a semaphore protocol, because its backward accumulates dQ across
cooperating programs and needs a deterministic ordering. AITER's fused backward computes
dQ in a separate per-Q-block phase that stores its tile directly, so there is nothing to
order: only the index lists are required.

The backward's dK/dV phase iterates the *transpose* (per KV block, which Q blocks attend
it). ``transpose_block_sparse`` derives that from the forward lists, so callers supply
only the forward direction.
"""

import weakref
from typing import NamedTuple, Optional, Tuple

import torch

__all__ = [
    "BlockSparseTensors",
    "backward_block_sparse",
    "combine_block_sparse",
    "create_block_sparse_from_mask_mod",
    "dense_to_block_sparse",
    "transpose_block_sparse",
]


class BlockSparseTensors(NamedTuple):
    """Ordered sparse block lists. Field names mirror flash_attn.cute's equivalent.

    ``mask_block_cnt``: int32 ``[B, H, num_q_blocks]`` -- number of partial KV blocks.
    ``mask_block_idx``: int32 ``[B, H, num_q_blocks, max_kv]`` -- their indices, packed
    left; entries at or past the count are ignored.
    ``full_block_cnt`` / ``full_block_idx``: same, for fully-unmasked KV blocks.
    ``block_size``: ``(q_block_size, kv_block_size)``.

    ``B`` and ``H`` may be 1 to broadcast across batch or heads.
    """

    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: Optional[torch.Tensor] = None
    full_block_idx: Optional[torch.Tensor] = None
    block_size: Tuple[int, int] = (128, 128)


def _ordered_from_dense(flags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B,H,R,C] bool -> (counts [B,H,R], left-packed indices [B,H,R,max_count])."""
    counts = flags.sum(dim=-1).to(torch.int32)
    max_count = int(counts.max().item()) if counts.numel() else 0
    max_count = max(max_count, 1)  # keep a non-degenerate trailing dim
    num_cols = flags.shape[-1]
    # Sort so that selected columns come first, each group in ascending column order.
    order = torch.argsort((~flags).to(torch.int32) * num_cols + torch.arange(
        num_cols, device=flags.device
    ), dim=-1, stable=True)
    idx = order[..., :max_count].to(torch.int32)
    return counts, idx.contiguous()


def dense_to_block_sparse(
    full_flags: torch.Tensor,
    partial_flags: torch.Tensor,
    block_size: Tuple[int, int],
) -> BlockSparseTensors:
    """Build BlockSparseTensors from two ``[B, H, num_q_blocks, num_kv_blocks]`` bool maps."""
    if full_flags.shape != partial_flags.shape:
        raise ValueError("full_flags and partial_flags must have the same shape")
    if (full_flags & partial_flags).any():
        raise ValueError("a block cannot be both full and partial")
    mask_cnt, mask_idx = _ordered_from_dense(partial_flags)
    full_cnt, full_idx = _ordered_from_dense(full_flags)
    return BlockSparseTensors(mask_cnt, mask_idx, full_cnt, full_idx, tuple(block_size))


def create_block_sparse_from_mask_mod(
    mask_fn,
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: Tuple[int, int] = (128, 128),
    device="cuda",
) -> BlockSparseTensors:
    """Build a block mask by evaluating ``mask_fn`` over the full grid.

    ``mask_fn(b, h, q_idx, kv_idx) -> bool`` is a plain *PyTorch* callable operating on
    broadcast index tensors (not the ``@triton.jit`` mask_mod you pass to the kernel,
    though they should express the same predicate).

    This materializes the full ``[B, H, seqlen_q, seqlen_k]`` predicate, so it is meant
    for tests and modest sequence lengths; for long sequences, evaluate the predicate at
    block granularity and use ``dense_to_block_sparse`` instead.
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
    per_block = blocked.permute(0, 1, 2, 4, 3, 5).reshape(
        batch, nheads, seqlen_q // q_bs, seqlen_k // kv_bs, q_bs * kv_bs
    )
    any_kept = per_block.any(dim=-1)
    all_kept = per_block.all(dim=-1)
    return dense_to_block_sparse(all_kept, any_kept & ~all_kept, block_size)


def combine_block_sparse(tensors: BlockSparseTensors, num_kv_blocks: int) -> BlockSparseTensors:
    """Merge the full and partial lists into a single partial list.

    The backward visits far fewer blocks than the forward's two-pass split is worth, and
    it applies ``mask_mod`` to every block it visits. Running mask_mod over a fully-kept
    block is a no-op (it returns all-True), so folding the full blocks into the partial
    list is correct and keeps the backward to one list per direction. Skipping them
    instead would silently drop their contribution to dQ/dK/dV.
    """
    mask_dense = _ordered_to_dense(tensors.mask_block_cnt, tensors.mask_block_idx, num_kv_blocks)
    if tensors.full_block_cnt is not None:
        mask_dense = mask_dense | _ordered_to_dense(
            tensors.full_block_cnt, tensors.full_block_idx, num_kv_blocks
        )
    cnt, idx = _ordered_from_dense(mask_dense)
    return BlockSparseTensors(cnt, idx, None, None, tensors.block_size)


def transpose_block_sparse(
    tensors: BlockSparseTensors, num_kv_blocks: int
) -> BlockSparseTensors:
    """Invert the mapping: per-Q-block KV lists -> per-KV-block Q lists.

    The backward's dK/dV phase fixes a KV block and sweeps the Q blocks that attend it,
    which is exactly this transpose. Derived here rather than demanded from the caller.
    """
    mask_dense = _ordered_to_dense(tensors.mask_block_cnt, tensors.mask_block_idx, num_kv_blocks)
    if tensors.full_block_cnt is not None:
        full_dense = _ordered_to_dense(
            tensors.full_block_cnt, tensors.full_block_idx, num_kv_blocks
        )
    else:
        full_dense = torch.zeros_like(mask_dense)
    # transpose the (q_block, kv_block) grid
    mask_t = mask_dense.transpose(-2, -1).contiguous()
    full_t = full_dense.transpose(-2, -1).contiguous()
    mask_cnt, mask_idx = _ordered_from_dense(mask_t)
    full_cnt, full_idx = _ordered_from_dense(full_t)
    return BlockSparseTensors(mask_cnt, mask_idx, full_cnt, full_idx, tensors.block_size)


def _ordered_to_dense(
    counts: torch.Tensor, indices: torch.Tensor, num_cols: int
) -> torch.Tensor:
    """(counts, left-packed indices) -> [B,H,R,num_cols] bool."""
    b, h, rows, max_entries = indices.shape
    device = indices.device
    valid = torch.arange(max_entries, device=device).view(1, 1, 1, -1) < counts.unsqueeze(-1)
    dense = torch.zeros(b, h, rows, num_cols + 1, dtype=torch.bool, device=device)
    safe = torch.where(valid, indices.long(), torch.full_like(indices.long(), num_cols))
    dense.scatter_(-1, safe, True)
    return dense[..., :num_cols]


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


def backward_block_sparse(
    tensors: BlockSparseTensors, num_kv_blocks: int
) -> Tuple[BlockSparseTensors, BlockSparseTensors]:
    """Return ``(dq_lists, dkdv_lists)`` for the backward, memoized per mask.

    ``dq_lists`` is the forward direction (per Q block -> KV blocks) with full and
    partial merged; ``dkdv_lists`` is its transpose (per KV block -> Q blocks).
    """
    slot = _cache_slot(tensors.mask_block_idx)
    key = (
        id(tensors.full_block_idx) if tensors.full_block_idx is not None else None,
        num_kv_blocks,
    )
    hit = slot.get(key)
    if hit is not None:
        return hit
    dq_lists = combine_block_sparse(tensors, num_kv_blocks)
    dkdv_lists = transpose_block_sparse(dq_lists, num_kv_blocks)
    slot[key] = (dq_lists, dkdv_lists)
    return dq_lists, dkdv_lists
