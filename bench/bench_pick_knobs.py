"""A/B: fixed default vs select-once-then-lean-launch, for the sparse path."""
import torch
import flex_attention._vendor.aiter_flash_attn.bwd as bwd_mod
import flex_attention._vendor.aiter_flash_attn.fwd_prefill as fwd_mod
import flex_attention._vendor.aiter_flash_attn.utils as u
from flex_attention import dense_to_block_sparse, flash_attn_func

DT = torch.bfloat16


def med(fn, w=45, r=25):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(r):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def make(seqlen, block, hd, heads, batch, grad, keep_every=4):
    dev = "cuda"
    torch.manual_seed(0)
    mk = lambda: (torch.randn(batch, seqlen, heads, hd, dtype=DT, device=dev) * 0.3).requires_grad_(grad)
    q, k, v = mk(), mk(), mk()
    n = seqlen // block
    qb = torch.arange(n, device=dev).view(1, 1, n, 1)
    kb = torch.arange(n, device=dev).view(1, 1, 1, n)
    keep = (((qb * 7 + kb) % keep_every) == 0).expand(batch, heads, n, n).contiguous()
    bs = dense_to_block_sparse(keep, torch.zeros_like(keep), (block, block))
    g = torch.ones(batch, seqlen, heads, hd, dtype=DT, device=dev)
    return q, k, v, bs, g


SHAPES = [
    ("seq2048 blk64  hd64",  (2048, 64, 64, 8, 2)),
    ("seq4096 blk64  hd64",  (4096, 64, 64, 8, 2)),
    ("seq8192 blk64  hd64",  (8192, 64, 64, 8, 2)),
    ("seq4096 blk128 hd64",  (4096, 128, 64, 8, 2)),
    ("seq8192 blk128 hd64",  (8192, 128, 64, 8, 2)),
    ("seq4096 blk64  hd128", (4096, 64, 128, 4, 2)),
]

print("arch:", u.get_arch().name)
for which in ("forward", "backward"):
    print("\n=== %s: fixed default vs select-once ===" % which)
    print("%26s %13s %13s %9s  %s" % ("shape", "fixed", "selected", "speedup", "chosen"))
    for label, (seqlen, block, hd, heads, batch) in SHAPES:
        q, k, v, bs, g = make(seqlen, block, hd, heads, batch, which == "backward")

        def step():
            if which == "backward":
                for t in (q, k, v):
                    t.grad = None
                flash_attn_func(q, k, v, block_sparse_tensors=bs).backward(g)
            else:
                with torch.no_grad():
                    flash_attn_func(q, k, v, block_sparse_tensors=bs)

        cache = fwd_mod._SPARSE_FWD_CHOICE if which == "forward" else bwd_mod._SPARSE_BWD_CHOICE
        saved = u.AUTOTUNE
        # fixed: force the AUTOTUNE=off path
        fwd_mod.AUTOTUNE = "off"; bwd_mod.AUTOTUNE = "off"; cache.clear()
        fixed = med(step)
        fwd_mod.AUTOTUNE = "on"; bwd_mod.AUTOTUNE = "on"; cache.clear()
        sel = med(step)
        chosen = list(cache.values())[0] if cache else {}
        cache.clear()
        fwd_mod.AUTOTUNE = saved; bwd_mod.AUTOTUNE = saved
        tag = "w%s s%s" % (chosen.get("waves_per_eu"), chosen.get("num_stages"))
        print("%26s %11.4fms %11.4fms %8.3fx  %s" % (label, fixed, sel, fixed / sel, tag))
