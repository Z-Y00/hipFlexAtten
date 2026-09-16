"""Ready-made score_mod / mask_mod callables for flash_attn_func / flash_attn_varlen_func.

All of these are ``@triton.jit`` functions inlined into the attention kernel at compile
time. Write your own the same way:

    @triton.jit
    def my_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx        # True = keep

    @triton.jit
    def my_score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_expr

    @triton.jit
    def my_score_mod_bwd(dscore, score, b, h, q_idx, kv_idx):
        return dscore * d_my_score_mod_d_score   # the VJP, written by hand

``q_idx`` / ``kv_idx`` are index tiles that broadcast against ``score``; ``b`` and ``h``
are scalars. Under varlen, indices are sequence-local, so these mods work unchanged for
both dense and varlen.

``score_mod`` is applied PRE-MASK (before causal/window masking), matching upstream
FA-4. That ordering is required for non-additive mods: ``tanh(-inf) = -1``, so a softcap
applied after masking would turn masked positions back on. ``mask_mod`` is applied after.

**Why the parameterized builders below generate source text.** Triton keys its compiled
kernels partly on a JIT function's *source*, so two closures produced from the same
``def`` with different captured constants hash identically, and the second silently
reuses the first's compiled code. Building each variant from source with the constant
baked in as a literal keeps them distinct. (Observed on MI300X: two different softcap
values in one process produced identical -- and therefore wrong -- results for the
second.) ``lru_cache`` then keeps one object per distinct value so repeated calls don't
trigger redundant compiles.
"""

import functools
import importlib.util
import os
import sys
import tempfile

import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = [
    "causal_mask_mod",
    "identity_score_mod_bwd",
    "make_sliding_window_mask_mod",
    "make_softcap_score_mod",
]


def _tag(value) -> str:
    """Identifier-safe tag for a number, used to keep generated function names unique."""
    return repr(value).replace("-", "neg").replace(".", "p").replace("+", "")


_GENERATED_DIR = None


def _generated_dir() -> str:
    """Directory holding generated mod modules, created lazily.

    They must be real files on disk: triton reads a JIT function's source with
    inspect.getsource, which rejects anything defined via exec of a string
    ("@jit functions should be defined in a Python file").
    """
    global _GENERATED_DIR
    if _GENERATED_DIR is None:
        _GENERATED_DIR = tempfile.mkdtemp(prefix="flex_attention_mods_")
    return _GENERATED_DIR


def _build_jit(src: str, names, tag):
    """Write a generated @triton.jit source to a real module and import it."""
    module_name = f"_flex_attention_mod_{tag}"
    path = os.path.join(_generated_dir(), f"{module_name}.py")
    header = (
        "import triton\n"
        "import triton.language as tl\n"
        "from triton.language.extra import libdevice\n"
    )
    with open(path, "w") as fh:
        fh.write(header + src)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: triton resolves a JIT function's module by name.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if isinstance(names, str):
        return getattr(module, names)
    return tuple(getattr(module, n) for n in names)


@triton.jit
def causal_mask_mod(b, h, q_idx, kv_idx):
    """Top-left-aligned causal mask (query i attends keys 0..i).

    NOTE: this is top-left aligned. ``causal=True`` in the kernels is *bottom-right*
    aligned (query i attends keys 0..i + (seqlen_k - seqlen_q)), which differs whenever
    seqlen_q != seqlen_k. They agree for the common square case.
    """
    return kv_idx <= q_idx


@triton.jit
def identity_score_mod_bwd(dscore, score, b, h, q_idx, kv_idx):
    """VJP for any purely *additive* score_mod (``score + f(b, h, q_idx, kv_idx)``).

    d/d(score) of (score + const) is 1, so the incoming gradient passes straight through.
    """
    return dscore


@functools.lru_cache(maxsize=None)
def make_sliding_window_mask_mod(window_left: int, window_right: int):
    """Build a mask_mod for a sliding window, as an alternative to ``window_size=``.

    Negative bounds mean "unbounded on that side", matching the ``window_size`` convention.

    NOTE: like ``causal_mask_mod``, this is top-left aligned, whereas the ``window_size=``
    kernel path is bottom-right aligned. They agree when seqlen_q == seqlen_k.
    """
    window_left, window_right = int(window_left), int(window_right)
    tag = f"{_tag(window_left)}_{_tag(window_right)}"
    name = f"sliding_window_mask_mod_{tag}"
    if window_left >= 0 and window_right >= 0:
        body = f"(rel >= {-window_left}) & (rel <= {window_right})"
    elif window_left >= 0:
        body = f"rel >= {-window_left}"
    elif window_right >= 0:
        body = f"rel <= {window_right}"
    else:
        body = "rel == rel"  # unbounded both sides: all-True tile shaped like rel
    src = f"""
@triton.jit
def {name}(b, h, q_idx, kv_idx):
    rel = kv_idx - q_idx
    return {body}
"""
    return _build_jit(src, name, tag)


@functools.lru_cache(maxsize=None)
def make_softcap_score_mod(softcap: float):
    """Build the (score_mod, score_mod_bwd) pair implementing logit softcapping.

    ``s = softcap * tanh(qk * softmax_scale / softcap)``, with VJP
    ``ds_in = ds_out * (1 - tanh(s_in / softcap)^2)``.

    Mirrors upstream FA-4, which also implements softcap as a score_mod pair
    (``create_softcap_scoremod`` / ``create_softcap_scoremod_bwd``) rather than as a
    dedicated kernel flag. ``flash_attn_func(..., softcap=...)`` wires this up for you;
    it is exported so callers can compose it explicitly.
    """
    if softcap <= 0:
        raise ValueError(f"softcap must be positive, got {softcap}")
    cap = float(softcap)
    inv = 1.0 / cap
    tag = _tag(cap)
    fwd_name = f"softcap_score_mod_{tag}"
    bwd_name = f"softcap_score_mod_bwd_{tag}"
    src = f"""
@triton.jit
def {fwd_name}(score, b, h, q_idx, kv_idx):
    return {cap!r} * libdevice.tanh(score * {inv!r})


@triton.jit
def {bwd_name}(dscore, score, b, h, q_idx, kv_idx):
    t = libdevice.tanh(score * {inv!r})
    return dscore * (1.0 - t * t)
"""
    return _build_jit(src, (fwd_name, bwd_name), f"softcap_{tag}")
