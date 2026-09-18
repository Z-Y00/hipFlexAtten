"""
Utilities for Flash Attention Triton AMD backend.

This module contains essential runtime utilities:
- GPU architecture detection
- Global configuration flags
"""

import functools
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

AutotuneMode = Literal["off", "on", "sweep"]

__all__ = [
    "AUTOTUNE",
    "DEBUG",
    "USE_EXP2",
    # Global config
    "AutotuneMode",
    # Runtime info
    "get_arch",
]


# -------------------------------
# GPU Architecture
# -------------------------------
ArchFamily = Literal["cdna", "rdna"]

CDNA_ARCHS = frozenset({"gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"})
RDNA_ARCHS = frozenset(
    {
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }
)


@dataclass(frozen=True)
class GpuArch:
    """GPU architecture information."""

    name: str  # e.g., "gfx942", "gfx1100"
    family: ArchFamily | None = None

    @property
    def is_rdna(self) -> bool:
        return self.family == "rdna"

    @property
    def cu_count(self) -> int:
        """Get the number of compute units on the current GPU."""
        return int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )


# -------------------------------
# Global Variables
# -------------------------------
# --- flex_attention addition (Phase 4 / MLA), not upstream AITER ---------------
# CDNA3 (gfx942) gives 64 KiB of LDS per workgroup. AITER's tuned configs assume
# head_dim <= 256, where the Q tile always fits; MLA shapes (head_dim_v up to 512,
# head_dim_qk up to 512) overflow it. Measured on MI300X, Triton's reported LDS
# requirement for attn_fwd is exactly BLOCK_M * padded_head_dim_qk * elem_size
# (131072 B for BLOCK_M=128, padded 512, bf16 -- and halving BLOCK_M made it fit),
# and for the fused backward it is the same product with BLOCK_N1 / BLOCK_M2 in
# place of BLOCK_M. So model the budget on that single dominant tile.
LDS_LIMIT_BYTES = 64 * 1024


def max_block_for_lds(padded_head_dim: int, elem_size: int) -> int:
    """Largest power-of-two sequence-block that keeps the dominant tile inside LDS.

    Returns 0 when even a block of 16 cannot fit, which the callers surface as a
    clear "head_dim too large" error rather than a raw Triton OutOfResources.
    """
    raw = LDS_LIMIT_BYTES // max(padded_head_dim * elem_size, 1)
    block = 1 << (raw.bit_length() - 1) if raw >= 1 else 0  # round down to pow2
    return block if block >= 16 else 0


AUTOTUNE: AutotuneMode = (
    "on"
    if os.environ.get("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "1").lower()
    in ("1", "true", "yes", "on")
    else "off"
)

# User override config json for attn_fwd.
# Note: Ignored if FLASH_ATTENTION_TRITON_AMD_AUTOTUNE is enabled.
#
# e.g. FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":32,"BLOCK_N":32,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":4}'
FWD_CONF_OVERRIDE = None
try:
    conf_json = os.getenv("FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON")
    if conf_json:
        conf = json.loads(conf_json)
        FWD_CONF_OVERRIDE = triton.Config(
            conf,
            num_stages=conf.pop("num_stages", 1),
            num_warps=conf.pop("num_warps", 4),
        )
except Exception as e:  # noqa: BLE001
    logger.warning(f"FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON parse error: {e}")

# Unified debug level:
#   0 = off (default)
#   1 = basic debug info (shapes, tensor stats, kernel params)
#   2 = detailed debug (includes Triton interpreter prints in kernels)
#
# Set via: FLASH_ATTENTION_TRITON_AMD_DEBUG=0|1|2
DEBUG: int = int(os.environ.get("FLASH_ATTENTION_TRITON_AMD_DEBUG", "0"))
# Printing every autotune result is debug output; it must not be forced on for the whole process.
if DEBUG > 0:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
if DEBUG >= 2:
    os.environ["TRITON_INTERPRET"] = "1"
USE_EXP2 = True

# -------------------------------
# Runtime info
# -------------------------------
@functools.cache
def get_arch() -> GpuArch:
    """Get the current GPU architecture."""
    try:
        name: str = triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        # No GPU available (e.g. import-only on Windows/CPU)
        return GpuArch(name="unknown")
    if name in CDNA_ARCHS:
        return GpuArch(name=name, family="cdna")
    elif name in RDNA_ARCHS:
        return GpuArch(name=name, family="rdna")
    else:
        return GpuArch(name=name)


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid
