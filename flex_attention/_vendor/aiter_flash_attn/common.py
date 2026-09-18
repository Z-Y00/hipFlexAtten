"""
Triton kernel helper functions shared across flash attention modules.

This module contains Triton JIT-compiled helper functions that are used within
the main attention kernels (fwd_prefill, fwd_decode, bwd). These are kept
separate from utils.py to allow stricter type checking on pure Python utilities.
"""
