# sitecustomize.py — auto-run when on sys.path
# Goal: On AMD DirectML or CPU-only runs, provide a safe xformers stub.

import os, sys, types

# When to disable real xformers
NO_XFORMERS = os.getenv("USE_DML") == "1" or os.getenv("FORCE_CPU") == "1"
if os.getenv("AUDIOCRAFT_USE_XFORMERS", "0") == "0":
    NO_XFORMERS = True

if NO_XFORMERS:
    # Hard-disable & avoid user-site shadowing
    os.environ["AUDIOCRAFT_USE_XFORMERS"] = "0"
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"

    # Remove any previously-imported real package
    for _m in ("xformers", "xformers.ops", "xformers.ops.fmha"):
        sys.modules.pop(_m, None)

    # Import torch pieces (optional; functions raise if missing)
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        torch, F = None, None

    # ---- minimal building blocks -------------------------------------------------
    class LowerTriangularMask:
        """Marker class only; consumers usually just require the symbol to exist."""
        pass

    def _require(obj, name):
        raise RuntimeError(
            f"xformers shim requires PyTorch '{name}', but it is unavailable."
        )

    def _sdpa(q, k, v, attn_bias=None, p: float = 0.0, scale=None):
        # Simple/portable fallback; ignores bias/scale.
        if F is None or not hasattr(F, "scaled_dot_product_attention"):
            return _require(F, "nn.functional.scaled_dot_product_attention")
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=p, is_causal=False
        )

    # Common helpers some builds call via xformers.ops
    def _unbind(x, dim=0):
        if torch is None or not hasattr(torch, "unbind"):
            return _require(torch, "unbind")
        return torch.unbind(x, dim)

    def _cat(seq, dim=0):
        if torch is None or not hasattr(torch, "cat"):
            return _require(torch, "cat")
        return torch.cat(seq, dim)

    def _stack(seq, dim=0):
        if torch is None or not hasattr(torch, "stack"):
            return _require(torch, "stack")
        return torch.stack(seq, dim)

    def _einsum(*args, **kwargs):
        if torch is None or not hasattr(torch, "einsum"):
            return _require(torch, "einsum")
        return torch.einsum(*args, **kwargs)

    def _softmax(x, dim=-1, dtype=None):
        if F is None or not hasattr(F, "softmax"):
            return _require(F, "softmax")
        return F.softmax(x, dim=dim, dtype=dtype)

    def _dropout(x, p=0.0, training=False, inplace=False):
        if F is None or not hasattr(F, "dropout"):
            return _require(F, "dropout")
        return F.dropout(x, p=p, training=training, inplace=inplace)

    def _matmul(a, b):
        if torch is None or not hasattr(torch, "matmul"):
            return _require(torch, "matmul")
        return torch.matmul(a, b)

    # Optional enum some code references
    class SdpAlgorithm:
        FLASH_ATTENTION = "flash"
        MATH = "math"
        EFFICIENT_ATTENTION = "mem_efficient"

    # ---- build a real package layout: xformers, xformers.ops, xformers.ops.fmha ----
    x_pkg = types.ModuleType("xformers")
    x_pkg.__path__ = []  # make it a package
    x_pkg.SdpAlgorithm = SdpAlgorithm

    ops = types.ModuleType("xformers.ops")
    ops.LowerTriangularMask = LowerTriangularMask
    ops.SdpAlgorithm = SdpAlgorithm
    ops.memory_efficient_attention = _sdpa
    ops.scaled_dot_product_attention = _sdpa
    ops.unbind = _unbind
    ops.cat = _cat
    ops.stack = _stack
    ops.einsum = _einsum
    ops.softmax = _softmax
    ops.dropout = _dropout
    ops.matmul = _matmul

    fmha = types.ModuleType("xformers.ops.fmha")
    fmha.memory_efficient_attention = _sdpa
    fmha.LowerTriangularMask = LowerTriangularMask
    fmha.SdpAlgorithm = SdpAlgorithm

    # Wire modules
    x_pkg.ops = ops
    sys.modules["xformers"] = x_pkg
    sys.modules["xformers.ops"] = ops
    sys.modules["xformers.ops.fmha"] = fmha