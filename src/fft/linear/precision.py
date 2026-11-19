"""CUDA specialization shared by FP32/FP64 linear kernels.

FP32 source is returned byte for byte. FP64 changes scalar/vector types,
intrinsics and literal suffixes; constants must be generated in FP64 first.
"""

import re

import numpy as np


def real_dtype(dtype):
    """Return the FP32 or FP64 real dtype associated with a supported scalar dtype."""
    dtype = np.dtype(dtype)
    if dtype in (np.dtype("float32"), np.dtype("complex64")):
        return np.dtype("float32")
    if dtype in (np.dtype("float64"), np.dtype("complex128")):
        return np.dtype("float64")
    raise ValueError("Linear CUDA kernels require FP32 or FP64")


def cuda_source(source, dtype):
    """Specialize FP32 CUDA source for the requested FP32 or FP64 precision."""
    if real_dtype(dtype) == np.dtype("float32"):
        return source
    for old, new in [
        ("make_float2", "make_double2"),
        ("make_float4", "make_double4"),
        ("float2", "double2"),
        ("float4", "double4"),
        ("float", "double"),
        ("sqrtf", "sqrt"),
        ("rsqrtf", "rsqrt"),
        ("fmaf", "fma"),
        ("nanf", "nan"),
        ("fabsf", "fabs"),
    ]:
        source = re.sub(r"\b" + old + r"\b", new, source)
    return re.sub(r"(\b(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)[fF]\b", r"\1", source)
