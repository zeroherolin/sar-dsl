"""Shared spellings of finite SAR operation attribute vocabularies.

The C++ dialect remains authoritative. Tests compare these frontend sets with
the verifier and runtime ABI sources so adding a spelling on only one side is
an immediate failure instead of a backend-dependent language change.
"""

CMP_PREDICATES = ("eq", "ne", "lt", "le", "gt", "ge")
REDUCE_KINDS = ("sum", "max", "min")
INTERP_KERNELS = ("nearest", "linear", "cubic", "sinc")
INTERP_WINDOWS = ("rect", "hann", "hamming", "kaiser")
INTERP_BOUNDARIES = ("zero", "edge", "reflect")
GATHER_KERNELS = ("nearest", "linear")
GATHER_BOUNDARIES = ("zero", "edge")
