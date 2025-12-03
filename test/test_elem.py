from mlir.dialects.sar.frontend import *
from mlir.dialects.sar import lower_to_linalg_text

@sar_func
def forward(arg0: float32[3, 3, 3], arg1: float32[3, 3, 3]) -> float32[3, 3, 3]:
	c1 = const([3, 3, 3], 1.0)
	c2 = const_like(arg1, 2.0)
	return (arg0 + c1) - (arg1 * c2)

print("=" * 50)
print("SAR Dialect Elem Test")
print("=" * 50)

module = forward()
print(module)

print("linalg text:", end="\n\n")
lowered = lower_to_linalg_text(str(module))
print(lowered)

