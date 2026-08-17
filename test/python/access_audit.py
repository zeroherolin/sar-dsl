"""Memory-access audit for the affine (HLS) lowering.

An access is *cross-row* when the innermost enclosing loop's induction
variable does not appear in the fastest-varying index of a
multi-dimensional memref. Consecutive iterations of the innermost loop
then stride by at least a whole row, which on an FPGA costs a memory
transaction per element instead of a burst, and forces the plane to stay
resident rather than stream.

The count is a regression gate rather than a target: a corner turn is
part of omega-K, and `sar-stage-transposes` already moves the striding
side onto a small on-chip block. What must not happen is the count
growing -- a pass reordering loops, or a fusion failing, puts full-scene
planes back on the strided side.
"""

import re

_FOR = re.compile(r'affine\.for\s+(%\w+)\s*=')
_ACCESS = re.compile(r'(affine|memref)\.(load|store)\s+(?:[^,\[]+,\s*)?'
                     r'(%\w+)\[([^\]]*)\]\s*:\s*memref<([^>]+)>')
_SSA = re.compile(r'%\w+')


class Access:
    """One memref access together with the loop nest enclosing it."""

    def __init__(self, line, dialect, op, buffer, indices, shape, ivs):
        self.line = line
        self.dialect = dialect
        self.op = op
        self.buffer = buffer
        self.indices = indices
        self.shape = shape
        self.ivs = ivs

    @property
    def rank(self):
        return len(self.shape)

    @property
    def elements(self):
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    def kind(self):
        """'row', 'cross', 'invariant', or 'flat' for rank < 2."""
        if self.rank < 2 or not self.indices or not self.ivs:
            return 'flat'
        innermost = self.ivs[-1]
        if innermost in set(_SSA.findall(self.indices[-1])):
            return 'row'
        for expr in self.indices[:-1]:
            if innermost in set(_SSA.findall(expr)):
                return 'cross'
        return 'invariant'

    def __repr__(self):
        return (f"L{self.line} {self.dialect}.{self.op} {self.buffer}"
                f"[{', '.join(self.indices)}] : "
                f"memref<{'x'.join(self.shape)}>")


def walk_accesses(mlir_text: str):
    """Yields an `Access` per memref access, in source order.

    The loop stack is tracked by brace depth, so an access reports the
    induction variables actually enclosing it.
    """
    ivs, opened_at, depth = [], [], 0
    for lineno, line in enumerate(mlir_text.splitlines(), 1):
        stripped = line.strip()

        loop = _FOR.search(stripped)
        if loop:
            ivs.append(loop.group(1))
            opened_at.append(depth)

        found = _ACCESS.search(stripped)
        if found:
            dialect, op, buffer, index_text, memref_type = found.groups()
            parts = memref_type.split('x')
            # Drop the element type, and any memory-space attribute after it.
            shape = [p for p in parts[:-1] if p.isdigit()]
            indices = ([s.strip() for s in index_text.split(',')]
                       if index_text.strip() else [])
            yield Access(lineno, dialect, op, buffer, indices, shape,
                         list(ivs))

        depth += line.count('{') - line.count('}')
        while opened_at and depth <= opened_at[-1]:
            opened_at.pop()
            ivs.pop()


def cross_row_accesses(mlir_text: str):
    """Every access whose innermost loop strides it by at least a row."""
    return [a for a in walk_accesses(mlir_text) if a.kind() == 'cross']
