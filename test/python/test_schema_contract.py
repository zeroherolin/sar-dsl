"""Frontend, dialect verifier, and runtime attribute-schema contracts."""

import re

from conftest import REPO_ROOT
from sar.language import _schema


def _quoted_values(text: str, start: str, end: str) -> tuple[str, ...]:
    body = text.split(start, 1)[1].split(end, 1)[0]
    return tuple(re.findall(r'"([a-z0-9_]+)"', body))


def test_frontend_vocabularies_match_dialect_verifiers():
    source = (REPO_ROOT / "lib/Dialect/SAR/IR/SARDialect.cpp").read_text()
    assert set(_quoted_values(source, "static const char *kinds[]", ";")) == \
        set(_schema.CMP_PREDICATES)
    reduce_body = source.split("LogicalResult ReduceOp::verify()",
                               1)[1].split("LogicalResult", 1)[0]
    assert set(re.findall(r'"(sum|max|min)"', reduce_body)) == \
        set(_schema.REDUCE_KINDS)
    interp = set(_quoted_values(source, "verifyInterpKernel",
                                "return success"))
    assert interp >= set(_schema.INTERP_KERNELS + _schema.INTERP_WINDOWS +
                         _schema.INTERP_BOUNDARIES)
    assert set(_quoted_values(source, "verifyGatherLike", "return success")) \
        >= set(_schema.GATHER_KERNELS + _schema.GATHER_BOUNDARIES)


def test_runtime_enum_and_lowering_cover_frontend_interp_vocabulary():
    enums = (REPO_ROOT / "include/sar/Runtime/RuntimeEnums.h").read_text()
    lowering = (REPO_ROOT /
                "lib/Conversion/SARSignalToRuntime/SARSignalToRuntime.cpp"
                ).read_text()
    expected = {
        name: "k" + "".join(part.capitalize() for part in name.split("_"))
        for name in (_schema.INTERP_KERNELS + _schema.INTERP_WINDOWS +
                     _schema.INTERP_BOUNDARIES)
    }
    defaults = {"sinc", "kaiser", "reflect"}
    for spelling, enum in expected.items():
        assert re.search(rf"\b{enum}\s*=", enums), enum
        if spelling in defaults:
            assert f".Default(sar_rt::{enum})" in lowering
        else:
            assert f'.Case("{spelling}", sar_rt::{enum})' in lowering


def test_every_verified_public_sar_op_has_invalid_ir_coverage():
    """Every operation with a custom verifier has an invalid IR test."""
    ops = (REPO_ROOT / "include/sar/Dialect/SAR/IR/SAROps.td").read_text()
    mappings = dict(
        re.findall(r"def SAR_(\w+)Op\s*:\s*[^;{]*?<\"([^\"]+)\"", ops, re.S))
    verifier = (REPO_ROOT / "lib/Dialect/SAR/IR/SARDialect.cpp").read_text()
    classes = set(re.findall(r"LogicalResult (\w+)Op::verify\(\)", verifier))
    declared = {mappings[name] for name in classes if name in mappings}
    internal = {"fft_split", "interp1d_split", "gather2d_split", "yield"}
    invalid = (REPO_ROOT / "test/Dialect/SAR/invalid.mlir").read_text()
    covered = set(re.findall(r"sar\.([a-z0-9_]+)", invalid))
    assert declared - internal <= covered, declared - internal - covered
