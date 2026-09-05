"""Device resources the constraints can be derived from.

The six resource budgets -- three memory tiers plus DSP, FF and LUT -- have
to describe the same device as `part`. Keeping them consistent by hand is a
resource mismatch: a VU13P budget paired with a Zynq part describes a device
several times larger than the synthesis target.

So the table below records what each supported part *has*, and
`budgets_for` sizes the six from it. Naming a part by itself asks `config.py`
to derive the budgets; a complete explicit contract may instead name that
part and all six values, so no budget is inherited from another device.

Primitive geometry differs by family too, and the placement passes charge
banks in whole primitives: `storage_primitives` reports the block sizes
the cost model must round to, rather than assuming UltraScale+ everywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = [
    "DEFAULT_UTILIZATION", "Device", "DEVICES", "bram18_count", "budgets_for",
    "find_device", "storage_primitives", "uram_count"
]

#: Share of a device the compiler plans against. Synthesis needs room for
#: interconnect and placement beyond what the design's own arrays claim,
#: and the shipped VU13P budgets are exactly this fraction of the part.
DEFAULT_UTILIZATION = 0.8

#: Bytes in one memory primitive, per family. A bank occupies a whole
#: primitive whatever it holds, which is what makes these the rounding
#: unit for every storage estimate.
_BRAM36_BYTES = 4608  # 36 Kb
_URAM_BYTES = 36864  # 288 Kb
_BRAM18_BYTES = 2304  # 18 Kb, the unit Vitis reports utilization in


@dataclass(frozen=True)
class Device:
    """What a part contains, before any utilization target is applied."""

    #: 36 Kb block RAMs.
    bram36: int
    #: UltraRAM blocks; zero on families without the primitive.
    uram: int
    dsp: int
    ff: int
    lut: int
    #: Distributed RAM, in kilobits, as the datasheets quote it.
    lutram_kb: int
    #: Bytes in one block RAM and one UltraRAM on this family.
    bram_block_bytes: int = _BRAM36_BYTES
    uram_block_bytes: int = _URAM_BYTES


#: Parts the budgets can be derived for, keyed by the device portion of the
#: part name (the speed grade and package do not change resources). Values
#: are datasheet counts for the whole device.
DEVICES: Dict[str, Device] = {
    # UltraScale+ VU13P: the reference target the defaults describe.
    "xcvu13p":
    Device(bram36=2688,
           uram=1280,
           dsp=12288,
           ff=3456000,
           lut=1728000,
           lutram_kb=28160),
    "xcvu9p":
    Device(bram36=2160,
           uram=960,
           dsp=6840,
           ff=2364480,
           lut=1182240,
           lutram_kb=36720),
    "xcvu11p":
    Device(bram36=2016,
           uram=960,
           dsp=9216,
           ff=2607360,
           lut=1303680,
           lutram_kb=21120),
    # Alveo cards, named by their device rather than the shell.
    "xcu250":
    Device(bram36=2688,
           uram=1280,
           dsp=12288,
           ff=3456000,
           lut=1728000,
           lutram_kb=28160),
    "xcu280":
    Device(bram36=2016,
           uram=960,
           dsp=9024,
           ff=2607360,
           lut=1303680,
           lutram_kb=21120),
    # Zynq UltraScale+ MPSoC and RFSoC.
    "xczu9eg":
    Device(bram36=912, uram=0, dsp=2520, ff=548160, lut=274080,
           lutram_kb=8820),
    "xczu28dr":
    Device(bram36=1080,
           uram=80,
           dsp=4272,
           ff=850560,
           lut=425280,
           lutram_kb=13500),
    # 7-series: no UltraRAM tier at all.
    "xc7z020":
    Device(bram36=140,
           uram=0,
           dsp=220,
           ff=106400,
           lut=53200,
           lutram_kb=1730,
           uram_block_bytes=0),
    "xc7k325t":
    Device(bram36=445,
           uram=0,
           dsp=840,
           ff=407600,
           lut=203800,
           lutram_kb=4000,
           uram_block_bytes=0),
}

#: A part name is `<device>-<package>-<speed>...`; only the leading device
#: token decides the resources.
_DEVICE_TOKEN = re.compile(r"^([a-z0-9]+?)(?:-.*)?$", re.I)


def find_device(part: Optional[str]) -> Optional[Device]:
    """The device a part name refers to, or None if it is not tabulated."""
    if not part:
        return None
    match = _DEVICE_TOKEN.match(part.strip())
    if match is None:
        return None
    return DEVICES.get(match.group(1).lower())


def budgets_for(part: Optional[str],
                utilization: float = DEFAULT_UTILIZATION) -> Dict[str, int]:
    """The six resource budgets `part` implies, or `{}` if it is unknown.

    `utilization` is the share of the device the design may claim, as a
    fraction. Whole primitives are budgeted, not bytes: a tier's cap is the
    byte size of the blocks the design may claim, so the count is floored
    before it is converted. Distributed RAM is quoted in kilobits by the
    datasheets and stored here in bytes.
    """
    device = find_device(part)
    if device is None:
        return {}
    return {
        "bram_bytes":
        int(device.bram36 * utilization) * device.bram_block_bytes,
        "uram_bytes": int(device.uram * utilization) * device.uram_block_bytes,
        "lutram_bytes": int(device.lutram_kb * 1024 // 8 * utilization),
        "dsp": int(device.dsp * utilization),
        "ff": int(device.ff * utilization),
        "lut": int(device.lut * utilization),
    }


def storage_primitives(part: Optional[str]) -> Tuple[int, int]:
    """(block RAM bytes, UltraRAM bytes) a bank is rounded up to.

    A family without UltraRAM reports zero for it, so a caller charging
    banked storage does not price a tier the device cannot provide.
    Unknown parts return ``(0, 0)``. Their primitive geometry must be stated
    explicitly with their budgets; silently assuming UltraScale+ would plan
    against a different device than the generated Tcl names.
    """
    if part is None:
        return _BRAM36_BYTES, _URAM_BYTES
    device = find_device(part)
    if device is None:
        return 0, 0
    return device.bram_block_bytes, device.uram_block_bytes


def bram18_count(bram_bytes: int) -> int:
    """Block RAM budget expressed in the 18 Kb units Vitis reports."""
    return int(bram_bytes) // _BRAM18_BYTES


def uram_count(uram_bytes: int, part: Optional[str] = None) -> int:
    """UltraRAM budget expressed in whole blocks, zero on families
    without the primitive."""
    block = storage_primitives(part)[1]
    return int(uram_bytes) // block if block else 0
