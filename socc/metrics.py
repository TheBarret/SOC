"""
Metrics bound directly to the C shared object.
"""

import ctypes

from constants import *
from ffi import SocEnergyAudit, _lib
from shape import Shape


def eta(shape_a: Shape, shape_b: Shape) -> float:
    return _lib.soc_eta(
        ctypes.byref(shape_a._struct),
        ctypes.byref(shape_b._struct),
    )


def y_rx(shape_in: Shape, target: Shape,
         formula: str = "spec", threshold: float = constants.ETA_THRESH) -> float:
    if formula not in ("spec", "fixed"):
        raise ValueError(f"formula must be 'spec' or 'fixed', got {formula}")

    c_formula = (
        constants.SOC_YRX_FORMULA_SPEC if formula == "spec"
        else constants.SOC_YRX_FORMULA_FIXED
    )

    return _lib.soc_y_rx(
        ctypes.byref(shape_in._struct),
        ctypes.byref(target._struct),
        int(c_formula),
        float(threshold),
        float(constants.GAMMA),
    )


def energy_audit(before: Shape, after: Shape) -> dict:
    audit = SocEnergyAudit()
    status = _lib.soc_energy_audit(
        ctypes.byref(before._struct),
        ctypes.byref(after._struct),
        ctypes.byref(audit),
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"energy_audit failed with status {status}")

    return {
        "energy_before": audit.energy_before,
        "energy_after": audit.energy_after,
        "energy_delta": audit.energy_delta,
        "norm_preserved": bool(audit.norm_preserved),
    }
