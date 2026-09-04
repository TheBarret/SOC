"""
Shape generators bound directly to the C shared object.
"""

import ctypes

from constants import *
from ffi import SOC_COMPLEX, _lib
from shape import Shape


def curve_to_coeffs(z_samples, provenance: str = "") -> Shape:
    z_samples = np.asarray(z_samples, dtype=np.complex128)
    n = len(z_samples)

    z_arr = (SOC_COMPLEX * n)()
    for i in range(n):
        z_arr[i][0] = z_samples[i].real
        z_arr[i][1] = z_samples[i].imag

    shape = Shape(provenance=provenance)
    status = _lib.soc_curve_to_coeffs(z_arr, n, ctypes.byref(shape._struct))
    if status != constants.SOC_OK:
        raise RuntimeError(f"curve_to_coeffs failed with status {status}")

    return shape


def gen_circle(provenance: str = "generator:circle") -> Shape:
    shape = Shape(provenance=provenance)
    status = _lib.soc_gen_circle(
        ctypes.byref(shape._struct), int(constants.M)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"gen_circle failed with status {status}")
    return shape


def gen_polygon(sides: int, provenance: str = "") -> Shape:
    if provenance == "":
        provenance = f"generator:polygon({sides})"
    shape = Shape(provenance=provenance)
    status = _lib.soc_gen_polygon(
        ctypes.byref(shape._struct), int(sides), int(constants.M)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"gen_polygon failed with status {status}")
    return shape


def gen_star(points: int = 5, inner_ratio: float = 0.45,
             provenance: str = "") -> Shape:
    if provenance == "":
        provenance = f"generator:star(points={points}, inner={inner_ratio})"
    shape = Shape(provenance=provenance)
    status = _lib.soc_gen_star(
        ctypes.byref(shape._struct),
        int(points),
        float(inner_ratio),
        int(constants.M),
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"gen_star failed with status {status}")
    return shape


def default_shapes() -> dict:
    return {
        "circle": gen_circle(),
        "triangle": gen_polygon(3),
        "square": gen_polygon(4),
        "star": gen_star(5),
    }
