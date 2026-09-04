"""
Operator functions bound directly to the C shared object.
"""

import ctypes

from constants import *
from ffi import SocShape, _lib
from shape import Shape


def phase_shift(shape: Shape, theta: float) -> Shape:
    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    status = _lib.soc_op_phase_shift(
        ctypes.byref(new_shape._struct), float(theta)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"phase_shift failed with status {status}")
    new_shape._provenance += f" | phase_shift({theta:.4f})"
    return new_shape


def freq_shift(shape: Shape, m: int, mode: str = "truncate") -> Shape:
    if mode not in ("truncate", "wrap"):
        raise ValueError(f"mode must be 'truncate' or 'wrap', got {mode}")

    c_mode = constants.SOC_SHIFT_TRUNCATE if mode == "truncate" else constants.SOC_SHIFT_WRAP
    new_shape = Shape(provenance=shape.provenance)

    status = _lib.soc_op_freq_shift(
        ctypes.byref(shape._struct),
        ctypes.byref(new_shape._struct),
        int(m),
        int(c_mode),
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"freq_shift failed with status {status}")

    new_shape._provenance += f" | freq_shift(m={m}, mode={mode})"
    return new_shape


def spectral_filter(shape: Shape, weights) -> Shape:
    if isinstance(weights, (int, float)):
        w = np.full(constants.COEFF_LENGTH, float(weights), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (constants.COEFF_LENGTH,):
            raise ValueError(
                f"weights must have shape ({constants.COEFF_LENGTH},), got {w.shape}"
            )

    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    w_arr = (ctypes.c_double * constants.COEFF_LENGTH)(*w)

    status = _lib.soc_op_spectral_filter(
        ctypes.byref(new_shape._struct), w_arr
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"spectral_filter failed with status {status}")

    new_shape._provenance += f" | filter(custom)"
    return new_shape


def uniform_gain(shape: Shape, gain: float) -> Shape:
    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    status = _lib.soc_op_uniform_gain(
        ctypes.byref(new_shape._struct), float(gain)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"uniform_gain failed with status {status}")
    new_shape._provenance += f" | gain({gain})"
    return new_shape


def lowpass(shape: Shape, cutoff: int) -> Shape:
    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    status = _lib.soc_op_lowpass(
        ctypes.byref(new_shape._struct), int(cutoff)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"lowpass failed with status {status}")
    new_shape._provenance += f" | lowpass(K={cutoff})"
    return new_shape


def highpass(shape: Shape, cutoff: int) -> Shape:
    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    status = _lib.soc_op_highpass(
        ctypes.byref(new_shape._struct), int(cutoff)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"highpass failed with status {status}")
    new_shape._provenance += f" | highpass(K={cutoff})"
    return new_shape


def dc_boost(shape: Shape, boost: float) -> Shape:
    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    status = _lib.soc_op_dc_boost(
        ctypes.byref(new_shape._struct), float(boost)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"dc_boost failed with status {status}")
    new_shape._provenance += f" | dc_boost({boost})"
    return new_shape


def attenuate(shape: Shape, distance: float, alpha: float = constants.ALPHA,
              noise_std: float = 0.0) -> Shape:
    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    status = _lib.soc_op_attenuate(
        ctypes.byref(new_shape._struct),
        float(distance),
        float(alpha),
        float(noise_std),
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"attenuate failed with status {status}")
    new_shape._provenance += f" | attenuate(d={distance:.2f})"
    return new_shape


def power_clamp(shape: Shape, p_max: float = constants.P_MAX) -> Shape:
    new_shape = Shape(shape.coeffs, provenance=shape.provenance)
    status = _lib.soc_op_power_clamp(
        ctypes.byref(new_shape._struct), float(p_max)
    )
    if status != constants.SOC_OK:
        raise RuntimeError(f"power_clamp failed with status {status}")
    new_shape._provenance += f" | clamp(P_max={p_max})"
    return new_shape
