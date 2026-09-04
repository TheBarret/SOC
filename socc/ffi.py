"""
    FFI Conduit
"""
from constants import *
import ctypes
from pathlib import Path

# Locate the shared object relative to this file.
_BUILD_DIR = Path(__file__).resolve().parent / "./build"
_LIB_PATH = _BUILD_DIR / "libsoc.so"

if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"libsoc.so not found at {_LIB_PATH}."
    )

_lib = ctypes.CDLL(str(_LIB_PATH))


# ctypes types


SOC_COMPLEX = ctypes.c_double * 2  # matches C99 double complex layout

SOC_COEFF_LENGTH = 17

class SocShape(ctypes.Structure):
    _fields_ = [
        ("coeffs", SOC_COMPLEX * SOC_COEFF_LENGTH),
    ]

class SocEnergyAudit(ctypes.Structure):
    _fields_ = [
        ("energy_before", ctypes.c_double),
        ("energy_after", ctypes.c_double),
        ("energy_delta", ctypes.c_double),
        ("norm_preserved", ctypes.c_int),
    ]


# shape.h signatures


_lib.soc_shape_init.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(SOC_COMPLEX),
]
_lib.soc_shape_init.restype = ctypes.c_int

_lib.soc_shape_zero.argtypes = [ctypes.POINTER(SocShape)]
_lib.soc_shape_zero.restype = ctypes.c_int

_lib.soc_shape_copy.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(SocShape),
]
_lib.soc_shape_copy.restype = ctypes.c_int

_lib.soc_shape_get.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    ctypes.POINTER(SOC_COMPLEX),
]
_lib.soc_shape_get.restype = ctypes.c_int

_lib.soc_shape_set.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    SOC_COMPLEX,
]
_lib.soc_shape_set.restype = ctypes.c_int

_lib.soc_shape_get_flat.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    ctypes.POINTER(SOC_COMPLEX),
]
_lib.soc_shape_get_flat.restype = ctypes.c_int

_lib.soc_shape_set_flat.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    SOC_COMPLEX,
]
_lib.soc_shape_set_flat.restype = ctypes.c_int

_lib.soc_shape_norm_full.argtypes = [ctypes.POINTER(SocShape)]
_lib.soc_shape_norm_full.restype = ctypes.c_double

_lib.soc_shape_norm_shape.argtypes = [ctypes.POINTER(SocShape)]
_lib.soc_shape_norm_shape.restype = ctypes.c_double

_lib.soc_shape_c0.argtypes = [ctypes.POINTER(SocShape)]
_lib.soc_shape_c0.restype = SOC_COMPLEX

_lib.soc_shape_reconstruct.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(SOC_COMPLEX),
]
_lib.soc_shape_reconstruct.restype = ctypes.c_int

_lib.soc_k_to_index.argtypes = [ctypes.c_int]
_lib.soc_k_to_index.restype = ctypes.c_int

_lib.soc_index_to_k.argtypes = [ctypes.c_int]
_lib.soc_index_to_k.restype = ctypes.c_int


# operators.h signatures


_lib.soc_op_phase_shift.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_double,
]
_lib.soc_op_phase_shift.restype = ctypes.c_int

_lib.soc_op_freq_shift.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    ctypes.c_int,
]
_lib.soc_op_freq_shift.restype = ctypes.c_int

_lib.soc_op_spectral_filter.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(ctypes.c_double),
]
_lib.soc_op_spectral_filter.restype = ctypes.c_int

_lib.soc_op_uniform_gain.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_double,
]
_lib.soc_op_uniform_gain.restype = ctypes.c_int

_lib.soc_op_lowpass.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
]
_lib.soc_op_lowpass.restype = ctypes.c_int

_lib.soc_op_highpass.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
]
_lib.soc_op_highpass.restype = ctypes.c_int

_lib.soc_op_dc_boost.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_double,
]
_lib.soc_op_dc_boost.restype = ctypes.c_int

_lib.soc_op_attenuate.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.soc_op_attenuate.restype = ctypes.c_int

_lib.soc_op_power_clamp.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_double,
]
_lib.soc_op_power_clamp.restype = ctypes.c_int


# metrics.h signatures


_lib.soc_eta.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(SocShape),
]
_lib.soc_eta.restype = ctypes.c_double

_lib.soc_y_rx.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.soc_y_rx.restype = ctypes.c_double

_lib.soc_energy_audit.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.POINTER(SocShape),
    ctypes.POINTER(SocEnergyAudit),
]
_lib.soc_energy_audit.restype = ctypes.c_int


# generators.h signatures


_lib.soc_curve_to_coeffs.argtypes = [
    ctypes.POINTER(SOC_COMPLEX),
    ctypes.c_int,
    ctypes.POINTER(SocShape),
]
_lib.soc_curve_to_coeffs.restype = ctypes.c_int

_lib.soc_gen_circle.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
]
_lib.soc_gen_circle.restype = ctypes.c_int

_lib.soc_gen_polygon.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    ctypes.c_int,
]
_lib.soc_gen_polygon.restype = ctypes.c_int

_lib.soc_gen_star.argtypes = [
    ctypes.POINTER(SocShape),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
]
_lib.soc_gen_star.restype = ctypes.c_int
