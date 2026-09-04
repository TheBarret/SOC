"""
Python Shape class wrapping the C SocShape struct.
"""

import ctypes
import numpy as np

from constants import *
from ffi import SocShape, SOC_COMPLEX, _lib


class Shape:
    """Closed contour represented as a band-limited Fourier coefficient vector."""

    __slots__ = ("_struct", "_provenance")

    def __init__(self, coeffs=None, provenance=""):
        self._struct = SocShape()
        self._provenance = provenance

        if coeffs is not None:
            coeffs = np.asarray(coeffs, dtype=np.complex128)
            if coeffs.shape != (constants.COEFF_LENGTH,):
                raise ValueError(
                    f"coeffs must have shape ({constants.COEFF_LENGTH},), "
                    f"got {coeffs.shape}"
                )
            self._set_coeffs(coeffs)
        else:
            _lib.soc_shape_zero(ctypes.byref(self._struct))

    def _set_coeffs(self, coeffs):
        arr = (SOC_COMPLEX * constants.COEFF_LENGTH)()
        for i in range(constants.COEFF_LENGTH):
            arr[i][0] = coeffs[i].real
            arr[i][1] = coeffs[i].imag
        _lib.soc_shape_init(ctypes.byref(self._struct), arr)

    @property
    def coeffs(self):
        arr = np.zeros(constants.COEFF_LENGTH, dtype=np.complex128)
        for i in range(constants.COEFF_LENGTH):
            c = SOC_COMPLEX()
            _lib.soc_shape_get_flat(
                ctypes.byref(self._struct), i, ctypes.byref(c)
            )
            arr[i] = complex(c[0], c[1])
        return arr

    @property
    def provenance(self):
        return self._provenance

    def norm_full(self):
        return _lib.soc_shape_norm_full(ctypes.byref(self._struct))

    def norm_shape(self):
        return _lib.soc_shape_norm_shape(ctypes.byref(self._struct))

    def c0(self):
        c = _lib.soc_shape_c0(ctypes.byref(self._struct))
        return complex(c[0], c[1])

    def reconstruct(self, t=None):
        if t is None:
            t = np.linspace(0, 1, constants.M, endpoint=False)
        t = np.asarray(t, dtype=np.float64)
        n = len(t)

        out = (SOC_COMPLEX * n)()
        t_arr = (ctypes.c_double * n)(*t)

        status = _lib.soc_shape_reconstruct(
            ctypes.byref(self._struct),
            t_arr,
            n,
            out,
        )
        if status != constants.SOC_OK:
            raise RuntimeError(f"soc_shape_reconstruct failed with status {status}")

        result = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            result[i] = complex(out[i][0], out[i][1])
        return result

    def __len__(self):
        return constants.COEFF_LENGTH

    def __repr__(self):
        return f"Shape(N={constants.N}, provenance='{self._provenance}')"
