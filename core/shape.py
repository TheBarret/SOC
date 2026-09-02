"""
Shape: immutable coefficient vector S in the Fourier basis.

S = (C_{-N}, ..., C_0, ..., C_N), complex128, shape (2N+1,).
The Shape object is immutable. Operators return new Shape instances.
"""

import numpy as np

from .constants import N, M, k_to_index, k_values


class Shape:
    """Closed contour represented as a band-limited Fourier coefficient vector."""

    __slots__ = ("_coeffs", "_provenance")

    def __init__(self, coeffs: np.ndarray, provenance: str = ""):
        """
        Args:
            coeffs: complex128 array of shape (2N+1,), indexed -N..N.
            provenance: optional string describing where this shape came from.
        """
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        if coeffs.shape != (2 * N + 1,):
            raise ValueError(f"coeffs must have shape ({2*N+1},), got {coeffs.shape}")
        # Store a copy to enforce immutability.
        self._coeffs = coeffs.copy()
        self._provenance = provenance

    #  basic access

    @property
    def coeffs(self) -> np.ndarray:
        """Read-only view of the coefficient vector."""
        return self._coeffs

    @property
    def provenance(self) -> str:
        return self._provenance

    @property
    def n(self) -> int:
        """Harmonic count N."""
        return N

    def __len__(self) -> int:
        return 2 * N + 1

    def __getitem__(self, idx: int) -> complex:
        """Access coefficient by flat index (0..2N) or by harmonic k (-N..N)."""
        if isinstance(idx, int):
            if idx < 0:
                idx = k_to_index(idx)
            return self._coeffs[idx]
        return self._coeffs[idx]

    def __repr__(self) -> str:
        return f"Shape(N={N}, provenance='{self._provenance}')"

    #  derived quantities -

    def norm_full(self) -> float:
        """||S||^2 over all harmonics, including C0."""
        return float(np.sum(np.abs(self._coeffs) ** 2))

    def norm_shape(self) -> float:
        """||S||^2 over the shape subspace (k != 0)."""
        mask = np.ones(2 * N + 1, dtype=bool)
        mask[k_to_index(0)] = False
        return float(np.sum(np.abs(self._coeffs[mask]) ** 2))

    def c0(self) -> complex:
        """The C0 term (position/bias, excluded from shape metrics)."""
        return self._coeffs[k_to_index(0)]

    #  reconstruction -

    def reconstruct(self, t: np.ndarray | None = None) -> np.ndarray:
        """
        Reconstruct the closed curve z(t) at sample points t.

        Args:
            t: sample points in [0,1). Defaults to M equally spaced points.

        Returns:
            complex128 array of shape (len(t),).
        """
        if t is None:
            t = np.linspace(0, 1, M, endpoint=False)
        ks = k_values()
        mat = np.exp(1j * 2 * np.pi * np.outer(t, ks))
        return mat @ self._coeffs

    #  serialization

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "N": N,
            "coeffs_real": self._coeffs.real.tolist(),
            "coeffs_imag": self._coeffs.imag.tolist(),
            "provenance": self._provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Shape":
        """Deserialize from a dict produced by to_dict()."""
        if d.get("N") != N:
            raise ValueError(f"serialized N={d.get('N')} does not match module N={N}")
        coeffs = np.array(d["coeffs_real"], dtype=np.float64) + \
                 1j * np.array(d["coeffs_imag"], dtype=np.float64)
        return cls(coeffs, provenance=d.get("provenance", ""))
