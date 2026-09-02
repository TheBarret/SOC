"""
Shape generators: produce Shape objects from geometric primitives.

The curve_to_coeffs transform is here because it's the bridge between
spatial domain (z(t) samples) and the Fourier coefficient domain (Shape).
"""

import numpy as np

from ..core.constants import N, M
from ..core.shape import Shape

# Transform: sampled curve -> coefficient vector

def curve_to_coeffs(z_samples: np.ndarray, provenance: str = "") -> Shape:
    """
    DFT of the sampled closed curve (spec §1).

    Args:
        z_samples: complex128 array of curve samples z(t), t in [0,1).
        provenance: passed through to the Shape.

    Returns:
        Shape with coefficients C_k for k = -N..N.

    Note:
        Uses normalized FFT: np.fft.fft(z) / len(z).
        Coefficients beyond +-N are discarded (band-limiting).
    """
    z_samples = np.asarray(z_samples, dtype=np.complex128)
    n_samples = len(z_samples)

    full = np.fft.fft(z_samples) / n_samples

    coeffs = np.zeros(2 * N + 1, dtype=np.complex128)
    for idx, k in enumerate(range(-N, N + 1)):
        coeffs[idx] = full[k % n_samples]

    return Shape(coeffs, provenance=provenance)

# Generators

def gen_circle(provenance: str = "generator:circle") -> Shape:
    """Unit circle: z(t) = e^{i 2pi t}."""
    t = np.linspace(0, 1, M, endpoint=False)
    z = np.exp(1j * 2 * np.pi * t)
    return curve_to_coeffs(z, provenance=provenance)


def gen_polygon(sides: int, provenance: str = "") -> Shape:
    """Regular polygon with given number of sides, circumradius 1."""
    if provenance == "":
        provenance = f"generator:polygon({sides})"

    t = np.linspace(0, 1, M, endpoint=False)
    theta = t * 2 * np.pi
    seg = 2 * np.pi / sides
    phi = np.mod(theta, seg) - seg / 2
    r = np.cos(seg / 2) / np.cos(phi)
    z = r * np.exp(1j * theta)

    return curve_to_coeffs(z, provenance=provenance)


def gen_star(points: int = 5, inner_ratio: float = 0.45,
             provenance: str = "") -> Shape:
    """Star shape with given number of points."""
    if provenance == "":
        provenance = f"generator:star(points={points}, inner={inner_ratio})"

    t = np.linspace(0, 1, M, endpoint=False)
    theta = t * 2 * np.pi
    seg = np.pi / points
    phi = np.mod(theta, 2 * seg)
    tri = np.abs(phi / seg - 1.0)
    r = inner_ratio + (1 - inner_ratio) * (1 - tri)
    z = r * np.exp(1j * theta)

    return curve_to_coeffs(z, provenance=provenance)

# Default shape set

def default_shapes() -> dict[str, Shape]:
    """Circle, triangle, square, star — the PoC's standard four."""
    return {
        "circle": gen_circle(),
        "triangle": gen_polygon(3),
        "square": gen_polygon(4),
        "star": gen_star(5),
    }
