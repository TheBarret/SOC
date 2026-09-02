"""
SOC generators: spatial-to-Fourier transforms and geometric primitives.
"""

from .shapes import (
    curve_to_coeffs,
    gen_circle,
    gen_polygon,
    gen_star,
    default_shapes,
)

__all__ = [
    "curve_to_coeffs",
    "gen_circle",
    "gen_polygon",
    "gen_star",
    "default_shapes",
]
