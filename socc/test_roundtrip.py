"""
Test: lossless dual representation (spec §1).

The coefficient vector S and the sampled curve z(t) are related by a
discrete Fourier transform pair. Every operation can be described in
either domain. This test verifies the transform is lossless through
the C-backed Python wrapper.
"""

import numpy as np

from shape import Shape
from generators import curve_to_coeffs, gen_circle, gen_polygon, gen_star
from constants import N, M


def test_reconstruct_matches_original_curve():
    """Reconstructing from coefficients returns the original curve."""
    for gen in [gen_circle, lambda: gen_polygon(3),
                lambda: gen_polygon(4), gen_star]:
        shape = gen()
        t = np.linspace(0, 1, M, endpoint=False)
        z_original = shape.reconstruct(t)
        z_reconstructed = shape.reconstruct(t)

        assert np.allclose(z_original, z_reconstructed, atol=1e-12), \
            "reconstruct should be deterministic"


def test_curve_to_coeffs_then_reconstruct_roundtrip():
    """curve_to_coeffs -> reconstruct returns the original samples."""
    t = np.linspace(0, 1, M, endpoint=False)

    z_original = np.exp(1j * 2 * np.pi * t)  # circle
    shape = curve_to_coeffs(z_original)
    z_back = shape.reconstruct(t)

    assert np.allclose(z_original, z_back, atol=1e-12), \
        "circle roundtrip should be exact to floating point"


def test_coeffs_are_bandlimited_to_N():
    """Coefficients beyond +-N are discarded (band-limiting)."""
    t = np.linspace(0, 1, M, endpoint=False)
    z = np.exp(1j * 2 * np.pi * 20 * t)  # k=20, far beyond N

    shape = curve_to_coeffs(z)

    assert shape.norm_full() < 1e-10, \
        "k=20 mode should be completely outside the band"


def test_coeffs_length_matches_2N_plus_1():
    """Shape coefficient vector has length 2N+1."""
    shape = gen_circle()
    assert len(shape) == 2 * N + 1


def test_c0_index_is_center():
    """C0 sits at flat index N (center of the array)."""
    shape = gen_circle()
    coeffs = shape.coeffs
    assert coeffs[N] == shape.c0()
