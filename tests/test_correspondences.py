"""
Test: correspondences (spec §6).

eta is the same formula as quantum state fidelity for normalized vectors.
Class A/B/C operators map to unitary/conditionally-unitary/non-unitary
operators. These tests verify the mathematical identity, not a metaphor.
"""

import numpy as np

from soc import eta, phase_shift, attenuate, freq_shift
from soc.generators import gen_circle
from soc.core.constants import k_to_index


def _normalized_shape_vector(shape):
    """Return the shape-subspace coefficients as a normalized vector."""
    mask = np.ones(len(shape), dtype=bool)
    mask[k_to_index(0)] = False
    vec = shape.coeffs[mask]
    norm = np.sqrt(np.sum(np.abs(vec) ** 2))
    return vec / norm


def test_eta_equals_quantum_fidelity():
    """eta(A, B) = |<psi_A|psi_B>| for normalized shape-subspace vectors."""
    from soc.generators import gen_polygon, gen_star

    a = gen_polygon(3)
    b = gen_star()

    psi_a = _normalized_shape_vector(a)
    psi_b = _normalized_shape_vector(b)

    fidelity = abs(np.sum(psi_a * np.conj(psi_b)))

    assert np.isclose(eta(a, b), fidelity, atol=1e-12), \
        f"eta should equal quantum fidelity: {eta(a, b)} vs {fidelity}"


def test_phase_shift_is_unitary():
    """Class A operator preserves inner products (unitary property)."""
    from soc.generators import gen_polygon

    a = gen_polygon(4)
    b = gen_polygon(3)

    a_rot = phase_shift(a, theta=1.234)

    # Unitary operators preserve inner products
    inner_before = np.sum(_normalized_shape_vector(a) *
                          np.conj(_normalized_shape_vector(b)))
    inner_after = np.sum(_normalized_shape_vector(a_rot) *
                         np.conj(_normalized_shape_vector(b)))

    assert np.isclose(abs(inner_before), abs(inner_after), atol=1e-12), \
        "phase shift should preserve inner product magnitude"


def test_attenuate_is_not_unitary():
    """Class C operator does not preserve inner products."""
    from soc.generators import gen_polygon

    a = gen_polygon(4)
    b = gen_polygon(3)

    a_att = attenuate(a, distance=0.5, noise_std=0.0)

    inner_before = np.sum(_normalized_shape_vector(a) *
                          np.conj(_normalized_shape_vector(b)))
    inner_after = np.sum(_normalized_shape_vector(a_att) *
                         np.conj(_normalized_shape_vector(b)))

    assert not np.isclose(abs(inner_before), abs(inner_after), atol=1e-12), \
        "attenuation should not preserve inner product magnitude"


def test_freq_shift_unitary_in_bounds():
    """Class B operator is unitary when no energy crosses the boundary."""
    from soc.generators import gen_polygon

    a = gen_polygon(4)
    b = gen_polygon(3)

    a_shift = freq_shift(a, m=1, mode="truncate")
    b_shift = freq_shift(b, m=1, mode="truncate")

    inner_before = np.sum(_normalized_shape_vector(a) *
                          np.conj(_normalized_shape_vector(b)))
    inner_after = np.sum(_normalized_shape_vector(a_shift) *
                         np.conj(_normalized_shape_vector(b_shift)))

    # freq_shift by 1 on a low-mode shape stays in bounds
    assert np.isclose(abs(inner_before), abs(inner_after), atol=1e-12)
    #assert np.isclose(abs(inner_before), abs(inner_after), atol=1e-12), \
    #    "in-bounds freq shift should preserve inner product magnitude"
