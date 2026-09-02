"""
Test: decoupled-term leak (spec §5).

A readout that multiplies eta by a full norm (including C0) leaks when
C0 is boosted. The fixed readout (shape norm only) stays flat. eta
itself never changes. This test constructs nonzero C0 explicitly.
"""

import numpy as np

from soc import Shape, eta, y_rx, dc_boost
from soc.generators import gen_circle
from soc.core.constants import k_to_index


def _shape_with_c0(shape, c0_value):
    """Return a copy of shape with a specific C0 value."""
    coeffs = shape.coeffs.copy()
    coeffs[k_to_index(0)] = c0_value
    return Shape(coeffs, provenance=f"{shape.provenance} | set_c0({c0_value})")


def test_eta_stays_exactly_one_under_c0_boost():
    """eta(S, S) == 1.0 even when C0 is boosted 20x."""
    shape = _shape_with_c0(gen_circle(), c0_value=0.5 + 0.2j)

    eta_before = eta(shape, shape)

    boosted = dc_boost(shape, 20.0)
    eta_after = eta(boosted, shape)

    assert np.isclose(eta_before, 1.0, atol=1e-12)
    assert np.isclose(eta_after, 1.0, atol=1e-12), \
        f"eta changed under C0 boost: {eta_before} -> {eta_after}"


def test_spec_readout_leaks_under_c0_boost():
    """y_rx with formula='spec' climbs when C0 is boosted."""
    shape = _shape_with_c0(gen_circle(), c0_value=0.5 + 0.2j)

    y_before = y_rx(shape, shape, formula="spec")

    boosted = dc_boost(shape, 20.0)
    y_after = y_rx(boosted, shape, formula="spec")

    # Spec: 20x C0 boost -> 100x y increase (norm includes |C0|^2)
    assert y_after > y_before * 50, \
        f"spec readout did not leak: {y_before} -> {y_after}"


def test_fixed_readout_stays_flat_under_c0_boost():
    """y_rx with formula='fixed' does not change when C0 is boosted."""
    shape = _shape_with_c0(gen_circle(), c0_value=0.5 + 0.2j)

    y_before = y_rx(shape, shape, formula="fixed")

    boosted = dc_boost(shape, 20.0)
    y_after = y_rx(boosted, shape, formula="fixed")

    assert np.isclose(y_before, y_after, atol=1e-12), \
        f"fixed readout leaked: {y_before} -> {y_after}"


def test_full_norm_includes_c0():
    """The full norm changes under C0 boost; the shape norm does not."""
    shape = _shape_with_c0(gen_circle(), c0_value=0.5 + 0.2j)

    full_before = shape.norm_full()
    shape_before = shape.norm_shape()

    boosted = dc_boost(shape, 20.0)

    assert not np.isclose(full_before, boosted.norm_full(), atol=1e-12), \
        "full norm should change under C0 boost"
    assert np.isclose(shape_before, boosted.norm_shape(), atol=1e-12), \
        "shape norm should not change under C0 boost"
