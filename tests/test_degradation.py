"""
Test: frequency-dependent degradation (spec §4).

Exponential attenuation damps high-|k| modes faster than low-|k| modes.
A state with energy concentrated in low modes degrades slower than one
with energy spread into high modes. This is a property of the decay law,
not of any specific shape.
"""

import numpy as np

from soc import attenuate, eta
from soc.generators import gen_circle, gen_star
from soc.core.constants import ALPHA


def _shape_with_energy_at_mode(shape, k, amplitude=1.0):
    """Return a copy of shape with extra energy injected at harmonic k."""
    from soc.core.constants import k_to_index
    coeffs = shape.coeffs.copy()
    coeffs[k_to_index(k)] += amplitude
    from soc import Shape
    return Shape(coeffs, provenance=f"{shape.provenance} | inject(k={k})")


def test_low_modes_decay_slower_than_high_modes():
    """Energy at low |k| survives attenuation better than high |k|."""
    shape = gen_circle()  # energy at k=±1 only

    low = _shape_with_energy_at_mode(shape, k=1, amplitude=1.0)
    high = _shape_with_energy_at_mode(shape, k=7, amplitude=1.0)

    distance = 1.0

    low_attenuated = attenuate(low, distance=distance, noise_std=0.0)
    high_attenuated = attenuate(high, distance=distance, noise_std=0.0)

    # Energy remaining at the injected mode after attenuation
    from soc.core.constants import k_to_index
    low_energy = abs(low_attenuated.coeffs[k_to_index(1)]) ** 2
    high_energy = abs(high_attenuated.coeffs[k_to_index(7)]) ** 2

    assert low_energy > high_energy, \
        f"low mode should retain more energy: {low_energy} vs {high_energy}"


def test_degradation_rate_matches_exponential_law():
    """The attenuation factor follows exp(-alpha * |k| * d) exactly."""
    from soc.core.constants import k_to_index

    shape = gen_circle()
    k_low, k_high = 2, 6
    distance = 1.5

    low = _shape_with_energy_at_mode(shape, k=k_low, amplitude=1.0)
    high = _shape_with_energy_at_mode(shape, k=k_high, amplitude=1.0)

    low_att = attenuate(low, distance=distance, noise_std=0.0)
    high_att = attenuate(high, distance=distance, noise_std=0.0)

    expected_low = np.exp(-ALPHA * k_low * distance)
    expected_high = np.exp(-ALPHA * k_high * distance)

    actual_low = abs(low_att.coeffs[k_to_index(k_low)])
    actual_high = abs(high_att.coeffs[k_to_index(k_high)])

    assert np.isclose(actual_low, expected_low, atol=1e-12), \
        f"low mode attenuation wrong: {actual_low} vs {expected_low}"
    assert np.isclose(actual_high, expected_high, atol=1e-12), \
        f"high mode attenuation wrong: {actual_high} vs {expected_high}"


def test_shape_with_low_mode_energy_is_more_robust():
    """A shape with energy in low modes retains more total energy."""
    low_shape = gen_circle()  # k=±1 only
    high_shape = gen_star()   # energy spread across many modes

    distance = 1.0

    low_att = attenuate(low_shape, distance=distance, noise_std=0.0)
    high_att = attenuate(high_shape, distance=distance, noise_std=0.0)

    low_retained = low_att.norm_shape() / low_shape.norm_shape()
    high_retained = high_att.norm_shape() / high_shape.norm_shape()

    assert low_retained > high_retained, \
        f"low-mode shape should retain more: {low_retained} vs {high_retained}"
