"""
Test: operator taxonomy by reversibility (spec §3).
    Class A: always unitary (norm-preserving, invertible)
    Class B: conditionally unitary (lossless iff no energy crosses +-N boundary)
    Class C: never unitary once active (discards or injects energy)
"""

import numpy as np

from soc import (
    phase_shift,
    freq_shift,
    spectral_filter,
    dc_boost,
    attenuate,
    power_clamp,
    reversibility_class,
)
from soc.generators import gen_circle, gen_polygon, gen_star, default_shapes
from soc.core.constants import N

def _energy(shape):
    return shape.norm_full()

# Class A: always unitary

def test_phase_shift_preserves_norm_exactly():
    """Class A: phase rotation preserves energy for any theta."""
    shape = gen_star()
    e_before = _energy(shape)

    for theta in np.linspace(0, 10 * np.pi, 200):
        rotated = phase_shift(shape, theta)
        assert np.isclose(_energy(rotated), e_before, atol=1e-12), \
            f"phase shift theta={theta} changed energy"


def test_phase_shift_invertible():
    """Class A: phase_shift(theta) then phase_shift(-theta) is identity."""
    shape = gen_polygon(3)
    theta = 1.234

    rotated = phase_shift(shape, theta)
    unrotated = phase_shift(rotated, -theta)

    assert np.allclose(shape.coeffs, unrotated.coeffs, atol=1e-12), \
        "phase_shift should be exactly invertible"



# Class B: conditionally unitary


def test_freq_shift_in_bounds_preserves_norm():
    """Class B: freq_shift with small m preserves energy."""
    shape = gen_circle()
    e_before = _energy(shape)

    for m in [-2, -1, 0, 1, 2]:
        shifted = freq_shift(shape, m, mode="truncate")
        # circle has energy only at k=±1, so small shifts stay in bounds
        assert np.isclose(_energy(shifted), e_before, atol=1e-12), \
            f"freq_shift m={m} should preserve energy for circle"


def test_freq_shift_out_of_bounds_truncates():
    """Class B: freq_shift past +-N boundary loses energy."""
    shape = gen_star()  # energy spread across many harmonics
    e_before = _energy(shape)

    # Shift by N pushes the lowest mode (k=-N) out of bounds
    shifted = freq_shift(shape, N, mode="truncate")

    assert _energy(shifted) < e_before, \
        "freq_shift past boundary should truncate energy"


def test_freq_shift_wrap_preserves_norm():
    """Class B: freq_shift with mode='wrap' is always unitary."""
    shape = gen_star()
    e_before = _energy(shape)

    for m in [-2 * N, -N, -1, 0, 1, N, 2 * N]:
        shifted = freq_shift(shape, m, mode="wrap")
        assert np.isclose(_energy(shifted), e_before, atol=1e-12), \
            f"wrap shift m={m} should preserve energy"



# Class C: never unitary once active


def test_spectral_filter_uniform_gain_preserves_norm():
    """Class C: uniform gain changes energy unless gain=1."""
    shape = gen_polygon(4)
    e_before = _energy(shape)

    # gain != 1 must change energy
    filtered = spectral_filter(shape, 0.5)
    assert not np.isclose(_energy(filtered), e_before, atol=1e-12), \
        "gain=0.5 should change energy"

    # gain = 1 is identity
    filtered = spectral_filter(shape, 1.0)
    assert np.isclose(_energy(filtered), e_before, atol=1e-12), \
        "gain=1.0 should be identity"


def test_spectral_filter_lowpass_removes_energy():
    from soc.core.constants import k_values
    """Class C: lowpass filter with K < N discards high-frequency energy."""
    shape = gen_star()
    e_before = _energy(shape)

    #filtered = spectral_filter(shape, 3, mode="low")
    weights = np.array([1.0 if abs(k) <= 3 else 0.0 for k in k_values()])
    filtered = spectral_filter(shape, weights)

    assert _energy(filtered) < e_before, \
        "lowpass filter should remove energy from high modes"

def test_spectral_filter_highpass_removes_energy():
    from soc.core.constants import k_values
    """Class C: highpass filter with K >= 0 discards low-frequency energy."""
    shape = gen_star()
    e_before = _energy(shape)

    #filtered = spectral_filter(shape, 3, mode="high")
    weights = np.array([1.0 if abs(k) > 3 else 0.0 for k in k_values()])
    filtered = spectral_filter(shape, weights)

    assert _energy(filtered) < e_before, \
        "highpass filter should remove energy from low modes"


def test_dc_boost_changes_energy():
    """Class C: DC boost changes full-norm energy unless boost=1."""
    from soc import Shape
    from soc.core.constants import k_to_index

    shape = gen_circle()
    coeffs = shape.coeffs.copy()
    coeffs[k_to_index(0)] = 0.5 + 0.2j
    shape_with_c0 = Shape(coeffs)

    e_baseline = _energy(shape_with_c0)

    boosted = dc_boost(shape_with_c0, 2.0)
    assert not np.isclose(_energy(boosted), e_baseline, atol=1e-12), \
        "dc_boost=2.0 should change full-norm energy"

    boosted = dc_boost(shape_with_c0, 1.0)
    assert np.isclose(_energy(boosted), e_baseline, atol=1e-12), \
        "dc_boost=1.0 should be identity"

def test_attenuate_removes_energy():
    """Class C: attenuation always removes energy at d > 0."""
    shape = gen_star()
    e_before = _energy(shape)

    attenuated = attenuate(shape, distance=1.0, noise_std=0.0)

    assert _energy(attenuated) < e_before, \
        "attenuation at d>0 should remove energy"


def test_attenuate_noise_injects_energy():
    """Class C: noise injection can increase energy."""
    shape = gen_circle()
    e_before = _energy(shape)

    # Run multiple times to catch the statistical effect
    for _ in range(10):
        noisy = attenuate(shape, distance=0.1, noise_std=0.5)
        assert not np.isclose(_energy(noisy), e_before, atol=1e-12), \
            "noise injection should change energy"


def test_power_clamp_caps_energy():
    """Class C: power_clamp caps energy at P_max."""
    shape = gen_star()
    # Ensure energy exceeds P_MAX
    if _energy(shape) < 40.0:
        shape = spectral_filter(shape, 10.0)

    clamped = power_clamp(shape, p_max=40.0)
    assert _energy(clamped) <= 40.0 + 1e-12, \
        "clamped energy should not exceed P_max"


def test_power_clamp_noop_when_below_max():
    """Class C: power_clamp is identity when energy < P_max."""
    shape = gen_circle()  # low energy

    clamped = power_clamp(shape, p_max=1000.0)
    assert np.allclose(shape.coeffs, clamped.coeffs, atol=1e-12), \
        "clamp below threshold should be identity"



# Reversibility class labels


def test_reversibility_class_labels():
    """Lock the class labels for every operator."""
    expected = {
        "phase_shift": "A",
        "freq_shift": "B",
        "spectral_filter": "C",
        "dc_boost": "C",
        "attenuate": "C",
        "power_clamp": "C",
    }
    for op_name, cls in expected.items():
        assert reversibility_class(op_name) == cls, \
            f"{op_name} should be Class {cls}"
