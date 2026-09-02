"""
Test: eta invariants (spec §2).

Consequences of the definitions, spec confirms them analytically and numerically to floating-point precision.

eta is invariant to:
    - global phase rotation: S -> S * e^{i*theta}
    - uniform gain: S -> g * S, g > 0
    - C0 modification: S -> S with only k=0 term changed
"""

import numpy as np

from soc import eta, phase_shift, spectral_filter, dc_boost
from soc.generators import gen_circle, gen_polygon, gen_star, default_shapes


def _all_pairs():
    """Return (shape_a, shape_b) for several meaningful pairs."""
    shapes = default_shapes()
    pairs = []
    for name_a, shape_a in shapes.items():
        for name_b, shape_b in shapes.items():
            if name_a != name_b:
                pairs.append((shape_a, shape_b, name_a, name_b))
    return pairs


def test_eta_invariant_to_phase_rotation():
    """eta(S, T) == eta(phase_shift(S, theta), T) for all theta."""
    shapes = default_shapes()
    star = shapes["star"]
    square = shapes["square"]

    base = eta(star, square)
    for theta in np.linspace(0, 2 * np.pi, 100):
        rotated = phase_shift(star, theta)
        assert np.isclose(eta(rotated, square), base, atol=1e-12), \
            f"phase rotation theta={theta} broke invariance"


def test_eta_invariant_to_uniform_gain():
    """eta(S, T) == eta(spectral_filter(S, g), T) for all g > 0."""
    shapes = default_shapes()
    star = shapes["star"]
    triangle = shapes["triangle"]

    base = eta(star, triangle)
    for g in np.linspace(0.1, 7.5, 50):
        scaled = spectral_filter(star, float(g))
        assert np.isclose(eta(scaled, triangle), base, atol=1e-12), \
            f"uniform gain g={g} broke invariance"


def test_eta_invariant_to_c0_modification():
    """eta(S, T) == eta(dc_boost(S, boost), T) for any boost factor."""
    shapes = default_shapes()
    circle = shapes["circle"]
    square = shapes["square"]

    base = eta(circle, square)
    for boost in [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]:
        boosted = dc_boost(circle, boost)
        assert np.isclose(eta(boosted, square), base, atol=1e-12), \
            f"DC boost={boost} broke invariance"


def test_eta_self_match_is_one():
    """eta(S, S) == 1.0 for any shape with nonzero shape-energy."""
    for shape in default_shapes().values():
        assert np.isclose(eta(shape, shape), 1.0, atol=1e-12), \
            "eta(S, S) should be exactly 1.0"


def test_eta_range_zero_to_one():
    """eta is always in [0, 1]."""
    for shape_a, shape_b, _, _ in _all_pairs():
        e = eta(shape_a, shape_b)
        assert 0.0 <= e <= 1.0, \
            f"eta out of range: {e}"


def test_eta_symmetric_up_to_phase():
    """eta(A, B) == eta(B, A) — absolute inner product is symmetric."""
    for shape_a, shape_b, _, _ in _all_pairs():
        assert np.isclose(eta(shape_a, shape_b), eta(shape_b, shape_a),
                          atol=1e-12), \
            "eta should be symmetric"
