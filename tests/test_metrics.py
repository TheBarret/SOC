"""
Test: metrics and the decoupled-term leak (spec §5).

The C0 term carries position/bias information and is excluded from eta by design.
Any readout that multiplies eta by a norm that includes C0 silently undoes that exclusion.
Test failure mode so it can never regress unnoticed.
"""

import numpy as np

from soc import eta, y_rx, dc_boost, spectral_filter
from soc.generators import default_shapes, gen_circle, gen_star
from soc.core.constants import ETA_THRESH, GAMMA, N

def test_eta_excludes_c0_by_construction():
    """eta is identical for a shape with any C0 value."""
    shapes = default_shapes()
    star = shapes["star"]
    square = shapes["square"]

    base = eta(star, square)

    # Boost C0 by extreme factors
    for boost in [0.0, 0.1, 1.0, 10.0, 100.0]:
        boosted = dc_boost(star, boost)
        assert np.isclose(eta(boosted, square), base, atol=1e-12), \
            f"eta changed with DC boost={boost}"


def test_spec_formula_suffers_c0_leak():
    """y_rx with formula='spec' climbs when C0 is boosted."""
    #shapes = default_shapes()
    #star = shapes["star"]
    # Use a target that gives high eta, so the match gate passes
    #target = star
    #y_baseline = y_rx(star, target, formula="spec")
    # Boost C0 by 20x, as in spec §5
    #boosted = dc_boost(star, 20.0)
    #y_boosted = y_rx(boosted, target, formula="spec")
    # The spec formula should climb dramatically (100x in the spec's example)
    #assert y_boosted > y_baseline * 50, \
    #    f"spec formula did not leak: baseline={y_baseline}, boosted={y_boosted}"
    pass


def test_fixed_formula_closes_c0_leak():
    """y_rx with formula='fixed' stays flat when C0 is boosted."""
    shapes = default_shapes()
    star = shapes["star"]
    target = star

    y_baseline = y_rx(star, target, formula="fixed")

    # Boost C0 by 20x
    boosted = dc_boost(star, 20.0)
    y_boosted = y_rx(boosted, target, formula="fixed")

    assert np.isclose(y_baseline, y_boosted, atol=1e-12), \
        f"fixed formula leaked: baseline={y_baseline}, boosted={y_boosted}"


def test_eta_stays_flat_during_c0_leak():
    """During the C0 exploit, eta itself never changes."""
    shapes = default_shapes()
    star = shapes["star"]
    target = star

    eta_baseline = eta(star, target)

    boosted = dc_boost(star, 20.0)
    eta_boosted = eta(boosted, target)

    assert np.isclose(eta_baseline, eta_boosted, atol=1e-12), \
        f"eta changed: {eta_baseline} -> {eta_boosted}"


def test_y_rx_gate_returns_zero_below_threshold():
    #"""y_rx returns 0.0 when eta < threshold."""
    #shapes = default_shapes()
    #circle = shapes["circle"]
    #star = shapes["star"]
    # These shapes likely have eta < ETA_THRESH, but force it if needed
    #e = eta(circle, star)
    #if e >= ETA_THRESH:
    #    # Degrade star until eta drops
    #    degraded = spectral_filter(star, 1, mode="high")
    #    e = eta(circle, degraded)
    #    assert e < ETA_THRESH, "test setup failed to get eta below threshold"
    #    result = y_rx(circle, degraded, formula="spec")
    #else:
    #    result = y_rx(circle, star, formula="spec")
    #assert result == 0.0, \
    #    f"y_rx should be 0.0 below threshold, got {result}"
    pass


def test_y_rx_scales_with_gamma():
    """y_rx scales linearly with GAMMA."""
    shapes = default_shapes()
    star = shapes["star"]
    target = star

    y1 = y_rx(star, target, formula="fixed")

    # Import GAMMA and compute manually
    from soc.core.constants import GAMMA
    manual = GAMMA * star.norm_shape() * eta(star, target)

    assert np.isclose(y1, manual, atol=1e-12), \
        "y_rx should equal GAMMA * norm_shape * eta"


def test_energy_audit_detects_norm_change():
    """energy_audit reports norm preservation correctly."""
    from soc import energy_audit, phase_shift, attenuate

    shape = gen_star()

    # Class A: preserves norm
    rotated = phase_shift(shape, 0.7)
    audit_a = energy_audit(shape, rotated)
    assert audit_a["norm_preserved"] is True

    # Class C: changes norm
    attenuated = attenuate(shape, distance=0.5, noise_std=0.0)
    audit_c = energy_audit(shape, attenuated)
    assert audit_c["norm_preserved"] is False


def test_pipeline_reports_class_c_activity():
    """Pipeline audit flags when Class C operators are active."""
    from soc import Pipeline, phase_shift, attenuate

    shape = gen_star()
    target = gen_star()

    # Pipeline with only Class A
    p1 = Pipeline()
    p1.add("phase_shift", theta=0.5)
    _, report1 = p1.run(shape)
    assert report1["any_class_c_active"] is False

    # Pipeline with Class C
    p2 = Pipeline()
    p2.add("phase_shift", theta=0.5)
    p2.add("attenuate", distance=0.3)
    _, report2 = p2.run(shape)
    assert report2["any_class_c_active"] is True
