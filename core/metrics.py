"""
Metrics and readouts, all operate on Shape objects.
The shape subspace is k != 0. C0 is excluded from eta by design (spec §2).
Any readout that claims to measure shape match must exclude C0 explicitly,
or it silently reintroduces the decoupled-term leak (spec §5).
"""

import math

import numpy as np

from .constants import N, GAMMA, ETA_THRESH, EPSILON, k_to_index
from .shape import Shape



# Core metric

def eta(shape_a: Shape, shape_b: Shape) -> float:
    """
    Shape similarity metric (spec §2).

    eta(A, B) = |<A, B>_shape| / (||A||_shape * ||B||_shape)

    Invariant to: global phase rotation, uniform gain, C0 modification.
    Range: [0, 1]. 1.0 = identical shape geometry.
    """
    mask = np.ones(2 * N + 1, dtype=bool)
    mask[k_to_index(0)] = False

    a = shape_a.coeffs[mask]
    b = shape_b.coeffs[mask]

    norm_a = math.sqrt(np.sum(np.abs(a) ** 2))
    norm_b = math.sqrt(np.sum(np.abs(b) ** 2))
    denom = norm_a * norm_b

    if denom < EPSILON:
        return 0.0

    inner = np.sum(a * np.conj(b))
    return float(abs(inner) / denom)



# Readout

def y_rx(shape_in: Shape, target: Shape,
         formula: str = "spec", threshold: float = ETA_THRESH) -> float:
    """
    Scalar readout with match gate (spec §3).

    Args:
        shape_in: received shape S_in.
        target: reference shape T.
        formula:
            "spec"  -> GAMMA * ||S_in||^2_full * eta   (literal spec §3.4)
            "fixed" -> GAMMA * ||S_in||^2_shape * eta  (excludes C0, closes leak §5)
        threshold: eta below this returns 0.0.
    """
    e = eta(shape_in, target)
    if e < threshold:
        return 0.0

    if formula == "spec":
        norm = shape_in.norm_full()
    elif formula == "fixed":
        norm = shape_in.norm_shape()
    else:
        raise ValueError(f"formula must be 'spec' or 'fixed', got {formula}")

    return float(GAMMA * norm * e)



# Energy audit

def energy_audit(before: Shape, after: Shape) -> dict:
    """
    Track energy flow across an operator (spec §3, §4C).

    Returns:
        dict with keys:
            energy_before, energy_after, energy_delta,
            norm_preserved (bool, within EPSILON),
            loss_is_truncation (bool, True if no Class C operator was active)
    """
    e_before = before.norm_full()
    e_after = after.norm_full()
    delta = e_after - e_before

    preserved = abs(delta) < EPSILON

    return {
        "energy_before": e_before,
        "energy_after": e_after,
        "energy_delta": delta,
        "norm_preserved": preserved,
    }
