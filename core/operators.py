"""
Operators
* Operators on Shape objects.
* Each returns a new Shape, never mutates.

Reversibility classes (spec §3):
    A: always unitary (norm-preserving, invertible)
    B: conditionally unitary (lossless iff no energy crosses the +-N boundary)
    C: never unitary once active (discards or injects energy)
"""

import numpy as np

from .constants import N, ALPHA, NOISE_STD, P_MAX, k_to_index, k_values
from .shape import Shape



# Class A: always unitary


def phase_shift(shape: Shape, theta: float) -> Shape:
    """
    R_theta: C_{k,out} = C_{k,in} * e^{i*theta}.
    Diagonal, unit-modulus, invertible by phase_shift(..., -theta).

    Reversibility class: A
    """
    return Shape(shape.coeffs * np.exp(1j * theta),
                 provenance=f"{shape.provenance} | phase_shift({theta:.4f})")



# Class B: conditionally unitary


def freq_shift(shape: Shape, m: int, mode: str = "truncate") -> Shape:
    """
    H_m: C_{k,out} = C_{k-m,in}.

    Two boundary behaviors (spec §7 open parameter):
        truncate: energy shifted past +-N is discarded (lossy).
        wrap:     energy shifted past +-N wraps around (lossless, circular).

    Reversibility class: B (unitary iff no energy crosses +-N in truncate mode,
                          always unitary in wrap mode).
    """
    if mode not in ("truncate", "wrap"):
        raise ValueError(f"mode must be 'truncate' or 'wrap', got {mode}")

    coeffs = np.zeros(2 * N + 1, dtype=np.complex128)
    for idx, k in enumerate(k_values()):
        src_k = k - m
        if mode == "truncate":
            if -N <= src_k <= N:
                coeffs[idx] = shape.coeffs[k_to_index(src_k)]
        else:  # wrap
            src_k = ((src_k + N) % (2 * N + 1)) - N
            coeffs[idx] = shape.coeffs[k_to_index(src_k)]

    return Shape(coeffs,
                 provenance=f"{shape.provenance} | freq_shift(m={m}, mode={mode})")



# Class C: never unitary (once active)


def spectral_filter(shape: Shape, weights: np.ndarray | list | float,
                    mode: str = "custom") -> Shape:
    """
    W: C_{k,out} = w_k * C_{k,in}.

    Args:
        weights:
            float -> uniform gain applied to all modes.
            array/list of length 2N+1 -> per-mode real weights.
        mode:
            "custom"  -> use weights as given.
            "low"     -> weights interpreted as cutoff K: keep |k| <= K.
            "high"    -> weights interpreted as cutoff K: keep |k| > K.

    Reversibility class: C (once any w_k != 1).
    """
    if isinstance(weights, (int, float)):
        w = np.full(2 * N + 1, float(weights), dtype=np.float64)
        mode_str = f"uniform({weights})"
    elif mode == "low":
        cutoff = int(weights)
        w = np.array([1.0 if abs(k) <= cutoff else 0.0 for k in k_values()])
        mode_str = f"lowpass(K={cutoff})"
    elif mode == "high":
        cutoff = int(weights)
        w = np.array([1.0 if abs(k) > cutoff else 0.0 for k in k_values()])
        mode_str = f"highpass(K={cutoff})"
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (2 * N + 1,):
            raise ValueError(f"weights must have shape ({2*N+1},), got {w.shape}")
        mode_str = "custom"

    return Shape(shape.coeffs * w,
                 provenance=f"{shape.provenance} | filter({mode_str})")


def dc_boost(shape: Shape, boost: float) -> Shape:
    """
    Scale only the C0 term by boost factor.
    Test probe for the decoupled-term leak (spec §5).

    Reversibility class: C (once boost != 1.0).
    """
    coeffs = shape.coeffs.copy()
    coeffs[k_to_index(0)] *= boost
    return Shape(coeffs,
                 provenance=f"{shape.provenance} | dc_boost({boost})")


def attenuate(shape: Shape, distance: float, alpha: float = ALPHA,
              noise_std: float = 0.0) -> Shape:
    """
    Exponential frequency-dependent attenuation + optional additive noise
    (spec §5.2).

    C_{k,out} = C_{k,in} * exp(-alpha * |k| * distance) + noise

    Reversibility class: C (attenuation always lossy; noise injects new DOF).
    """
    ks = k_values()
    atten = np.exp(-alpha * np.abs(ks) * distance)
    coeffs = shape.coeffs * atten

    if noise_std > 0:
        noise = (np.random.normal(0, noise_std, size=coeffs.shape)
                 + 1j * np.random.normal(0, noise_std, size=coeffs.shape))
        coeffs = coeffs + noise

    prov = f"{shape.provenance} | attenuate(d={distance:.2f}"
    if noise_std > 0:
        prov += f", noise={noise_std}"
    prov += ")"

    return Shape(coeffs, provenance=prov)


def power_clamp(shape: Shape, p_max: float = P_MAX) -> Shape:
    """
    Scale down uniformly if ||S||^2 exceeds p_max (spec §5.1).

    Reversibility class: C (lossy only when active; identity otherwise).
    """
    p = shape.norm_full()
    if p > p_max:
        scale = np.sqrt(p_max / p)
        return Shape(shape.coeffs * scale,
                     provenance=f"{shape.provenance} | clamp(P_max={p_max})")
    return Shape(shape.coeffs.copy(),
                 provenance=f"{shape.provenance} | clamp(no-op)")



# Reversibility lookup

REVERSIBILITY = {
    "phase_shift": "A",
    "freq_shift": "B",
    "spectral_filter": "C",
    "dc_boost": "C",
    "attenuate": "C",
    "power_clamp": "C",
}


def reversibility_class(op_name: str) -> str:
    """Return the reversibility class ('A', 'B', or 'C') for an operator."""
    return REVERSIBILITY.get(op_name, "?")
