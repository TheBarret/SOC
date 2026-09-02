"""
SOC, Spectral Operator Chain
Numerical conventions and constants.

    Every convention in this file is load-bearing.
    The entire model rests on these choices being consistent across all modules.
    If you change anything here, you are changing the mathematical model, not just the code.

    References to spec sections (§) refer to the SOC specification document.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Core dimensional parameters
# ---------------------------------------------------------------------------

# Number of harmonics. Coefficient vector S has shape (2N + 1,),
# indexed k = -N .. N inclusive.
N = 8

# Length of the coefficient vector.
COEFF_LENGTH = 2 * N + 1

# Sample resolution for curve <-> coefficient transforms.
# The sampled curve z(t) is evaluated at M equally spaced points t in [0, 1).
M = 300

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def k_to_index(k: int) -> int:
    """
    Map harmonic index k (range -N .. N) to flat array index (range 0 .. 2N).

    Flat array layout:
        index 0  -> k = -N
        index 1  -> k = -N+1
        ...
        index N  -> k = 0
        ...
        index 2N -> k = N
    """
    return k + N


def index_to_k(idx: int) -> int:
    """Inverse of k_to_index."""
    return idx - N


def k_values() -> np.ndarray:
    """Return array of harmonic indices k = -N .. N."""
    return np.arange(-N, N + 1, dtype=int)


# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------

# Denominator floor for eta calculation. Prevents division-by-zero when
# either shape has zero energy in the k != 0 subspace.
EPSILON = 1e-12

# ---------------------------------------------------------------------------
# Model parameters (from spec)
# ---------------------------------------------------------------------------

# Attenuation coefficient in the exponential decay law:
#   C_{k,out} = C_{k,in} * exp(-ALPHA * |k| * d)
# Higher ALPHA = faster high-frequency degradation with distance.
ALPHA = 0.15

# Default thermal noise standard deviation (complex Gaussian, per mode).
NOISE_STD = 0.02

# Power ceiling for the clamp operator (spec §5.1).
P_MAX = 40.0

# eta threshold for the y_RX match gate (spec §3).
ETA_THRESH = 0.85

# Scalar gain in the y_RX readout formula (spec §3.4).
GAMMA = 1.0

# ---------------------------------------------------------------------------
# Numerical conventions
# ---------------------------------------------------------------------------

# 1. FFT normalization:
#    curve_to_coeffs uses np.fft.fft(z) / M, so coefficients are the normalized DFT of the sampled curve.
#    This makes the coefficient magnitudes independent of the sample count M.
#
# 2. Inner product convention:
#    <S, T> = sum_k S_k * conj(T_k)
#    This is the standard complex inner product, linear in first argument, conjugate-linear in second.
#
# 3. Coefficient dtype:
#    complex128 throughout. No exceptions.
#
# 4. Shape subspace:
#    The "shape" subspace is all k != 0. The k = 0 term (C0) carries position/bias information
#    and is excluded from eta by design (§2). Any metric or readout that claims to measure
#    "shape match" must exclude k = 0 explicitly, or it silently reintroduces the decoupled-term leak (§5).
