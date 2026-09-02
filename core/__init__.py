"""
SOC core: Spectral Operator Chain.

Public API:
    Shape, Pipeline
    operators: phase_shift, freq_shift, spectral_filter, dc_boost,
               attenuate, power_clamp
    metrics: eta, y_rx, energy_audit
"""

from .shape import Shape
from .pipeline import Pipeline, run_pipeline

from .operators import (
    phase_shift,
    freq_shift,
    spectral_filter,
    dc_boost,
    attenuate,
    power_clamp,
    reversibility_class,
)

from .metrics import (
    eta,
    y_rx,
    energy_audit,
)

__all__ = [
    # objects
    "Shape",
    "Pipeline",
    # pipeline convenience
    "run_pipeline",
    # operators
    "phase_shift",
    "freq_shift",
    "spectral_filter",
    "dc_boost",
    "attenuate",
    "power_clamp",
    "reversibility_class",
    # metrics
    "eta",
    "y_rx",
    "energy_audit",
]
