"""
Pipeline, ordered composition of operators with audit trail.
The pipeline is the only place where the reversibility taxonomy (spec §3)
becomes operational. Each stage records energy flow, and the final report
flags where loss occurred and whether it was structural (Class A),
boundary truncation (Class B), or degradation (Class C).
"""

import numpy as np

from .shape import Shape
from . import operators
from .metrics import energy_audit


class Pipeline:
    """
    Immutable ordered list of operator calls applied to a Shape.
    """

    def __init__(self):
        self._stages: list[dict] = []

    def add(self, op_name: str, **kwargs) -> "Pipeline":
        """
        Append an operator stage.

        Args:
            op_name: function name in operators module (e.g. "phase_shift").
            **kwargs: arguments passed to that operator.

        Returns:
            self, for chaining.
        """
        op_fn = getattr(operators, op_name, None)
        if op_fn is None:
            raise ValueError(f"unknown operator: {op_name}")

        rev_class = operators.reversibility_class(op_name)

        self._stages.append({
            "op_name": op_name,
            "op_fn": op_fn,
            "kwargs": kwargs,
            "reversibility": rev_class,
        })
        return self

    def run(self, shape: Shape) -> tuple[Shape, dict]:
        """
        Apply all stages in order to the input shape.

        Returns:
            (final_shape, audit_report)

        audit_report keys:
            stages: list of per-stage dicts with:
                op_name, kwargs, reversibility,
                energy_before, energy_after, energy_delta,
                norm_preserved
            total_energy_delta: float
            any_class_c_active: bool
            any_boundary_loss: bool (Class B with energy drop)
        """
        current = shape
        stage_reports = []
        any_class_c = False
        any_boundary_loss = False

        for stage in self._stages:
            op_fn = stage["op_fn"]
            kwargs = stage["kwargs"]

            next_shape = op_fn(current, **kwargs)
            audit = energy_audit(current, next_shape)

            report = {
                "op_name": stage["op_name"],
                "kwargs": kwargs,
                "reversibility": stage["reversibility"],
                **audit,
            }
            stage_reports.append(report)

            if stage["reversibility"] == "C":
                any_class_c = True
            if (stage["reversibility"] == "B"
                    and not audit["norm_preserved"]):
                any_boundary_loss = True

            current = next_shape

        total_delta = current.norm_full() - shape.norm_full()

        return current, {
            "stages": stage_reports,
            "total_energy_delta": total_delta,
            "any_class_c_active": any_class_c,
            "any_boundary_loss": any_boundary_loss,
        }


def run_pipeline(shape: Shape, stages: list[tuple[str, dict]]) -> tuple[Shape, dict]:
    """
    Convenience: build and run a pipeline in one call.

    Args:
        shape: input Shape.
        stages: list of (op_name, kwargs) tuples, applied in order.

    Returns:
        (final_shape, audit_report) — same as Pipeline.run().
    """
    pipe = Pipeline()
    for op_name, kwargs in stages:
        pipe.add(op_name, **kwargs)
    return pipe.run(shape)
