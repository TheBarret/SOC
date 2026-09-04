"""
Pipeline: ordered composition of operators with audit trail.
Python-side composition. Each stage calls into C operators.
"""

from constants import *
from operators import *
from metrics import energy_audit
from shape import Shape


class Pipeline:
    def __init__(self):
        self._stages = []

    def add(self, op_name: str, **kwargs) -> "Pipeline":
        op_fn = getattr(operators, op_name, None)
        if op_fn is None:
            raise ValueError(f"unknown operator: {op_name}")

        rev_class = operators.REVERSIBILITY.get(op_name, "?")

        self._stages.append({
            "op_name": op_name,
            "op_fn": op_fn,
            "kwargs": kwargs,
            "reversibility": rev_class,
        })
        return self

    def run(self, shape: Shape):
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


def run_pipeline(shape: Shape, stages: list) -> tuple:
    pipe = Pipeline()
    for op_name, kwargs in stages:
        pipe.add(op_name, **kwargs)
    return pipe.run(shape)
