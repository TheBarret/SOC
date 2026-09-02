"""
SOC io: serialization utilities.
"""

from .serialization import (
    shape_to_json,
    shape_from_json,
    save_shape,
    load_shape,
    save_shape_set,
    load_shape_set,
)

__all__ = [
    "shape_to_json",
    "shape_from_json",
    "save_shape",
    "load_shape",
    "save_shape_set",
    "load_shape_set",
]
