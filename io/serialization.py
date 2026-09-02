"""
Serialization: JSON save/load for Shape objects.

Uses the Shape.to_dict() / Shape.from_dict() methods, which are already
JSON-friendly. This module adds file-level convenience.
"""

import json

from ..core.shape import Shape


def shape_to_json(shape: Shape, indent: int = 2) -> str:
    """Serialize a Shape to a JSON string."""
    return json.dumps(shape.to_dict(), indent=indent)


def shape_from_json(json_str: str) -> Shape:
    """Deserialize a Shape from a JSON string."""
    d = json.loads(json_str)
    return Shape.from_dict(d)


def save_shape(shape: Shape, filepath: str) -> None:
    """Write a Shape to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(shape_to_json(shape))


def load_shape(filepath: str) -> Shape:
    """Read a Shape from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return shape_from_json(f.read())


def save_shape_set(shapes: dict[str, Shape], filepath: str) -> None:
    """Write a dict of named Shapes to a single JSON file."""
    payload = {name: shape.to_dict() for name, shape in shapes.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_shape_set(filepath: str) -> dict[str, Shape]:
    """Read a dict of named Shapes from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {name: Shape.from_dict(d) for name, d in payload.items()}
