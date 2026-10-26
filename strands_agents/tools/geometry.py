"""Strands-native tools for basic three-dimensional point operations."""

from strands import tool


def _validate_point(point: list[float], name: str) -> None:
    """Verify that a tool argument represents exactly one 3D point.

    Args:
        point: Coordinate sequence to validate.
        name: Argument name included in validation errors.

    Raises:
        ValueError: If the sequence does not contain exactly three coordinates.
    """
    if len(point) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates [x, y, z]")


@tool
def add_3d_points(point_a: list[float], point_b: list[float]) -> list[float]:
    """Add two 3D points component by component.

    Args:
        point_a: First point in [x, y, z] form.
        point_b: Second point in [x, y, z] form.

    Returns:
        The component-wise sum in [x, y, z] form.
    """
    _validate_point(point_a, "point_a")
    _validate_point(point_b, "point_b")
    return [a + b for a, b in zip(point_a, point_b)]


@tool
def subtract_3d_points(point_a: list[float], point_b: list[float]) -> list[float]:
    """Subtract one 3D point from another component by component.

    Args:
        point_a: Point to subtract from, in [x, y, z] form.
        point_b: Point to subtract, in [x, y, z] form.

    Returns:
        The component-wise difference point_a - point_b in [x, y, z] form.
    """
    _validate_point(point_a, "point_a")
    _validate_point(point_b, "point_b")
    return [a - b for a, b in zip(point_a, point_b)]
