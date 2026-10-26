import pytest

from tools.geometry import add_3d_points, subtract_3d_points


def test_add_3d_points() -> None:
    assert add_3d_points([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_subtract_3d_points() -> None:
    assert subtract_3d_points([5, 7, 9], [1, 2, 3]) == [4, 5, 6]


def test_point_requires_three_coordinates() -> None:
    with pytest.raises(ValueError, match="exactly three coordinates"):
        add_3d_points([1, 2], [3, 4, 5])
