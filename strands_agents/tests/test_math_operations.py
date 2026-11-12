import pytest

from mcp_servers import math_operations


def test_operations() -> None:
    assert math_operations.add(2, 3) == 5
    assert math_operations.subtract(10, 4) == 6
    assert math_operations.multiply(6, 7) == 42
    assert math_operations.divide(8, 2) == 4
    assert math_operations.power(2, 8) == 256


def test_divide_by_zero() -> None:
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        math_operations.divide(1, 0)

