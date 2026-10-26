"""Pure math operations used by the MCP server."""


def add(a: float, b: float) -> float:
    """Return the sum of two numbers.

    Args:
        a: First addend.
        b: Second addend.

    Returns:
        The value of ``a + b``.
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """Return ``b`` subtracted from ``a``.

    Args:
        a: Starting value.
        b: Value to subtract.

    Returns:
        The value of ``a - b``.
    """
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers.

    Args:
        a: First factor.
        b: Second factor.

    Returns:
        The value of ``a * b``.
    """
    return a * b


def divide(a: float, b: float) -> float:
    """Return ``a`` divided by ``b``.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The quotient ``a / b``.

    Raises:
        ValueError: If ``b`` is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(base: float, exponent: float) -> float:
    """Return ``base`` raised to ``exponent``.

    Args:
        base: Value to raise.
        exponent: Power applied to the base.

    Returns:
        The value of ``base ** exponent``.
    """
    return base**exponent
