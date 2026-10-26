"""MCP server that exposes basic math tools over standard input/output."""

from fastmcp import FastMCP

import math_operations

mcp = FastMCP("Math Tools")


@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers and return their sum.

    Args:
        a: First addend.
        b: Second addend.
    """
    return math_operations.add(a, b)


@mcp.tool
def subtract(a: float, b: float) -> float:
    """Subtract ``b`` from ``a`` and return the difference.

    Args:
        a: Starting value.
        b: Value to subtract.
    """
    return math_operations.subtract(a, b)


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return their product.

    Args:
        a: First factor.
        b: Second factor.
    """
    return math_operations.multiply(a, b)


@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide ``a`` by nonzero ``b`` and return the quotient.

    Args:
        a: Dividend.
        b: Nonzero divisor.

    Raises:
        ValueError: If ``b`` is zero.
    """
    return math_operations.divide(a, b)


@mcp.tool
def power(base: float, exponent: float) -> float:
    """Raise ``base`` to ``exponent`` and return the result.

    Args:
        base: Value to raise.
        exponent: Power applied to the base.
    """
    return math_operations.power(base, exponent)


if __name__ == "__main__":
    # stdio is ideal for a local MCP server launched by an agent.
    # Never print to stdout here because stdout carries MCP messages.
    mcp.run()
