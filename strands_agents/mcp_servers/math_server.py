"""Expose basic math tools over MCP Streamable HTTP."""

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_servers import math_operations
from mcp_servers.tool_cache import tool_cache


@lifespan
async def server_lifespan(_):
    """Release the Redis pool during graceful server shutdown."""
    yield None
    await tool_cache.close()


mcp = FastMCP("Math Tools", lifespan=server_lifespan)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """Return container readiness without opening an MCP session."""
    return JSONResponse({"status": "ok", "service": "math-mcp"})


@mcp.tool
async def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return await tool_cache.get_or_compute(
        "add", {"a": a, "b": b}, lambda: math_operations.add(a, b)
    )


@mcp.tool
async def subtract(a: float, b: float) -> float:
    """Subtract ``b`` from ``a`` and return the difference."""
    return await tool_cache.get_or_compute(
        "subtract", {"a": a, "b": b}, lambda: math_operations.subtract(a, b)
    )


@mcp.tool
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return their product."""
    return await tool_cache.get_or_compute(
        "multiply", {"a": a, "b": b}, lambda: math_operations.multiply(a, b)
    )


@mcp.tool
async def divide(a: float, b: float) -> float:
    """Divide ``a`` by nonzero ``b`` and return the quotient."""
    return await tool_cache.get_or_compute(
        "divide", {"a": a, "b": b}, lambda: math_operations.divide(a, b)
    )


@mcp.tool
async def power(base: float, exponent: float) -> float:
    """Raise ``base`` to ``exponent`` and return the result."""
    return await tool_cache.get_or_compute(
        "power",
        {"base": base, "exponent": exponent},
        lambda: math_operations.power(base, exponent),
    )


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
    )
