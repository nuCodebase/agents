import pytest
from fastmcp import Client

from math_server import mcp


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_fastmcp_tools_in_memory() -> None:
    """Discover and invoke a tool entirely through FastMCP."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
        }

        result = await client.call_tool("multiply", {"a": 6, "b": 7})
        assert result.data == 42
