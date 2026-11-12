"""Load the network configuration for the standalone math MCP service."""

import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("mcp.json")
DEFAULT_MATH_MCP_URL = "http://127.0.0.1:8001/mcp"


def load_mcp_config() -> dict[str, Any]:
    """Load MCP configuration using ``MATH_MCP_URL`` when provided."""
    math_mcp_url = os.getenv("MATH_MCP_URL", DEFAULT_MATH_MCP_URL)
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    for server in config["mcpServers"].values():
        server["url"] = server["url"].replace("${MATH_MCP_URL}", math_mcp_url)
    return config
