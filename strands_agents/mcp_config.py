"""Load the shared MCP server configuration."""

import json
import os
import sys
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("mcp.json").resolve()


def load_mcp_config() -> dict[str, Any]:
    """Load the shared MCP configuration and resolve runtime paths.

    Returns:
        Parsed MCP configuration with the active Python executable and absolute
        ``math_server.py`` path substituted into the JSON placeholders.

    Raises:
        FileNotFoundError: If ``mcp.json`` does not exist beside this module.
        json.JSONDecodeError: If the configuration is not valid JSON.
    """
    values = {
        "MCP_PYTHON": sys.executable,
        "MCP_SERVER_PATH": str(CONFIG_PATH.with_name("math_server.py")),
    }

    def resolve(value: Any) -> Any:
        """Recursively replace runtime placeholders in a parsed JSON value.

        Args:
            value: String, list, dictionary, or scalar from ``mcp.json``.

        Returns:
            A value of the same shape with supported placeholders replaced.
        """
        if isinstance(value, str):
            for name, replacement in values.items():
                value = value.replace(f"${{{name}}}", replacement)
            return os.path.expanduser(value)
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        return value

    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return resolve(json.load(config_file))
