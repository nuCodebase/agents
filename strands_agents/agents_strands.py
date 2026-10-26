"""Run a hierarchical Strands agent using Ollama and specialized workers."""

import argparse
from pathlib import Path
from typing import Any

import httpx
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.loader import load_tools_from_file_path
from strands.tools.mcp import MCPClient

from agent_logging import attach_tool_logging, configure_logging
from mcp_config import load_mcp_config

OLLAMA_HOST = "http://127.0.0.1:11434"
GEOMETRY_TOOLS_PATH = Path(__file__).with_name("tools") / "geometry.py"

MATH_AGENT_PROMPT = (
    "You are the math operations specialist. Solve only arithmetic tasks. A "
    "request may contain several operations. Always use the provided MCP math "
    "tools instead of calculating mentally. When operations are independent, "
    "call all required tools together in the same model turn so they execute "
    "concurrently. Run operations sequentially only when one result is an input "
    "to another. Call each requested operation exactly once. After a tool returns "
    "status=success, treat its result as final and never call that operation "
    "again. Retry an operation only when its tool result explicitly has "
    "status=error. As soon as every requested operation has a successful result, "
    "immediately return a concise answer without making another tool call. Return "
    "every result with an unambiguous operation label."
)
GEOMETRY_AGENT_PROMPT = (
    "You are the 3D operations specialist. Solve only three-dimensional point "
    "tasks. A request may contain several operations. Always use the provided "
    "geometry tools instead of calculating mentally. When operations are "
    "independent, call all required tools together in the same model turn so "
    "they execute concurrently. Run operations sequentially only when one "
    "result feeds another. Call each requested operation exactly once. After a "
    "tool returns status=success, treat its result as final and never call that "
    "operation again. Retry an operation only when its tool result explicitly "
    "has status=error. As soon as every requested operation has a successful "
    "result, immediately return a concise answer without making another tool "
    "call. Return every result with coordinates in [x, y, z] form."
)
SUPERVISOR_PROMPT = (
    "You are the high-level supervisor. Identify the tasks in each request and "
    "delegate arithmetic work to math_operations_agent and 3D point work to "
    "geometry_operations_agent. You have no operation tools of your own. Bundle "
    "all independent arithmetic tasks into ONE math_operations_agent call, and "
    "bundle all independent 3D tasks into ONE geometry_operations_agent call. "
    "Never call the same worker more than once concurrently because a worker "
    "instance accepts only one invocation at a time. When both batches are "
    "independent, call the math and geometry workers together in the same model "
    "turn so they run concurrently. Use additional worker calls only for genuine "
    "dependencies where an earlier result is required. Combine the returned "
    "results into one clear final answer. Never perform operations yourself."
)


def ensure_ollama_ready(model_id: str) -> None:
    """Verify that Ollama is reachable and has the requested model.

    Args:
        model_id: Ollama model tag, such as ``qwen3:4b``.

    Raises:
        SystemExit: If Ollama is unreachable or the model has not been pulled.
    """
    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise SystemExit(
            f"Cannot connect to Ollama at {OLLAMA_HOST}.\n"
            "Install and start Ollama, then run:\n"
            f"  ollama pull {model_id}"
        ) from error

    installed = {item["name"] for item in response.json().get("models", [])}
    if model_id not in installed and f"{model_id}:latest" not in installed:
        raise SystemExit(
            f"Ollama is running, but model {model_id!r} is not installed.\n"
            f"Run: ollama pull {model_id}"
        )


def create_math_client() -> MCPClient:
    """Create the Strands MCP client described by ``mcp.json``.

    Returns:
        The configured math-server client. The caller must start/stop it or use
        it as a context manager.
    """
    clients = MCPClient.load_servers(load_mcp_config())
    return clients[0]


def create_supervisor_agent(
    model: Any,
    math_tools: list[Any],
    geometry_tools: list[Any],
    *,
    preserve_worker_context: bool = False,
) -> Agent:
    """Build a supervisor with isolated math and 3D worker agents.

    The supervisor receives only agent-as-tool wrappers. Consequently it cannot
    bypass delegation and invoke an operation tool directly.
    """
    math_agent = Agent(
        name="math-operations-agent",
        description="Handles arithmetic using the math_operations MCP tools.",
        model=model,
        tools=math_tools,
        system_prompt=MATH_AGENT_PROMPT,
        callback_handler=None,
    )
    geometry_agent = Agent(
        name="geometry-operations-agent",
        description="Handles addition and subtraction of three-dimensional points.",
        model=model,
        tools=geometry_tools,
        system_prompt=GEOMETRY_AGENT_PROMPT,
        callback_handler=None,
    )
    supervisor = Agent(
        name="operations-supervisor",
        description="Routes requests to specialized math and 3D operation agents.",
        model=model,
        tools=[
            math_agent.as_tool(
                name="math_operations_agent",
                description=(
                    "Solve one or many arithmetic tasks as a single batch using "
                    "MCP math tools. Put all independent arithmetic operations "
                    "into this one call."
                ),
                preserve_context=preserve_worker_context,
            ),
            geometry_agent.as_tool(
                name="geometry_operations_agent",
                description=(
                    "Solve one or many 3D point tasks as a single batch using "
                    "geometry tools. Put all independent 3D operations into "
                    "this one call."
                ),
                preserve_context=preserve_worker_context,
            ),
        ],
        system_prompt=SUPERVISOR_PROMPT,
        callback_handler=None,
    )
    for agent in (math_agent, geometry_agent, supervisor):
        attach_tool_logging(agent)
    return supervisor


def main() -> None:
    """Run one CLI prompt through a locally hosted Strands agent.

    The function verifies Ollama, opens the MCP subprocess, combines its math
    tools with tools auto-discovered from ``tools/``, attaches lifecycle
    logging, invokes the agent, and closes MCP during context-manager cleanup.
    """
    log_path = configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="*",
        default=["What is (27 + 15) multiplied by 3?"],
        help="Question to ask the Strands agent",
    )
    parser.add_argument(
        "--model",
        default="qwen3:4b",
        help="Local Ollama model (default: qwen3:4b)",
    )
    args = parser.parse_args()

    ensure_ollama_ready(args.model)
    model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=args.model,
        temperature=0,
    )

    math_client = create_math_client()
    with math_client:
        math_tools = math_client.list_tools_sync()
        geometry_tools = load_tools_from_file_path(str(GEOMETRY_TOOLS_PATH))
        supervisor = create_supervisor_agent(
            model, math_tools, geometry_tools
        )
        print(f"Detailed logs: {log_path}")
        supervisor(" ".join(args.prompt))


if __name__ == "__main__":
    main()
