"""Run a credential-free LangChain agent with the local FastMCP tools."""

import argparse
import asyncio
import os

import httpx
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from mcp_config import load_mcp_config


async def ensure_ollama_ready(model: str) -> None:
    """Verify asynchronously that Ollama and a requested model are available.

    Args:
        model: Ollama model tag, such as ``qwen3:4b``.

    Raises:
        SystemExit: If the local service is unreachable or lacks the model.
    """
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get("http://127.0.0.1:11434/api/tags")
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise SystemExit(
            "Cannot connect to Ollama at http://127.0.0.1:11434.\n"
            "Install and start Ollama, then run:\n"
            f"  ollama pull {model}"
        ) from error

    installed = {item["name"] for item in response.json().get("models", [])}
    if model not in installed and f"{model}:latest" not in installed:
        raise SystemExit(
            f"Ollama is running, but model {model!r} is not installed.\n"
            f"Run: ollama pull {model}"
        )


async def main() -> None:
    """Build the LangChain MCP agent and execute one command-line prompt.

    The MCP adapter launches ``math_server.py`` using the shared configuration,
    converts discovered MCP tools to LangChain tools, and sends the prompt to a
    deterministic local Ollama chat model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="*",
        default=["What is (27 + 15) multiplied by 3?"],
        help="Question to ask the agent",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        help="Local Ollama model (default: OLLAMA_MODEL or qwen3:4b)",
    )
    args = parser.parse_args()
    await ensure_ollama_ready(args.model)

    # LangChain's adapter launches our FastMCP server over local stdio and
    # converts its discovered MCP tools into LangChain tools.
    config = load_mcp_config()
    mcp_client = MultiServerMCPClient(config["mcpServers"])
    tools = await mcp_client.get_tools()

    print("All available tools: ", [tool.name for tool in tools])

    agent = create_agent(
        model=ChatOllama(model=args.model, temperature=0),
        tools=tools,
        system_prompt=(
            "You are a careful math assistant. Use the provided math tools "
            "for arithmetic instead of calculating mentally."
        ),
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": " ".join(args.prompt)}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
