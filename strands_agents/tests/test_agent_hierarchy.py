from pathlib import Path

from strands import tool
from strands.models.ollama import OllamaModel
from strands.tools.loader import load_tools_from_file_path

from agents_strands import (
    GEOMETRY_AGENT_PROMPT,
    MATH_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
    create_supervisor_agent,
)


@tool
def fake_math_tool(a: float, b: float) -> float:
    """Stand-in for one MCP math operation."""
    return a + b


def test_supervisor_exposes_workers_not_operation_tools() -> None:
    model = OllamaModel(host="http://127.0.0.1:11434", model_id="test-model")
    geometry_path = Path(__file__).parents[1] / "tools" / "geometry.py"
    geometry_tools = load_tools_from_file_path(str(geometry_path))

    supervisor = create_supervisor_agent(model, [fake_math_tool], geometry_tools)

    assert set(supervisor.tool_names) == {
        "math_operations_agent",
        "geometry_operations_agent",
    }
    math_worker = supervisor.tool_registry.registry["math_operations_agent"].agent
    geometry_worker = supervisor.tool_registry.registry[
        "geometry_operations_agent"
    ].agent
    assert math_worker.tool_names == ["fake_math_tool"]
    assert set(geometry_worker.tool_names) == {
        "add_3d_points",
        "subtract_3d_points",
    }


def test_prompts_require_safe_parallel_batches() -> None:
    assert "ONE math_operations_agent call" in SUPERVISOR_PROMPT
    assert "ONE geometry_operations_agent call" in SUPERVISOR_PROMPT
    assert "same model turn" in MATH_AGENT_PROMPT
    assert "same model turn" in GEOMETRY_AGENT_PROMPT
    assert "exactly once" in MATH_AGENT_PROMPT
    assert "exactly once" in GEOMETRY_AGENT_PROMPT
    assert "status=error" in MATH_AGENT_PROMPT
    assert "status=error" in GEOMETRY_AGENT_PROMPT
    assert "without making another tool call" in MATH_AGENT_PROMPT
    assert "without making another tool call" in GEOMETRY_AGENT_PROMPT
