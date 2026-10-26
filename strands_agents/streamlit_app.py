"""Interactive Streamlit chat for the Strands + FastMCP math agent."""

import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.loader import load_tools_from_file_path

from agent_logging import LOG_PATH, configure_logging, request_log_context
from agents_strands import (
    OLLAMA_HOST,
    create_math_client,
    create_supervisor_agent,
    ensure_ollama_ready,
)

DEFAULT_MODEL = "qwen3:4b"
GEOMETRY_TOOLS_PATH = Path(__file__).with_name("tools") / "geometry.py"

configure_logging()


def close_mcp_client() -> None:
    """Stop and forget this browser session's MCP subprocess.

    The operation is safe when the session has not created a client. Strands'
    ``stop`` method receives empty exception details because this is normal
    application cleanup rather than exception unwinding.
    """
    client = st.session_state.pop("mcp_client", None)
    if client is not None:
        client.stop(None, None, None)


def reset_conversation() -> None:
    """Clear the visible transcript and all agent-side conversation state.

    Closing the MCP client first prevents the old subprocess from surviving a
    model change or explicit **New conversation** action.
    """
    close_mcp_client()
    st.session_state.pop("agent", None)
    st.session_state.pop("agent_model", None)
    st.session_state.messages = []


def get_agent(model_id: str) -> Agent:
    """Return the stateful Strands agent for this browser session.

    Args:
        model_id: Installed Ollama model tag selected in the sidebar.

    Returns:
        Existing agent when the model is unchanged, otherwise a newly created
        agent connected to the FastMCP math server and native geometry tools.

    Raises:
        SystemExit: If Ollama or the selected model is unavailable.
        Exception: If MCP startup, tool discovery, or agent construction fails.
    """
    existing = st.session_state.get("agent")
    if existing is not None and st.session_state.get("agent_model") == model_id:
        return existing

    reset_conversation()
    ensure_ollama_ready(model_id)

    # Keep the client in session state for the entire conversation. Concrete
    # MCP tool wrappers require its stdio session to remain open.
    mcp_client = create_math_client()
    mcp_client.start()
    try:
        math_tools = mcp_client.list_tools_sync()
        model = OllamaModel(
            host=OLLAMA_HOST,
            model_id=model_id,
            temperature=0,
        )
        geometry_tools = load_tools_from_file_path(str(GEOMETRY_TOOLS_PATH))
        agent = create_supervisor_agent(
            model,
            math_tools,
            geometry_tools,
            preserve_worker_context=True,
        )
    except Exception:
        mcp_client.stop(None, None, None)
        raise

    st.session_state.mcp_client = mcp_client
    st.session_state.agent = agent
    st.session_state.agent_model = model_id
    return agent


def render_message(message: dict[str, Any]) -> None:
    """Render one stored user or assistant message.

    Args:
        message: Dictionary containing Streamlit-supported ``role`` and Markdown
        ``content`` values.
    """
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


st.set_page_config(page_title="Math MCP Chat", page_icon="🧮")
st.title("🧮 Math MCP Chat")
st.caption("Strands + Ollama + FastMCP")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "logging_session_id" not in st.session_state:
    st.session_state.logging_session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Settings")
    model_id = st.text_input("Ollama model", value=DEFAULT_MODEL)
    if st.button("New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()
    st.caption("The Ollama service must be running and the model must be pulled.")
    show_logs = st.checkbox("Show recent logs", value=False)

for saved_message in st.session_state.messages:
    render_message(saved_message)

if prompt := st.chat_input("Ask a math question"):
    request_id = str(uuid.uuid4())
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Working..."):
                with request_log_context(
                    st.session_state.logging_session_id,
                    request_id,
                ):
                    response = str(get_agent(model_id)(prompt))
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        except SystemExit as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"The agent could not answer: {error}")

if show_logs:
    with st.sidebar.expander("Strands and tool-call logs", expanded=True):
        if LOG_PATH.exists():
            recent_lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-150:]
            st.code("\n".join(recent_lines), language="text")
        else:
            st.caption("No logs have been written yet.")
