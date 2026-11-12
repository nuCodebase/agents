"""Interactive Streamlit client for the Strands agent FastAPI service."""

import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

from agent_logging import LOG_PATH, configure_logging

DEFAULT_SUPERVISOR_MODEL = "qwen3:1.7b"
DEFAULT_WORKER_MODEL = "qwen3:0.6b"
DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL = os.getenv("AGENT_API_URL", DEFAULT_API_URL).rstrip("/")
API_TIMEOUT = httpx.Timeout(300.0, connect=5.0)

configure_logging()


def api_request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Call the agent API and turn transport failures into readable errors."""
    try:
        response = httpx.request(
            method,
            f"{API_URL}{path}",
            timeout=API_TIMEOUT,
            **kwargs,
        )
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        raise RuntimeError(
            f"Agent API returned {error.response.status_code}: {detail}"
        ) from error
    except httpx.RequestError as error:
        raise RuntimeError(
            f"Could not reach the agent API at {API_URL}. "
            "Start the containers with `docker compose up --build -d`."
        ) from error


def api_headers() -> dict[str, str]:
    """Return the ownership header for this browser session."""
    return {"X-User-ID": st.session_state.user_id}


def create_conversation(supervisor_model_id: str, worker_model_id: str) -> str:
    """Create an API-owned conversation and return its identifier."""
    response = api_request(
        "POST",
        "/v1/conversations",
        headers=api_headers(),
        json={
            "supervisor_model": supervisor_model_id,
            "worker_model": worker_model_id,
        },
    )
    return str(response.json()["conversation_id"])


def delete_conversation() -> None:
    """Delete the current API conversation, if one exists."""
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        api_request(
            "DELETE",
            f"/v1/conversations/{conversation_id}",
            headers=api_headers(),
        )


def reset_conversation() -> None:
    """Delete API state and clear the locally rendered transcript."""
    delete_conversation()
    st.session_state.pop("conversation_id", None)
    st.session_state.pop("conversation_models", None)
    st.session_state.messages = []


def stream_message(
    message: str, supervisor_model_id: str, worker_model_id: str
) -> Iterator[str]:
    """Yield assistant text chunks from the FastAPI streaming endpoint."""
    conversation_id = st.session_state.get("conversation_id")
    if not conversation_id:
        conversation_id = create_conversation(supervisor_model_id, worker_model_id)
        st.session_state.conversation_id = conversation_id
        st.session_state.conversation_models = (
            supervisor_model_id,
            worker_model_id,
        )

    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/conversations/{conversation_id}/messages/stream",
            headers=api_headers(),
            json={"message": message},
            timeout=API_TIMEOUT,
        ) as response:
            if response.is_error:
                detail = response.read().decode(errors="replace")
                raise RuntimeError(
                    f"Agent API returned {response.status_code}: {detail}"
                )
            received_delta = False
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError as error:
                    raise RuntimeError("Agent API returned an invalid stream event") from error
                if event["type"] == "delta":
                    received_delta = True
                    yield str(event["text"])
                elif event["type"] == "complete" and not received_delta:
                    yield str(event["response"])
                elif event["type"] == "error":
                    raise RuntimeError(f"Agent stream failed: {event['detail']}")
    except httpx.RequestError as error:
        raise RuntimeError(
            f"The response stream from {API_URL} was interrupted."
        ) from error


def render_message(message: dict[str, Any]) -> None:
    """Render one cached user or assistant message."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


st.set_page_config(page_title="Math MCP Chat", page_icon="🧮")
st.title("🧮 Math MCP Chat")
st.caption(f"Streamlit → FastAPI ({API_URL}) → Strands + Ollama + FastMCP")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "supervisor_model_id" not in st.session_state:
    st.session_state.supervisor_model_id = DEFAULT_SUPERVISOR_MODEL
if "worker_model_id" not in st.session_state:
    st.session_state.worker_model_id = DEFAULT_WORKER_MODEL

with st.sidebar:
    st.header("Settings")
    supervisor_model_id = st.text_input(
        "Supervisor model", key="supervisor_model_id"
    )
    worker_model_id = st.text_input(
        "Math and geometry model", key="worker_model_id"
    )
    if st.button("New conversation", use_container_width=True):
        try:
            reset_conversation()
        except RuntimeError as error:
            # Still allow a local reset if the API process is unavailable or
            # has already lost its in-memory conversation.
            st.session_state.pop("conversation_id", None)
            st.session_state.pop("conversation_models", None)
            st.session_state.messages = []
            st.warning(str(error))
        st.rerun()
    st.caption("The Docker services and Ollama must be running.")
    show_logs = st.checkbox("Show recent logs", value=False)

selected_models = (supervisor_model_id, worker_model_id)
active_models = st.session_state.get("conversation_models")
if active_models is not None and active_models != selected_models:
    try:
        reset_conversation()
    except RuntimeError as error:
        st.session_state.pop("conversation_id", None)
        st.session_state.pop("conversation_models", None)
        st.session_state.messages = []
        st.warning(str(error))

for saved_message in st.session_state.messages:
    render_message(saved_message)

if prompt := st.chat_input("Ask a math question"):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                stream_message(prompt, supervisor_model_id, worker_model_id)
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        except RuntimeError as error:
            st.error(str(error))

if show_logs:
    with st.sidebar.expander("Strands and tool-call logs", expanded=True):
        if LOG_PATH.exists():
            recent_lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-150:]
            st.code("\n".join(recent_lines), language="text")
        else:
            st.caption("No logs have been written yet.")
