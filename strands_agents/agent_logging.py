"""Full lifecycle logging shared by the Strands applications."""

import json
import logging
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from strands import Agent
from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    AfterToolsEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    BeforeToolsEvent,
    MessageAddedEvent,
)

LOG_PATH = Path(__file__).with_name("logs") / "strands.log"
FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "session_id=%(session_id)s | request_id=%(request_id)s | %(message)s"
)
SESSION_ID: ContextVar[str] = ContextVar("log_session_id", default="-")
REQUEST_ID: ContextVar[str] = ContextVar("log_request_id", default="-")


class _RequestContextFilter(logging.Filter):
    """Add the active Streamlit request identifiers to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = SESSION_ID.get()
        record.request_id = REQUEST_ID.get()
        return True


@contextmanager
def request_log_context(session_id: str, request_id: str) -> Iterator[None]:
    """Make identifiers available to supervisor, worker, and tool logs."""
    session_token = SESSION_ID.set(session_id)
    request_token = REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        REQUEST_ID.reset(request_token)
        SESSION_ID.reset(session_token)


def _logger_name(agent: Agent) -> str:
    """Return a readable, logger-safe lifecycle namespace for an agent."""
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", agent.name).strip("-")
    return f"agent.lifecycle.{name or 'unnamed'}"


def configure_logging() -> Path:
    """Enable verbose SDK and lifecycle logging.

    The function is idempotent: handlers marked with ``_math_agent_handler``
    are reused across Streamlit reruns instead of being added repeatedly.

    Returns:
        Absolute path of the log file written by the file handler.
    """
    LOG_PATH.parent.mkdir(exist_ok=True)
    root = logging.getLogger()

    handlers = [
        handler
        for handler in root.handlers
        if getattr(handler, "_math_agent_handler", False)
    ]
    if not handlers:
        for handler in (logging.StreamHandler(), logging.FileHandler(LOG_PATH)):
            handler._math_agent_handler = True  # type: ignore[attr-defined]
            root.addHandler(handler)
            handlers.append(handler)

    # Reapply these on Streamlit reruns so a process started with the previous
    # format picks up request context without accumulating duplicate handlers.
    for handler in handlers:
        handler.setFormatter(logging.Formatter(FORMAT))
        if not any(isinstance(item, _RequestContextFilter) for item in handler.filters):
            handler.addFilter(_RequestContextFilter())

    root.setLevel(logging.INFO)
    logging.getLogger("strands").setLevel(logging.DEBUG)
    logging.getLogger("fastmcp").setLevel(logging.INFO)
    logging.getLogger("mcp").setLevel(logging.INFO)
    return LOG_PATH


def _json(value: Any) -> str:
    """Serialize a value for one-line structured logging.

    Args:
        value: JSON-compatible data or an arbitrary SDK object.

    Returns:
        JSON text. Objects unsupported by JSON fall back to ``str(value)``.
    """
    return json.dumps(value, default=str, ensure_ascii=False)


def _content(message: Any) -> Any:
    """Extract content blocks from a Strands message when possible.

    Args:
        message: A Strands message dictionary or arbitrary event value.

    Returns:
        The message's ``content`` value, or the original value when it is not a
        message dictionary.
    """
    if isinstance(message, dict):
        return message.get("content", message)
    return message


def _model_stop_response(response: Any) -> tuple[Any, Any]:
    """Normalize model-stop responses across Strands SDK representations.

    Args:
        response: ``None``, a mapping, or a dataclass-like response object.

    Returns:
        A ``(message, stop_reason)`` tuple. Missing fields become ``None``.
    """
    if response is None:
        return None, None
    if isinstance(response, Mapping):
        return response.get("message"), response.get("stop_reason")
    return getattr(response, "message", None), getattr(response, "stop_reason", None)


def attach_agent_logging(agent: Agent) -> None:
    """Attach non-mutating hooks for the complete observable agent lifecycle.

    Args:
        agent: Strands agent whose events should be logged.

    Notes:
        The hooks record request boundaries, model-visible content/reasoning,
        tool selection, tool input/output, errors, metrics, and timings. They do
        not alter messages, retry flags, tool arguments, or model responses.
    """
    # The configured log format includes the logger name, so using a child
    # namespace makes nested supervisor/worker events easy to distinguish and
    # filter without changing every event message.
    logger = logging.getLogger(_logger_name(agent))
    invocation_started = 0.0
    model_started = 0.0

    def before_invocation(event: BeforeInvocationEvent) -> None:
        """Record the incoming request and start end-to-end timing."""
        nonlocal invocation_started
        invocation_started = time.perf_counter()
        logger.info(
            "REQUEST_RECEIVED | messages=%s | invocation_state=%s",
            _json(event.messages),
            _json(event.invocation_state),
        )

    def message_added(event: MessageAddedEvent) -> None:
        """Record each user, assistant, and tool-result conversation message."""
        role = event.message.get("role", "unknown")
        logger.info(
            "MESSAGE_ADDED | role=%s | content=%s",
            role,
            _json(_content(event.message)),
        )

    def before_model_call(event: BeforeModelCallEvent) -> None:
        """Record model input sizing and start model-call timing."""
        nonlocal model_started
        model_started = time.perf_counter()
        logger.info(
            "MODEL_START | projected_input_tokens=%s | conversation_messages=%s",
            event.projected_input_tokens,
            len(event.agent.messages),
        )

    def after_model_call(event: AfterModelCallEvent) -> None:
        """Record model output, stop reason, duration, or failure."""
        duration = time.perf_counter() - model_started if model_started else None
        if event.exception is not None:
            logger.exception(
                "MODEL_END | status=error | duration_seconds=%s | error=%r",
                duration,
                event.exception,
                exc_info=event.exception,
            )
            return

        message, stop_reason = _model_stop_response(event.stop_response)
        logger.info(
            "MODEL_END | status=success | duration_seconds=%s | "
            "stop_reason=%s | content=%s",
            duration,
            stop_reason,
            _json(_content(message)),
        )

    def before_tools(event: BeforeToolsEvent) -> None:
        """Record the complete batch of tools selected by the model."""
        tool_uses = [
            block["toolUse"]
            for block in event.message.get("content", [])
            if "toolUse" in block
        ]
        logger.info("TOOLS_SELECTED | calls=%s", _json(tool_uses))

    def before_tool_call(event: BeforeToolCallEvent) -> None:
        """Record one tool name and its arguments before execution."""
        logger.info("TOOL_START | tool_use=%s", _json(event.tool_use))

    def after_tool_call(event: AfterToolCallEvent) -> None:
        """Record one tool's result, status, error, and duration."""
        status = "error" if event.exception is not None else "success"
        logger.info(
            "TOOL_END | name=%s | status=%s | duration_seconds=%s | "
            "result=%s | error=%r",
            event.tool_use.get("name", "unknown"),
            status,
            event.duration,
            _json(event.result),
            event.exception,
        )

    def after_tools(event: AfterToolsEvent) -> None:
        """Record the combined result message after a tool batch."""
        logger.info(
            "TOOLS_FINISHED | result_message=%s",
            _json(_content(event.message)),
        )

    def after_invocation(event: AfterInvocationEvent) -> None:
        """Record the final response, metrics, stop reason, and total duration."""
        duration = time.perf_counter() - invocation_started if invocation_started else None
        result = event.result
        logger.info(
            "REQUEST_FINISHED | duration_seconds=%s | stop_reason=%s | "
            "response=%s | metrics=%s",
            duration,
            getattr(result, "stop_reason", None),
            _json(_content(getattr(result, "message", None))),
            _json(getattr(result, "metrics", None)),
        )

    hooks = (
        (before_invocation, BeforeInvocationEvent),
        (message_added, MessageAddedEvent),
        (before_model_call, BeforeModelCallEvent),
        (after_model_call, AfterModelCallEvent),
        (before_tools, BeforeToolsEvent),
        (before_tool_call, BeforeToolCallEvent),
        (after_tool_call, AfterToolCallEvent),
        (after_tools, AfterToolsEvent),
        (after_invocation, AfterInvocationEvent),
    )
    for callback, event_type in hooks:
        agent.add_hook(callback, event_type)


def attach_tool_logging(agent: Agent) -> None:
    """Attach lifecycle logging using the original helper name.

    Args:
        agent: Strands agent whose lifecycle should be logged.

    Notes:
        Kept as a compatibility alias for callers created before logging was
        expanded beyond tool events.
    """
    attach_agent_logging(agent)
