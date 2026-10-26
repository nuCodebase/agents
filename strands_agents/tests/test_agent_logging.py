from dataclasses import dataclass

import logging

from agent_logging import (
    _RequestContextFilter,
    _logger_name,
    _model_stop_response,
    request_log_context,
)


class NamedAgent:
    name = "math operations agent"


@dataclass
class ObjectResponse:
    message: dict
    stop_reason: str


def test_model_stop_response_object() -> None:
    message = {"role": "assistant", "content": [{"text": "done"}]}
    assert _model_stop_response(ObjectResponse(message, "end_turn")) == (
        message,
        "end_turn",
    )


def test_model_stop_response_mapping() -> None:
    message = {"role": "assistant", "content": [{"text": "done"}]}
    assert _model_stop_response(
        {"message": message, "stop_reason": "tool_use"}
    ) == (message, "tool_use")


def test_model_stop_response_none() -> None:
    assert _model_stop_response(None) == (None, None)


def test_agent_logger_name_identifies_agent() -> None:
    assert _logger_name(NamedAgent()) == "agent.lifecycle.math-operations-agent"


def test_request_context_filter_adds_ids() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
    context_filter = _RequestContextFilter()

    with request_log_context("session-123", "request-456"):
        assert context_filter.filter(record)

    assert record.session_id == "session-123"
    assert record.request_id == "request-456"
