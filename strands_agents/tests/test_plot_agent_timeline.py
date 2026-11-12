"""Tests for model-invocation extraction in timeline reports."""

from datetime import datetime, timedelta

from plot_agent_timeline import LogEvent, build_model_invocations, render_model_invocations


def event(seconds: float, name: str, details: str) -> LogEvent:
    return LogEvent(
        timestamp=datetime(2026, 1, 1) + timedelta(seconds=seconds),
        agent="operations-supervisor",
        session_id="session",
        request_id="request",
        event=name,
        details=details,
    )


def test_model_invocation_includes_messages_output_and_usage() -> None:
    request = [
        event(0, "REQUEST_RECEIVED", "messages=[] | invocation_state={}"),
        event(
            1,
            "MODEL_START",
            'model=qwen3:1.7b | projected_input_tokens=42 | '
            'conversation_messages=1 | input_messages=[{"role":"user","content":[{"text":"2 + 2"}]}]',
        ),
        event(
            3,
            "MODEL_END",
            'status=success | duration_seconds=2.0 | stop_reason=end_turn | '
            'content=[{"text":"4"}]',
        ),
        event(
            3.1,
            "REQUEST_FINISHED",
            "duration_seconds=3.1 | metrics=EventLoopMetrics("
            "agent_invocations=[AgentInvocation(cycles=[EventLoopCycleMetric("
            "event_loop_cycle_id='cycle', usage={'inputTokens': 10, "
            "'outputTokens': 5, 'totalTokens': 15})])])",
        ),
    ]

    calls = build_model_invocations(request)

    assert len(calls) == 1
    assert calls[0].model == "qwen3:1.7b"
    assert calls[0].input_messages == [
        {"role": "user", "content": [{"text": "2 + 2"}]}
    ]
    assert calls[0].output == [{"text": "4"}]
    assert (calls[0].input_tokens, calls[0].output_tokens, calls[0].total_tokens) == (
        10,
        5,
        15,
    )

    document = render_model_invocations(calls)
    assert "qwen3:1.7b" in document
    assert "Input messages" in document
    assert "Model output" in document
    assert "10 input · 5 output · 15 total" in document


def test_legacy_model_start_reconstructs_messages() -> None:
    request = [
        event(0, "MESSAGE_ADDED", 'role=user | content=[{"text":"hello"}]'),
        event(
            1,
            "MODEL_START",
            "projected_input_tokens=7 | conversation_messages=1",
        ),
        event(
            2,
            "MODEL_END",
            'status=success | duration_seconds=1 | stop_reason=end_turn | '
            'content=[{"text":"hi"}]',
        ),
    ]

    calls = build_model_invocations(request)

    assert calls[0].model == "not recorded"
    assert calls[0].input_messages == [
        {"role": "user", "content": [{"text": "hello"}]}
    ]
