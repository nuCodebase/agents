"""Tests for newline-delimited streaming chat responses."""

import asyncio
import json

from agent_api import Conversation, stream_agent_response, stream_event


class FakeResult:
    """Minimal Strands result used by the stream test."""

    stop_reason = "end_turn"

    def __str__(self) -> str:
        return "The answer is 42."


class FakeStreamingAgent:
    """Agent stand-in that emits two text deltas and a final result."""

    async def stream_async(self, message, *, cancel_signal):
        assert message == "What is 6 times 7?"
        assert not cancel_signal.is_set()
        yield {"data": "The answer "}
        yield {"data": "is 42."}
        yield {"result": FakeResult()}


def test_stream_event_is_one_json_record() -> None:
    record = stream_event("delta", text="hello")

    assert record.endswith("\n")
    assert json.loads(record) == {"type": "delta", "text": "hello"}


def test_stream_agent_response_persists_completed_transcript() -> None:
    conversation = Conversation(
        user_id="alice",
        supervisor_model_id="test-supervisor-model",
        worker_model_id="test-worker-model",
        agent=FakeStreamingAgent(),  # type: ignore[arg-type]
    )

    async def collect() -> list[dict[str, object]]:
        return [
            json.loads(record)
            async for record in stream_agent_response(
                conversation,
                "What is 6 times 7?",
            )
        ]

    events = asyncio.run(collect())

    assert events == [
        {"type": "start"},
        {"type": "delta", "text": "The answer "},
        {"type": "delta", "text": "is 42."},
        {
            "type": "complete",
            "response": "The answer is 42.",
            "stop_reason": "end_turn",
        },
    ]
    assert conversation.messages == [
        {"role": "user", "content": "What is 6 times 7?"},
        {"role": "assistant", "content": "The answer is 42."},
    ]
