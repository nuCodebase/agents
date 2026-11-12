"""Multi-user HTTP API for the Strands math and geometry agent."""

import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.loader import load_tools_from_file_path

from agent_logging import configure_logging
from agents_strands import (
    OLLAMA_HOST,
    create_math_client,
    create_supervisor_agent,
    ensure_ollama_ready,
)

DEFAULT_SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "qwen3:1.7b")
DEFAULT_WORKER_MODEL = os.getenv("WORKER_MODEL", "qwen3:0.6b")
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "1000"))
GEOMETRY_TOOLS_PATH = Path(__file__).with_name("tools") / "geometry.py"


class ConversationCreate(BaseModel):
    """Request body used to select the model for a new conversation."""
    supervisor_model: str = Field(default=DEFAULT_SUPERVISOR_MODEL, min_length=1)
    worker_model: str = Field(default=DEFAULT_WORKER_MODEL, min_length=1)


class ChatRequest(BaseModel):
    """Validated user message submitted to an existing conversation."""
    message: str = Field(min_length=1, max_length=10_000)


class ChatResponse(BaseModel):
    """Final agent response returned by the synchronous chat endpoint."""
    conversation_id: str
    user_id: str
    response: str
    stop_reason: str | None = None


class ConversationResponse(BaseModel):
    """Identifiers and model information returned after conversation creation."""
    conversation_id: str
    user_id: str
    supervisor_model: str
    worker_model: str


@dataclass
class Conversation:
    """Mutable state owned by one user conversation.

    ``lock`` serializes messages within this conversation. Different
    conversations have distinct locks and can therefore run concurrently.
    """
    user_id: str
    supervisor_model_id: str
    worker_model_id: str
    agent: Agent
    messages: list[dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    updated_at: float = field(default_factory=time.monotonic)


class ConversationStore:
    """In-memory isolated agent registry with bounded LRU-style state.

    The registry lock protects dictionary membership and ownership checks. It is
    deliberately separate from each conversation lock so long model calls do
    not prevent other users from creating or retrieving conversations.
    """

    def __init__(
        self,
        agent_factory: Callable[[str, str], Agent],
        max_conversations: int = MAX_CONVERSATIONS,
    ) -> None:
        """Initialize an empty conversation registry.

        Args:
            agent_factory: Callable that creates a fresh agent for a model ID.
            max_conversations: Maximum resident conversations before eviction.
        """
        self._agent_factory = agent_factory
        self._max_conversations = max_conversations
        self._items: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()

    async def create(
        self, user_id: str, supervisor_model_id: str, worker_model_id: str
    ) -> tuple[str, Conversation]:
        """Create an isolated conversation, evicting the oldest at capacity.

        Args:
            user_id: Ownership identifier supplied by the authenticated layer.
            supervisor_model_id: Ollama model used by the supervisor.
            worker_model_id: Ollama model shared by the worker agents.

        Returns:
            Newly generated conversation UUID and its state object.
        """
        async with self._lock:
            if len(self._items) >= self._max_conversations:
                oldest_id = min(
                    self._items,
                    key=lambda item_id: self._items[item_id].updated_at,
                )
                del self._items[oldest_id]

            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                user_id=user_id,
                supervisor_model_id=supervisor_model_id,
                worker_model_id=worker_model_id,
                agent=self._agent_factory(supervisor_model_id, worker_model_id),
            )
            self._items[conversation_id] = conversation
            return conversation_id, conversation

    async def get(self, conversation_id: str, user_id: str) -> Conversation:
        """Return and touch a conversation owned by the requesting user.

        Args:
            conversation_id: UUID returned by ``create``.
            user_id: Expected owner identifier.

        Returns:
            Matching conversation state.

        Raises:
            KeyError: If the conversation is absent or belongs to another user.
        """
        async with self._lock:
            conversation = self._items.get(conversation_id)
            if conversation is None or conversation.user_id != user_id:
                raise KeyError(conversation_id)
            conversation.updated_at = time.monotonic()
            return conversation

    async def delete(self, conversation_id: str, user_id: str) -> None:
        """Delete a user-owned conversation from the in-memory registry.

        Raises:
            KeyError: If the conversation is absent or belongs to another user.
        """
        await self.get(conversation_id, user_id)
        async with self._lock:
            del self._items[conversation_id]

    @property
    def count(self) -> int:
        """Return the number of active in-memory conversations."""
        return len(self._items)


def create_conversation_agent(
    supervisor_model_id: str,
    worker_model_id: str,
    math_tools: list[Any],
    geometry_tools: list[Any],
) -> Agent:
    """Build one stateful, fully logged Strands agent for a conversation.

    Args:
        supervisor_model_id: Installed Ollama model used for routing.
        worker_model_id: Installed Ollama model shared by operation workers.
        math_tools: Shared MCP math tool wrappers.
        geometry_tools: Native Strands 3D tool wrappers.

    Returns:
        A new agent with empty, conversation-local message history.
    """
    supervisor_model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=supervisor_model_id,
        temperature=0,
    )
    worker_model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=worker_model_id,
        temperature=0,
    )
    return create_supervisor_agent(
        supervisor_model, worker_model, math_tools, geometry_tools
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize process-wide resources and release them during shutdown.

    Startup verifies the default Ollama model, opens one remote MCP connection,
    loads MCP/native tool definitions, and constructs the conversation store.
    The MCP client is always stopped in ``finally``, including failed startup.

    Args:
        app: FastAPI application receiving initialized objects in ``app.state``.

    Yields:
        Control to FastAPI while the service is accepting requests.
    """
    configure_logging()
    ensure_ollama_ready(DEFAULT_SUPERVISOR_MODEL)
    ensure_ollama_ready(DEFAULT_WORKER_MODEL)

    mcp_client = create_math_client()
    mcp_client.start()
    try:
        # Tool definitions are immutable and safe to share; conversation history
        # remains isolated because every registry entry owns a separate Agent.
        mcp_tools = mcp_client.list_tools_sync()
        geometry_tools = load_tools_from_file_path(str(GEOMETRY_TOOLS_PATH))
        app.state.mcp_client = mcp_client
        app.state.tool_names = {
            "math_operations_agent": sorted(tool.tool_name for tool in mcp_tools),
            "geometry_operations_agent": sorted(
                tool.tool_name for tool in geometry_tools
            ),
        }
        app.state.conversations = ConversationStore(
            lambda supervisor_model_id, worker_model_id: create_conversation_agent(
                supervisor_model_id, worker_model_id, mcp_tools, geometry_tools
            )
        )
        yield
    finally:
        mcp_client.stop(None, None, None)


app = FastAPI(
    title="Math and Geometry Agent API",
    version="1.0.0",
    lifespan=lifespan,
)


def store(request: Request) -> ConversationStore:
    """Retrieve the initialized registry from application state.

    Args:
        request: Current FastAPI request.

    Returns:
        Store created during application lifespan startup.
    """
    return request.app.state.conversations


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Report readiness, default models, and active conversation count."""
    return {
        "status": "ok",
        "supervisor_model": DEFAULT_SUPERVISOR_MODEL,
        "worker_model": DEFAULT_WORKER_MODEL,
        "active_conversations": store(request).count,
    }


@app.get("/v1/tools")
async def list_tools(request: Request) -> dict[str, dict[str, list[str]]]:
    """List operation tools grouped by the worker that owns them."""
    return {"agents": request.app.state.tool_names}


@app.post(
    "/v1/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    body: ConversationCreate,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=200)],
) -> ConversationResponse:
    """Create a conversation owned by the supplied user identity.

    Args:
        body: Requested Ollama model configuration.
        request: Current FastAPI request.
        user_id: Value of the ``X-User-ID`` header.

    Returns:
        New conversation identifier, owner, and model.

    Raises:
        SystemExit: If Ollama or the requested model is unavailable.
    """
    ensure_ollama_ready(body.supervisor_model)
    ensure_ollama_ready(body.worker_model)
    conversation_id, conversation = await store(request).create(
        user_id, body.supervisor_model, body.worker_model
    )
    return ConversationResponse(
        conversation_id=conversation_id,
        user_id=user_id,
        supervisor_model=conversation.supervisor_model_id,
        worker_model=conversation.worker_model_id,
    )


@app.post(
    "/v1/conversations/{conversation_id}/messages",
    response_model=ChatResponse,
)
async def send_message(
    conversation_id: str,
    body: ChatRequest,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=200)],
) -> ChatResponse:
    """Send one message to an isolated agent and return its final response.

    The synchronous Strands invocation runs in a worker thread so it does not
    block FastAPI's event loop. The conversation lock preserves message order
    when one client submits concurrent requests to the same conversation.

    Raises:
        HTTPException: With status 404 when the conversation is missing or the
        supplied user does not own it.
    """
    try:
        conversation = await store(request).get(conversation_id, user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error

    async with conversation.lock:
        result = await asyncio.to_thread(conversation.agent, body.message)
        response_text = str(result).strip()
        conversation.messages.extend(
            [
                {"role": "user", "content": body.message},
                {"role": "assistant", "content": response_text},
            ]
        )
        conversation.updated_at = time.monotonic()
        return ChatResponse(
            conversation_id=conversation_id,
            user_id=user_id,
            response=response_text,
            stop_reason=getattr(result, "stop_reason", None),
        )


def stream_event(event_type: str, **payload: Any) -> str:
    """Encode one response event as a newline-delimited JSON record."""
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


async def stream_agent_response(
    conversation: Conversation,
    message: str,
) -> AsyncIterator[str]:
    """Stream one agent response while preserving conversation ordering.

    The lock remains held until the agent finishes or the HTTP client
    disconnects. Closing the HTTP stream closes this async generator and sets
    the Strands cancellation signal.
    """
    cancel_signal = threading.Event()
    try:
        async with conversation.lock:
            yield stream_event("start")
            result = None
            async for event in conversation.agent.stream_async(
                message,
                cancel_signal=cancel_signal,
            ):
                if "result" in event:
                    result = event["result"]
                    continue

                text = event.get("data")
                if isinstance(text, str) and text:
                    yield stream_event("delta", text=text)

            if result is None:
                if cancel_signal.is_set():
                    return
                raise RuntimeError("Agent stream ended without a final result")

            response_text = str(result).strip()
            stop_reason = getattr(result, "stop_reason", None)
            if stop_reason != "cancelled":
                conversation.messages.extend(
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": response_text},
                    ]
                )
                conversation.updated_at = time.monotonic()
                yield stream_event(
                    "complete",
                    response=response_text,
                    stop_reason=stop_reason,
                )
    except Exception as error:
        yield stream_event("error", detail=str(error))
    finally:
        cancel_signal.set()


@app.post("/v1/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str,
    body: ChatRequest,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=200)],
) -> StreamingResponse:
    """Stream assistant text as newline-delimited JSON events."""
    try:
        conversation = await store(request).get(conversation_id, user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error

    return StreamingResponse(
        stream_agent_response(conversation, body.message),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=200)],
) -> dict[str, Any]:
    """Return the API transcript for a user-owned conversation.

    Raises:
        HTTPException: With status 404 for missing or foreign conversations.
    """
    try:
        conversation = await store(request).get(conversation_id, user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error
    return {"conversation_id": conversation_id, "messages": conversation.messages}


@app.delete(
    "/v1/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: str,
    request: Request,
    user_id: Annotated[str, Header(alias="X-User-ID", min_length=1, max_length=200)],
) -> None:
    """Permanently remove a user-owned in-memory conversation.

    Raises:
        HTTPException: With status 404 for missing or foreign conversations.
    """
    try:
        await store(request).delete(conversation_id, user_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Conversation not found") from error
