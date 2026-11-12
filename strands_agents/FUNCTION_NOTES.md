# Function notes

This reference describes the project functions by responsibility. Private
helpers are included when they affect lifecycle, logging, or portability.

## `mcp_servers/math_operations.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `add(a, b)` | Add two numbers. | Pure function with no external state. |
| `subtract(a, b)` | Calculate `a - b`. | Operand order matters. |
| `multiply(a, b)` | Multiply two numbers. | Pure function. |
| `divide(a, b)` | Calculate `a / b`. | Raises `ValueError` when `b == 0`. |
| `power(base, exponent)` | Calculate `base ** exponent`. | Accepts floating-point exponents. |

## `mcp_servers/math_server.py`

`add`, `subtract`, `multiply`, `divide`, and `power` are FastMCP tool handlers.
Each delegates to the corresponding pure function in
`mcp_servers/math_operations.py`.
FastMCP derives the tools' JSON schemas from their type annotations and
docstrings. The module serves Streamable HTTP at `/mcp` and exposes `/health`
for container readiness. Each handler delegates through the shared Redis cache
before invoking its pure operation.

## `mcp_servers/tool_cache.py`

| Function or method | Purpose | Important behavior |
|---|---|---|
| `RedisToolCache.key(tool_name, arguments)` | Create a deterministic cache key. | Sorts and compacts JSON arguments, hashes them with SHA-256, and includes a version namespace. |
| `RedisToolCache.get_or_compute(...)` | Read-through cache a tool result. | Returns JSON on a hit; computes and stores with a TTL on a miss; fails open on Redis or serialization errors. |
| `RedisToolCache.close()` | Release Redis resources. | Called by the FastMCP server lifespan during graceful shutdown. |

Caching is enabled when `REDIS_URL` is present. Compose supplies
`redis://redis:6379/0` and configures `TOOL_CACHE_TTL_SECONDS`, defaulting to
3,600 seconds.

## `tools/geometry.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `_validate_point(point, name)` | Validate one point argument. | Requires exactly three coordinates; raises `ValueError` otherwise. |
| `add_3d_points(point_a, point_b)` | Component-wise 3D point addition. | Strands-native `@tool`; returns `[x, y, z]`. |
| `subtract_3d_points(point_a, point_b)` | Component-wise `point_a - point_b`. | Strands-native `@tool`; operand order matters. |

## `mcp_servers/mcp_config.py`

### `load_mcp_config()`

Reads `mcp_servers/mcp.json` and resolves `${MATH_MCP_URL}` from the environment.
The default endpoint is `http://127.0.0.1:8001/mcp`.

## `agents_strands.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `ensure_ollama_ready(model_id)` | Check Ollama connectivity and model availability. | Uses configurable `OLLAMA_HOST`; exits with an actionable message when unavailable. |
| `create_math_client()` | Build a Strands network MCP client. | Returns the configured math server client; callers own its lifecycle. |
| `create_supervisor_agent(supervisor_model, worker_model, math_tools, geometry_tools)` | Build the three-agent hierarchy. | Gives each worker only its operation tools and gives the supervisor only the two worker wrappers. |
| `main()` | Run the command-line Strands experience. | Connects to MCP, loads both tool families, builds the supervisor, and sends it one prompt. |

## `agent_logging.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `configure_logging()` | Configure terminal and file handlers. | Enables `strands` DEBUG logs and writes `logs/strands.log` without adding duplicate handlers. |
| `_json(value)` | Safely serialize SDK values for logs. | Falls back to `str()` for non-JSON SDK objects. |
| `_content(message)` | Extract a message's content blocks. | Accepts message dictionaries or arbitrary values. |
| `_model_stop_response(response)` | Normalize Strands model-stop responses. | Supports both dataclass/object and mapping representations. |
| `attach_agent_logging(agent)` | Register the complete lifecycle audit trail. | Logs requests, model calls, model-visible content/reasoning, tool selection, execution, responses, metrics, and timing. |
| `attach_tool_logging(agent)` | Compatibility alias. | Calls `attach_agent_logging`; retained for existing imports. |

The callbacks defined inside `attach_agent_logging` correspond to Strands hook
events. They observe behavior only and do not modify prompts, results, or retry
flags.

## `plot_agent_timeline.py`

The report generator selects the latest completed request by default, builds
model/tool timing spans, and renders a standalone HTML file. Each model span is
also represented by an expandable invocation card containing its model ID,
agent, elapsed time, exact model-visible input messages, observable output,
stop reason, and per-call token usage. Older logs remain supported; fields that
were not recorded by the older lifecycle format are labeled accordingly.

## `streamlit_app.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `api_request(method, path, **kwargs)` | Call the FastAPI service. | Applies timeouts and translates HTTP/transport errors into readable UI errors. |
| `api_headers()` | Build the API ownership header. | Uses the browser session's stable UUID as `X-User-ID`. |
| `create_conversation(model_id)` | Create API conversation state. | Returns the identifier generated by FastAPI. |
| `delete_conversation()` | Remove API conversation state. | Safe when the browser session has not created a conversation. |
| `reset_conversation()` | Clear UI and API history. | Deletes the remote conversation before clearing the local display cache. |
| `stream_message(message, model_id)` | Stream chat through FastAPI. | Lazily creates a conversation, parses NDJSON, and yields assistant text deltas. |
| `render_message(message)` | Render one chat message. | Expects `role` and `content` keys. |

Streamlit reruns the module after UI interactions. Long-lived objects are kept
in `st.session_state`, which is isolated per browser session. Agent, MCP, and
model resources live only in the FastAPI process.

## Docker services

`docker-compose.yml` launches `redis`, `math-mcp`, and `agent-api`. Math MCP
waits for Redis, and the API waits for the MCP health check. The services
connect over the Compose network. `docker/FastAPI.Dockerfile` packages the
FastAPI agent, while `mcp_servers/Dockerfile` packages the math MCP service.
Compose sets the API container's `OLLAMA_HOST` from
`API_OLLAMA_HOST`, defaulting to `http://host.docker.internal:11434`.

## `agent_api.py`

### Agent and resource construction

| Function | Purpose | Important behavior |
|---|---|---|
| `create_conversation_agent(model_id, math_tools, geometry_tools)` | Build one supervisor hierarchy per conversation. | Each supervisor owns isolated history; tool definitions and the MCP connection are shared. |
| `lifespan(app)` | Manage process-wide API resources. | Verifies Ollama, connects to MCP once, loads worker tools, initializes the store, and closes the client during shutdown. |
| `store(request)` | Access the initialized registry. | Reads it from FastAPI application state. |

### `ConversationStore`

| Method | Purpose | Concurrency and ownership |
|---|---|---|
| `create(user_id, model_id)` | Create a UUID conversation and agent. | Protected by the registry lock; evicts the least-recently-used item at capacity. |
| `get(conversation_id, user_id)` | Retrieve and touch a conversation. | Returns only when the user owns it; otherwise raises `KeyError`. |
| `delete(conversation_id, user_id)` | Remove a conversation. | Performs the ownership check before deletion. |
| `count` | Report active conversation count. | Used by `/health`. |

Each `Conversation` also owns an `asyncio.Lock`. Messages to the same
conversation are serialized, while different conversations can execute
concurrently. Agent calls run in worker threads so they do not block FastAPI's
event loop.

### HTTP endpoint functions

| Function | Endpoint | Purpose |
|---|---|---|
| `health(request)` | `GET /health` | Return readiness, model, and conversation count. |
| `list_tools(request)` | `GET /v1/tools` | Return all MCP and native tool names. |
| `create_conversation(...)` | `POST /v1/conversations` | Create an isolated user-owned conversation. |
| `send_message(...)` | `POST /v1/conversations/{id}/messages` | Invoke the conversation's agent and return the final response. |
| `stream_message(...)` | `POST /v1/conversations/{id}/messages/stream` | Stream assistant text as NDJSON while the agent runs. |
| `get_messages(...)` | `GET /v1/conversations/{id}/messages` | Return the API transcript. |
| `delete_conversation(...)` | `DELETE /v1/conversations/{id}` | Remove the in-memory conversation. |

`stream_agent_response(...)` holds the conversation lock for the complete
Strands `stream_async()` invocation, forwards only text deltas, records the
completed response in the API transcript, and signals cancellation when the
HTTP response generator closes.

`X-User-ID` is an ownership key for this example, not authentication. Production
code must derive identity from a verified token or trusted gateway.
