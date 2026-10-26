# Function notes

This reference describes the project functions by responsibility. Private
helpers are included when they affect lifecycle, logging, or portability.

## `math_operations.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `add(a, b)` | Add two numbers. | Pure function with no external state. |
| `subtract(a, b)` | Calculate `a - b`. | Operand order matters. |
| `multiply(a, b)` | Multiply two numbers. | Pure function. |
| `divide(a, b)` | Calculate `a / b`. | Raises `ValueError` when `b == 0`. |
| `power(base, exponent)` | Calculate `base ** exponent`. | Accepts floating-point exponents. |

## `math_server.py`

`add`, `subtract`, `multiply`, `divide`, and `power` are FastMCP tool handlers.
Each delegates to the corresponding pure function in `math_operations.py`.
FastMCP derives the tools' JSON schemas from their type annotations and
docstrings. The module starts a stdio MCP server only when run as a script.

## `tools/geometry.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `_validate_point(point, name)` | Validate one point argument. | Requires exactly three coordinates; raises `ValueError` otherwise. |
| `add_3d_points(point_a, point_b)` | Component-wise 3D point addition. | Strands-native `@tool`; returns `[x, y, z]`. |
| `subtract_3d_points(point_a, point_b)` | Component-wise `point_a - point_b`. | Strands-native `@tool`; operand order matters. |

## `mcp_config.py`

### `load_mcp_config()`

Reads `mcp.json` and resolves `${MCP_PYTHON}` to the current interpreter and
`${MCP_SERVER_PATH}` to the absolute `math_server.py` path. This makes the MCP
configuration work without activating the virtual environment or depending on
the caller's current directory.

The nested `resolve(value)` helper recursively handles strings, lists, and
dictionaries from the JSON document.

## `agents_strands.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `ensure_ollama_ready(model_id)` | Check Ollama connectivity and model availability. | Exits with an actionable message when the service or model is unavailable. |
| `create_math_client()` | Build a Strands MCP client from `mcp.json`. | Returns the configured math server client; callers own its lifecycle. |
| `create_supervisor_agent(model, math_tools, geometry_tools)` | Build the three-agent hierarchy. | Gives each worker only its operation tools and gives the supervisor only the two worker wrappers. |
| `main()` | Run the command-line Strands experience. | Starts MCP, loads both tool families, builds the supervisor, and sends it one prompt. |

## `agent.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `ensure_ollama_ready(model)` | Async Ollama readiness check for LangChain. | Uses a three-second HTTP timeout. |
| `main()` | Run the LangChain command-line agent. | Loads MCP tools and invokes the selected local Ollama model asynchronously. |

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

## `streamlit_app.py`

| Function | Purpose | Important behavior |
|---|---|---|
| `close_mcp_client()` | Stop the current browser session's MCP process. | Safe when no client exists. |
| `reset_conversation()` | Clear UI and agent history. | Also closes MCP resources and forces a new agent on the next prompt. |
| `get_agent(model_id)` | Get or create the session's Strands agent. | Reuses the agent only when the model is unchanged; retains follow-up context. |
| `render_message(message)` | Render one chat message. | Expects `role` and `content` keys. |

Streamlit reruns the module after UI interactions. Long-lived objects are kept
in `st.session_state`, which is isolated per browser session.

## `agent_api.py`

### Agent and resource construction

| Function | Purpose | Important behavior |
|---|---|---|
| `create_conversation_agent(model_id, math_tools, geometry_tools)` | Build one supervisor hierarchy per conversation. | Each supervisor owns isolated history; tool definitions and the MCP connection are shared. |
| `lifespan(app)` | Manage process-wide API resources. | Verifies Ollama, starts MCP once, loads worker tools, initializes the store, and closes MCP during shutdown. |
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
| `get_messages(...)` | `GET /v1/conversations/{id}/messages` | Return the API transcript. |
| `delete_conversation(...)` | `DELETE /v1/conversations/{id}` | Remove the in-memory conversation. |

`X-User-ID` is an ownership key for this example, not authentication. Production
code must derive identity from a verified token or trusted gateway.
