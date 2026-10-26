# Math MCP server + LangChain agent

See [`FUNCTION_NOTES.md`](FUNCTION_NOTES.md) for a function-by-function
developer reference covering the MCP server, agents, logging, Streamlit UI, and
multi-user API.

This small Python project uses the standalone FastMCP package to expose five
math functions as MCP tools, then gives those tools to a LangChain agent over
MCP's local `stdio` transport.

## How it fits together

1. `math_server.py` registers functions with FastMCP's `@mcp.tool`.
2. `mcp.json` contains the shared stdio server configuration.
3. `agents_strands.py` builds a high-level supervisor and two specialized workers.
4. The math worker receives only the discovered MCP arithmetic tools.
5. The 3D worker receives only the native geometry tools.
6. The supervisor receives the workers as tools, delegates the identified tasks,
   and combines their responses. It cannot invoke operation tools directly.

Both `agent.py` and `agents_strands.py` load the same `mcp.json`. Its
`${MCP_PYTHON}` and `${MCP_SERVER_PATH}` placeholders are resolved at runtime,
so the configuration continues to work if this project directory is moved.

All server implementation and direct protocol testing use FastMCP. LangChain's
official MCP adapter is the framework bridge that exposes MCP tools to its agent.

## Setup

Python 3.10 or newer is required. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Any Python 3.10+ executable can replace `python3.12` above.

Activation is optional. If you prefer it, run `source .venv/bin/activate` from
`zsh` or `bash`; do not execute the `activate` file directly. The commands in
this guide use `.venv/bin/python` explicitly so shell configuration cannot
interfere.

## Run without cloud credentials

Install [Ollama](https://ollama.com), start it, and download the default model:

```bash
ollama pull qwen3:4b
```

Then run:

```bash
.venv/bin/python agent.py "What is 144 divided by 12, then raised to the third power?"
```

Choose another installed Ollama model with `--model`, for example:

```bash
.venv/bin/python agent.py --model llama3.1:8b "Multiply 25 by 16"
```

## Strands agent with Ollama

`agents_strands.py` uses the same FastMCP math server with a Strands agent and
the local `qwen3:4b` Ollama model, so it does not require AWS credentials:

```bash
.venv/bin/python agents_strands.py "What is 81 divided by 9, then raised to 3?"
```

The Strands entry point also accepts `--model` to select another installed
Ollama model.

The Strands hierarchy separates two tool sources:

- `math-operations-agent` uses arithmetic tools supplied by `math_server.py`
  over MCP using `mcp.json`.
- `geometry-operations-agent` uses `add_3d_points` and `subtract_3d_points`
  loaded from `tools/geometry.py`.
- `operations-supervisor` identifies one or more tasks, calls the appropriate
  worker agents, and returns a consolidated response.

Independent work is batched by specialization. The supervisor makes at most
one concurrent call to each worker: all independent arithmetic tasks go into a
single math-worker request, while all independent 3D tasks go into a single
geometry-worker request. The two workers can run concurrently, and each worker
requests all independent underlying tools in the same model turn. Dependent
operations remain sequential so later calls can consume earlier results.

For example:

```bash
.venv/bin/python agents_strands.py \
  "Add the 3D points [1, 2, 3] and [4, 5, 6], then multiply the x coordinate by 10"
```

## Streamlit chat app

Run a persistent browser chat backed by the Strands agent:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`). Each
browser session retains its conversation and Strands agent, so follow-up math
questions can refer to earlier answers. Use **New conversation** in the sidebar
to clear both the UI transcript and agent history.

### Logging

Both Strands entry points enable DEBUG logging for the complete `strands`
namespace. Logs are printed in the terminal and saved to `logs/strands.log`.
The custom lifecycle trace includes `REQUEST_RECEIVED`, `MESSAGE_ADDED`,
`MODEL_START`, `MODEL_END`, `TOOLS_SELECTED`, `TOOL_START`, `TOOL_END`,
`TOOLS_FINISHED`, and `REQUEST_FINISHED`. It records model-visible reasoning and
content blocks, stop reasons, tool arguments/results, errors, timings, and
metrics. Private model chain-of-thought is not exposed by Strands or Ollama and
therefore cannot be logged. In Streamlit, enable **Show recent logs** in the
sidebar to view the latest entries.

The logger-name column identifies the agent that emitted each lifecycle event:
`agent.lifecycle.operations-supervisor`,
`agent.lifecycle.math-operations-agent`, or
`agent.lifecycle.geometry-operations-agent`. This also makes the log easy to
filter by agent name when requests contain multiple delegated tasks.

Streamlit assigns a persistent `session_id` to each browser session and a new
`request_id` to every submitted chat message. Both values are included on all
supervisor, worker, model, and tool log records created while that request is
processed, allowing interleaved sessions to be filtered reliably.

Generate a standalone sequence timeline for the latest completed request:

```bash
.venv/bin/python plot_agent_timeline.py
```

This writes `agent_timeline.html`, which can be opened directly in a browser.
Select an earlier completed request or another destination with:

```bash
.venv/bin/python plot_agent_timeline.py \
  --request-index 1 \
  --output reports/previous-request.html
```

Select one exact Streamlit message using the identifiers printed in every log
record:

```bash
.venv/bin/python plot_agent_timeline.py \
  --session-id SESSION_UUID \
  --request-id REQUEST_UUID \
  --output reports/request-timeline.html
```

Omit `--request-id` to include every completed request from the selected
session in one HTML report:

```bash
.venv/bin/python plot_agent_timeline.py \
  --session-id SESSION_UUID \
  --output reports/session-timelines.html
```

The generated timelines display both identifiers and include only records from
the selected session, even when multiple browser sessions wrote interleaved log
records. `--request-id` requires `--session-id`.

## Multi-user agent API

`agent_api.py` exposes the Strands agent as a FastAPI service. It creates one
isolated agent and conversation history per `(X-User-ID, conversation_id)`,
while sharing the MCP connection and loaded tool definitions across users.

Start Ollama and the API:

```bash
ollama serve
.venv/bin/python -m uvicorn agent_api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Create a conversation:

```bash
curl -X POST http://127.0.0.1:8000/v1/conversations \
  -H 'Content-Type: application/json' \
  -H 'X-User-ID: alice' \
  -d '{"model":"qwen3:4b"}'
```

Use the returned `conversation_id` for multiple messages:

```bash
curl -X POST http://127.0.0.1:8000/v1/conversations/CONVERSATION_ID/messages \
  -H 'Content-Type: application/json' \
  -H 'X-User-ID: alice' \
  -d '{"message":"Add [1,2,3] and [4,5,6]."}'

curl -X POST http://127.0.0.1:8000/v1/conversations/CONVERSATION_ID/messages \
  -H 'Content-Type: application/json' \
  -H 'X-User-ID: alice' \
  -d '{"message":"Now multiply its x coordinate by 10."}'
```

Other endpoints:

- `GET /health`
- `GET /v1/tools`
- `GET /v1/conversations/{conversation_id}/messages`
- `DELETE /v1/conversations/{conversation_id}`

`X-User-ID` provides conversation ownership inside this example, but it is
client supplied and is not authentication. A production multi-tenant service
must derive the user identity from a verified token or trusted API gateway.

The FastMCP server needs no separate terminal: LangChain's MCP adapter launches
and shuts down its stdio subprocess for each operation.

## Test the math layer

```bash
.venv/bin/python -m pytest
```

To inspect the MCP server itself with the MCP Inspector:

```bash
npx -y @modelcontextprotocol/inspector .venv/bin/python math_server.py
```

You can also use FastMCP's own CLI:

```bash
fastmcp list math_server.py --input-schema
fastmcp call math_server.py multiply a=6 b=7
```

Do not add `print()` calls to the stdio server's stdout; MCP uses stdout for its
protocol messages. Use logging to stderr instead.
