# Math MCP server + LangChain agent

See [`FUNCTION_NOTES.md`](FUNCTION_NOTES.md) for a function-by-function
developer reference covering the MCP server, agents, logging, Streamlit UI, and
multi-user API.

This small Python project uses FastMCP to expose five math functions as MCP
tools from a Docker container. The agents access them over Streamable HTTP.

## How it fits together

1. `mcp_servers/math_server.py` registers functions with FastMCP's `@mcp.tool`.
2. `mcp_servers/mcp.json` contains the Streamable HTTP configuration.
3. `agents_strands.py` builds a high-level supervisor and two specialized workers.
4. The math worker receives only the discovered MCP arithmetic tools, whose
   deterministic results are shared through Redis.
5. The 3D worker receives only the native geometry tools.
6. The supervisor receives the workers as tools, delegates the identified tasks,
   and combines their responses. It cannot invoke operation tools directly.

Both `agent.py` and `agents_strands.py` load the same configuration. The MCP URL
defaults to `http://127.0.0.1:8001/mcp` and can be changed with
`MATH_MCP_URL`.

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

Install [Ollama](https://ollama.com), start it, and download the default models:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
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

`agents_strands.py` uses the containerized FastMCP server with a Strands agent,
using `qwen3:1.7b` for the supervisor and `qwen3:0.6b` for both workers, so it
does not require AWS credentials:

```bash
.venv/bin/python agents_strands.py "What is 81 divided by 9, then raised to 3?"
```

The Strands entry point accepts `--supervisor-model` and `--worker-model` to
select different installed Ollama models for routing and operation work.

The Strands hierarchy separates two tool sources:

- `math-operations-agent` uses arithmetic tools supplied by
  `mcp_servers/math_server.py` over Streamable HTTP MCP.
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

Make sure Ollama is running on the host, then build and start Redis, math MCP,
and FastAPI together:

```bash
ollama serve
docker compose up --build -d
```

Its readiness endpoint is `http://127.0.0.1:8001/health`, and its MCP endpoint
is `http://127.0.0.1:8001/mcp`. The host port can be changed with
`MATH_MCP_PORT`. FastAPI is published at `http://127.0.0.1:8000` and can be
changed with `AGENT_API_PORT`. Inside Compose, FastAPI reaches MCP through the
service address `http://math-mcp:8000/mcp`.

In a second terminal, start the Streamlit client:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

The request path is:

```text
Streamlit -> FastAPI agent -> math MCP container -> Redis cache
```

Streamlit sends conversation creation, messages, and deletion through FastAPI.
The FastAPI agent sends arithmetic tool discovery and calls to the MCP
container. Streamlit does not call the MCP server directly because that server
exposes math tools, not a chat-agent endpoint. To use a different API address,
set `AGENT_API_URL` before starting Streamlit.

Assistant output is streamed from FastAPI to Streamlit as it is generated.
FastAPI uses newline-delimited JSON (`application/x-ndjson`) with `start`,
`delta`, `complete`, and `error` event types. The original non-streaming message
endpoint remains available for clients that need one final JSON response.

Open the URL printed by Streamlit (normally `http://localhost:8501`). Each
browser session retains its API conversation identifier, so follow-up math
questions can refer to earlier answers. Use **New conversation** in the sidebar
to delete the API conversation and clear the UI transcript.

### Tool-result cache

Docker Compose starts a private Redis service automatically. Every math MCP
tool uses a key shaped like `math-tool:v1:<tool>:<sha256-of-canonical-inputs>`.
On a hit, the tool returns the stored JSON result without executing the pure
operation again. On a miss, it performs the operation and stores the result.

Results expire after 3,600 seconds by default. Change that when starting the
stack:

```bash
TOOL_CACHE_TTL_SECONDS=86400 docker compose up --build -d
```

Inspect cache activity through the math service logs and Redis:

```bash
docker compose logs -f math-mcp
docker compose exec redis redis-cli DBSIZE
docker compose exec redis redis-cli --scan --pattern 'math-tool:*'
```

Clear only tool-result entries when needed:

```bash
docker compose exec redis sh -c \
  "redis-cli --scan --pattern 'math-tool:*' | xargs -r redis-cli DEL"
```

Redis is configured as a bounded, non-persistent cache with an `allkeys-lru`
eviction policy. Cache reads and writes fail open: if Redis is unavailable, MCP
still computes and returns the math result. When tool behavior changes in a way
that affects results, increment the `v1` namespace in `tool_cache.py` to avoid
reusing incompatible values.

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

Generate a standalone sequence timeline for the latest completed request:

```bash
.venv/bin/python plot_agent_timeline.py
```

This writes `agent_timeline.html`, which can be opened directly in a browser.
Below the timeline, the report includes an expandable card for every model
invocation with the model ID, agent, duration, token usage, input messages,
observable model output, and stop reason. Private chain-of-thought is not
available; tool calls and normal text returned by the model are included.

The API image also contains the report generator. Generate the latest report
inside the running container and copy it to the host with:

```bash
docker compose exec -T agent-api \
  python plot_agent_timeline.py --output /tmp/agent_timeline.html
docker compose cp agent-api:/tmp/agent_timeline.html ./agent_timeline.html
```
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
while sharing the network MCP connection and loaded tool definitions across
users.

Start Ollama, then launch both the MCP and API containers through Compose:

```bash
ollama serve
docker compose up --build -d
```

Compose waits for the math service health check before starting FastAPI.
FastAPI reaches host Ollama through `http://host.docker.internal:11434` by
default. Override `API_OLLAMA_HOST` if Ollama is available at another address.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Create a conversation:

```bash
curl -X POST http://127.0.0.1:8000/v1/conversations \
  -H 'Content-Type: application/json' \
  -H 'X-User-ID: alice' \
  -d '{"supervisor_model":"qwen3:1.7b","worker_model":"qwen3:0.6b"}'
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
- `POST /v1/conversations/{conversation_id}/messages/stream`
- `GET /v1/conversations/{conversation_id}/messages`
- `DELETE /v1/conversations/{conversation_id}`

Stream a response with curl's output buffering disabled:

```bash
curl -N -X POST \
  http://127.0.0.1:8000/v1/conversations/CONVERSATION_ID/messages/stream \
  -H 'Content-Type: application/json' \
  -H 'X-User-ID: alice' \
  -d '{"message":"What is 6 multiplied by 7?"}'
```

Each output line is an independent JSON event. Only assistant text is exposed
as `delta` events; internal reasoning and tool-call events are not sent to the
client.

`X-User-ID` provides conversation ownership inside this example, but it is
client supplied and is not authentication. A production multi-tenant service
must derive the user identity from a verified token or trusted API gateway.

Both backend services are managed by Docker Compose. Inspect them with
`docker compose ps` or `docker compose logs -f`, and stop them with
`docker compose down`.

## Test the math layer

```bash
.venv/bin/python -m pytest
```

To inspect the running MCP server with MCP Inspector:

```bash
npx -y @modelcontextprotocol/inspector http://127.0.0.1:8001/mcp
```

