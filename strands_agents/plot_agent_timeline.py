"""Generate a standalone HTML timeline from Strands lifecycle logs."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LINE_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r" \| \w+ \| agent\.lifecycle\.(?P<agent>[^ ]+)"
    r"(?: \| session_id=(?P<session_id>[^ ]+)"
    r" \| request_id=(?P<request_id>[^ ]+))?"
    r" \| (?P<event>[A-Z_]+) \| (?P<details>.*)$"
)
DURATION_RE = re.compile(r"duration_seconds=(?P<duration>\d+(?:\.\d+)?)")
TOOL_NAME_RE = re.compile(r"(?:name=|\"name\":\s*\")(?P<name>[a-zA-Z0-9_-]+)")
USAGE_RE = re.compile(
    r"EventLoopCycleMetric\(.*?usage=\{'inputTokens': (?P<input>\d+), "
    r"'outputTokens': (?P<output>\d+), 'totalTokens': (?P<total>\d+)\}\)"
)


@dataclass(frozen=True)
class LogEvent:
    timestamp: datetime
    agent: str
    session_id: str
    request_id: str
    event: str
    details: str


@dataclass(frozen=True)
class Span:
    lane: str
    start: float
    end: float
    label: str
    detail: str
    kind: str


@dataclass
class ModelInvocation:
    """One completed model call and the model-visible request/response."""

    agent: str
    model: str
    start: float
    end: float
    duration: float
    projected_input_tokens: int | None
    input_messages: object
    output: object
    stop_reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def parse_events(log_path: Path) -> list[LogEvent]:
    """Read lifecycle events from a Strands log file."""
    events: list[LogEvent] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(line)
        if not match:
            continue
        events.append(
            LogEvent(
                timestamp=datetime.strptime(
                    match.group("time"), "%Y-%m-%d %H:%M:%S,%f"
                ),
                agent=match.group("agent"),
                session_id=match.group("session_id") or "-",
                request_id=match.group("request_id") or "-",
                event=match.group("event"),
                details=match.group("details"),
            )
        )
    return events


def completed_requests(events: list[LogEvent]) -> list[list[LogEvent]]:
    """Return completed supervisor request slices in chronological order."""
    requests: list[list[LogEvent]] = []

    # Context-aware logs can be grouped exactly even when browser sessions run
    # concurrently and their records interleave in the shared file.
    grouped: dict[tuple[str, str], list[LogEvent]] = {}
    for event in events:
        if event.request_id == "-":
            continue
        key = (event.session_id, event.request_id)
        grouped.setdefault(key, []).append(event)
    for request in grouped.values():
        if any(
            event.agent == "operations-supervisor"
            and event.event == "REQUEST_FINISHED"
            for event in request
        ):
            requests.append(request)

    # Retain support for logs written before session/request IDs were added.
    legacy_events = [event for event in events if event.request_id == "-"]
    start_index: int | None = None
    for index, event in enumerate(legacy_events):
        is_supervisor = event.agent == "operations-supervisor"
        if is_supervisor and event.event == "REQUEST_RECEIVED":
            start_index = index
        elif (
            start_index is not None
            and is_supervisor
            and event.event == "REQUEST_FINISHED"
        ):
            requests.append(legacy_events[start_index : index + 1])
            start_index = None
    return sorted(requests, key=lambda request: request[0].timestamp)


def build_spans(request: list[LogEvent]) -> tuple[list[Span], float]:
    """Convert one request into model and tool execution spans."""
    origin = request[0].timestamp
    total = (request[-1].timestamp - origin).total_seconds()
    spans: list[Span] = []

    for event in request:
        if event.event not in {"MODEL_END", "TOOL_END"}:
            continue
        duration_match = DURATION_RE.search(event.details)
        if not duration_match:
            continue
        duration = float(duration_match.group("duration"))
        end = (event.timestamp - origin).total_seconds()
        start = max(0.0, end - duration)

        if event.event == "MODEL_END":
            stop_reason = re.search(r"stop_reason=([^ |]+)", event.details)
            reason = stop_reason.group(1) if stop_reason else "unknown"
            spans.append(
                Span(
                    lane=f"{event.agent} / model",
                    start=start,
                    end=end,
                    label=f"model → {reason}",
                    detail=event.details,
                    kind="model",
                )
            )
        else:
            name_match = TOOL_NAME_RE.search(event.details)
            name = name_match.group("name") if name_match else "unknown-tool"
            is_error = '"status": "error"' in event.details or "status=error" in event.details
            spans.append(
                Span(
                    lane=f"{event.agent} / tool:{name}",
                    start=start,
                    end=end,
                    label=f"{name}{' (error)' if is_error else ''}",
                    detail=event.details,
                    kind="error" if is_error else "tool",
                )
            )
    return spans, total


def _json_after(details: str, marker: str) -> object | None:
    """Decode a JSON value stored after a final lifecycle detail marker."""
    _, separator, value = details.partition(marker)
    if not separator:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def build_model_invocations(request: list[LogEvent]) -> list[ModelInvocation]:
    """Pair model start/end events and recover per-call messages and token usage."""
    origin = request[0].timestamp
    histories: dict[str, list[dict[str, object]]] = defaultdict(list)
    pending: dict[str, list[dict[str, object]]] = defaultdict(list)
    invocations: list[ModelInvocation] = []

    for event in request:
        if event.event == "MESSAGE_ADDED":
            role_match = re.search(r"role=([^ |]+)", event.details)
            content = _json_after(event.details, "content=")
            if role_match and content is not None:
                histories[event.agent].append(
                    {"role": role_match.group(1), "content": content}
                )
            continue

        if event.event == "MODEL_START":
            model_match = re.search(r"(?:^| \| )model=([^ |]+)", event.details)
            projected_match = re.search(
                r"projected_input_tokens=([^ |]+)", event.details
            )
            recorded_messages = _json_after(event.details, "input_messages=")
            pending[event.agent].append(
                {
                    "timestamp": event.timestamp,
                    "model": model_match.group(1) if model_match else "not recorded",
                    "projected": (
                        int(projected_match.group(1))
                        if projected_match and projected_match.group(1).isdigit()
                        else None
                    ),
                    "messages": (
                        recorded_messages
                        if recorded_messages is not None
                        else list(histories[event.agent])
                    ),
                }
            )
            continue

        if event.event != "MODEL_END" or not pending[event.agent]:
            continue

        started = pending[event.agent].pop(0)
        duration_match = DURATION_RE.search(event.details)
        duration = (
            float(duration_match.group("duration")) if duration_match else 0.0
        )
        stop_match = re.search(r"stop_reason=([^ |]+)", event.details)
        invocations.append(
            ModelInvocation(
                agent=event.agent,
                model=str(started["model"]),
                start=(started["timestamp"] - origin).total_seconds(),
                end=(event.timestamp - origin).total_seconds(),
                duration=duration,
                projected_input_tokens=started["projected"],
                input_messages=started["messages"],
                output=_json_after(event.details, "content="),
                stop_reason=stop_match.group(1) if stop_match else "unknown",
            )
        )

    # Strands exposes exact per-cycle token counts on REQUEST_FINISHED. Assign
    # the final N cycle usages to the N model calls from this request/agent;
    # this also handles stateful agents whose metrics include earlier cycles.
    for event in request:
        if event.event != "REQUEST_FINISHED":
            continue
        agent_calls = [call for call in invocations if call.agent == event.agent]
        usages = list(USAGE_RE.finditer(event.details))
        for call, usage in zip(agent_calls, usages[-len(agent_calls) :]):
            call.input_tokens = int(usage.group("input"))
            call.output_tokens = int(usage.group("output"))
            call.total_tokens = int(usage.group("total"))

    return sorted(invocations, key=lambda call: call.start)


def render_model_invocations(invocations: list[ModelInvocation]) -> str:
    """Render expandable model-call inputs, outputs, timing, and token counts."""
    if not invocations:
        return '<p class="empty">No completed model calls were found.</p>'

    cards: list[str] = []
    for index, call in enumerate(invocations, start=1):
        token_text = (
            f"{call.input_tokens:,} input · {call.output_tokens:,} output · "
            f"{call.total_tokens:,} total"
            if call.total_tokens is not None
            else "token usage not recorded"
        )
        projected = (
            str(call.projected_input_tokens)
            if call.projected_input_tokens is not None
            else "not recorded"
        )
        input_json = json.dumps(call.input_messages, indent=2, ensure_ascii=False)
        output_json = json.dumps(call.output, indent=2, ensure_ascii=False)
        cards.append(
            '<details class="model-call">'
            f'<summary><span class="call-number">#{index}</span> '
            f'<strong>{html.escape(call.agent)}</strong> · '
            f'{html.escape(call.model)} · {call.duration:.3f}s · '
            f'{html.escape(token_text)} · {html.escape(call.stop_reason)}</summary>'
            '<div class="call-meta">'
            f'Elapsed: {call.start:.3f}s–{call.end:.3f}s · '
            f'Projected input tokens: {html.escape(projected)}</div>'
            '<div class="io-grid"><section><h3>Input messages</h3>'
            f'<pre>{html.escape(input_json)}</pre></section>'
            '<section><h3>Model output</h3>'
            f'<pre>{html.escape(output_json)}</pre></section></div>'
            '</details>'
        )
    return '<div class="model-calls">' + "".join(cards) + "</div>"


def render_html(
    spans: list[Span],
    total: float,
    title: str,
    session_id: str,
    request_id: str,
    invocations: list[ModelInvocation] | None = None,
) -> str:
    """Render spans as a dependency-free standalone SVG timeline."""
    lanes = list(dict.fromkeys(span.lane for span in spans))
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    width, left, right, top, row_height, bottom = 1200, 260, 30, 70, 48, 65
    plot_width = width - left - right
    height = top + len(lanes) * row_height + bottom

    def x(value: float) -> float:
        return left + (value / total * plot_width if total else 0)

    palette = {"model": "#4f7cff", "tool": "#26a269", "error": "#d64545"}
    ticks = 8
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<text class="heading" x="0" y="24">{html.escape(title)}</text>',
        f'<text class="subtitle" x="0" y="47">Session: {html.escape(session_id)} · '
        f'Request: {html.escape(request_id)} · Total wall time: {total:.3f} seconds</text>',
    ]

    for index in range(ticks + 1):
        value = total * index / ticks
        px = x(value)
        svg_parts.extend(
            [
                f'<line class="grid" x1="{px:.2f}" x2="{px:.2f}" '
                f'y1="{top}" y2="{top + len(lanes) * row_height}"/>',
                f'<text class="tick" x="{px:.2f}" y="{height - 22}" '
                f'text-anchor="middle">{value:.1f}s</text>',
            ]
        )

    for lane, index in lane_index.items():
        y = top + index * row_height
        svg_parts.append(
            f'<text class="lane" x="{left - 12}" y="{y + 29}" '
            f'text-anchor="end">{html.escape(lane)}</text>'
        )

    for span in spans:
        y = top + lane_index[span.lane] * row_height + 10
        natural_width = x(span.end) - x(span.start)
        bar_width = max(4.0, natural_width)
        tooltip = (
            f"{span.label} | {span.start:.3f}s–{span.end:.3f}s | "
            f"duration {span.end - span.start:.3f}s"
        )
        svg_parts.append(
            f'<g class="span"><title>{html.escape(tooltip)}</title>'
            f'<rect x="{x(span.start):.2f}" y="{y}" width="{bar_width:.2f}" '
            f'height="28" rx="3" fill="{palette[span.kind]}"/>'
        )
        if natural_width >= 105:
            svg_parts.append(
                f'<text class="bar-label" x="{x(span.start) + 7:.2f}" '
                f'y="{y + 19}">{html.escape(span.label)}</text>'
            )
        svg_parts.append("</g>")

    svg_parts.extend(
        [
            f'<text class="axis-title" x="{left + plot_width / 2:.2f}" '
            f'y="{height - 2}" text-anchor="middle">Elapsed time from request receipt</text>',
            "</svg>",
        ]
    )
    svg = "\n".join(svg_parts)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 24px; font-family: system-ui, sans-serif; background: Canvas; color: CanvasText; }}
  svg {{ width: 100%; min-width: 720px; height: auto; }}
  .heading {{ font-size: 20px; font-weight: 600; fill: CanvasText; }}
  .subtitle, .tick, .axis-title {{ font-size: 12px; fill: GrayText; }}
  .lane {{ font-size: 12px; fill: CanvasText; }}
  .grid {{ stroke: color-mix(in srgb, CanvasText 18%, transparent); stroke-width: 1; }}
  .bar-label {{ font-size: 11px; fill: white; pointer-events: none; }}
  .span:hover rect {{ stroke: CanvasText; stroke-width: 2; }}
  .scroller {{ overflow-x: auto; }}
  h2 {{ margin-top: 32px; }}
  .model-call {{ border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius: 7px; margin: 12px 0; padding: 10px 14px; }}
  .model-call summary {{ cursor: pointer; line-height: 1.5; }}
  .call-number {{ color: GrayText; }}
  .call-meta, .empty {{ color: GrayText; margin: 12px 0; }}
  .io-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }}
  .io-grid h3 {{ font-size: 14px; }}
  pre {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 12px; border-radius: 6px; background: color-mix(in srgb, CanvasText 8%, Canvas); font-size: 12px; }}
  @media (max-width: 900px) {{ .io-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body><div class="scroller">{svg}</div>
<h2>Model invocations</h2>
{render_model_invocations(invocations or [])}
</body>
</html>
"""


def render_session_html(
    plots: list[
        tuple[list[Span], float, str, str, str, list[ModelInvocation]]
    ],
    session_id: str,
) -> str:
    """Render every completed request in one session into one HTML report."""
    sections: list[str] = []
    for spans, total, title, plot_session_id, request_id, invocations in plots:
        document = render_html(
            spans, total, title, plot_session_id, request_id, invocations
        )
        svg_start = document.index("<svg ")
        svg_end = document.index("</svg>", svg_start) + len("</svg>")
        sections.append(
            '<section class="request-timeline">'
            f"<div class=\"scroller\">{document[svg_start:svg_end]}</div>"
            '<h2>Model invocations</h2>'
            f'{render_model_invocations(invocations)}'
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent timelines — session {html.escape(session_id)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 24px; font-family: system-ui, sans-serif; background: Canvas; color: CanvasText; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .summary {{ color: GrayText; margin-bottom: 28px; }}
  .request-timeline {{ margin-bottom: 34px; border-bottom: 1px solid color-mix(in srgb, CanvasText 18%, transparent); }}
  svg {{ width: 100%; min-width: 720px; height: auto; }}
  .heading {{ font-size: 20px; font-weight: 600; fill: CanvasText; }}
  .subtitle, .tick, .axis-title {{ font-size: 12px; fill: GrayText; }}
  .lane {{ font-size: 12px; fill: CanvasText; }}
  .grid {{ stroke: color-mix(in srgb, CanvasText 18%, transparent); stroke-width: 1; }}
  .bar-label {{ font-size: 11px; fill: white; pointer-events: none; }}
  .span:hover rect {{ stroke: CanvasText; stroke-width: 2; }}
  .scroller {{ overflow-x: auto; }}
  .model-call {{ border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius: 7px; margin: 12px 0; padding: 10px 14px; }}
  .model-call summary {{ cursor: pointer; line-height: 1.5; }}
  .call-number {{ color: GrayText; }}
  .call-meta, .empty {{ color: GrayText; margin: 12px 0; }}
  .io-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }}
  .io-grid h3 {{ font-size: 14px; }}
  pre {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 12px; border-radius: 6px; background: color-mix(in srgb, CanvasText 8%, Canvas); font-size: 12px; }}
  @media (max-width: 900px) {{ .io-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Session request timelines</h1>
<div class="summary">Session: {html.escape(session_id)} · {len(plots)} completed request(s)</div>
{"".join(sections)}
</body>
</html>
"""


def main() -> None:
    """Parse arguments and generate a timeline for a completed request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).with_name("logs") / "strands.log",
        help="Strands log path (default: logs/strands.log)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agent_timeline.html"),
        help="Standalone HTML output path (default: agent_timeline.html)",
    )
    parser.add_argument(
        "--request-index",
        type=int,
        default=0,
        help="Completed request counted from newest: 0=latest, 1=previous",
    )
    parser.add_argument(
        "--session-id",
        help="Plot all completed requests belonging to this Streamlit session UUID",
    )
    parser.add_argument(
        "--request-id",
        help="Select this message/request UUID within --session-id",
    )
    args = parser.parse_args()

    requests = completed_requests(parse_events(args.log))
    if args.request_id is not None and args.session_id is None:
        raise SystemExit("--request-id requires --session-id.")

    if args.session_id is not None:
        matching = [
            request
            for request in requests
            if request[0].session_id == args.session_id
        ]
        if args.request_id is not None:
            matching = [
                request
                for request in matching
                if request[0].request_id == args.request_id
            ]
        if not matching:
            request_detail = (
                f", request_id={args.request_id!r}"
                if args.request_id is not None
                else ""
            )
            raise SystemExit(
                "No completed supervisor request found for "
                f"session_id={args.session_id!r}{request_detail}."
            )
        selected_requests = matching
    else:
        if args.request_index < 0 or args.request_index >= len(requests):
            raise SystemExit(
                f"Request index {args.request_index} is unavailable; "
                f"found {len(requests)} completed supervisor request(s)."
            )
        selected_requests = [requests[-1 - args.request_index]]

    plots: list[
        tuple[list[Span], float, str, str, str, list[ModelInvocation]]
    ] = []
    for request in selected_requests:
        spans, total = build_spans(request)
        invocations = build_model_invocations(request)
        started = request[0].timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        title = f"Agent request timeline — {started}"
        plots.append(
            (
                spans,
                total,
                title,
                request[0].session_id,
                request[0].request_id,
                invocations,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.session_id is not None and args.request_id is None:
        document = render_session_html(plots, args.session_id)
    else:
        document = render_html(*plots[0])
    args.output.write_text(document, encoding="utf-8")
    span_count = sum(len(plot[0]) for plot in plots)
    invocation_count = sum(len(plot[5]) for plot in plots)
    print(
        f"Wrote {args.output.resolve()} "
        f"({len(plots)} request(s), {span_count} spans, "
        f"{invocation_count} model invocation(s))"
    )


if __name__ == "__main__":
    main()
