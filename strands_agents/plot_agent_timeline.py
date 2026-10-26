"""Generate a standalone HTML timeline from Strands lifecycle logs."""

from __future__ import annotations

import argparse
import html
import re
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


def render_html(
    spans: list[Span],
    total: float,
    title: str,
    session_id: str,
    request_id: str,
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
</style>
</head>
<body><div class="scroller">{svg}</div></body>
</html>
"""


def render_session_html(
    plots: list[tuple[list[Span], float, str, str, str]], session_id: str
) -> str:
    """Render every completed request in one session into one HTML report."""
    sections: list[str] = []
    for spans, total, title, plot_session_id, request_id in plots:
        document = render_html(
            spans, total, title, plot_session_id, request_id
        )
        svg_start = document.index("<svg ")
        svg_end = document.index("</svg>", svg_start) + len("</svg>")
        sections.append(
            '<section class="request-timeline">'
            f"<div class=\"scroller\">{document[svg_start:svg_end]}</div>"
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

    plots: list[tuple[list[Span], float, str, str, str]] = []
    for request in selected_requests:
        spans, total = build_spans(request)
        started = request[0].timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        title = f"Agent request timeline — {started}"
        plots.append(
            (
                spans,
                total,
                title,
                request[0].session_id,
                request[0].request_id,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.session_id is not None and args.request_id is None:
        document = render_session_html(plots, args.session_id)
    else:
        document = render_html(*plots[0])
    args.output.write_text(document, encoding="utf-8")
    span_count = sum(len(plot[0]) for plot in plots)
    print(
        f"Wrote {args.output.resolve()} "
        f"({len(plots)} request(s), {span_count} spans)"
    )


if __name__ == "__main__":
    main()
