"""OpenTelemetry tracing for agent nodes. Spans stay in memory and can be dumped to JSON lines.

Point UPW_OTEL_CONSOLE=1 to also print spans, or swap the exporter for OTLP to ship them to a
collector (Jaeger, Tempo, Phoenix) without touching the agent code.
"""

from __future__ import annotations

import functools
import json
import statistics
from collections import defaultdict
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER = InMemorySpanExporter()
_PROVIDER: TracerProvider | None = None


def setup(console: bool = False) -> None:
    global _PROVIDER
    if _PROVIDER is not None:
        return
    _PROVIDER = TracerProvider()
    _PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    if console:
        _PROVIDER.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(_PROVIDER)


def tracer() -> trace.Tracer:
    setup()
    return trace.get_tracer("upstream_watch")


def traced(name: str):
    """Wrap a LangGraph node so each call is a span with the run mode and outcome."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            with tracer().start_as_current_span(f"node.{name}") as span:
                span.set_attribute("upw.mode", str(state.get("mode", "")))
                span.set_attribute("upw.intent", str(state.get("intent", "")))
                out = fn(state, *args, **kwargs)
                if isinstance(out, dict):
                    for key in ("grounded", "fallback_used"):
                        if key in out:
                            span.set_attribute(f"upw.{key}", bool(out[key]))
                    if "hits" in out:
                        span.set_attribute("upw.n_hits", len(out["hits"]))
                return out

        return wrapper

    return deco


def clear() -> None:
    _EXPORTER.clear()


def latency_summary() -> dict[str, dict[str, float]]:
    """p50 and p95 latency in milliseconds per span name."""
    durations: dict[str, list[float]] = defaultdict(list)
    for span in _EXPORTER.get_finished_spans():
        durations[span.name].append((span.end_time - span.start_time) / 1e6)
    out = {}
    for name, values in sorted(durations.items()):
        values.sort()
        out[name] = {
            "n": len(values),
            "p50_ms": round(statistics.median(values), 2),
            "p95_ms": round(values[int(0.95 * (len(values) - 1))], 2),
        }
    return out


def dump(path: Path) -> int:
    spans = _EXPORTER.get_finished_spans()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for s in spans:
            fh.write(
                json.dumps(
                    {
                        "name": s.name,
                        "trace_id": f"{s.context.trace_id:032x}",
                        "span_id": f"{s.context.span_id:016x}",
                        "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                        "duration_ms": round((s.end_time - s.start_time) / 1e6, 3),
                        "attributes": dict(s.attributes or {}),
                    }
                )
                + "\n"
            )
    return len(spans)
