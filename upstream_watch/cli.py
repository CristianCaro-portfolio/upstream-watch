"""Command line: python -m upstream_watch <command>."""

from __future__ import annotations

import json
import logging
import platform
import time
from importlib.metadata import version
from pathlib import Path

import typer

from . import config, evaluate, hydro, pipeline, replay, report, telemetry
from .agent import ask, resume_brief, start_brief

app = typer.Typer(add_completion=False, help="Upstream gauges as early warning for the towns downstream.")
log = logging.getLogger("upstream_watch")

REPLAY_EVENT = "2025-05-15"  # Western Maryland flash flood, minor flood at Little Falls


def _setup(verbose: bool) -> config.Settings:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = config.get_settings()
    telemetry.setup(console=settings.otel_console)
    return settings


@app.command()
def ingest(offline: bool = typer.Option(False, help="Only use cached responses in data/raw.")) -> None:
    """Download the network, events and hourly stage into data/curated."""
    settings = _setup(True)
    typer.echo(json.dumps(pipeline.ingest(settings, offline=offline), indent=1))


@app.command()
def run(
    sample: bool = typer.Option(False, help="Use data/sample even if data/curated exists."),
    out_dir: Path = typer.Option(None, help="Defaults to reports/."),
    embedder: str = typer.Option(None, help="fastembed (default) or hash."),
) -> None:
    """Backtest, warning replay, retrieval evaluation, brief replay and figures."""
    settings = _setup(True)
    out = out_dir or settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    agent, kit, model, (gauges, edges, stage) = pipeline.agent_from_settings(settings, sample, embedder)
    build_s = time.perf_counter() - t0
    by_lid = {g.lid: g for g in gauges}
    outlet = config.OUTLET_LID

    crest_map = hydro.crests(stage)
    backtest = hydro.backtest(edges, crest_map, outlet)
    lead_times = [
        hydro.warning_lead_times(stage, by_lid, edges, max_hops=h, trigger=t)
        for t, h in (("action", 1), ("action", 3), ("minor", 1), ("minor", 3))
    ]

    lags = {(e.upstream, e.downstream): round(model.edge_lag(e), 1) for e in edges}
    telemetry.clear()
    modes = ["dense", "hybrid", "graph"]
    generated = evaluate.run_eval(agent, evaluate.build_questions(gauges, edges, outlet), modes, lags)
    handwritten = evaluate.run_eval(agent, evaluate.handwritten_questions(gauges, edges, outlet), modes, lags)
    spans = telemetry.latency_summary()
    telemetry.dump(out / "traces.jsonl")

    event_id = (
        REPLAY_EVENT if REPLAY_EVENT in set(stage["event_id"].to_list()) else sorted(stage["event_id"].unique())[-1]
    )
    as_of = replay.busiest_hour(stage, event_id, by_lid)
    readings = replay.readings_at(stage, event_id, as_of)
    state, pending = start_brief(agent, readings, as_of, thread_id=f"replay-{event_id}")
    brief = state["brief"]
    final = (
        resume_brief(
            agent, f"replay-{event_id}", {"approve": True, "reviewer": "replay-bot", "note": "auto approved in replay"}
        )
        if pending
        else state
    )

    metrics = {
        "environment": {
            "python": platform.python_version(),
            "embedder": kit.index.embedder.__class__.__name__,
            "llm_provider": kit.provider.name,
            "packages": {
                p: version(p)
                for p in ("langgraph", "real_ladybug", "qdrant-client", "polars", "mcp", "opentelemetry-sdk")
            },
            "build_seconds": round(build_s, 1),
        },
        "data": {
            "gauges": len(gauges),
            "edges": len(edges),
            "impact_statements": sum(len(g.impacts) for g in gauges),
            "events": sorted(stage["event_id"].unique().to_list()),
            "hourly_stage_rows": stage.height,
            "graph": kit.graph.counts(),
        },
        "travel_time_model": {
            "celerity_kmh": round(model.celerity_kmh, 2),
            "edges_with_learned_lag": len(model.lag_h),
            "edges_total": len(edges),
        },
        "backtest": backtest,
        "lead_times": lead_times,
        "eval": {
            "generated": {"modes": generated["modes"]},
            "handwritten": {"modes": handwritten["modes"]},
        },
        "node_latency": spans,
        "replay": {
            "event_id": event_id,
            "as_of": as_of,
            "gauges_reporting": len(readings),
            "gauges_at_or_above_action": len(brief["gauges_at_or_above_action"]),
            "paused_for_review": pending is not None,
            "status": final.get("status"),
            **replay.score_brief(brief, stage, event_id),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    (out / "eval_rows.json").write_text(json.dumps(generated["rows"] + handwritten["rows"], indent=1))
    (out / "brief.txt").write_text(replay.render_brief(brief))
    d = metrics["data"]
    report.architecture(
        out / "architecture.png",
        f"upstream-watch: {d['gauges']} Potomac basin gauges, {d['edges']} river edges, "
        f"{d['impact_statements']} NWS impact statements, {len(d['events'])} flood events (2024 to 2026)",
    )
    report.results(metrics, out / "results.png")
    typer.echo(_summary(metrics))
    (out / "summary.txt").write_text(_summary(metrics))


def _summary(m: dict) -> str:
    g = m["eval"]["generated"]["modes"]
    h = m["eval"]["handwritten"]["modes"]
    bt = m["backtest"]
    lines = [
        f"data: {m['data']['gauges']} gauges, {m['data']['edges']} edges, {m['data']['impact_statements']} impacts, "
        f"{len(m['data']['events'])} events, {m['data']['hourly_stage_rows']} hourly rows",
        f"embedder {m['environment']['embedder']}, llm {m['environment']['llm_provider']}",
        "retrieval (generated / handwritten answer accuracy, context recall):",
    ]
    for mode in g:
        lines.append(
            f"  {mode:7s} {g[mode]['answer_accuracy']:.3f} / {h[mode]['answer_accuracy']:.3f}  "
            f"recall {g[mode]['context_recall']:.3f}  p50 {g[mode]['latency_p50_ms']} ms"
        )
    lines.append(
        f"backtest MAE h: next gauge learned {bt['one_hop']['learned']['mae_h']} vs celerity "
        f"{bt['one_hop']['celerity']['mae_h']}; to outlet learned {bt['to_outlet']['learned']['mae_h']} vs "
        f"celerity {bt['to_outlet']['celerity']['mae_h']}"
    )
    for r in m["lead_times"]:
        lines.append(
            f"warning {r['trigger']} {r['max_hops']} hop: {r['warned_in_advance']}/{r['floods']} floods, "
            f"median lead {r['median_lead_h']} h, precision {r['precision']}"
        )
    rp = m["replay"]
    lines.append(
        f"replay {rp['event_id']} at {rp['as_of']}: {rp['gauges_at_or_above_action']} gauges at action, "
        f"ETA MAE {rp['eta_mae_h']} h over {rp['eta_checked']} downstream crests, status {rp['status']}"
    )
    return "\n".join(lines)


@app.command("ask")
def ask_cmd(
    question: str,
    mode: str = typer.Option("graph", help="dense, hybrid or graph"),
    sample: bool = typer.Option(False),
) -> None:
    """Ask the agent a question."""
    settings = _setup(False)
    agent, *_ = pipeline.agent_from_settings(settings, sample)
    state = ask(agent, question, mode=mode)
    typer.echo(state["answer"])
    typer.echo(f"\nsources: {', '.join(state.get('hits', []))}\ngrounded: {state.get('grounded')}")


@app.command()
def brief(
    event: str = typer.Option(REPLAY_EVENT, help="Event id, the outlet peak day."),
    as_of: str = typer.Option(None, help="UTC hour, defaults to the busiest hour of the event."),
    approve: bool = typer.Option(None, "--approve/--reject", help="Answer the review instead of stopping."),
    reviewer: str = typer.Option("duty-officer"),
    sample: bool = typer.Option(False),
) -> None:
    """Replay readings from a past event into the brief flow, pause for review, then resume."""
    settings = _setup(False)
    agent, kit, model, (gauges, _, stage) = pipeline.agent_from_settings(settings, sample)
    by_lid = {g.lid: g for g in gauges}
    hour = as_of or replay.busiest_hour(stage, event, by_lid)
    state, pending = start_brief(agent, replay.readings_at(stage, event, hour), hour, thread_id=f"brief-{event}")
    typer.echo(replay.render_brief(state["brief"]))
    if pending is None:
        return
    typer.echo(f"\npaused for review, expected answer schema: {json.dumps(pending.response_schema)}")
    if approve is None:
        return
    final = resume_brief(agent, f"brief-{event}", {"approve": approve, "reviewer": reviewer})
    typer.echo(f"status: {final['status']}")


@app.command()
def mcp() -> None:
    """Serve the gauge graph over MCP (stdio)."""
    from .mcp_server import main

    main()


if __name__ == "__main__":  # pragma: no cover
    app()
