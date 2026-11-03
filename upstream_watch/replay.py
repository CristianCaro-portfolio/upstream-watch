"""Replay a past event: readings at one hour go into the brief flow, then the brief is scored
against what the gauges actually did next."""

from __future__ import annotations

import statistics
from datetime import datetime

import polars as pl

from .hydro import crests


def readings_at(stage: pl.DataFrame, event_id: str, as_of: str) -> dict[str, float]:
    t = datetime.fromisoformat(as_of)
    snap = (
        stage.filter((pl.col("event_id") == event_id) & (pl.col("time") <= t))
        .sort("time")
        .group_by("lid")
        .agg(pl.col("stage_ft").last(), pl.col("time").last().alias("t"))
        .filter((pl.lit(t) - pl.col("t")) <= pl.duration(hours=2))  # stale readings are left out
    )
    return {r["lid"]: float(r["stage_ft"]) for r in snap.iter_rows(named=True)}


def busiest_hour(stage: pl.DataFrame, event_id: str, gauges: dict) -> str:
    """Hour with the most gauges at or above action stage, used as the default replay time."""
    levels = pl.DataFrame(
        [{"lid": lid, "action": g.thresholds.get("action") or g.thresholds.get("minor")} for lid, g in gauges.items()]
    )
    counts = (
        stage.filter(pl.col("event_id") == event_id)
        .join(levels, on="lid")
        .filter(pl.col("stage_ft") >= pl.col("action"))
        .group_by("time")
        .len()
        .sort(["len", "time"], descending=[True, False])
    )
    return counts["time"][0].isoformat()


def score_brief(brief: dict, stage: pl.DataFrame, event_id: str) -> dict:
    """For every downstream ETA in the brief, compare with the crest that gauge actually had."""
    as_of = datetime.fromisoformat(brief["as_of"])
    crest_map = crests(stage.filter(pl.col("event_id") == event_id))
    errors, checked, already_past = [], 0, 0
    seen = set()
    for g in brief["gauges_at_or_above_action"]:
        for d in g["downstream"]:
            key = (g["lid"], d["lid"])
            if key in seen:
                continue
            seen.add(key)
            crest = crest_map.get((d["lid"], event_id))
            if crest is None:
                continue
            if crest.time <= as_of:
                already_past += 1
                continue
            checked += 1
            errors.append(abs((datetime.fromisoformat(d["eta"]) - crest.time).total_seconds() / 3600.0))
    return {
        "eta_checked": checked,
        "eta_mae_h": round(statistics.fmean(errors), 1) if errors else None,
        "eta_median_ae_h": round(statistics.median(errors), 1) if errors else None,
        "downstream_crest_already_passed": already_past,
    }


def render_brief(brief: dict, limit: int = 8) -> str:
    lines = [f"Flood watch brief as of {brief['as_of']} UTC"]
    rows = brief["gauges_at_or_above_action"]
    if not rows:
        return lines[0] + "\nNo gauge at or above action stage."
    lines.append(f"{len(rows)} gauges at or above action stage, most severe first:")
    for g in rows[:limit]:
        nxt = ", ".join(f"{d['name']} ~{d['eta_h']:g} h" for d in g["downstream"][:2]) or "none (outlet)"
        lines.append(f"- {g['name']} ({g['lid']}) {g['stage_ft']:g} ft, {g['category']}. Next: {nxt}.")
        if g["impact"]:
            lines.append(f"  NWS impact at {g['impact']['stage_ft']:g} ft: {g['impact']['statement']}")
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more.")
    return "\n".join(lines)
