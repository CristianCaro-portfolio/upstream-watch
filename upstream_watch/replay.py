"""Replay a past event: readings at one hour go into the brief flow, then the brief is scored
against what the gauges actually did next."""

from __future__ import annotations

import statistics
from datetime import datetime

import polars as pl

from .hydro import crests
from .lang import EN, Lang


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
        [{"lid": lid, "action": g.thresholds.get("action") or g.thresholds.get("minor")} for lid, g in gauges.items()],
        schema={"lid": pl.Utf8, "action": pl.Float64},
    )
    counts = (
        stage.filter(pl.col("event_id") == event_id)
        .join(levels, on="lid")
        .filter(pl.col("stage_ft") >= pl.col("action"))
        .group_by("time")
        .len()
        .sort(["len", "time"], descending=[True, False])
    )
    if counts.height == 0:  # nobody crossed: use the hour with the highest stage anywhere
        part = stage.filter(pl.col("event_id") == event_id).sort("stage_ft", descending=True)
        return part["time"][0].isoformat()
    return counts["time"][0].isoformat()


def gauges_over(stage: pl.DataFrame, event_id: str, gauges: dict) -> int:
    """How many gauges were at or above their first level at the busiest hour of an event."""
    hour = busiest_hour(stage, event_id, gauges)
    readings = readings_at(stage, event_id, hour)
    count = 0
    for lid, value in readings.items():
        g = gauges.get(lid)
        level = g and (g.thresholds.get("action") or g.thresholds.get("minor"))
        if level is not None and value >= level:
            count += 1
    return count


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


def render_brief(brief: dict, limit: int = 8, lang: Lang = EN) -> str:
    lines = [lang.t("brief_title", as_of=brief["as_of"])]
    rows = brief["gauges_at_or_above_action"]
    if not rows:
        return lines[0] + "\n" + lang.text["brief_none"]
    lines.append(lang.t("brief_count", n=len(rows)))
    for g in rows[:limit]:
        nxt = ", ".join(lang.t("brief_next", name=d["name"], hours=f"{d['eta_h']:g}") for d in g["downstream"][:2])
        lines.append(
            lang.t(
                "brief_row", name=g["name"], lid=g["lid"], stage=f"{g['stage_ft']:g}", unit=lang.unit,
                category=g["category"], nxt=nxt or lang.text["brief_outlet"],
            )
        )  # fmt: skip
        if g["impact"]:
            lines.append(
                lang.t(
                    "brief_impact",
                    stage=f"{g['impact']['stage_ft']:g}",
                    unit=lang.unit,
                    statement=g["impact"]["statement"],
                )
            )
    if len(rows) > limit:
        lines.append(lang.t("brief_more", n=len(rows) - limit))
    return "\n".join(lines)
