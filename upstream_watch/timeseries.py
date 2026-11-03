"""Event detection at the outlet and hourly stage series for every gauge."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from . import sources
from .network import Gauge

log = logging.getLogger(__name__)

STAGE = "00065"  # gage height, ft
DISCHARGE = "00060"  # discharge, ft3/s


@dataclass(frozen=True)
class Event:
    event_id: str
    peak_day: str
    peak_cfs: float
    start: str
    end: str


def detect_events(
    daily: list[dict],
    min_peak_cfs: float = 35000.0,
    min_gap_days: int = 10,
    lead_days: int = 6,
    tail_days: int = 3,
) -> list[Event]:
    """Independent high flow events from the outlet daily mean discharge.

    A day is a peak when its value is the maximum within +/- min_gap_days and above
    min_peak_cfs. The window opens lead_days earlier so upstream crests are inside it.
    """
    rows = sorted(
        ((r["time"][:10], float(r["value"])) for r in daily if r.get("value") not in (None, "")),
        key=lambda x: x[0],
    )
    events: list[Event] = []
    for i, (day, val) in enumerate(rows):
        if val < min_peak_cfs:
            continue
        lo, hi = max(0, i - min_gap_days), min(len(rows), i + min_gap_days + 1)
        if val < max(v for _, v in rows[lo:hi]):
            continue
        if any(v == val for _, v in rows[lo:i]):  # ties: keep the first day
            continue
        peak = date.fromisoformat(day)
        events.append(
            Event(
                event_id=peak.strftime("%Y-%m-%d"),
                peak_day=day,
                peak_cfs=val,
                start=(peak - timedelta(days=lead_days)).isoformat() + "T00:00:00Z",
                end=(peak + timedelta(days=tail_days)).isoformat() + "T00:00:00Z",
            )
        )
    return events


def to_hourly(raw: list[dict], lid: str, event_id: str) -> pl.DataFrame:
    """15 minute USGS values to an hourly mean series. Provisional values are kept and flagged."""
    schema = {"lid": pl.Utf8, "event_id": pl.Utf8, "time": pl.Datetime("us", "UTC"), "stage_ft": pl.Float64}
    clean = [
        {"time": r["time"], "value": r["value"], "approved": r.get("approval_status") == "Approved"}
        for r in raw
        if r.get("value") not in (None, "")
    ]
    if not clean:
        return pl.DataFrame(schema=schema | {"approved_share": pl.Float64})
    df = (
        pl.DataFrame(clean)
        .with_columns(
            pl.col("time").str.to_datetime(time_zone="UTC", time_unit="us"),
            pl.col("value").cast(pl.Float64).alias("stage_ft"),
        )
        .filter(pl.col("stage_ft") > -100)  # USGS uses large negative sentinels for missing
        .sort("time")
        .group_by_dynamic("time", every="1h")
        .agg(pl.col("stage_ft").mean(), pl.col("approved").cast(pl.Float64).mean().alias("approved_share"))
        .with_columns(pl.lit(lid).alias("lid"), pl.lit(event_id).alias("event_id"))
    )
    return df.select("lid", "event_id", "time", "stage_ft", "approved_share")


def fetch_event_stages(
    client: sources.CachedClient, gauges: list[Gauge], events: list[Event], workers: int = 4
) -> pl.DataFrame:
    jobs = [(g, e) for e in events for g in gauges]

    def one(job: tuple[Gauge, Event]) -> pl.DataFrame | None:
        gauge, event = job
        try:
            raw = sources.usgs_continuous(client, gauge.usgs_id, STAGE, event.start, event.end)
        except sources.SourceError as exc:
            log.warning("no stage for %s in %s: %s", gauge.lid, event.event_id, exc)
            return None
        return to_hourly(raw, gauge.lid, event.event_id)

    with ThreadPoolExecutor(workers) as pool:
        frames = [f for f in pool.map(one, jobs) if f is not None and f.height]
    log.info("hourly stage for %d gauge-event pairs out of %d", len(frames), len(jobs))
    return pl.concat(frames) if frames else pl.DataFrame()


def write_partitioned(df: pl.DataFrame, root: Path) -> None:
    """Hive style layout: root/event_id=YYYY-MM-DD/part.parquet."""
    root.mkdir(parents=True, exist_ok=True)
    for (event_id,), part in df.group_by("event_id"):
        out = root / f"event_id={event_id}"
        out.mkdir(parents=True, exist_ok=True)
        part.drop("event_id").write_parquet(out / "part.parquet")


def read_partitioned(root: Path) -> pl.DataFrame:
    return pl.read_parquet(root / "**" / "*.parquet", hive_partitioning=True).with_columns(
        pl.col("event_id").cast(pl.Utf8)
    )
