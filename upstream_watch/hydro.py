"""Crest detection, travel time learning per river edge, and a leave one event out backtest."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import polars as pl

from .network import Edge, Gauge


@dataclass(frozen=True)
class Crest:
    lid: str
    event_id: str
    time: object  # datetime
    stage_ft: float
    rise_ft: float


def crests(stage: pl.DataFrame, min_rise_ft: float = 1.0, edge_hours: int = 3) -> dict[tuple[str, str], Crest]:
    """Highest hourly stage per gauge and event, kept only if it is a real crest.

    A crest needs a rise of at least min_rise_ft above the first hours of the window and must
    not sit on the last hours of the window, where the true crest could be outside it.
    """
    out: dict[tuple[str, str], Crest] = {}
    for (lid, event_id), part in stage.sort("time").group_by(["lid", "event_id"]):
        if part.height < 24:
            continue
        values = part["stage_ft"].to_list()
        times = part["time"].to_list()
        idx = max(range(len(values)), key=values.__getitem__)
        base = min(values[: min(24, len(values))])
        rise = values[idx] - base
        if rise < min_rise_ft or idx >= len(values) - edge_hours:
            continue
        out[(lid, event_id)] = Crest(lid, event_id, times[idx], float(values[idx]), float(rise))
    return out


def edge_lags(edges: list[Edge], crest_map: dict[tuple[str, str], Crest]) -> pl.DataFrame:
    """Observed crest to crest travel time for every edge and event where both crests exist."""
    rows = []
    events = sorted({e for _, e in crest_map})
    for edge in edges:
        for event_id in events:
            up = crest_map.get((edge.upstream, event_id))
            down = crest_map.get((edge.downstream, event_id))
            if not up or not down:
                continue
            lag_h = (down.time - up.time).total_seconds() / 3600.0
            # A downstream crest more than 4 days later, or more than 12 h before, belongs to another wave.
            if -12 <= lag_h <= 96:
                rows.append(
                    {
                        "upstream": edge.upstream,
                        "downstream": edge.downstream,
                        "event_id": event_id,
                        "river_km": edge.river_km,
                        "lag_h": lag_h,
                    }
                )
    schema = {
        "upstream": pl.Utf8,
        "downstream": pl.Utf8,
        "event_id": pl.Utf8,
        "river_km": pl.Float64,
        "lag_h": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


@dataclass
class TravelTimeModel:
    """Median crest lag per edge, with a constant celerity fallback for edges never observed."""

    lag_h: dict[tuple[str, str], float]
    n_obs: dict[tuple[str, str], int]
    celerity_kmh: float

    @classmethod
    def fit(cls, lags: pl.DataFrame, min_obs: int = 1) -> TravelTimeModel:
        """min_obs: an edge needs that many observed crest pairs to use its own lag; with fewer
        it falls back to the basin wave speed. 1 for the US data, 3 for noisier networks."""
        lag_h: dict[tuple[str, str], float] = {}
        n_obs: dict[tuple[str, str], int] = {}
        for (up, down), part in lags.group_by(["upstream", "downstream"]):
            values = part["lag_h"].to_list()
            n_obs[(up, down)] = len(values)
            if len(values) >= min_obs:
                lag_h[(up, down)] = float(statistics.median(values))
        positive = lags.filter(pl.col("lag_h") > 1)
        celerity = (
            float(statistics.median((positive["river_km"] / positive["lag_h"]).to_list())) if positive.height else 4.0
        )
        return cls(lag_h=lag_h, n_obs=n_obs, celerity_kmh=celerity)

    def edge_lag(self, edge: Edge) -> float:
        learned = self.lag_h.get((edge.upstream, edge.downstream))
        return learned if learned is not None else edge.river_km / self.celerity_kmh

    def celerity_lag(self, edge: Edge) -> float:
        return edge.river_km / self.celerity_kmh


def path_to(edges: list[Edge], source: str, target: str) -> list[Edge] | None:
    nxt = {e.upstream: e for e in edges}
    path, cur = [], source
    while cur != target:
        edge = nxt.get(cur)
        if edge is None:
            return None
        path.append(edge)
        cur = edge.downstream
    return path


def backtest(edges: list[Edge], crest_map: dict[tuple[str, str], Crest], outlet: str, min_obs: int = 1) -> dict:
    """Leave one event out: fit on the other events, predict crest arrival on the held out one.

    Two targets: the next gauge downstream (one hop) and the outlet (whole path). Two methods:
    learned per edge median lag, and the constant celerity baseline fitted on the same data.
    """
    all_lags = edge_lags(edges, crest_map)
    events = sorted(all_lags["event_id"].unique().to_list())
    one_hop = {"learned": [], "celerity": []}
    to_outlet = {"learned": [], "celerity": []}
    for held in events:
        model = TravelTimeModel.fit(all_lags.filter(pl.col("event_id") != held), min_obs=min_obs)
        for row in all_lags.filter(pl.col("event_id") == held).iter_rows(named=True):
            edge = Edge(row["upstream"], row["downstream"], row["river_km"])
            one_hop["learned"].append(abs(model.edge_lag(edge) - row["lag_h"]))
            one_hop["celerity"].append(abs(model.celerity_lag(edge) - row["lag_h"]))
        out_crest = crest_map.get((outlet, held))
        if out_crest is None:
            continue
        for (lid, event_id), crest in crest_map.items():
            if event_id != held or lid == outlet:
                continue
            path = path_to(edges, lid, outlet)
            if not path:
                continue
            actual = (out_crest.time - crest.time).total_seconds() / 3600.0
            if not -12 <= actual <= 120:
                continue
            to_outlet["learned"].append(abs(sum(model.edge_lag(e) for e in path) - actual))
            to_outlet["celerity"].append(abs(sum(model.celerity_lag(e) for e in path) - actual))

    def summary(errors: list[float]) -> dict:
        if not errors:
            return {"n": 0}
        errors = sorted(errors)
        return {
            "n": len(errors),
            "mae_h": round(statistics.fmean(errors), 2),
            "median_ae_h": round(statistics.median(errors), 2),
            "p90_ae_h": round(errors[int(0.9 * (len(errors) - 1))], 2),
        }

    return {
        "events": events,
        "edge_observations": all_lags.height,
        "one_hop": {k: summary(v) for k, v in one_hop.items()},
        "to_outlet": {k: summary(v) for k, v in to_outlet.items()},
    }


def upstream_of(edges: list[Edge], target: str, max_hops: int = 3) -> dict[str, int]:
    """Gauges upstream of target within max_hops, with their hop count."""
    parents: dict[str, list[str]] = {}
    for e in edges:
        parents.setdefault(e.downstream, []).append(e.upstream)
    found: dict[str, int] = {}
    frontier = [(target, 0)]
    while frontier:
        node, hops = frontier.pop()
        if hops == max_hops:
            continue
        for p in parents.get(node, []):
            if p not in found or found[p] > hops + 1:
                found[p] = hops + 1
                frontier.append((p, hops + 1))
    return found


def first_crossing(series: pl.DataFrame, level: float):
    hit = series.filter(pl.col("stage_ft") >= level).sort("time")
    return hit["time"][0] if hit.height else None


def warning_lead_times(
    stage: pl.DataFrame,
    gauges: dict[str, Gauge],
    edges: list[Edge],
    max_hops: int = 3,
    trigger: str = "action",
) -> dict:
    """Replay every event hour by hour, as an upstream watch would have seen it.

    For each gauge that reached its minor flood stage, the warning is the first hour any gauge
    within max_hops upstream crossed its own trigger stage (action or minor; minor is used when
    a gauge has no action stage).
    Lead time is how much earlier that was than the target's own minor crossing. Targets with
    an upstream warning but no flooding count as false alarms.
    """
    leads, misses, false_alarms, alarms = [], 0, 0, 0
    by_key = {k: part for k, part in stage.group_by(["lid", "event_id"])}
    events = sorted(stage["event_id"].unique().to_list())
    for event_id in events:
        for lid, gauge in gauges.items():
            ups = upstream_of(edges, lid, max_hops)
            if not ups:
                continue
            warn_times = []
            for up in ups:
                part = by_key.get((up, event_id))
                level = gauges[up].thresholds.get(trigger) or gauges[up].thresholds.get("minor")
                if part is None or level is None:
                    continue
                t = first_crossing(part, level)
                if t is not None:
                    warn_times.append(t)
            warn = min(warn_times) if warn_times else None
            target = by_key.get((lid, event_id))
            minor = gauge.thresholds.get("minor")
            if minor is None:
                continue
            flood_at = first_crossing(target, minor) if target is not None else None
            if warn is not None:
                alarms += 1
            if flood_at is None:
                if warn is not None:
                    false_alarms += 1
                continue
            if warn is None:
                misses += 1
                continue
            leads.append((flood_at - warn).total_seconds() / 3600.0)
    floods = len(leads) + misses
    positive = [x for x in leads if x > 0]
    return {
        "trigger": trigger,
        "max_hops": max_hops,
        "floods": floods,
        "warned_in_advance": len(positive),
        "missed_or_late": misses + (len(leads) - len(positive)),
        "median_lead_h": round(statistics.median(positive), 1) if positive else None,
        "alarms": alarms,
        "false_alarms": false_alarms,
        "precision": round((alarms - false_alarms) / alarms, 3) if alarms else None,
    }
