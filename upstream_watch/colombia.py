"""Colombia region: IDEAM river level sensors from datos.gov.co and the HydroRIVERS network.

What changes from the US region:

* Readings come from the IDEAM "Nivel Instantaneo del Rio" open dataset (datos.gov.co, id
  bdmn-sqnh, CC BY-SA 4.0), queried through the Socrata API with no key. Times are local
  Colombia time (UTC-5) and levels are in meters.
* The river network comes from HydroRIVERS v1.0 (HydroSHEDS): each station is snapped to the
  largest river reach within 2 km, and the next station downstream is found by following
  NEXT_DOWN. Distances use DIST_DN_KM.
* There are no published per station alert thresholds or impact statements in open data, so
  the reference levels are statistical: "vigilancia" is the 90th percentile of daily maximum
  level and "alto" the 98th percentile, over the downloaded history. They are not IDEAM alert
  levels and the README says so.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from . import sources
from .network import Edge, Gauge
from .timeseries import Event

log = logging.getLogger(__name__)

SODA_LEVELS = "https://www.datos.gov.co/resource/bdmn-sqnh.json"
HYDRORIVERS_SA = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_sa_shp.zip"
STATION_FIELDS = "codigoestacion,nombreestacion,departamento,municipio,zonahidrografica,latitud,longitud"
UTC_OFFSET_H = 5  # Colombia has no daylight saving time

# Cauca basin box, used to cut HydroRIVERS and to pick candidate stations.
CAUCA_BBOX = (-77.8, 1.3, -74.3, 9.5)
CAUCA_ZONES = ("Cauca", "Bajo Magdalena- Cauca -San Jorge", "Nechí")


@dataclass(frozen=True)
class Station:
    code: str
    raw_name: str
    department: str
    municipality: str
    zone: str
    lat: float
    lon: float


def clean_name(raw: str) -> str:
    """'LA VIRGINIA - AUT [26177030]' -> 'La Virginia'."""
    name = re.sub(r"\[[^\]]*\]", "", raw)
    name = re.sub(r"\s*-?\s*AUT\b", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip(" -")
    small = {"de", "del", "la", "las", "el", "los", "y"}
    words = [w.lower() if i and w.lower() in small else w.capitalize() for i, w in enumerate(name.split())]
    return " ".join(words)


def station_label(st: Station) -> str:
    place = clean_name(st.raw_name)
    muni = clean_name(re.sub(r"\([^)]*\)", "", st.municipality))
    return place if place.lower() == muni.lower() else f"{place} ({muni})"


def short_code(code: str) -> str:
    return code.lstrip("0") or code


# datos.gov.co ---------------------------------------------------------------------------


def active_stations(client: sources.CachedClient, since: str) -> list[Station]:
    params = {
        "$select": f"{STATION_FIELDS},count(*) as n",
        "$where": f"fechaobservacion > '{since}'",
        "$group": STATION_FIELDS,
        "$limit": 5000,
    }
    rows = client.get_json(SODA_LEVELS, params)
    # A station can appear with two locations (a placeholder like 8, -75 and the real one).
    # Keep the location that carries most of the readings.
    best: dict[str, dict] = {}
    for r in rows:
        if not r.get("latitud"):
            continue
        prev = best.get(r["codigoestacion"])
        if prev is None or int(r.get("n", 0)) > int(prev.get("n", 0)):
            best[r["codigoestacion"]] = r
    seen: dict[str, Station] = {}
    for r in best.values():
        seen[r["codigoestacion"]] = Station(
            code=r["codigoestacion"],
            raw_name=r["nombreestacion"],
            department=r.get("departamento", ""),
            municipality=r.get("municipio", ""),
            zone=r.get("zonahidrografica", ""),
            lat=float(r["latitud"]),
            lon=float(r["longitud"]),
        )
    return sorted(seen.values(), key=lambda s: s.code)


def station_levels(client: sources.CachedClient, code: str, start: str, end: str, page: int = 50000) -> list[dict]:
    """All readings of one station between two local timestamps, paged by offset."""
    out: list[dict] = []
    offset = 0
    while True:
        params = {
            "$select": "fechaobservacion,valorobservado",
            "$where": f"codigoestacion='{code}' AND fechaobservacion between '{start}' and '{end}'",
            "$order": "fechaobservacion",
            "$limit": page,
            "$offset": offset,
        }
        rows = client.get_json(SODA_LEVELS, params)
        out.extend(rows)
        if len(rows) < page:
            return out
        offset += page


def hourly_levels(rows: list[dict], lid: str) -> pl.DataFrame:
    """Readings to hourly means in UTC. Values outside a plausible gauge range are dropped."""
    schema = {"lid": pl.Utf8, "time": pl.Datetime("us", "UTC"), "stage_ft": pl.Float64}
    clean = [r for r in rows if r.get("valorobservado") not in (None, "")]
    if not clean:
        return pl.DataFrame(schema=schema)
    df = (
        pl.DataFrame(clean)
        .with_columns(
            (
                pl.col("fechaobservacion").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="us")
                + pl.duration(hours=UTC_OFFSET_H)
            )
            .dt.replace_time_zone("UTC")
            .alias("time"),
            pl.col("valorobservado").cast(pl.Float64, strict=False).alias("v"),
        )
        .filter(pl.col("v").is_between(-5, 60))
        .sort("time")
        .group_by_dynamic("time", every="1h")
        .agg(pl.col("v").median().alias("stage_ft"))
        .with_columns(pl.lit(lid).alias("lid"))
    )
    return df.select("lid", "time", "stage_ft")


def despike(hourly: pl.DataFrame, jump_m: float = 1.0, settle_m: float = 0.5) -> pl.DataFrame:
    """Drop isolated readings: more than jump_m away from both neighbours while the neighbours
    agree within settle_m. A real rise lasts more than one hour, a bad reading does not."""
    marked = hourly.sort("lid", "time").with_columns(
        pl.col("stage_ft").shift(1).over("lid").alias("_p"),
        pl.col("stage_ft").shift(-1).over("lid").alias("_n"),
    )
    spike = (
        ((pl.col("stage_ft") - pl.col("_p")).abs() > jump_m)
        & ((pl.col("stage_ft") - pl.col("_n")).abs() > jump_m)
        & ((pl.col("_p") - pl.col("_n")).abs() < settle_m)
    )
    return marked.filter(~spike.fill_null(False)).drop("_p", "_n")


def level_shift_stations(hourly: pl.DataFrame, max_jump_m: float = 8.0, min_hours: int = 200) -> dict[str, float]:
    """Stations whose monthly median jumps more than max_jump_m between consecutive months.
    That is a change of gauge datum or units, not the river, so thresholds and lags from that
    station would be wrong. They stay in the network but are not used for numbers."""
    monthly = (
        hourly.with_columns(pl.col("time").dt.strftime("%Y-%m").alias("month"))
        .group_by("lid", "month")
        .agg(pl.col("stage_ft").median().alias("med"), pl.len().alias("n"))
        .filter(pl.col("n") >= min_hours)
        .sort("lid", "month")
        .with_columns(pl.col("med").diff().abs().over("lid").alias("jump"))
        .group_by("lid")
        .agg(pl.col("jump").max())
    )
    return {r["lid"]: round(r["jump"], 2) for r in monthly.iter_rows(named=True) if (r["jump"] or 0) > max_jump_m}


# HydroRIVERS ------------------------------------------------------------------------------


def load_reaches(shp: Path, bbox: tuple[float, float, float, float], min_upland_km2: float = 100.0):
    import shapefile
    from shapely.geometry import LineString

    reader = shapefile.Reader(str(shp))
    reaches: dict[int, dict] = {}
    geoms: dict[int, LineString] = {}
    for sr in reader.iterShapeRecords(bbox=list(bbox)):
        rec = sr.record.as_dict()
        if rec["UPLAND_SKM"] < min_upland_km2:
            continue
        reaches[rec["HYRIV_ID"]] = rec
        geoms[rec["HYRIV_ID"]] = LineString(sr.shape.points)
    return reaches, geoms


def snap_stations(stations: list[Station], geoms: dict, reaches: dict, radius_km: float = 2.0) -> dict[str, int]:
    """Each station to the largest reach within radius_km. Gauges sit on the river they measure,
    and the largest nearby reach avoids snapping a main stem gauge to a side creek."""
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    ids = list(geoms)
    tree = STRtree([geoms[i] for i in ids])
    out = {}
    for st in stations:
        p = Point(st.lon, st.lat)
        cands = []
        for idx in tree.query(p.buffer(radius_km / 111.0)):
            rid = ids[int(idx)]
            d_km = geoms[rid].distance(p) * 111.0
            if d_km <= radius_km:
                cands.append((reaches[rid]["UPLAND_SKM"], -d_km, rid))
        if cands:
            out[st.code] = max(cands)[2]
    return out


def downstream_chain(reaches: dict, rid: int, limit: int = 20000) -> list[int]:
    chain, cur = [], rid
    while cur in reaches and len(chain) < limit:
        chain.append(cur)
        cur = reaches[cur]["NEXT_DOWN"]
    return chain


def build_network_co(
    stations: list[Station], reaches: dict, geoms: dict, zones: tuple[str, ...]
) -> tuple[list[Station], dict[str, int], list[Edge], str]:
    snapped = snap_stations([s for s in stations if s.zone in zones], geoms, reaches)
    if not snapped:
        raise RuntimeError("no station could be snapped to the river network")
    outlet = max(snapped, key=lambda c: reaches[snapped[c]]["UPLAND_SKM"])
    outlet_reach = snapped[outlet]
    members = {c: r for c, r in snapped.items() if outlet_reach in downstream_chain(reaches, r)}
    at_reach: dict[int, list[str]] = {}
    for code, rid in members.items():
        at_reach.setdefault(rid, []).append(code)
    edges = []
    for code, rid in members.items():
        for nxt in downstream_chain(reaches, rid)[1:]:
            if nxt in at_reach:
                down = sorted(at_reach[nxt])[0]
                km = reaches[rid]["DIST_DN_KM"] - reaches[nxt]["DIST_DN_KM"]
                edges.append(Edge(short_code(code), short_code(down), round(max(km, 0.1), 1)))
                break
    kept = [s for s in stations if s.code in members]
    log.info("%d stations drain to %s, %d edges", len(kept), outlet, len(edges))
    return kept, members, sorted(edges, key=lambda e: e.upstream), short_code(outlet)


# Thresholds and events -----------------------------------------------------------------


def statistical_thresholds(hourly: pl.DataFrame, min_days: int = 180) -> dict[str, dict]:
    """P90 and P98 of daily maximum level per station, plus the record and its date."""
    daily = hourly.group_by("lid", pl.col("time").dt.date().alias("day")).agg(pl.col("stage_ft").max().alias("dmax"))
    out = {}
    for (lid,), part in daily.group_by("lid"):
        if part.height < min_days:
            continue
        top = part.sort("dmax", descending=True).row(0, named=True)
        out[lid] = {
            "action": round(float(part["dmax"].quantile(0.90)), 2),
            "minor": round(float(part["dmax"].quantile(0.98)), 2),
            "record": round(float(top["dmax"]), 2),
            "record_date": top["day"].isoformat(),
            "days": part.height,
        }
    return out


def detect_events_co(
    hourly: pl.DataFrame, thresholds: dict[str, dict], top_n: int = 12, min_gap_days: int = 10
) -> list[Event]:
    """Days when the largest share of stations exceeded their own 'alto' level, kept apart."""
    level = pl.DataFrame([{"lid": k, "minor": v["minor"]} for k, v in thresholds.items()])
    daily = (
        hourly.join(level, on="lid")
        .group_by("lid", pl.col("time").dt.date().alias("day"))
        .agg((pl.col("stage_ft").max() >= pl.col("minor").first()).alias("over"))
        .group_by("day")
        .agg(pl.col("over").sum().alias("n_over"))
        .sort("n_over", descending=True)
    )
    chosen: list[date] = []
    for row in daily.iter_rows(named=True):
        if row["n_over"] < 2:
            break
        if all(abs((row["day"] - d).days) > min_gap_days for d in chosen):
            chosen.append(row["day"])
        if len(chosen) == top_n:
            break
    return [
        Event(
            event_id=d.isoformat(),
            peak_day=d.isoformat(),
            peak_cfs=float(daily.filter(pl.col("day") == d)["n_over"][0]),  # stations above 'alto' that day
            start=(d - timedelta(days=6)).isoformat() + "T00:00:00Z",
            end=(d + timedelta(days=3)).isoformat() + "T00:00:00Z",
        )
        for d in sorted(chosen)
    ]


def event_slices(hourly: pl.DataFrame, events: list[Event]) -> pl.DataFrame:
    parts = []
    for e in events:
        start = pl.lit(e.start).str.to_datetime(time_zone="UTC", time_unit="us")
        end = pl.lit(e.end).str.to_datetime(time_zone="UTC", time_unit="us")
        parts.append(
            hourly.filter((pl.col("time") >= start) & (pl.col("time") < end)).with_columns(
                pl.lit(e.event_id).alias("event_id"), pl.lit(1.0).alias("approved_share")
            )
        )
    return pl.concat(parts).select("lid", "event_id", "time", "stage_ft", "approved_share")


# Orchestration -----------------------------------------------------------------------------


def ingest_cauca(
    client: sources.CachedClient,
    hydrorivers_shp: Path,
    start: str = "2022-11-01T00:00:00",
    end: str = "2026-11-02T23:59:59",
    workers: int = 4,
) -> tuple[list[Gauge], list[Edge], list[Event], pl.DataFrame, pl.DataFrame, str, dict]:
    stations = active_stations(client, since="2026-10-01T00:00:00")
    reaches, geoms = load_reaches(hydrorivers_shp, CAUCA_BBOX)
    members, snapped, edges, outlet = build_network_co(stations, reaches, geoms, CAUCA_ZONES)

    def series(st: Station) -> pl.DataFrame:
        return hourly_levels(station_levels(client, st.code, start, end), short_code(st.code))

    with ThreadPoolExecutor(workers) as pool:
        hourly = despike(pl.concat([f for f in pool.map(series, members) if f.height]))
    shifted = level_shift_stations(hourly)
    if shifted:
        log.warning("level shifts, not used for thresholds or lags: %s", shifted)
        hourly = hourly.filter(~pl.col("lid").is_in(list(shifted)))
    thresholds = statistical_thresholds(hourly)
    gauges = []
    for st in members:
        lid = short_code(st.code)
        th = thresholds.get(lid)
        gauges.append(
            Gauge(
                lid=lid,
                usgs_id=st.code,
                name=station_label(st),
                river=st.zone,
                state=st.department,
                county=clean_name(re.sub(r"\([^)]*\)", "", st.municipality)),
                lat=st.lat,
                lon=st.lon,
                comid=int(snapped[st.code]),
                measure=50.0,
                thresholds={"action": th["action"], "minor": th["minor"]} if th else {},
                impacts=[],
                record_stage_ft=th["record"] if th else None,
                record_date=th["record_date"] if th else None,
            )
        )
    events = detect_events_co(hourly, thresholds)
    stage = event_slices(hourly, events)
    quality = {
        "level_shift_m": shifted,
        "no_thresholds": sorted(g.lid for g in gauges if not g.thresholds),
        "days_of_history": {k: v["days"] for k, v in sorted(thresholds.items())},
    }
    return sorted(gauges, key=lambda g: g.lid), edges, events, stage, hourly, outlet, quality
