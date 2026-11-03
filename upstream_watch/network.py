"""Build the gauge network: forecast points, flood thresholds, impacts and river topology."""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import sources

log = logging.getLogger(__name__)

CATEGORIES = ("action", "minor", "moderate", "major")


@dataclass
class Impact:
    lid: str
    stage_ft: float
    statement: str

    @property
    def doc_id(self) -> str:
        return f"{self.lid}@{self.stage_ft:g}"


@dataclass
class Gauge:
    lid: str
    usgs_id: str
    name: str
    river: str
    state: str
    county: str
    lat: float
    lon: float
    comid: int
    measure: float
    thresholds: dict[str, float]
    impacts: list[Impact] = field(default_factory=list)
    record_stage_ft: float | None = None
    record_date: str | None = None

    def category_for(self, stage_ft: float) -> str:
        """Highest NWS flood category reached at a stage, or 'none'."""
        reached = "none"
        for cat in CATEGORIES:
            level = self.thresholds.get(cat)
            if level is not None and stage_ft >= level:
                reached = cat
        return reached

    def to_dict(self) -> dict:
        d = asdict(self)
        d["impacts"] = [asdict(i) for i in self.impacts]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Gauge:
        d = dict(d)
        d["impacts"] = [Impact(**i) for i in d.get("impacts", [])]
        return cls(**d)


@dataclass
class Edge:
    upstream: str
    downstream: str
    river_km: float


def river_name(gauge_name: str) -> str:
    """'South Fork Shenandoah River at Luray' -> 'South Fork Shenandoah River'."""
    for sep in (" at ", " near ", " above ", " below ", " abv ", " blw "):
        idx = gauge_name.lower().find(sep)
        if idx > 0:
            return gauge_name[:idx].strip()
    return gauge_name.strip()


def _thresholds(detail: dict) -> dict[str, float]:
    cats = (detail.get("flood") or {}).get("categories") or {}
    out = {}
    for cat in CATEGORIES:
        stage = (cats.get(cat) or {}).get("stage")
        if stage is not None and stage > -999:
            out[cat] = float(stage)
    return out


def _record(detail: dict) -> tuple[float | None, str | None]:
    crests = ((detail.get("flood") or {}).get("crests") or {}).get("historic") or []
    valid = [c for c in crests if c.get("stage") and not c.get("olddatum")]
    if not valid:
        return None, None
    top = max(valid, key=lambda c: c["stage"])
    return float(top["stage"]), str(top["occurredTime"])[:10]


def gauge_from_detail(detail: dict, site: dict) -> Gauge:
    props = site["properties"]
    impacts = [
        Impact(lid=detail["lid"], stage_ft=float(i["stage"]), statement=" ".join(i["statement"].split()))
        for i in (detail.get("flood") or {}).get("impacts") or []
        if i.get("statement") and i.get("stage") is not None
    ]
    record_stage, record_date = _record(detail)
    lon, lat = site["geometry"]["coordinates"][:2]
    return Gauge(
        lid=detail["lid"],
        usgs_id=detail["usgsId"],
        name=detail["name"].strip(),
        river=river_name(detail["name"]),
        state=(detail.get("state") or {}).get("abbreviation", ""),
        county=detail.get("county") or "",
        lat=float(lat),
        lon=float(lon),
        comid=int(props["comid"]),
        measure=float(props.get("measure") or 50.0),
        thresholds=_thresholds(detail),
        impacts=sorted(impacts, key=lambda i: i.stage_ft),
        record_stage_ft=record_stage,
        record_date=record_date,
    )


def _haversine_km(coords_a: tuple[float, float], coords_b: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, coords_a)
    lon2, lat2 = map(math.radians, coords_b)
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _line_km(coords: list[list[float]]) -> float:
    return sum(_haversine_km(tuple(a[:2]), tuple(b[:2])) for a, b in zip(coords, coords[1:], strict=False))


def order_flowlines(features: list[dict], start_comid: int) -> list[tuple[int, float]]:
    """Chain NLDI flowlines from start_comid downstream. Returns (comid, length_km) in order.

    NLDI does not guarantee the order of the features, so the chain is rebuilt by
    matching the last vertex of one flowline with the first vertex of the next.
    """
    by_start: dict[tuple[float, float], dict] = {}
    by_comid: dict[int, dict] = {}
    for f in features:
        coords = f["geometry"]["coordinates"]
        if f["geometry"]["type"] == "MultiLineString":
            coords = [pt for part in coords for pt in part]
        f["_coords"] = coords
        by_start[(round(coords[0][0], 6), round(coords[0][1], 6))] = f
        by_comid[int(f["properties"]["nhdplus_comid"])] = f
    current = by_comid.get(start_comid)
    chain: list[tuple[int, float]] = []
    seen: set[int] = set()
    while current is not None:
        comid = int(current["properties"]["nhdplus_comid"])
        if comid in seen:
            break
        seen.add(comid)
        coords = current["_coords"]
        chain.append((comid, _line_km(coords)))
        end = (round(coords[-1][0], 6), round(coords[-1][1], 6))
        current = by_start.get(end)
    return chain


def downstream_neighbor(
    gauge: Gauge, chain: list[tuple[int, float]], gauges_by_comid: dict[int, list[Gauge]]
) -> tuple[Gauge, float] | None:
    """First gauge downstream on the chain and the river distance to it.

    NLDI measures run from 0 at the downstream end of a flowline to 100 at the top, so
    the share of the first flowline still ahead of the gauge is measure / 100.
    """
    travelled = 0.0
    for idx, (comid, length_km) in enumerate(chain):
        for other in sorted(gauges_by_comid.get(comid, []), key=lambda g: -g.measure):
            if other.lid == gauge.lid:
                continue
            if idx == 0:
                if other.measure < gauge.measure:
                    km = length_km * (gauge.measure - other.measure) / 100.0
                    return other, max(km, 0.1)
                continue
            km = travelled + length_km * (100.0 - other.measure) / 100.0
            return other, km
        travelled += length_km * (gauge.measure / 100.0 if idx == 0 else 1.0)
    return None


def build_network(
    client: sources.CachedClient,
    bbox: tuple[float, float, float, float],
    outlet_comid: int,
    workers: int = 6,
) -> tuple[list[Gauge], list[Edge]]:
    """Forecast points in the basin with flood stages, plus the downstream edges between them."""
    listing = sources.nwps_gauges_in_bbox(client, bbox)
    lids = [g["lid"] for g in listing if (g.get("rfc") or {}).get("abbreviation") == "MARFC"]
    with ThreadPoolExecutor(workers) as pool:
        details = list(pool.map(lambda lid: _safe(sources.nwps_gauge, client, lid), lids))
    candidates = [
        d
        for d in details
        if d and d.get("usgsId") and _thresholds(d).get("minor") and d.get("inService", {}).get("enabled")
    ]
    log.info("%d NWPS gauges listed, %d with a USGS id and a minor flood stage", len(lids), len(candidates))

    def resolve(detail: dict) -> tuple[Gauge, list[tuple[int, float]]] | None:
        site = _safe(sources.nldi_site, client, detail["usgsId"])
        if not site or site["properties"].get("comid") is None:
            return None
        flowlines = _safe(sources.nldi_downstream_flowlines, client, detail["usgsId"]) or []
        gauge = gauge_from_detail(detail, site)
        chain = order_flowlines(flowlines, gauge.comid)
        if outlet_comid not in {c for c, _ in chain}:
            return None
        return gauge, chain

    with ThreadPoolExecutor(workers) as pool:
        resolved = [r for r in pool.map(resolve, candidates) if r]
    gauges = sorted((g for g, _ in resolved), key=lambda g: g.lid)
    by_comid: dict[int, list[Gauge]] = {}
    for g in gauges:
        by_comid.setdefault(g.comid, []).append(g)
    edges: list[Edge] = []
    for gauge, chain in resolved:
        hit = downstream_neighbor(gauge, chain, by_comid)
        if hit:
            other, km = hit
            edges.append(Edge(upstream=gauge.lid, downstream=other.lid, river_km=round(km, 1)))
    log.info("%d gauges drain to the outlet, %d edges", len(gauges), len(edges))
    return gauges, sorted(edges, key=lambda e: e.upstream)


def _safe(fn, *args):
    try:
        return fn(*args)
    except sources.SourceError as exc:
        log.warning("skipping %s: %s", args[-1], exc)
        return None
