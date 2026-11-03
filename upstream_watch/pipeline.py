"""End to end steps: ingest public data, build the graph and index, wire the agent."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

import polars as pl

from . import config, hydro, network, sources
from . import timeseries as ts
from .agent import Toolkit, build_agent
from .graph import GaugeGraph
from .lang import get as get_lang
from .llm import make_provider
from .retrieval import GaugeLinker, GraphRetriever, HybridIndex, build_corpus, make_embedder

log = logging.getLogger(__name__)


def ingest(
    settings: config.Settings, offline: bool = False, start: str = "2023-10-01", end: str = "2026-10-31"
) -> dict:
    """Download and curate the configured region."""
    if settings.region.key == "cauca":
        return ingest_cauca(settings, offline)
    return ingest_potomac(settings, offline, start, end)


def _write_curated(out: Path, gauges, edges, events) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gauges.json").write_text(json.dumps([g.to_dict() for g in gauges], indent=1, ensure_ascii=False))
    (out / "edges.json").write_text(json.dumps([asdict(e) for e in edges], indent=1))
    (out / "events.json").write_text(json.dumps([asdict(e) for e in events], indent=1))


def ingest_cauca(settings: config.Settings, offline: bool = False) -> dict:
    """IDEAM levels from datos.gov.co and the HydroRIVERS network for the Cauca basin."""
    from . import colombia

    shp_dir = settings.raw_dir / "hydrorivers"
    shp = shp_dir / "HydroRIVERS_v10_sa_shp" / "HydroRIVERS_v10_sa.shp"
    if not shp.exists():
        if offline:
            raise sources.SourceError("offline mode and HydroRIVERS is not in data/raw/hydrorivers")
        _download_and_unzip(colombia.HYDRORIVERS_SA, shp_dir)
    client = sources.CachedClient(settings.raw_dir / "http_co", 180.0, offline=offline)
    try:
        gauges, edges, events, stage, _, outlet, quality = colombia.ingest_cauca(client, shp)
    finally:
        client.close()
    if outlet != settings.region.outlet_lid:
        log.warning("outlet is %s, the region config expects %s", outlet, settings.region.outlet_lid)
    out = settings.curated_dir
    _write_curated(out, gauges, edges, events)
    stage.sort("event_id", "lid", "time").write_parquet(out / "stage_hourly.parquet", compression="zstd")
    (out / "quality.json").write_text(json.dumps(quality, indent=1))
    summary = {"gauges": len(gauges), "edges": len(edges), "events": len(events), "stage_rows": stage.height}
    log.info("ingest done: %s", summary)
    return summary


def _download_and_unzip(url: str, dest: Path) -> None:
    import zipfile

    import httpx

    dest.mkdir(parents=True, exist_ok=True)
    target = dest / url.rsplit("/", 1)[-1]
    with httpx.stream("GET", url, timeout=600, follow_redirects=True) as resp:
        resp.raise_for_status()
        with target.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    with zipfile.ZipFile(target) as zf:
        zf.extractall(dest)


def ingest_potomac(settings: config.Settings, offline: bool, start: str, end: str) -> dict:
    """Network from NWPS and NLDI, events from the outlet daily flow, hourly stage per event."""
    client = sources.CachedClient(
        settings.raw_dir / "http", settings.http_timeout_s, api_key=settings.usgs_api_key, offline=offline
    )
    try:
        gauges, edges = network.build_network(client, config.BASIN_BBOX, config.OUTLET_COMID)
        daily = sources.usgs_daily(client, config.OUTLET_USGS_ID, ts.DISCHARGE, start, end)
        events = ts.detect_events(daily)
        stage = ts.fetch_event_stages(client, gauges, events)
    finally:
        client.close()
    if not gauges or stage.is_empty():
        raise RuntimeError("ingest produced no gauges or no stage data, nothing written")
    out = settings.curated_dir
    _write_curated(out, gauges, edges, events)
    stage_dir = out / "stage_hourly"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    ts.write_partitioned(stage, stage_dir)
    summary = {"gauges": len(gauges), "edges": len(edges), "events": len(events), "stage_rows": stage.height}
    log.info("ingest done: %s", summary)
    return summary


def load_curated(data_dir: Path) -> tuple[list[network.Gauge], list[network.Edge], pl.DataFrame]:
    gauges = [network.Gauge.from_dict(d) for d in json.loads((data_dir / "gauges.json").read_text(encoding="utf-8"))]
    edges = [network.Edge(**d) for d in json.loads((data_dir / "edges.json").read_text())]
    single = data_dir / "stage_hourly.parquet"
    stage = pl.read_parquet(single) if single.exists() else ts.read_partitioned(data_dir / "stage_hourly")
    return gauges, edges, stage


def resolve_data_dir(settings: config.Settings, sample: bool) -> Path:
    if sample or not (settings.curated_dir / "gauges.json").exists():
        return settings.region.sample_dir
    return settings.curated_dir


def build_toolkit(
    gauges: list[network.Gauge],
    edges: list[network.Edge],
    stage: pl.DataFrame,
    settings: config.Settings,
    embedder_name: str | None = None,
) -> tuple[Toolkit, hydro.TravelTimeModel]:
    crest_map = hydro.crests(stage)
    model = hydro.TravelTimeModel.fit(hydro.edge_lags(edges, crest_map), min_obs=settings.region.min_lag_obs)
    graph = GaugeGraph()
    graph.load(gauges, edges, model)
    lags = {(e.upstream, e.downstream): round(model.edge_lag(e), 1) for e in edges}
    lang = get_lang(settings.region.lang)
    corpus = build_corpus(gauges, edges, lags, lang)
    index = HybridIndex(corpus, make_embedder(embedder_name or settings.embedder))
    linker = GaugeLinker(gauges)
    kit = Toolkit(
        graph=graph,
        index=index,
        linker=linker,
        retriever=GraphRetriever(graph, index, linker),
        provider=make_provider(settings.llm_provider, settings.llm_model, settings.llm_base_url),
        names={g.lid: g.name for g in gauges},
        top_k=settings.top_k,
        lang=lang,
    )
    return kit, model


def agent_from_settings(settings: config.Settings, sample: bool = False, embedder_name: str | None = None):
    gauges, edges, stage = load_curated(resolve_data_dir(settings, sample))
    kit, model = build_toolkit(gauges, edges, stage, settings, embedder_name)
    return build_agent(kit), kit, model, (gauges, edges, stage)
