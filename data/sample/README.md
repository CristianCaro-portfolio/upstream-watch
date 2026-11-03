# Bundled data

The full curated dataset used for every number in the README, small enough to ship.

gauges.json: 76 NWPS forecast points in the Potomac basin above Little Falls, with flood stages, record crests and 458 NWS impact statements.
edges.json: 75 river edges from each gauge to the next one downstream, with the river distance from USGS NLDI.
events.json: 12 high flow events detected from the Little Falls daily discharge, October 2023 to October 2026.
stage_hourly.parquet: hourly mean gage height for every gauge in every event window, from the USGS Water Data OGC API.

Sources are US Government works in the public domain. Downloaded on 2026-11-03.
Rebuild it with: python -m upstream_watch ingest
