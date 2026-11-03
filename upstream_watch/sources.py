"""HTTP clients for the three public sources.

* USGS Water Data OGC API (api.waterdata.usgs.gov), the replacement for the legacy
  WaterServices API that USGS retires in February 2027.
* USGS NLDI (api.water.usgs.gov/nldi) for the NHDPlus river network.
* NOAA National Water Prediction Service (api.water.noaa.gov/nwps) for flood stages,
  impact statements and the forecast point list.

All responses are cached as JSON under data/raw so a rerun is offline and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

USGS_OGC = "https://api.waterdata.usgs.gov/ogcapi/v0"
NLDI = "https://api.water.usgs.gov/nldi"
NWPS = "https://api.water.noaa.gov/nwps/v1"

RETRY_STATUS = {429, 500, 502, 503, 504}


class SourceError(RuntimeError):
    """Raised when a source keeps failing after retries."""


class CachedClient:
    """Small JSON client with a disk cache and exponential backoff."""

    def __init__(
        self,
        cache_dir: Path,
        timeout_s: float = 60.0,
        max_attempts: int = 4,
        api_key: str = "",
        offline: bool = False,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 2.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.offline = offline
        headers = {"User-Agent": "upstream-watch/0.1 (portfolio project)"}
        if api_key:
            headers["X-Api-Key"] = api_key
        self.backoff_s = backoff_s
        self._http = httpx.Client(timeout=timeout_s, headers=headers, follow_redirects=True, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._cache_path(url, params)
        if path.exists():
            return json.loads(path.read_text())
        if self.offline:
            raise SourceError(f"offline mode and no cached response for {url}")
        delay = self.backoff_s
        last_error: str = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                last_error = repr(exc)
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    path.write_text(json.dumps(data))
                    return data
                if resp.status_code not in RETRY_STATUS:
                    raise SourceError(f"{resp.status_code} from {url}: {resp.text[:200]}")
                last_error = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            log.warning("attempt %d/%d failed for %s (%s)", attempt, self.max_attempts, url, last_error)
            if attempt < self.max_attempts:
                time.sleep(delay)
                delay *= 2
        raise SourceError(f"giving up on {url}: {last_error}")


# NWPS -----------------------------------------------------------------------------------


def nwps_gauges_in_bbox(client: CachedClient, bbox: tuple[float, float, float, float]) -> list[dict]:
    xmin, ymin, xmax, ymax = bbox
    params = {
        "bbox.xmin": xmin,
        "bbox.ymin": ymin,
        "bbox.xmax": xmax,
        "bbox.ymax": ymax,
        "srid": "EPSG_4326",
    }
    return client.get_json(f"{NWPS}/gauges", params)["gauges"]


def nwps_gauge(client: CachedClient, lid: str) -> dict:
    return client.get_json(f"{NWPS}/gauges/{lid}")


# NLDI -----------------------------------------------------------------------------------


def nldi_site(client: CachedClient, usgs_id: str) -> dict | None:
    data = client.get_json(f"{NLDI}/linked-data/nwissite/USGS-{usgs_id}", {"f": "json"})
    feats = data.get("features") or []
    return feats[0] if feats else None


def nldi_downstream_flowlines(client: CachedClient, usgs_id: str, distance_km: int = 700) -> list[dict]:
    url = f"{NLDI}/linked-data/nwissite/USGS-{usgs_id}/navigation/DM/flowlines"
    return client.get_json(url, {"distance": distance_km, "f": "json"}).get("features") or []


# USGS Water Data OGC API ---------------------------------------------------------------


def _paged_items(client: CachedClient, collection: str, params: dict[str, Any]) -> Iterator[dict]:
    url = f"{USGS_OGC}/collections/{collection}/items"
    offset = 0
    limit = int(params.get("limit", 10000))
    while True:
        page = client.get_json(url, {**params, "f": "json", "limit": limit, "offset": offset})
        feats = page.get("features") or []
        yield from feats
        if len(feats) < limit:
            return
        offset += limit


def usgs_daily(client: CachedClient, usgs_id: str, parameter: str, start: str, end: str) -> list[dict]:
    params = {
        "monitoring_location_id": f"USGS-{usgs_id}",
        "parameter_code": parameter,
        "statistic_id": "00003",
        "datetime": f"{start}/{end}",
    }
    return [f["properties"] for f in _paged_items(client, "daily", params)]


def usgs_continuous(client: CachedClient, usgs_id: str, parameter: str, start: str, end: str) -> list[dict]:
    params = {
        "monitoring_location_id": f"USGS-{usgs_id}",
        "parameter_code": parameter,
        "datetime": f"{start}/{end}",
    }
    return [f["properties"] for f in _paged_items(client, "continuous", params)]
