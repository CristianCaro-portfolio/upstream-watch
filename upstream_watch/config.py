"""Settings, read from environment variables with the UPW_ prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample"

# Potomac River near Washington DC (Little Falls). Every gauge in the network drains to it.
OUTLET_USGS_ID = "01646500"
OUTLET_COMID = 4512772
OUTLET_LID = "BRKM2"
# Rough box around the Potomac basin above Little Falls, used only to list NWPS candidates.
BASIN_BBOX = (-80.6, 38.0, -76.9, 40.1)


@dataclass(frozen=True)
class Region:
    """One river basin: where its data lives, its language and its outlet gauge."""

    key: str
    title: str
    lang: str
    outlet_lid: str
    replay_event: str
    sample_dir: Path
    curated_name: str
    outlet_label: str = "outlet"
    min_lag_obs: int = 1  # crest pairs an edge needs before its own lag replaces the basin wave speed


REGIONS = {
    "potomac": Region(
        key="potomac",
        title="Potomac River above Washington DC (USGS, NOAA NWPS)",
        lang="en",
        outlet_lid=OUTLET_LID,
        replay_event="2025-05-15",  # western Maryland flash flood, minor flood at Little Falls
        sample_dir=SAMPLE_DIR,
        curated_name="curated",
        outlet_label="Little Falls",
    ),
    "cauca": Region(
        key="cauca",
        title="Río Cauca, Colombia (IDEAM, datos.gov.co)",
        lang="es",
        # La Raya: last Cauca gauge with clean data. Las Varas, just below, repeats an 8.0x m flag value.
        outlet_lid="25027910",
        replay_event="",  # the event with most stations above 'alto' is used
        sample_dir=ROOT / "data" / "sample_cauca",
        curated_name="curated_cauca",
        outlet_label="La Raya",
        min_lag_obs=5,  # 4 years with gaps and two dams: few crest pairs per edge, see README
    ),
}


def _env(name: str, default: str) -> str:
    return os.environ.get(f"UPW_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    reports_dir: Path = field(default_factory=lambda: Path(_env("REPORTS_DIR", str(ROOT / "reports"))))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "rule"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen2.5:1.5b-instruct"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434"))
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "fastembed"))
    usgs_api_key: str = field(default_factory=lambda: _env("USGS_API_KEY", ""))
    http_timeout_s: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT_S", "60")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "8")))
    otel_console: bool = field(default_factory=lambda: _env("OTEL_CONSOLE", "0") == "1")
    region_key: str = field(default_factory=lambda: _env("REGION", "potomac"))

    @property
    def region(self) -> Region:
        if self.region_key not in REGIONS:
            raise ValueError(f"unknown region {self.region_key!r}, use one of {sorted(REGIONS)}")
        return REGIONS[self.region_key]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / self.region.curated_name


def get_settings() -> Settings:
    return Settings()
