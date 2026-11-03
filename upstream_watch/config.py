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

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"


def get_settings() -> Settings:
    return Settings()
