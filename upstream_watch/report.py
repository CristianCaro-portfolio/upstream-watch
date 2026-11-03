"""Figures for the README and the post: architecture and measured results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK = "#1f2933"
MUTED = "#616e7c"
COLORS = {"dense": "#9aa5b1", "hybrid": "#4c78a8", "graph": "#e45756"}
STAGE_FILL = ["#e3f2fd", "#e8f5e9", "#fff3e0", "#fce4ec", "#ede7f6"]


def architecture(path: Path, footer: str = "") -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    columns = [
        (
            "Public sources",
            ["USGS Water Data\nOGC API (new)", "USGS NLDI\nriver network", "NOAA NWPS\nstages + impacts"],
        ),
        (
            "Ingest",
            ["httpx + disk cache\nretries, offline", "Polars hourly stage\nHive Parquet", "crest detection\nper event"],
        ),
        (
            "Knowledge",
            [
                "LadybugDB graph\nGauge-FLOWS_TO-Gauge",
                "learned crest lag\nper river edge",
                "Qdrant hybrid\ndense + BM25, RRF",
            ],
        ),
        (
            "LangGraph agent",
            [
                "ask: understand,\nretrieve, compose",
                "verify: grounding\ncheck, fallback",
                "brief: assess,\nhuman review",
            ],
        ),
        ("Out", ["MCP server\n(SDK v2)", "OpenTelemetry\nspans", "eval + backtest\nreports"]),
    ]
    x0, w, gap = 0.25, 2.25, 0.32
    for ci, (title, boxes) in enumerate(columns):
        x = x0 + ci * (w + gap)
        ax.text(x + w / 2, 4.85, title, ha="center", va="center", fontsize=11, weight="bold", color=INK)
        for bi, label in enumerate(boxes):
            y = 3.55 - bi * 1.35
            ax.add_patch(
                FancyBboxPatch(
                    (x, y), w, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                    facecolor=STAGE_FILL[ci], edgecolor="#9aa5b1", linewidth=1,
                )
            )  # fmt: skip
            ax.text(x + w / 2, y + 0.5, label, ha="center", va="center", fontsize=9, color=INK)
        if ci < len(columns) - 1:
            ax.annotate(
                "", xy=(x + w + gap - 0.04, 2.7), xytext=(x + w + 0.04, 2.7),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.6},
            )  # fmt: skip
    ax.text(
        6.5, 0.12,
        footer,
        ha="center", fontsize=9, color=MUTED,
    )  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def results(metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.6, 1, 1]})

    ax = axes[0]
    modes = list(metrics["eval"]["generated"]["modes"])
    intents = list(metrics["eval"]["generated"]["modes"][modes[0]]["accuracy_by_intent"])
    width = 0.8 / len(modes)
    for mi, mode in enumerate(modes):
        acc = metrics["eval"]["generated"]["modes"][mode]["accuracy_by_intent"]
        xs = [i + (mi - (len(modes) - 1) / 2) * width for i in range(len(intents))]
        ax.bar(xs, [acc[k] for k in intents], width * 0.92, label=mode, color=COLORS[mode])
    ax.set_xticks(range(len(intents)), intents)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("answer accuracy")
    n = metrics["eval"]["generated"]["modes"][modes[0]]["questions"]
    ax.set_title(f"Retrieval mode vs question type ({n} questions)", fontsize=11, color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center")

    ax = axes[1]
    bt = metrics["backtest"]
    labels = ["next gauge", "to Little Falls"]
    learned = [bt["one_hop"]["learned"]["mae_h"], bt["to_outlet"]["learned"]["mae_h"]]
    celerity = [bt["one_hop"]["celerity"]["mae_h"], bt["to_outlet"]["celerity"]["mae_h"]]
    xs = range(len(labels))
    ax.bar([x - 0.2 for x in xs], celerity, 0.38, label="constant wave speed", color=COLORS["dense"])
    ax.bar([x + 0.2 for x in xs], learned, 0.38, label="learned lag per edge", color=COLORS["graph"])
    for x, v in zip(xs, learned, strict=True):
        ax.text(x + 0.2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    for x, v in zip(xs, celerity, strict=True):
        ax.text(x - 0.2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs), labels)
    ax.set_ylim(0, max(celerity + learned) * 1.4)
    ax.set_ylabel("crest arrival error, MAE hours")
    ax.set_title("Leave one event out backtest", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    rows = metrics["lead_times"]
    names = [f"{r['trigger']}, {r['max_hops']} hop" for r in rows]
    ax.barh(names, [r["median_lead_h"] or 0 for r in rows], color=COLORS["hybrid"])
    for i, r in enumerate(rows):
        ax.text(
            (r["median_lead_h"] or 0) + 0.4, i,
            f"{r['median_lead_h']:g} h, {r['warned_in_advance']}/{r['floods']} floods, precision {r['precision']:.2f}",
            va="center", fontsize=8.5, color=INK,
        )  # fmt: skip
    ax.set_xlim(0, max((r["median_lead_h"] or 0) for r in rows) * 2.3)
    ax.set_xlabel("median warning lead time, hours")
    ax.set_title("Upstream trigger replay, 12 events", fontsize=11, color=INK)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
