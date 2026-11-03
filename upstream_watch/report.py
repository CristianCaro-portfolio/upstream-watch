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


LABELS = {
    "en": {
        "modes": {"dense": "dense", "hybrid": "hybrid", "graph": "graph"},
        "intents": {k: k for k in ("downstream", "impact", "stages", "travel", "upstream")},
        "accuracy": "answer accuracy",
        "retrieval_title": "Retrieval mode vs question type ({n} questions)",
        "next": "next gauge",
        "outlet": "to {name}",
        "celerity": "constant wave speed",
        "learned": "learned lag per edge",
        "mae": "crest arrival error, MAE hours",
        "backtest": "Leave one event out backtest",
        "levels": {"action": "action", "minor": "minor"},
        "trigger": "{trigger}, {hops} hop",
        "lead_row": "{lead} h, {ok}/{n} floods, precision {p:.2f}",
        "lead_x": "median warning lead time, hours",
        "lead_title": "Upstream trigger replay, {n} events",
    },
    "es": {
        "modes": {"dense": "densa", "hybrid": "híbrida", "graph": "grafo"},
        "intents": {
            "downstream": "aguas abajo", "impact": "nivel actual", "stages": "umbrales",
            "travel": "tiempo de viaje", "upstream": "aguas arriba",
        },
        "accuracy": "respuestas correctas",
        "retrieval_title": "Recuperación por tipo de pregunta ({n} preguntas)",
        "next": "siguiente estación",
        "outlet": "hasta {name}",
        "celerity": "velocidad constante",
        "learned": "rezago aprendido por tramo",
        "mae": "error en hora de llegada, horas (MAE)",
        "backtest": "Validación dejando un evento fuera",
        "levels": {"action": "vigilancia", "minor": "alto"},
        "trigger": "{trigger}, {hops} salto(s)",
        "lead_row": "{lead} h, {ok}/{n} eventos, precisión {p:.2f}",
        "lead_x": "anticipación mediana del aviso, horas",
        "lead_title": "Aviso desde aguas arriba, {n} eventos",
    },
}  # fmt: skip


def results(metrics: dict, path: Path, lang: str = "en") -> None:
    t = LABELS[lang]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.6, 1, 1]})

    ax = axes[0]
    modes = list(metrics["eval"]["generated"]["modes"])
    intents = list(metrics["eval"]["generated"]["modes"][modes[0]]["accuracy_by_intent"])
    width = 0.8 / len(modes)
    for mi, mode in enumerate(modes):
        acc = metrics["eval"]["generated"]["modes"][mode]["accuracy_by_intent"]
        xs = [i + (mi - (len(modes) - 1) / 2) * width for i in range(len(intents))]
        ax.bar(xs, [acc[k] for k in intents], width * 0.92, label=t["modes"][mode], color=COLORS[mode])
    ax.set_xticks(range(len(intents)), [t["intents"][k] for k in intents])
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(t["accuracy"])
    n = metrics["eval"]["generated"]["modes"][modes[0]]["questions"]
    ax.set_title(t["retrieval_title"].format(n=n), fontsize=11, color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center")

    ax = axes[1]
    bt = metrics["backtest"]
    labels = [t["next"], t["outlet"].format(name=metrics.get("outlet_name", "outlet"))]
    learned = [bt["one_hop"]["learned"].get("mae_h", 0.0), bt["to_outlet"]["learned"].get("mae_h", 0.0)]
    celerity = [bt["one_hop"]["celerity"].get("mae_h", 0.0), bt["to_outlet"]["celerity"].get("mae_h", 0.0)]
    xs = range(len(labels))
    ax.bar([x - 0.2 for x in xs], celerity, 0.38, label=t["celerity"], color=COLORS["dense"])
    ax.bar([x + 0.2 for x in xs], learned, 0.38, label=t["learned"], color=COLORS["graph"])
    for x, v in zip(xs, learned, strict=True):
        ax.text(x + 0.2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    for x, v in zip(xs, celerity, strict=True):
        ax.text(x - 0.2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs), labels)
    ax.set_ylim(0, max(celerity + learned) * 1.4)
    ax.set_ylabel(t["mae"])
    ax.set_title(t["backtest"], fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    rows = metrics["lead_times"]
    names = [t["trigger"].format(trigger=t["levels"][r["trigger"]], hops=r["max_hops"]) for r in rows]
    ax.barh(names, [r["median_lead_h"] or 0 for r in rows], color=COLORS["hybrid"])
    for i, r in enumerate(rows):
        ax.text(
            (r["median_lead_h"] or 0) + 0.4, i,
            t["lead_row"].format(
                lead=f"{r['median_lead_h'] or 0:g}", ok=r["warned_in_advance"], n=r["floods"], p=r["precision"] or 0
            ),
            va="center", fontsize=8.5, color=INK,
        )  # fmt: skip
    ax.set_xlim(0, max([(r["median_lead_h"] or 0) for r in rows] + [1]) * 2.3)
    ax.set_xlabel(t["lead_x"])
    ax.set_title(t["lead_title"].format(n=len(metrics["data"]["events"])), fontsize=11, color=INK)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def network_map(gauges, edges, brief: dict, path: Path, title: str) -> None:
    """Stations on the river network, coloured by their level in a replayed brief (Spanish)."""
    by = {g.lid: g for g in gauges}
    level = {r["lid"]: r["level"] for r in brief["gauges_at_or_above_action"]}
    colors = {"minor": "#e45756", "action": "#f2a541"}
    fig, ax = plt.subplots(figsize=(8.0, 9.0), dpi=150)
    for e in edges:
        a, b = by[e.upstream], by[e.downstream]
        ax.plot([a.lon, b.lon], [a.lat, b.lat], color="#9fc3e7", lw=2.2, zorder=1)
    key_towns = {"26027200", "25027910", "26237040", "26207080", "26127040", "26237100"}
    order = sorted(gauges, key=lambda g: (level.get(g.lid) != "minor", g.lid not in key_towns, g.lid not in level))
    placed: list[tuple[float, float]] = []
    for g in order:
        c = colors.get(level.get(g.lid), "#4c78a8" if g.thresholds else "#c0c7cf")
        ax.scatter(g.lon, g.lat, s=70 if g.lid in level else 40, color=c, edgecolor="white", lw=0.8, zorder=3)
        wanted = g.lid in level or g.lid in key_towns
        free = all(abs(g.lat - y) > 0.12 or abs(g.lon - x) > 0.9 for x, y in placed)
        if wanted and free:
            placed.append((g.lon, g.lat))
            ax.annotate(g.name, (g.lon, g.lat), xytext=(6, 2), textcoords="offset points", fontsize=7.5, color=INK)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=colors["minor"], label="nivel alto"),
        plt.Line2D([], [], marker="o", ls="", color=colors["action"], label="nivel de vigilancia"),
        plt.Line2D([], [], marker="o", ls="", color="#4c78a8", label="normal"),
        plt.Line2D([], [], marker="o", ls="", color="#c0c7cf", label="sin datos suficientes"),
        plt.Line2D([], [], color="#9fc3e7", lw=2.2, label="tramo de río entre estaciones"),
    ]
    lons = [g.lon for g in gauges]
    ax.set_xlim(min(lons) - 0.3, max(lons) + 1.8)
    ax.legend(handles=handles, frameon=False, loc="center right", fontsize=8)
    ax.text(
        0.99, 0.12, "Esquema: líneas rectas entre estaciones,\nla distancia real va por el río (HydroRIVERS)",
        transform=ax.transAxes, ha="right", fontsize=7, color=MUTED,
    )  # fmt: skip
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlabel("longitud", color=MUTED)
    ax.set_ylabel("latitud", color=MUTED)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
