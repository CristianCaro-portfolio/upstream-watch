"""Deterministic answer writer and the grounding check.

The writer only uses what is in the retrieved documents. When the context is missing a fact
it says so instead of filling the gap, which is what makes retrieval quality visible in the
evaluation.
"""

from __future__ import annotations

import re

from .network import CATEGORIES
from .retrieval import Doc, GaugeLinker

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
CITATION = re.compile(r"\[([^\]]+)\]")
NOT_FOUND = "The retrieved context does not contain"


def category_phrase(category: str) -> str:
    if category == "none":
        return "below flood stage"
    return "at action stage" if category == "action" else f"in {category} flood"


def _cat(thresholds: dict, stage: float) -> str:
    reached = "none"
    for cat in CATEGORIES:
        if cat in thresholds and stage >= thresholds[cat]:
            reached = cat
    return category_phrase(reached)


def _fmt(x: float) -> str:
    return f"{x:g}"


def compose(intent: str, lids: list[str], stage_ft: float | None, docs: list[Doc], names: dict[str, str]) -> str:
    by_id = {d.doc_id: d for d in docs}
    focus = lids[0] if lids else None
    if focus is None:
        return f"{NOT_FOUND} a gauge I can identify in the question."
    name = names.get(focus, focus)

    if intent in ("impact", "stages"):
        stages = by_id.get(f"stages:{focus}")
        parts = []
        if stages:
            t = stages.payload["thresholds"]
            levels = ", ".join(f"{c} {_fmt(t[c])} ft" for c in CATEGORIES if c in t)
            if stage_ft is not None:
                parts.append(f"At {_fmt(stage_ft)} ft {name} is {_cat(t, stage_ft)} ({levels}) [{stages.doc_id}].")
            else:
                parts.append(f"Flood stages for {name}: {levels} [{stages.doc_id}].")
        if intent == "impact" and stage_ft is not None:
            impacts = [d for d in docs if d.kind == "impact" and d.lid == focus and d.payload["stage_ft"] <= stage_ft]
            if impacts:
                top = max(impacts, key=lambda d: d.payload["stage_ft"])
                parts.append(
                    f"NWS impact at {_fmt(top.payload['stage_ft'])} ft: {top.payload['statement']} [{top.doc_id}]"
                )
            else:
                parts.append(f"{NOT_FOUND} an impact statement for {name} at or below {_fmt(stage_ft)} ft.")
        return " ".join(parts) if parts else f"{NOT_FOUND} flood stages for {name}."

    flows = [d for d in docs if d.kind == "flow"]
    nxt = {d.payload["upstream"]: d for d in flows}

    if intent == "travel":
        if len(lids) < 2:
            return f"{NOT_FOUND} two gauges to measure travel time between."
        for src, dst in ((lids[0], lids[1]), (lids[1], lids[0])):
            path, cur = [], src
            while cur in nxt and cur != dst and len(path) < 20:
                path.append(nxt[cur])
                cur = nxt[cur].payload["downstream"]
            if cur == dst and path and all(p.payload["lag_h"] is not None for p in path):
                hours = sum(p.payload["lag_h"] for p in path)
                km = sum(p.payload["river_km"] for p in path)
                cites = "".join(f"[{p.doc_id}]" for p in path)
                return (
                    f"A crest at {names.get(src, src)} typically reaches {names.get(dst, dst)} after about "
                    f"{_fmt(round(hours, 1))} hours, over {_fmt(round(km, 1))} km of river {cites}."
                )
        return f"{NOT_FOUND} the full river path between {names.get(lids[0])} and {names.get(lids[1])}."

    if intent == "downstream":
        chain, cur, total = [], focus, 0.0
        while cur in nxt and len(chain) < 2:
            d = nxt[cur]
            total += d.payload["lag_h"] or 0.0
            chain.append((d, total))
            cur = d.payload["downstream"]
        if not chain:
            return f"{NOT_FOUND} the gauges downstream of {name}."
        steps = "; then ".join(
            f"{names.get(d.payload['downstream'])} after about {_fmt(round(t, 1))} hours [{d.doc_id}]" for d, t in chain
        )
        return f"Downstream of {name} the crest reaches {steps}."

    if intent == "upstream":
        feeders = [d for d in flows if d.payload["downstream"] == focus]
        if not feeders:
            return f"{NOT_FOUND} the gauges upstream of {name}."
        items = "; ".join(
            f"{names.get(d.payload['upstream'])}, crest lag about {_fmt(d.payload['lag_h'])} hours [{d.doc_id}]"
            for d in sorted(feeders, key=lambda d: d.payload["upstream"])
        )
        return f"Gauges that feed {name} directly: {items}."

    return f"{NOT_FOUND} an answer I can ground for this question."


def grounded(answer: str, question: str, docs: list[Doc], linker: GaugeLinker | None = None) -> tuple[bool, list[str]]:
    """Every number in the answer must appear in the context or the question, and every
    citation must be a retrieved document. Returns (ok, problems)."""
    source_numbers = set()
    for text in [question] + [d.text for d in docs] + [str(d.payload) for d in docs]:
        source_numbers.update(float(n) for n in NUMBER.findall(text))
    problems = []
    cited_ids = {d.doc_id for d in docs}
    for cite in CITATION.findall(answer):
        if cite not in cited_ids:
            problems.append(f"unknown citation {cite}")
    stripped = CITATION.sub("", answer)
    for n in NUMBER.findall(stripped):
        value = float(n)
        if value in source_numbers:
            continue
        # Sums of lags and distances are computed, accept them if they are a sum of context numbers
        # rounded to one decimal, which the writer prints with its citations.
        if _is_sum_of(value, docs):
            continue
        problems.append(f"ungrounded number {n}")
    return (not problems), problems


def _is_sum_of(value: float, docs: list[Doc]) -> bool:
    """True when value is a cumulative lag or distance along a chain of retrieved flow documents."""
    nxt = {d.payload["upstream"]: d for d in docs if d.kind == "flow"}
    for start in nxt:
        lag = km = 0.0
        cur, steps = start, 0
        while cur in nxt and steps < 30:
            d = nxt[cur]
            lag += d.payload.get("lag_h") or 0.0
            km += d.payload.get("river_km") or 0.0
            if abs(round(lag, 1) - value) < 0.051 or abs(round(km, 1) - value) < 0.051:
                return True
            cur, steps = d.payload["downstream"], steps + 1
    return False
