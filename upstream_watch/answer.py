"""Deterministic answer writer and the grounding check.

The writer only uses what is in the retrieved documents. When the context is missing a fact
it says so instead of filling the gap, which is what makes retrieval quality visible in the
evaluation.
"""

from __future__ import annotations

import re

from .lang import EN, Lang
from .network import CATEGORIES
from .retrieval import Doc, GaugeLinker

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
CITATION = re.compile(r"\[([^\]]+)\]")
NOT_FOUND = EN.not_found


def category_phrase(category: str, lang: Lang = EN) -> str:
    return lang.phrase(category)


def reached_category(thresholds: dict, stage: float) -> str:
    reached = "none"
    for cat in CATEGORIES:
        if cat in thresholds and stage >= thresholds[cat]:
            reached = cat
    return reached


def _cat(thresholds: dict, stage: float, lang: Lang = EN) -> str:
    return lang.phrase(reached_category(thresholds, stage))


def _fmt(x: float) -> str:
    return f"{x:g}"


def levels_text(thresholds: dict, lang: Lang = EN) -> str:
    return ", ".join(f"{lang.labels[c]} {_fmt(thresholds[c])} {lang.unit}" for c in CATEGORIES if c in thresholds)


def compose(
    intent: str,
    lids: list[str],
    stage_ft: float | None,
    docs: list[Doc],
    names: dict[str, str],
    lang: Lang = EN,
) -> str:
    """Answer from the retrieved documents only. stage_ft is the stage in the region unit."""
    nf = lang.not_found
    by_id = {d.doc_id: d for d in docs}
    focus = lids[0] if lids else None
    if focus is None:
        return lang.t("no_gauge", nf=nf)
    name = names.get(focus, focus)
    unit = lang.unit

    if intent in ("impact", "stages"):
        stages = by_id.get(f"stages:{focus}")
        parts = []
        if stages:
            t = stages.payload["thresholds"]
            levels = levels_text(t, lang)
            if stage_ft is not None:
                parts.append(
                    lang.t(
                        "at_stage", stage=_fmt(stage_ft), unit=unit, name=name, phrase=_cat(t, stage_ft, lang),
                        levels=levels, cite=stages.doc_id,
                    )
                )  # fmt: skip
            else:
                parts.append(lang.t("stages", name=name, levels=levels, cite=stages.doc_id))
        if intent == "impact" and stage_ft is not None:
            impacts = [d for d in docs if d.kind == "impact" and d.lid == focus and d.payload["stage_ft"] <= stage_ft]
            if impacts:
                top = max(impacts, key=lambda d: d.payload["stage_ft"])
                parts.append(
                    lang.t(
                        "impact", stage=_fmt(top.payload["stage_ft"]), unit=unit,
                        statement=top.payload["statement"], cite=top.doc_id,
                    )
                )  # fmt: skip
            elif lang.code == "en" or any(d.kind == "impact" for d in docs):
                parts.append(lang.t("no_impact", nf=nf, name=name, stage=_fmt(stage_ft), unit=unit))
        return " ".join(parts) if parts else lang.t("no_stages", nf=nf, name=name)

    flows = [d for d in docs if d.kind == "flow"]
    nxt = {d.payload["upstream"]: d for d in flows}

    if intent == "travel":
        if len(lids) < 2:
            return lang.t("need_two", nf=nf)
        for src, dst in ((lids[0], lids[1]), (lids[1], lids[0])):
            path, cur = [], src
            while cur in nxt and cur != dst and len(path) < 20:
                path.append(nxt[cur])
                cur = nxt[cur].payload["downstream"]
            if cur == dst and path and all(p.payload["lag_h"] is not None for p in path):
                hours = sum(p.payload["lag_h"] for p in path)
                km = sum(p.payload["river_km"] for p in path)
                cites = "".join(f"[{p.doc_id}]" for p in path)
                return lang.t(
                    "travel", src=names.get(src, src), dst=names.get(dst, dst), hours=_fmt(round(hours, 1)),
                    km=_fmt(round(km, 1)), cites=cites,
                )  # fmt: skip
        return lang.t("no_path", nf=nf, a=names.get(lids[0]), b=names.get(lids[1]))

    if intent == "downstream":
        chain, cur, total = [], focus, 0.0
        while cur in nxt and len(chain) < 2:
            d = nxt[cur]
            total += d.payload["lag_h"] or 0.0
            chain.append((d, total))
            cur = d.payload["downstream"]
        if not chain:
            return lang.t("no_down", nf=nf, name=name)
        steps = lang.text["down_join"].join(
            lang.t("down_step", name=names.get(d.payload["downstream"]), hours=_fmt(round(t, 1)), cite=d.doc_id)
            for d, t in chain
        )
        return lang.t("down", name=name, steps=steps)

    if intent == "upstream":
        feeders = [d for d in flows if d.payload["downstream"] == focus]
        if not feeders:
            return lang.t("no_up", nf=nf, name=name)
        items = "; ".join(
            lang.t("up_item", name=names.get(d.payload["upstream"]), hours=_fmt(d.payload["lag_h"]), cite=d.doc_id)
            for d in sorted(feeders, key=lambda d: d.payload["upstream"])
        )
        return lang.t("up", name=name, items=items)

    return lang.t("other", nf=nf)


def grounded(answer: str, question: str, docs: list[Doc], linker: GaugeLinker | None = None) -> tuple[bool, list[str]]:
    """Every number in the answer must appear in the context or the question, and every
    citation must be a retrieved document. Returns (ok, problems)."""
    source_numbers = set()
    question = re.sub(r"(\d),(\d)", r"\1.\2", question)  # "5,5 m" is 5.5 in Spanish
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
