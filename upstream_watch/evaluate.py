"""Question set with ground truth taken from the data, and the retrieval mode comparison.

Questions are generated from the graph so every expected answer is known exactly. Each mode
gets the same question, the same parser and the same deterministic writer, so differences
come from what each retriever put in the context.
"""

from __future__ import annotations

import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field

from .agent import ask
from .answer import category_phrase
from .hydro import path_to
from .lang import EN, Lang
from .network import Edge, Gauge


@dataclass
class EvalQuestion:
    qid: str
    intent: str
    question: str
    gold_docs: list[str]
    expect: dict = field(default_factory=dict)


TEMPLATES = {
    "impact": [
        "What happens at {name} when the river reaches {stage} feet?",
        "The gauge {name} reads {stage} ft. What is flooded there?",
        "Impacts expected at {name} at a stage of {stage} ft?",
    ],
    "stages": [
        "What are the flood stages for {name}?",
        "At what level does {name} go into minor flood stage?",
        "Give me the flood category thresholds for {name}.",
    ],
    "travel": [
        "How long does a crest take to travel from {a} to {b}?",
        "If {a} crests now, how many hours until the crest gets from there to {b}?",
        "What is the travel time from {a} down to {b}?",
    ],
    "downstream": [
        "{name} is rising fast. Which gauges downstream should expect it next?",
        "Where does the water from {name} go next downstream, and when?",
        "Which gauges will see the crest from {name} after it passes?",
    ],
    "upstream": [
        "Which upstream gauges should I watch for early warning at {name}?",
        "What gauges flow into {name}?",
        "List the upstream gauges that feed {name}.",
    ],
}


TEMPLATES_ES = {
    "impact": [
        "¿Qué pasa en {name} cuando el río llega a {stage} m?",
        "{name} marca {stage} m, ¿cómo está?",
    ],
    "stages": [
        "¿Cuáles son los niveles de referencia de {name}?",
        "¿Cuál es el umbral de nivel alto en {name}?",
    ],
    "travel": [
        "¿Cuánto tarda una creciente en llegar de {a} a {b}?",
        "¿Cuántas horas hay de {a} a {b}?",
    ],
    "downstream": [
        "{name} está subiendo, ¿qué estaciones siguen aguas abajo?",
        "¿Hacia dónde va el agua de {name} aguas abajo?",
    ],
    "upstream": [
        "¿Qué estaciones aguas arriba debo vigilar para {name}?",
        "¿Qué estaciones alimentan a {name}?",
    ],
}


def build_questions(
    gauges: list[Gauge], edges: list[Edge], outlet: str, seed: int = 7, per_type: int = 24, lang: Lang = EN
) -> list[EvalQuestion]:
    rng = random.Random(seed)
    by_lid = {g.lid: g for g in gauges}
    qs: list[EvalQuestion] = []
    templates = TEMPLATES_ES if lang.code == "es" else TEMPLATES

    def pick(template_key: str) -> str:
        return rng.choice(templates[template_key])

    with_impacts = [g for g in gauges if len(g.impacts) >= 2]
    for g in rng.sample(with_impacts, min(per_type, len(with_impacts))):
        idx = rng.randrange(len(g.impacts))
        imp = g.impacts[idx]
        ceiling = g.impacts[idx + 1].stage_ft if idx + 1 < len(g.impacts) else imp.stage_ft + 1
        stage = round(imp.stage_ft + rng.uniform(0, max(0.0, min(0.9, ceiling - imp.stage_ft - 0.1))), 1)
        qs.append(
            EvalQuestion(
                qid=f"impact-{g.lid}",
                intent="impact",
                question=pick("impact").format(name=g.name, stage=f"{stage:g}"),
                gold_docs=[f"impact:{imp.doc_id}", f"stages:{g.lid}"],
                expect={"cite": f"impact:{imp.doc_id}", "category": g.category_for(stage)},
            )
        )
    with_minor = [g for g in gauges if "minor" in g.thresholds]
    for g in rng.sample(with_minor, min(per_type, len(with_minor))):
        qs.append(
            EvalQuestion(
                qid=f"stages-{g.lid}",
                intent="stages",
                question=pick("stages").format(name=g.name),
                gold_docs=[f"stages:{g.lid}"],
                expect={"number": f"{g.thresholds['minor']:g} ft"},
            )
        )
    pairs = []
    for g in gauges:
        path = path_to(edges, g.lid, outlet) or []
        for n in (2, 3, 4):
            if len(path) >= n:
                pairs.append((g.lid, path[n - 1].downstream, path[:n]))
    for a, b, path in rng.sample(pairs, min(per_type, len(pairs))):
        qs.append(
            EvalQuestion(
                qid=f"travel-{a}-{b}",
                intent="travel",
                question=pick("travel").format(a=by_lid[a].name, b=by_lid[b].name),
                gold_docs=[f"flow:{e.upstream}->{e.downstream}" for e in path],
                expect={"path": [(e.upstream, e.downstream) for e in path]},
            )
        )
    two_hop = [g for g in gauges if len(path_to(edges, g.lid, outlet) or []) >= 2]
    for g in rng.sample(two_hop, min(per_type, len(two_hop))):
        path = (path_to(edges, g.lid, outlet) or [])[:2]
        qs.append(
            EvalQuestion(
                qid=f"downstream-{g.lid}",
                intent="downstream",
                question=pick("downstream").format(name=g.name),
                gold_docs=[f"flow:{e.upstream}->{e.downstream}" for e in path],
                expect={"names": [by_lid[e.downstream].name for e in path]},
            )
        )
    feeders: dict[str, list[str]] = {}
    for e in edges:
        feeders.setdefault(e.downstream, []).append(e.upstream)
    targets = [lid for lid, ups in feeders.items() if len(ups) >= 2]
    for lid in rng.sample(targets, min(per_type, len(targets))):
        qs.append(
            EvalQuestion(
                qid=f"upstream-{lid}",
                intent="upstream",
                question=pick("upstream").format(name=by_lid[lid].name),
                gold_docs=[f"flow:{u}->{lid}" for u in sorted(feeders[lid])],
                expect={"names": [by_lid[u].name for u in sorted(feeders[lid])]},
            )
        )
    return qs


def correct(q: EvalQuestion, answer: str, lags: dict[tuple[str, str], float], lang: Lang = EN) -> bool:
    if q.intent == "impact":
        return f"[{q.expect['cite']}]" in answer and category_phrase(q.expect["category"], lang) in answer
    if q.intent == "stages":
        return q.expect["number"].replace(" ft", f" {lang.unit}") in answer
    if q.intent == "travel":
        hours = round(sum(lags[p] for p in map(tuple, q.expect["path"])), 1)
        return re.search(rf"(?<![\d.]){re.escape(f'{hours:g}')} (?:hours|horas)", answer) is not None
    return all(name in answer for name in q.expect["names"])


def run_eval(
    agent, questions: list[EvalQuestion], modes: list[str], lags: dict[tuple[str, str], float], lang: Lang = EN
) -> dict:
    results: dict[str, dict] = {}
    rows = []
    for mode in modes:
        recalls, full, acc, ctx, lat, grounded, intent_ok, fallback = [], [], [], [], [], [], [], []
        per_intent: dict[str, list[bool]] = {}
        for q in questions:
            t0 = time.perf_counter()
            state = ask(agent, q.question, mode=mode, thread_id=f"eval-{mode}-{q.qid}")
            lat.append((time.perf_counter() - t0) * 1000)
            hits = set(state.get("hits", []))
            got = sum(1 for d in q.gold_docs if d in hits)
            recalls.append(got / len(q.gold_docs))
            full.append(got == len(q.gold_docs))
            ok = correct(q, state["answer"], lags, lang)
            acc.append(ok)
            grounded.append(bool(state.get("grounded")))
            intent_ok.append(state.get("intent") == q.intent)
            fallback.append(bool(state.get("fallback_used")))
            ctx.append(len(hits))
            per_intent.setdefault(q.intent, []).append(ok)
            rows.append(
                {"mode": mode, "qid": q.qid, "intent": q.intent, "correct": ok, "recall": got / len(q.gold_docs)}
            )
        lat.sort()
        results[mode] = {
            "questions": len(questions),
            "context_recall": round(statistics.fmean(recalls), 3),
            "full_context_rate": round(statistics.fmean(full), 3),
            "answer_accuracy": round(statistics.fmean(acc), 3),
            "grounded_rate": round(statistics.fmean(grounded), 3),
            "writer_fallback_rate": round(statistics.fmean(fallback), 3),
            "intent_accuracy": round(statistics.fmean(intent_ok), 3),
            "mean_context_docs": round(statistics.fmean(ctx), 2),
            "latency_p50_ms": round(statistics.median(lat), 1),
            "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
            "accuracy_by_intent": {k: round(statistics.fmean(v), 3) for k, v in sorted(per_intent.items())},
        }
    return {"modes": results, "rows": rows, "questions": [asdict(q) for q in questions]}


# Written by hand the way a duty officer would type them: short place names, station ids,
# no full NWS names. Gold documents were checked against the NWPS data by reading it.
HANDWRITTEN: list[tuple[str, str, dict]] = [
    ("impact", "Point of Rocks is reading 13.5 ft, what is flooding there?", {"lid": "PORM2", "impact": 13.0}),
    ("impact", "Little Falls just hit 11.7 feet. What is under water?", {"lid": "BRKM2", "impact": 11.5}),
    ("impact", "Hancock gauge at 24.3 ft, what are the impacts?", {"lid": "HNKM2", "impact": 24.0}),
    ("impact", "What does 15 ft at Front Royal mean on the South Fork?", {"lid": "FROV2", "impact": 14.0}),
    ("impact", "BRKM2 reads 9.2 ft, anything flooded?", {"lid": "BRKM2", "impact": 9.0}),
    ("stages", "When does Goose Creek near Leesburg go into minor flood stage?", {"lid": "LEEV2"}),
    ("stages", "What are the flood stages at Edwards Ferry?", {"lid": "EDFM2"}),
    ("travel", "How long does it take a crest to get from the North Branch at Cumberland to Little Falls?", {"a": "CBEM2", "b": "BRKM2"}),
    ("travel", "Travel time from Millville to Edwards Ferry?", {"a": "MILW2", "b": "EDFM2"}),
    ("travel", "How many hours from Shepherdstown to Little Falls?", {"a": "SHEW2", "b": "BRKM2"}),
    ("travel", "How long from PAWW2 to SHEW2?", {"a": "PAWW2", "b": "SHEW2"}),
    ("downstream", "Paw Paw is cresting, which gauges downstream get it next?", {"lid": "PAWW2"}),
    ("downstream", "Front Royal is rising. Where does the water go next downstream?", {"lid": "FROV2"}),
    ("upstream", "Which gauges flow into Point of Rocks?", {"lid": "PORM2"}),
    ("upstream", "Which upstream gauges should I watch for early warning at Little Falls?", {"lid": "BRKM2"}),
    ("upstream", "What gauges flow into Goose Creek near Leesburg?", {"lid": "LEEV2"}),
]  # fmt: skip


HANDWRITTEN_ES: list[tuple[str, str, dict]] = [
    ("impact", "Puerto Valdivia marca 6 metros, ¿cómo está?", {"lid": "26237040"}),
    ("impact", "Bolombolo está en 5,5 m, ¿qué pasa?", {"lid": "26207080"}),
    ("stages", "¿Cuáles son los niveles de referencia de Puerto Valdivia?", {"lid": "26237040"}),
    ("stages", "¿Cuál es el umbral de nivel alto en La Coquera?", {"lid": "26247020"}),
    ("stages", "Niveles de referencia de la estación 25027910", {"lid": "25027910"}),
    ("travel", "¿Cuánto tarda una creciente en llegar de Puerto Valdivia a La Raya?", {"a": "26237040", "b": "25027910"}),
    ("travel", "¿Cuántas horas hay de Bolombolo a Puerto Valdivia?", {"a": "26207080", "b": "26237040"}),
    ("travel", "¿Cuánto se demora una creciente de Olaya a Las Varas?", {"a": "26237100", "b": "25027200"}),
    ("downstream", "Olaya está subiendo, ¿hacia dónde va el agua aguas abajo?", {"lid": "26237100"}),
    ("downstream", "Puente Iglesias está crecido, ¿qué estaciones siguen aguas abajo?", {"lid": "26207030"}),
    ("upstream", "¿Qué estaciones aguas arriba debo vigilar para La Raya?", {"lid": "25027910"}),
    ("upstream", "¿Qué estaciones alimentan a Bolombolo?", {"lid": "26207080"}),
]  # fmt: skip


def handwritten_questions(
    gauges: list[Gauge], edges: list[Edge], outlet: str, region: str = "potomac"
) -> list[EvalQuestion]:
    by_lid = {g.lid: g for g in gauges}
    feeders: dict[str, list[str]] = {}
    for e in edges:
        feeders.setdefault(e.downstream, []).append(e.upstream)
    out = []
    for i, (intent, text, spec) in enumerate(HANDWRITTEN_ES if region == "cauca" else HANDWRITTEN):
        qid = f"hw-{i:02d}-{intent}"
        if intent == "impact":
            g = by_lid[spec["lid"]]
            stage = float(re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ft|feet|m|metros)\b", text).group(1).replace(",", "."))
            gold = f"impact:{g.lid}@{spec['impact']:g}" if "impact" in spec else f"stages:{g.lid}"
            out.append(
                EvalQuestion(
                    qid,
                    intent,
                    text,
                    sorted({gold, f"stages:{g.lid}"}),
                    {"cite": gold, "category": g.category_for(stage)},
                )
            )
        elif intent == "stages":
            g = by_lid[spec["lid"]]
            out.append(
                EvalQuestion(qid, intent, text, [f"stages:{g.lid}"], {"number": f"{g.thresholds['minor']:g} ft"})
            )
        elif intent == "travel":
            path = path_to(edges, spec["a"], spec["b"]) or []
            gold = [f"flow:{e.upstream}->{e.downstream}" for e in path]
            out.append(EvalQuestion(qid, intent, text, gold, {"path": [(e.upstream, e.downstream) for e in path]}))
        elif intent == "downstream":
            path = (path_to(edges, spec["lid"], outlet) or [])[:2]
            gold = [f"flow:{e.upstream}->{e.downstream}" for e in path]
            out.append(EvalQuestion(qid, intent, text, gold, {"names": [by_lid[e.downstream].name for e in path]}))
        else:
            ups = sorted(feeders[spec["lid"]])
            gold = [f"flow:{u}->{spec['lid']}" for u in ups]
            out.append(EvalQuestion(qid, intent, text, gold, {"names": [by_lid[u].name for u in ups]}))
    return out
