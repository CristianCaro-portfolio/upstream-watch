"""Answer and brief wording per region language. English for the US data, Spanish for Colombia."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lang:
    code: str
    unit: str
    not_found: str
    phrases: dict[str, str]
    labels: dict[str, str]
    text: dict[str, str] = field(default_factory=dict)

    def phrase(self, category: str) -> str:
        return self.phrases.get(category, self.phrases["none"])

    def t(self, key: str, **kw) -> str:
        return self.text[key].format(**kw)


EN = Lang(
    code="en",
    unit="ft",
    not_found="The retrieved context does not contain",
    phrases={
        "none": "below flood stage",
        "action": "at action stage",
        "minor": "in minor flood",
        "moderate": "in moderate flood",
        "major": "in major flood",
    },
    labels={"action": "action", "minor": "minor", "moderate": "moderate", "major": "major"},
    text={
        "no_gauge": "{nf} a gauge I can identify in the question.",
        "at_stage": "At {stage} {unit} {name} is {phrase} ({levels}) [{cite}].",
        "stages": "Flood stages for {name}: {levels} [{cite}].",
        "impact": "NWS impact at {stage} {unit}: {statement} [{cite}]",
        "no_impact": "{nf} an impact statement for {name} at or below {stage} {unit}.",
        "no_stages": "{nf} flood stages for {name}.",
        "need_two": "{nf} two gauges to measure travel time between.",
        "travel": "A crest at {src} typically reaches {dst} after about {hours} hours, over {km} km of river {cites}.",
        "no_path": "{nf} the full river path between {a} and {b}.",
        "no_down": "{nf} the gauges downstream of {name}.",
        "down_step": "{name} after about {hours} hours [{cite}]",
        "down_join": "; then ",
        "down": "Downstream of {name} the crest reaches {steps}.",
        "no_up": "{nf} the gauges upstream of {name}.",
        "up_item": "{name}, crest lag about {hours} hours [{cite}]",
        "up": "Gauges that feed {name} directly: {items}.",
        "other": "{nf} an answer I can ground for this question.",
        "doc_stages": "Flood stages for {name} ({lid}), {state}: {levels}.{record}",
        "doc_record": " Record crest {stage} {unit} on {date}.",
        "doc_impact": "{name} ({lid}), {state}. At {stage} {unit}: {statement}",
        "doc_flow": "{up} ({up_lid}) flows downstream to {down} ({down_lid}), {km} km of river.{lag}",
        "doc_lag": " Typical crest travel time {lag} hours.",
        "brief_title": "Flood watch brief as of {as_of} UTC",
        "brief_none": "No gauge at or above action stage.",
        "brief_count": "{n} gauges at or above action stage, most severe first:",
        "brief_row": "- {name} ({lid}) {stage} {unit}, {category}. Next: {nxt}.",
        "brief_next": "{name} ~{hours} h",
        "brief_outlet": "none (outlet)",
        "brief_impact": "  NWS impact at {stage} {unit}: {statement}",
        "brief_more": "... and {n} more.",
    },
)

ES = Lang(
    code="es",
    unit="m",
    not_found="El contexto recuperado no contiene",
    phrases={
        "none": "por debajo del nivel de vigilancia",
        "action": "en nivel de vigilancia",
        "minor": "en nivel alto",
        "moderate": "en nivel muy alto",
        "major": "en nivel extremo",
    },
    labels={"action": "vigilancia", "minor": "alto", "moderate": "muy alto", "major": "extremo"},
    text={
        "no_gauge": "{nf} una estación que pueda identificar en la pregunta.",
        "at_stage": "Con {stage} {unit}, {name} está {phrase} ({levels}) [{cite}].",
        "stages": "Niveles de referencia de {name}: {levels} [{cite}].",
        "impact": "Referencia a {stage} {unit}: {statement} [{cite}]",
        "no_impact": "{nf} una descripción de impacto para {name} a {stage} {unit} o menos.",
        "no_stages": "{nf} niveles de referencia para {name}.",
        "need_two": "{nf} dos estaciones para medir el tiempo de viaje.",
        "travel": "Una creciente en {src} suele llegar a {dst} en unas {hours} horas, por {km} km de río {cites}.",
        "no_path": "{nf} el recorrido completo del río entre {a} y {b}.",
        "no_down": "{nf} las estaciones aguas abajo de {name}.",
        "down_step": "{name} en unas {hours} horas [{cite}]",
        "down_join": "; luego ",
        "down": "Aguas abajo de {name} la creciente llega a {steps}.",
        "no_up": "{nf} las estaciones aguas arriba de {name}.",
        "up_item": "{name}, unas {hours} horas antes [{cite}]",
        "up": "Estaciones que alimentan directamente a {name}: {items}.",
        "other": "{nf} una respuesta que pueda sustentar para esta pregunta.",
        "doc_stages": "Niveles de referencia de {name} ({lid}), {state}: {levels}.{record}",
        "doc_record": " Máximo registrado {stage} {unit} el {date}.",
        "doc_impact": "{name} ({lid}), {state}. A {stage} {unit}: {statement}",
        "doc_flow": "{up} ({up_lid}) drena aguas abajo hacia {down} ({down_lid}), {km} km de río.{lag}",
        "doc_lag": " Tiempo típico de viaje de la creciente {lag} horas.",
        "brief_title": "Boletín de vigilancia de ríos, corte {as_of} UTC",
        "brief_none": "Ninguna estación en nivel de vigilancia o superior.",
        "brief_count": "{n} estaciones en nivel de vigilancia o superior, primero las más altas:",
        "brief_row": "- {name} ({lid}) {stage} {unit}, {category}. Siguen: {nxt}.",
        "brief_next": "{name} ~{hours} h",
        "brief_outlet": "ninguna (salida de la cuenca)",
        "brief_impact": "  Referencia a {stage} {unit}: {statement}",
        "brief_more": "... y {n} más.",
    },
)

LANGS = {"en": EN, "es": ES}


def get(code: str) -> Lang:
    return LANGS.get(code, EN)
