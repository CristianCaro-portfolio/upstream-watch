import pydantic
import pytest

from upstream_watch import replay
from upstream_watch.agent import ask, build_agent, resume_brief, start_brief
from upstream_watch.answer import grounded


def test_graph_mode_answers_with_citations(agent):
    state = ask(agent, "Little Falls just hit 11.7 feet. What is under water?")
    assert state["intent"] == "impact" and state["grounded"]
    assert "[impact:BRKM2@11.5]" in state["answer"] and "in minor flood" in state["answer"]


def test_travel_time_sums_learned_lags(agent):
    state = ask(agent, "How many hours from Shepherdstown to Little Falls?")
    assert "[flow:SHEW2->PORM2][flow:PORM2->EDFM2][flow:EDFM2->BRKM2]" in state["answer"]
    assert state["grounded"]


def test_dense_mode_admits_missing_context(agent):
    state = ask(agent, "Which gauges flow into Point of Rocks?", mode="dense", thread_id="dense")
    assert state["hits"] and state["grounded"]


class _Liar:
    name = "liar"

    def parse(self, question):
        from upstream_watch.llm import rule_parse

        return rule_parse(question)

    def write(self, question, context, draft):
        return "Little Falls is at 99.9 ft and rising [impact:FAKE@1]."


def test_ungrounded_model_output_falls_back_to_the_draft(kit):
    kit_liar = type(kit)(**{**kit.__dict__, "provider": _Liar()})
    state = ask(build_agent(kit_liar), "Little Falls just hit 11.7 feet. What is under water?")
    assert state["fallback_used"] and not state["grounded"]
    assert "99.9" not in state["answer"] and "[impact:BRKM2@11.5]" in state["answer"]
    assert any("unknown citation" in p for p in state["problems"])


def test_grounding_accepts_cumulative_sums(kit):
    docs = [kit.index.docs[d] for d in ("flow:SHEW2->PORM2", "flow:PORM2->EDFM2")]
    total = round(docs[0].payload["lag_h"] + docs[1].payload["lag_h"], 1)
    ok, problems = grounded(f"about {total:g} hours [flow:SHEW2->PORM2]", "q", docs)
    assert ok, problems


def test_brief_pauses_for_review_and_validates_the_resume(agent, sample):
    gauges, _, stage = sample
    by_lid = {g.lid: g for g in gauges}
    hour = replay.busiest_hour(stage, "2025-05-15", by_lid)
    state, pending = start_brief(agent, replay.readings_at(stage, "2025-05-15", hour), hour, "t-brief")
    assert pending is not None and pending.response_schema["required"] == ["approve", "reviewer"]
    assert state["brief"]["gauges_at_or_above_action"]
    with pytest.raises(pydantic.ValidationError):
        resume_brief(agent, "t-brief", {"approve": "maybe"})
    final = resume_brief(agent, "t-brief", {"approve": False, "reviewer": "duty officer", "note": "hold"})
    assert final["status"] == "rejected"


def test_quiet_readings_do_not_pause(agent):
    state, pending = start_brief(agent, {"BRKM2": 2.0}, "2025-05-14T00:00:00+00:00", "t-quiet")
    assert pending is None and state["status"] == "nothing to send"
