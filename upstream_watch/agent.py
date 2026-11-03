"""LangGraph agent with two flows.

ask:   understand -> retrieve -> compose -> verify
brief: assess -> review (human in the loop, typed resume) -> finalize

The question flow answers duty officer questions with citations. The brief flow turns a
snapshot of live gauge readings into a flood watch brief that a person must approve before
it goes anywhere. Both are checkpointed, so a paused brief survives a restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, Field

from . import answer as writer
from .graph import GaugeGraph
from .llm import Provider
from .retrieval import Doc, GaugeLinker, GraphRetriever, HybridIndex
from .telemetry import traced

Mode = Literal["dense", "hybrid", "graph"]


class ReviewDecision(BaseModel):
    """What the duty officer sends back to resume a paused brief."""

    approve: bool
    reviewer: str = Field(min_length=1, max_length=60)
    note: str = Field(default="", max_length=500)


class State(TypedDict, total=False):
    question: str
    mode: Mode
    intent: str
    stage_ft: float | None
    lids: list[str]
    hits: list[str]
    draft: str
    answer: str
    grounded: bool
    problems: list[str]
    fallback_used: bool
    readings: dict[str, float]
    as_of: str
    brief: dict[str, Any]
    review: dict[str, Any]
    status: str


@dataclass
class Toolkit:
    graph: GaugeGraph
    index: HybridIndex
    linker: GaugeLinker
    retriever: GraphRetriever
    provider: Provider
    names: dict[str, str]
    top_k: int = 8


def build_agent(kit: Toolkit, checkpointer=None):
    def route(state: State) -> str:
        return "assess" if state.get("readings") else "understand"

    @traced("understand")
    def understand(state: State) -> State:
        parsed = kit.provider.parse(state["question"])
        return {"intent": parsed.intent, "stage_ft": parsed.stage_ft, "lids": kit.linker.link(state["question"])}

    @traced("retrieve")
    def retrieve(state: State) -> State:
        mode = state.get("mode", "graph")
        q, k = state["question"], kit.top_k
        if mode == "dense":
            hits = kit.index.dense(q, k)
        elif mode == "hybrid":
            hits = kit.index.hybrid(q, k)
        else:
            hits = kit.retriever.retrieve(q, state["intent"], k, state.get("stage_ft"))
        return {"hits": [h.doc.doc_id for h in hits]}

    def _docs(state: State) -> list[Doc]:
        return [kit.index.docs[i] for i in state.get("hits", [])]

    @traced("compose")
    def compose(state: State) -> State:
        docs = _docs(state)
        draft = writer.compose(state["intent"], state.get("lids", []), state.get("stage_ft"), docs, kit.names)
        context = "\n".join(f"[{d.doc_id}] {d.text}" for d in docs)
        return {"draft": draft, "answer": kit.provider.write(state["question"], context, draft)}

    @traced("verify")
    def verify(state: State) -> State:
        ok, problems = writer.grounded(state["answer"], state["question"], _docs(state))
        if ok:
            return {"grounded": True, "problems": [], "fallback_used": False}
        # The model wrote something the context does not support: fall back to the draft.
        return {"answer": state["draft"], "grounded": False, "problems": problems, "fallback_used": True}

    @traced("assess")
    def assess(state: State) -> State:
        as_of = datetime.fromisoformat(state["as_of"])
        rising = []
        for lid, stage in sorted(state["readings"].items()):
            g = kit.graph.gauge(lid)
            if not g:
                continue
            action = g["action_ft"] if g["action_ft"] is not None else g["minor_ft"]
            if action is None or stage < action:
                continue
            downstream = [
                {
                    "lid": h.lid,
                    "name": h.name,
                    "eta": (as_of + timedelta(hours=h.eta_h)).isoformat(timespec="minutes"),
                    "eta_h": h.eta_h,
                }
                for h in kit.graph.downstream(lid, max_hops=3)
            ]
            impacts = kit.graph.impacts_at(lid, stage, limit=1)
            rising.append(
                {
                    "lid": lid,
                    "name": g["name"],
                    "stage_ft": round(stage, 2),
                    "category": writer._cat(
                        {k: g[f"{k}_ft"] for k in ("action", "minor", "moderate", "major") if g[f"{k}_ft"] is not None},
                        stage,
                    ),
                    "impact": impacts[0] if impacts else None,
                    "downstream": downstream,
                }
            )
        # Most severe flood category first; feet are not comparable across gauges.
        severity = {"major": 4, "moderate": 3, "minor": 2, "action": 1}
        rising.sort(key=lambda r: (-max((v for k, v in severity.items() if k in r["category"]), default=0), r["lid"]))
        return {"brief": {"as_of": state["as_of"], "gauges_at_or_above_action": rising}}

    @traced("review")
    def review(state: State) -> State:
        if not state["brief"]["gauges_at_or_above_action"]:
            return {"review": {"approve": False, "reviewer": "system", "note": "nothing at action stage"}}
        decision = interrupt(
            {"task": "approve flood watch brief", "brief": state["brief"]},
            response_schema=ReviewDecision,
        )
        return {"review": decision.model_dump()}

    @traced("finalize")
    def finalize(state: State) -> State:
        if state["review"]["approve"]:
            return {"status": "approved"}
        return {"status": "nothing to send" if state["review"]["reviewer"] == "system" else "rejected"}

    builder = StateGraph(State)
    retry = RetryPolicy(max_attempts=2, initial_interval=0.2)
    builder.add_node("understand", understand)
    builder.add_node("retrieve", retrieve, retry_policy=retry)
    builder.add_node("compose", compose, retry_policy=retry)
    builder.add_node("verify", verify)
    builder.add_node("assess", assess)
    builder.add_node("review", review)
    builder.add_node("finalize", finalize)
    builder.add_conditional_edges(START, route, {"assess": "assess", "understand": "understand"})
    builder.add_edge("understand", "retrieve")
    builder.add_edge("retrieve", "compose")
    builder.add_edge("compose", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("assess", "review")
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def ask(agent, question: str, mode: Mode = "graph", thread_id: str = "ask") -> State:
    return agent.invoke({"question": question, "mode": mode}, {"configurable": {"thread_id": thread_id}})


def start_brief(agent, readings: dict[str, float], as_of: str, thread_id: str) -> tuple[State, Any]:
    """Run until the review pause. Returns the state and the pending interrupt (or None)."""
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.invoke({"readings": readings, "as_of": as_of}, config)
    pending = state.get("__interrupt__")
    return state, (pending[0] if pending else None)


def resume_brief(agent, thread_id: str, decision: dict) -> State:
    return agent.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})
