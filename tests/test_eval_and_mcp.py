import asyncio

from mcp import Client

from upstream_watch import evaluate
from upstream_watch.mcp_server import build_server


def test_question_set_shape(sample):
    gauges, edges, _ = sample
    qs = evaluate.build_questions(gauges, edges, "BRKM2", per_type=5)
    assert {q.intent for q in qs} == {"impact", "stages", "travel", "downstream", "upstream"}
    assert evaluate.build_questions(gauges, edges, "BRKM2", per_type=5)[0].question == qs[0].question
    hw = evaluate.handwritten_questions(gauges, edges, "BRKM2")
    assert len(hw) == len(evaluate.HANDWRITTEN) and all(q.gold_docs for q in hw)


def test_graph_mode_beats_dense_on_a_small_run(agent, kit_and_model, sample):
    gauges, edges, _ = sample
    _, model = kit_and_model
    lags = {(e.upstream, e.downstream): round(model.edge_lag(e), 1) for e in edges}
    qs = evaluate.handwritten_questions(gauges, edges, "BRKM2")
    res = evaluate.run_eval(agent, qs, ["dense", "graph"], lags)["modes"]
    assert res["graph"]["answer_accuracy"] == 1.0
    assert res["graph"]["answer_accuracy"] > res["dense"]["answer_accuracy"]


def test_mcp_tools_in_process(monkeypatch):
    monkeypatch.setenv("UPW_DATA_DIR", "/nonexistent")  # forces the bundled sample

    async def go():
        async with Client(build_server(embedder_name="hash")) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            res = await client.call_tool("downstream_eta", {"gauge": "Paw Paw", "max_hops": 2})
            return names, res

    names, res = asyncio.run(go())
    assert names == {"gauge_info", "downstream_eta", "impacts_at", "ask_question"}
    assert not res.is_error and "HNKM2" in str(res.content)
