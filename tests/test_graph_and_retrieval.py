import pytest

from upstream_watch.retrieval import HashingEmbedder, sparse_vector, tokens


def test_graph_counts_match_sample(kit, sample):
    gauges, edges, _ = sample
    counts = kit.graph.counts()
    assert counts["gauges"] == len(gauges)
    assert counts["flows_to"] == len(edges)
    assert counts["impacts"] == sum(len(g.impacts) for g in gauges)


def test_downstream_eta_is_cumulative(kit):
    hops = kit.graph.downstream("CBEM2")
    assert [h.lid for h in hops][-1] == "BRKM2"
    etas = [h.eta_h for h in hops]
    assert etas == sorted(etas) and all(h.hops == i + 1 for i, h in enumerate(hops))


def test_impacts_at_returns_highest_reached(kit):
    rows = kit.graph.impacts_at("BRKM2", 11.7, limit=2)
    assert [r["stage_ft"] for r in rows] == [11.5, 11.0]
    assert kit.graph.impacts_at("BRKM2", 0.5) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Little Falls just hit 11.7 feet", ["BRKM2"]),
        ("How long from the North Branch at Cumberland to Little Falls?", ["CBEM2", "BRKM2"]),
        ("How long from PAWW2 to SHEW2?", ["PAWW2", "SHEW2"]),
        ("from South Fork Catoctin Creek near Waterford to Point of Rocks", ["WATV2", "PORM2"]),
        ("Opequon Creek near Martinsburg", ["MBGW2"]),
    ],
)
def test_linker(kit, text, expected):
    assert kit.linker.link(text) == expected


def test_tokens_and_sparse_vector():
    assert tokens("At 11.5 ft the C&O towpath") == ["11.5", "ft", "c", "o", "towpath"]
    vec = sparse_vector("flood flood towpath")
    assert len(vec.indices) == 2 and max(vec.values) > min(vec.values)


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(64)
    a, b = emb.embed(["towpath flooded"]), emb.embed(["towpath flooded"])
    assert (a == b).all() and abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5


def test_hybrid_filters_by_gauge(kit):
    hits = kit.index.hybrid("towpath flooded", 5, lids=["BRKM2"])
    assert hits and all(h.doc.lid == "BRKM2" for h in hits)
