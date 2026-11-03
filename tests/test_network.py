from upstream_watch.network import Gauge, downstream_neighbor, order_flowlines, river_name


def _gauge(lid, comid, measure, **kw):
    return Gauge(
        lid=lid, usgs_id="0", name=lid, river="r", state="MD", county="", lat=0.0, lon=0.0,
        comid=comid, measure=measure, thresholds=kw.get("thresholds", {}),
    )  # fmt: skip


def _line(comid, coords):
    return {"properties": {"nhdplus_comid": comid}, "geometry": {"type": "LineString", "coordinates": coords}}


def test_river_name():
    assert river_name("South Fork Shenandoah River at Luray") == "South Fork Shenandoah River"
    assert river_name("Goose Creek near Leesburg") == "Goose Creek"
    assert river_name("Plain Name") == "Plain Name"


def test_order_flowlines_rebuilds_the_chain_from_shuffled_features():
    feats = [
        _line(3, [[0.2, 0.0], [0.3, 0.0]]),
        _line(1, [[0.0, 0.0], [0.1, 0.0]]),
        _line(2, [[0.1, 0.0], [0.2, 0.0]]),
    ]
    chain = order_flowlines(feats, 1)
    assert [c for c, _ in chain] == [1, 2, 3]
    assert all(abs(km - 11.1) < 0.2 for _, km in chain)


def test_downstream_neighbor_uses_measure_on_shared_and_later_flowlines():
    a = _gauge("A", 1, 80.0)
    b = _gauge("B", 1, 20.0)
    c = _gauge("C", 2, 50.0)
    chain = [(1, 10.0), (2, 10.0)]
    hit, km = downstream_neighbor(a, chain, {1: [a, b], 2: [c]})
    assert hit.lid == "B" and abs(km - 6.0) < 1e-6
    hit, km = downstream_neighbor(b, chain, {1: [a, b], 2: [c]})
    assert hit.lid == "C" and abs(km - (2.0 + 5.0)) < 1e-6


def test_category_for():
    g = _gauge("A", 1, 50.0, thresholds={"action": 5.0, "minor": 10.0, "moderate": 12.0, "major": 14.0})
    assert g.category_for(4.9) == "none"
    assert g.category_for(5.0) == "action"
    assert g.category_for(12.5) == "moderate"
    assert g.category_for(20) == "major"


def test_sample_network_is_a_tree_draining_to_little_falls(sample):
    gauges, edges, _ = sample
    lids = {g.lid for g in gauges}
    assert len(gauges) == len(edges) + 1
    assert {e.upstream for e in edges} == lids - {"BRKM2"}
    assert all(e.downstream in lids and e.river_km > 0 for e in edges)
