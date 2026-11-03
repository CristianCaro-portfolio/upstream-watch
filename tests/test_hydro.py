from datetime import UTC, datetime, timedelta

import polars as pl

from upstream_watch import hydro
from upstream_watch.network import Edge


def _wave(lid, event, peak_hour, n=72, base=2.0, height=5.0):
    t0 = datetime(2025, 5, 10, tzinfo=UTC)
    return [
        {
            "lid": lid,
            "event_id": event,
            "time": t0 + timedelta(hours=h),
            "stage_ft": base + height * max(0, 1 - abs(h - peak_hour) / 10),
        }
        for h in range(n)
    ]


def test_crests_and_lags_on_synthetic_waves():
    rows = _wave("A", "e1", 20) + _wave("B", "e1", 30) + _wave("A", "e2", 15) + _wave("B", "e2", 27)
    stage = pl.DataFrame(rows)
    cm = hydro.crests(stage)
    lags = hydro.edge_lags([Edge("A", "B", 40.0)], cm)
    assert sorted(lags["lag_h"].to_list()) == [10.0, 12.0]
    model = hydro.TravelTimeModel.fit(lags)
    assert model.edge_lag(Edge("A", "B", 40.0)) == 11.0
    assert model.edge_lag(Edge("X", "Y", 40.0)) == 40.0 / model.celerity_kmh


def test_crest_at_window_end_is_rejected():
    stage = pl.DataFrame(_wave("A", "e1", 71))
    assert hydro.crests(stage) == {}


def test_path_and_upstream():
    edges = [Edge("A", "B", 1), Edge("B", "C", 1), Edge("D", "C", 1)]
    assert [e.downstream for e in hydro.path_to(edges, "A", "C")] == ["B", "C"]
    assert hydro.path_to(edges, "C", "A") is None
    assert hydro.upstream_of(edges, "C", max_hops=2) == {"B": 1, "D": 1, "A": 2}


def test_backtest_on_real_sample_beats_constant_wave_speed(sample):
    _, edges, stage = sample
    result = hydro.backtest(edges, hydro.crests(stage), "BRKM2")
    assert len(result["events"]) >= 10
    assert result["one_hop"]["learned"]["mae_h"] < result["one_hop"]["celerity"]["mae_h"]
    assert result["to_outlet"]["learned"]["mae_h"] < result["to_outlet"]["celerity"]["mae_h"]
