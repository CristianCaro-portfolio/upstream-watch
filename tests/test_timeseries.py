from datetime import date, timedelta

import polars as pl

from upstream_watch.timeseries import detect_events, read_partitioned, to_hourly, write_partitioned


def _daily(values):
    start = date(2025, 1, 1)
    return [{"time": (start + timedelta(days=i)).isoformat(), "value": str(v)} for i, v in enumerate(values)]


def test_detect_events_keeps_separated_peaks_only():
    values = [1000] * 40
    values[5], values[6], values[30] = 50000, 60000, 40000
    events = detect_events(_daily(values), min_peak_cfs=35000, min_gap_days=10)
    assert [e.peak_day for e in events] == ["2025-01-07", "2025-01-31"]
    assert events[0].start.startswith("2025-01-01")


def test_to_hourly_means_and_drops_sentinels():
    raw = [
        {"time": "2025-05-14T00:00:00+00:00", "value": "2.0", "approval_status": "Approved"},
        {"time": "2025-05-14T00:30:00+00:00", "value": "4.0", "approval_status": "Provisional"},
        {"time": "2025-05-14T01:15:00+00:00", "value": "-999999", "approval_status": "Approved"},
        {"time": "2025-05-14T01:30:00+00:00", "value": "5.0", "approval_status": "Approved"},
    ]
    df = to_hourly(raw, "X", "e1")
    assert df["stage_ft"].to_list() == [3.0, 5.0]
    assert df["approved_share"].to_list() == [0.5, 1.0]


def test_partition_roundtrip(tmp_path):
    df = to_hourly([{"time": "2025-05-14T00:00:00+00:00", "value": "2.0"}], "X", "2025-05-15")
    write_partitioned(df, tmp_path)
    back = read_partitioned(tmp_path)
    assert back.select("lid", "event_id", "stage_ft").to_dicts() == [
        {"lid": "X", "event_id": "2025-05-15", "stage_ft": 2.0}
    ]
    assert isinstance(back, pl.DataFrame)
