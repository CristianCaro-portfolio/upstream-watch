from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from upstream_watch import colombia, config, evaluate, pipeline
from upstream_watch.agent import ask, build_agent
from upstream_watch.lang import ES
from upstream_watch.llm import rule_parse


def test_clean_name_and_label():
    assert colombia.clean_name("LA VIRGINIA - AUT [26177030]") == "La Virginia"
    assert colombia.clean_name("PUENTE CARRETERA  - AUT [26027200]") == "Puente Carretera"
    st = colombia.Station(
        "0026237040", "PUERTO VALDIVIA - AUT [26237040]", "Antioquia", "Valdivia", "Cauca", 7.3, -75.4
    )
    assert colombia.station_label(st) == "Puerto Valdivia (Valdivia)"
    assert colombia.short_code("0026237040") == "26237040"


def _hourly(lid, values):
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    return pl.DataFrame({"lid": lid, "time": [t0 + timedelta(hours=i) for i in range(len(values))], "stage_ft": values})


def test_despike_drops_isolated_readings_only():
    df = _hourly("A", [2.0, 2.1, 9.0, 2.1, 2.2, 3.5, 4.8, 5.9, 6.0])
    kept = colombia.despike(df)["stage_ft"].to_list()
    assert 9.0 not in kept
    assert kept[-4:] == [3.5, 4.8, 5.9, 6.0]  # a real rise survives


def test_level_shift_detection():
    month = 24 * 30
    shifted = _hourly("A", [3.0] * month + [20.0] * month)
    steady = _hourly("B", [3.0] * month + [4.0] * month)
    assert set(colombia.level_shift_stations(pl.concat([shifted, steady]), max_jump_m=8.0)) == {"A"}


def test_statistical_thresholds_are_ordered():
    days = 400
    values = [1.0 + (i % 24) / 24 + (i // 24 % 50) / 10 for i in range(24 * days)]
    th = colombia.statistical_thresholds(_hourly("A", values), min_days=180)["A"]
    assert th["action"] < th["minor"] <= th["record"]


def test_network_follows_next_down():
    LineString = pytest.importorskip("shapely.geometry").LineString  # noqa: N806

    reaches = {
        1: {"HYRIV_ID": 1, "NEXT_DOWN": 2, "UPLAND_SKM": 500, "DIST_DN_KM": 300.0},
        2: {"HYRIV_ID": 2, "NEXT_DOWN": 3, "UPLAND_SKM": 900, "DIST_DN_KM": 250.0},
        3: {"HYRIV_ID": 3, "NEXT_DOWN": 0, "UPLAND_SKM": 1500, "DIST_DN_KM": 200.0},
        9: {"HYRIV_ID": 9, "NEXT_DOWN": 0, "UPLAND_SKM": 800, "DIST_DN_KM": 10.0},
    }
    geoms = {
        1: LineString([(0, 0), (0.1, 0)]),
        2: LineString([(0.1, 0), (0.2, 0)]),
        3: LineString([(0.2, 0), (0.3, 0)]),
        9: LineString([(5, 5), (5.1, 5)]),
    }
    st = [
        colombia.Station("001", "UP", "D", "M", "Cauca", 0.0, 0.05),
        colombia.Station("002", "DOWN", "D", "M", "Cauca", 0.0, 0.25),
        colombia.Station("003", "ELSEWHERE", "D", "M", "Cauca", 5.0, 5.05),
    ]
    members, snapped, edges, outlet = colombia.build_network_co(st, reaches, geoms, ("Cauca",))
    assert outlet == "2" and {m.code for m in members} == {"001", "002"}
    assert [(e.upstream, e.downstream, e.river_km) for e in edges] == [("1", "2", 100.0)]


@pytest.mark.parametrize(
    ("text", "intent", "stage"),
    [
        ("¿Cuánto tarda una creciente en llegar de Olaya a La Raya?", "travel", None),
        ("Puerto Valdivia marca 6 metros, ¿cómo está?", "impact", 6.0),
        ("Bolombolo está en 5,5 m, ¿qué pasa?", "impact", 5.5),
        ("¿Qué estaciones aguas arriba debo vigilar para La Raya?", "upstream", None),
        ("Olaya está subiendo, ¿hacia dónde va el agua aguas abajo?", "downstream", None),
        ("¿Cuál es el umbral de nivel alto en La Coquera?", "stages", None),
    ],
)
def test_spanish_parsing(text, intent, stage):
    parsed = rule_parse(text)
    assert parsed.intent == intent and parsed.stage_ft == stage


@pytest.fixture(scope="module")
def cauca():
    settings = config.Settings(llm_provider="rule", embedder="hash", region_key="cauca")
    gauges, edges, stage = pipeline.load_curated(settings.region.sample_dir)
    kit, model = pipeline.build_toolkit(gauges, edges, stage, settings, embedder_name="hash")
    return gauges, edges, kit, model


def test_cauca_sample_answers_in_spanish_with_citations(cauca):
    gauges, edges, kit, model = cauca
    agent = build_agent(kit)
    lags = {(e.upstream, e.downstream): round(model.edge_lag(e), 1) for e in edges}
    qs = evaluate.handwritten_questions(gauges, edges, "25027910", region="cauca")
    res = evaluate.run_eval(agent, qs, ["dense", "graph"], lags, ES)["modes"]
    assert res["graph"]["answer_accuracy"] == 1.0 and res["graph"]["grounded_rate"] == 1.0
    state = ask(agent, "Puerto Valdivia marca 6 metros, ¿cómo está?", thread_id="es")
    assert "en nivel alto" in state["answer"] and "[stages:26237040]" in state["answer"]


def test_cauca_sample_quality_flags(cauca):
    gauges, _, _, _ = cauca
    by_lid = {g.lid: g for g in gauges}
    assert by_lid["26177030"].thresholds == {}  # La Virginia: sensor level shift, no numbers from it
    assert all(g.thresholds["action"] < g.thresholds["minor"] for g in gauges if g.thresholds)
