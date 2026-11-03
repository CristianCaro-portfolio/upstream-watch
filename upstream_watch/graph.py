"""Property graph of the gauge network in LadybugDB (embedded, Cypher).

Nodes: Gauge, River, Impact. Relationships: FLOWS_TO (gauge to the next gauge downstream,
with river distance and learned crest travel time), ON_RIVER and HAS_IMPACT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import real_ladybug as lb

from .hydro import TravelTimeModel
from .network import Edge, Gauge

SCHEMA = [
    """CREATE NODE TABLE Gauge(
        lid STRING PRIMARY KEY, usgs_id STRING, name STRING, river STRING, state STRING, county STRING,
        lat DOUBLE, lon DOUBLE, action_ft DOUBLE, minor_ft DOUBLE, moderate_ft DOUBLE, major_ft DOUBLE,
        record_ft DOUBLE, record_date STRING)""",
    "CREATE NODE TABLE River(name STRING PRIMARY KEY)",
    "CREATE NODE TABLE Impact(doc_id STRING PRIMARY KEY, stage_ft DOUBLE, statement STRING)",
    "CREATE REL TABLE FLOWS_TO(FROM Gauge TO Gauge, river_km DOUBLE, lag_h DOUBLE, n_obs INT64)",
    "CREATE REL TABLE ON_RIVER(FROM Gauge TO River)",
    "CREATE REL TABLE HAS_IMPACT(FROM Gauge TO Impact)",
]


@dataclass(frozen=True)
class Hop:
    lid: str
    name: str
    hops: int
    eta_h: float
    river_km: float


class GaugeGraph:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.db = lb.Database(str(path))
        self.conn = lb.Connection(self.db)

    def _rows(self, query: str, params: dict | None = None) -> list[list]:
        result = self.conn.execute(query, params or {})
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    # build -----------------------------------------------------------------------------

    def load(self, gauges: list[Gauge], edges: list[Edge], model: TravelTimeModel | None) -> None:
        for stmt in SCHEMA:
            self.conn.execute(stmt)
        for river in sorted({g.river for g in gauges}):
            self.conn.execute("CREATE (:River {name: $name})", {"name": river})
        for g in gauges:
            t = g.thresholds
            self.conn.execute(
                """CREATE (:Gauge {lid: $lid, usgs_id: $usgs, name: $name, river: $river, state: $state,
                   county: $county, lat: $lat, lon: $lon, action_ft: $action, minor_ft: $minor,
                   moderate_ft: $moderate, major_ft: $major, record_ft: $record, record_date: $rdate})""",
                {
                    "lid": g.lid,
                    "usgs": g.usgs_id,
                    "name": g.name,
                    "river": g.river,
                    "state": g.state,
                    "county": g.county,
                    "lat": g.lat,
                    "lon": g.lon,
                    "action": t.get("action"),
                    "minor": t.get("minor"),
                    "moderate": t.get("moderate"),
                    "major": t.get("major"),
                    "record": g.record_stage_ft,
                    "rdate": g.record_date,
                },
            )
            self.conn.execute(
                "MATCH (g:Gauge {lid: $lid}), (r:River {name: $river}) CREATE (g)-[:ON_RIVER]->(r)",
                {"lid": g.lid, "river": g.river},
            )
            for imp in g.impacts:
                self.conn.execute(
                    "CREATE (:Impact {doc_id: $id, stage_ft: $stage, statement: $text})",
                    {"id": imp.doc_id, "stage": imp.stage_ft, "text": imp.statement},
                )
                self.conn.execute(
                    "MATCH (g:Gauge {lid: $lid}), (i:Impact {doc_id: $id}) CREATE (g)-[:HAS_IMPACT]->(i)",
                    {"lid": g.lid, "id": imp.doc_id},
                )
        for e in edges:
            lag = model.edge_lag(e) if model else None
            n_obs = model.n_obs.get((e.upstream, e.downstream), 0) if model else 0
            self.conn.execute(
                """MATCH (a:Gauge {lid: $up}), (b:Gauge {lid: $down})
                   CREATE (a)-[:FLOWS_TO {river_km: $km, lag_h: $lag, n_obs: $n}]->(b)""",
                {"up": e.upstream, "down": e.downstream, "km": e.river_km, "lag": lag, "n": n_obs},
            )

    # queries ---------------------------------------------------------------------------

    def gauge(self, lid: str) -> dict | None:
        rows = self._rows("MATCH (g:Gauge {lid: $lid}) RETURN g", {"lid": lid})
        return dict(rows[0][0]) if rows else None

    def all_gauges(self) -> list[dict]:
        return [dict(r[0]) for r in self._rows("MATCH (g:Gauge) RETURN g ORDER BY g.lid")]

    def downstream(self, lid: str, max_hops: int = 12) -> list[Hop]:
        """Gauges downstream in order, with cumulative crest travel time and distance."""
        rows = self._rows(
            f"""MATCH p = (a:Gauge {{lid: $lid}})-[:FLOWS_TO*1..{max_hops}]->(b:Gauge)
                RETURN b.lid, b.name, length(p), rels(p) ORDER BY length(p)""",
            {"lid": lid},
        )
        hops = []
        for b_lid, b_name, length, rels in rows:
            hops.append(
                Hop(
                    lid=b_lid,
                    name=b_name,
                    hops=int(length),
                    eta_h=round(sum(r["lag_h"] for r in rels), 1),
                    river_km=round(sum(r["river_km"] for r in rels), 1),
                )
            )
        return hops

    def upstream(self, lid: str, max_hops: int = 2) -> list[Hop]:
        rows = self._rows(
            f"""MATCH p = (a:Gauge)-[:FLOWS_TO*1..{max_hops}]->(b:Gauge {{lid: $lid}})
                RETURN a.lid, a.name, length(p), rels(p) ORDER BY length(p), a.lid""",
            {"lid": lid},
        )
        return [
            Hop(
                lid=a_lid,
                name=a_name,
                hops=int(length),
                eta_h=round(sum(r["lag_h"] for r in rels), 1),
                river_km=round(sum(r["river_km"] for r in rels), 1),
            )
            for a_lid, a_name, length, rels in rows
        ]

    def impacts_at(self, lid: str, stage_ft: float, limit: int = 2) -> list[dict]:
        """Impact statements already reached at a stage, highest first."""
        rows = self._rows(
            """MATCH (g:Gauge {lid: $lid})-[:HAS_IMPACT]->(i:Impact) WHERE i.stage_ft <= $stage
               RETURN i.doc_id, i.stage_ft, i.statement ORDER BY i.stage_ft DESC LIMIT $limit""",
            {"lid": lid, "stage": float(stage_ft), "limit": limit},
        )
        return [{"doc_id": r[0], "stage_ft": r[1], "statement": r[2]} for r in rows]

    def counts(self) -> dict[str, int]:
        q = {
            "gauges": "MATCH (g:Gauge) RETURN count(*)",
            "rivers": "MATCH (r:River) RETURN count(*)",
            "impacts": "MATCH (i:Impact) RETURN count(*)",
            "flows_to": "MATCH ()-[f:FLOWS_TO]->() RETURN count(*)",
        }
        return {k: int(self._rows(v)[0][0]) for k, v in q.items()}
