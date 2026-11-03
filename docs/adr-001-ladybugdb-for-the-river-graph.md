# ADR 001: LadybugDB for the river graph

## Context

The questions a river forecast desk asks are mostly about topology: what is downstream of this
gauge, which gauges feed that one, how long does a crest take to get from A to B. The answers
need multi hop traversal over a tree of 76 gauges and 75 river edges, plus the impact statements
hanging off each gauge.

Options considered:

* **networkx in memory.** Fine for traversal, but no query language, nothing to hand to another
  tool, and the graph would be rebuilt by custom code every time.
* **Neo4j in Docker.** The reference choice for Cypher, but a server, a JVM and a container for a
  graph that fits in a few hundred kilobytes. It also makes CI slower and the laptop story worse.
* **Kuzu.** Embedded and Cypher, but the upstream repository was archived in October 2025.
* **LadybugDB (real_ladybug).** The community continuation of Kuzu: same embedded columnar engine,
  same Cypher dialect, wheels for Linux, macOS and Windows on Python 3.10 to 3.14, actively
  released (0.15.3 is used here).

## Decision

Use LadybugDB in process. Schema: `Gauge`, `River`, `Impact` nodes and `FLOWS_TO`, `ON_RIVER`,
`HAS_IMPACT` relationships. `FLOWS_TO` carries the river distance from NLDI and the learned crest
lag, so a variable length path query returns cumulative travel time in one statement.

## Consequences

* Graph build takes about one second from the curated JSON, so it is rebuilt on every run and
  never goes stale.
* Queries are plain Cypher, so moving to Neo4j or Memgraph later is a driver change, not a rewrite.
* Risk: a young fork. The graph code is one module (`graph.py`) behind a small class, and the
  tests cover every query, so swapping the engine is contained.
