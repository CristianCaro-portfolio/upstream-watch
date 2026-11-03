# ADR 004: A Colombia region on the same engine, with statistical levels and data quality gates

## Context

The Potomac version leans on three things the US publishes as open data: a river network
service (NLDI), official flood stages per gauge and NWS impact statements. Colombia publishes the
readings (IDEAM "Nivel Instantaneo del Rio" on datos.gov.co, near real time, CC BY-SA 4.0) but
not the other two in machine readable form.

## Decision

One codebase, a `--region` switch. The graph, retrieval, agent, review step, MCP server and
evaluation stay the same. The Colombia region swaps the inputs:

* **Network:** HydroRIVERS v1.0. Each station is snapped to the largest reach within 2 km and
  the next station downstream is found by following `NEXT_DOWN`. River distance is the
  difference in `DIST_DN_KM`.
* **Levels:** "vigilancia" and "alto" are the 90th and 98th percentile of each station's daily
  maximum since November 2022, with at least 180 days of data. They are labelled as statistical
  everywhere, never as IDEAM alert levels.
* **Quality gates before any number:** isolated spikes are removed; a station whose monthly
  median jumps more than 8 m is kept in the network but gets no thresholds and no lags; a
  station listed with two locations keeps the one with most readings.
* **Travel time:** an edge uses its own median crest lag only with at least 5 observed crest
  pairs; otherwise the basin wave speed.
* **Language:** answers, briefs and figures in Spanish through a small wording table, so the
  grounding check and the tests are the same in both languages.

## Consequences

* The graph retriever keeps its advantage (79 of 79 generated, 12 of 12 hand written questions
  against 57 and 67 percent for plain vector search).
* With four years of data only 3 of 26 edges qualify for a learned lag, and arrival time error to
  La Raya is about 23 hours either way. Learning every edge made it 40 hours. Reservoir operation
  on the Cauca likely adds to that. Longer history or IDEAM rating curves would be the fix.
* The automatic upstream trigger is weak here (precision 0.18), so the useful output is the
  brief with human review, not an automatic alarm.
* If IDEAM publishes official thresholds per station, they replace the percentiles with no other
  change.
