# ADR 003: Learned crest lags, a grounding check on every answer, and a person on every brief

## Context

Three things decide whether a flood watch brief can be trusted: the arrival times it quotes, the
facts in its text, and who signs it off.

## Decision

**Arrival times.** For each river edge the crest to crest lag is measured on every past event
where both gauges had a real crest (rise of at least 1 ft, not on the edge of the window), and
the median is used. Edges never observed fall back to the median wave speed of the basin. This is
compared against that constant wave speed baseline with a leave one event out backtest. It is a
statistical routing of crests, not a hydraulic model, and it is presented that way.

**Facts.** The answer writer is swappable (deterministic templates, a local transformers model,
Ollama, or any OpenAI compatible server). Whatever writes the answer, a verify node checks that
every number appears in the retrieved context or the question (cumulative sums along a retrieved
path are allowed) and that every citation is a retrieved document. If not, the agent returns the
deterministic draft instead. Model parse output is validated with Pydantic and falls back to the
rule parser on failure.

**Sign off.** The brief flow ends in a LangGraph `interrupt` with a `response_schema`
(`ReviewDecision`, new in langgraph 1.2.12). The paused run is checkpointed, the client gets the
JSON schema of the expected answer, and a malformed resume raises a validation error without
losing the pause.

## Consequences

* The README numbers come from the deterministic writer, so they are reproducible bit for bit.
  Model runs are reported separately with the fallback rate, which is the honest measure of how
  often a small model would have said something unsupported.
* Lags for small creeks feeding a main stem gauge measure "how long after the creek peaks does the
  river peak", not wave travel. That is still the number an operator wants, but it is noisier.
