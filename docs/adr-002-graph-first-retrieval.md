# ADR 002: Graph first retrieval, with Qdrant hybrid search inside the graph scope

## Context

The corpus is small (76 flood stage documents, 458 NWS impact statements, 75 river segment
documents) but the questions are structured. A pure vector search finds documents that look like
the question, which works for "what happens at Little Falls at 11.7 ft" only when the right stage
happens to rank first, and fails for anything that needs a chain of documents.

## Decision

Three retrievers behind the same agent so they can be compared on the same questions:

1. **dense:** BAAI/bge-small-en-v1.5 through fastembed (ONNX, CPU), cosine search in Qdrant.
2. **hybrid:** the same dense vector plus a BM25 style sparse vector in the same Qdrant point,
   fused with Reciprocal Rank Fusion through the Query API. IDF is applied by Qdrant
   (`Modifier.IDF`), so the sparse side needs no model download.
3. **graph:** link the gauges named in the question (station id, full name, or an unambiguous
   place name such as "Point of Rocks"), walk the graph for the exact facts the intent needs
   (path, neighbours, impact at or below a stage), then fill the remaining slots with hybrid
   search filtered to the linked gauges.

Qdrant runs in local mode (in process) so there is nothing to deploy; the same code works against
a Qdrant server by changing the location string.

## Consequences

* On 112 generated and 16 hand written questions the graph retriever gets every gold document;
  dense and hybrid miss most multi hop questions (numbers in the README).
* The generated questions are built from the same graph, which favours the graph retriever. The
  hand written set, typed with short place names and station ids, is there to check that the
  gap is not only an artefact of that.
* The entity linker is the single point of failure for the graph path. When it finds nothing,
  the retriever falls back to plain hybrid search instead of returning an empty context.
