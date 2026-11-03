"""Document corpus and retrievers: dense, hybrid (dense + BM25 sparse with RRF in Qdrant) and graph."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient, models

from .graph import GaugeGraph
from .lang import EN, Lang
from .network import Edge, Gauge

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = {
    "the", "a", "an", "of", "at", "to", "in", "on", "and", "or", "is", "are", "be", "will", "what",
    "when", "which", "how", "does", "do", "for", "from", "with", "near", "it", "its", "this", "that",
    "el", "la", "los", "las", "de", "del", "en", "y", "que", "cuanto", "cuantas", "cual", "cuales",
    "por", "para", "con", "se", "es", "un", "una", "hasta", "desde",
}  # fmt: skip


@dataclass(frozen=True)
class Doc:
    doc_id: str
    kind: str  # impact | stages | flow
    lid: str
    text: str
    payload: dict = field(default_factory=dict)


def build_corpus(
    gauges: list[Gauge], edges: list[Edge], lags: dict[tuple[str, str], float], lang: Lang = EN
) -> list[Doc]:
    by_lid = {g.lid: g for g in gauges}
    docs: list[Doc] = []
    for g in gauges:
        t = g.thresholds
        levels = ", ".join(
            f"{lang.labels[c]} {t[c]:g} {lang.unit}" for c in ("action", "minor", "moderate", "major") if c in t
        )
        record = (
            lang.t("doc_record", stage=f"{g.record_stage_ft:g}", unit=lang.unit, date=g.record_date)
            if g.record_stage_ft
            else ""
        )
        docs.append(
            Doc(
                doc_id=f"stages:{g.lid}",
                kind="stages",
                lid=g.lid,
                text=lang.t("doc_stages", name=g.name, lid=g.lid, state=g.state, levels=levels, record=record),
                payload={"thresholds": t, "record_ft": g.record_stage_ft},
            )
        )
        for imp in g.impacts:
            docs.append(
                Doc(
                    doc_id=f"impact:{imp.doc_id}",
                    kind="impact",
                    lid=g.lid,
                    text=lang.t(
                        "doc_impact", name=g.name, lid=g.lid, state=g.state, stage=f"{imp.stage_ft:g}",
                        unit=lang.unit, statement=imp.statement,
                    ),
                    payload={"stage_ft": imp.stage_ft, "statement": imp.statement},
                )
            )  # fmt: skip
    for e in edges:
        up, down = by_lid[e.upstream], by_lid[e.downstream]
        lag = lags.get((e.upstream, e.downstream))
        lag_text = lang.t("doc_lag", lag=f"{lag:g}") if lag is not None else ""
        docs.append(
            Doc(
                doc_id=f"flow:{e.upstream}->{e.downstream}",
                kind="flow",
                lid=e.upstream,
                text=lang.t(
                    "doc_flow", up=up.name, up_lid=e.upstream, down=down.name, down_lid=e.downstream,
                    km=f"{e.river_km:g}", lag=lag_text,
                ),
                payload={"upstream": e.upstream, "downstream": e.downstream, "river_km": e.river_km, "lag_h": lag},
            )
        )  # fmt: skip
    return docs


# Embedders ------------------------------------------------------------------------------


class DenseEmbedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag of words embedding. No download, used by tests and CI."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[row, h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 through fastembed (ONNX Runtime, CPU, about 65 MB)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model)
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)


def make_embedder(name: str) -> DenseEmbedder:
    return FastEmbedder() if name == "fastembed" else HashingEmbedder()


def fold(text: str) -> str:
    """Lowercase without accents, so 'Popayán' and 'popayan' match."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(fold(text)) if t not in STOP]


def sparse_vector(text: str) -> models.SparseVector:
    """Term frequencies with BM25 saturation. Qdrant applies IDF on its side (Modifier.IDF)."""
    counts = Counter(tokens(text))
    k1 = 1.2
    idx, val = [], []
    for tok, tf in sorted(counts.items()):
        idx.append(int(hashlib.md5(tok.encode()).hexdigest()[:8], 16))
        val.append(tf * (k1 + 1) / (tf + k1))
    return models.SparseVector(indices=idx, values=val)


# Index ---------------------------------------------------------------------------------


@dataclass
class Hit:
    doc: Doc
    score: float


class HybridIndex:
    """Qdrant collection with a dense and a sparse vector per document."""

    COLLECTION = "gauge_docs"

    def __init__(self, docs: list[Doc], embedder: DenseEmbedder, location: str = ":memory:") -> None:
        self.docs = {d.doc_id: d for d in docs}
        self.embedder = embedder
        self.client = QdrantClient(location=location)
        self.client.create_collection(
            self.COLLECTION,
            vectors_config={"dense": models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        vectors = embedder.embed([d.text for d in docs])
        self.client.upsert(
            self.COLLECTION,
            points=[
                models.PointStruct(
                    id=i,
                    vector={"dense": vectors[i].tolist(), "bm25": sparse_vector(d.text)},
                    payload={"doc_id": d.doc_id, "lid": d.lid, "kind": d.kind},
                )
                for i, d in enumerate(docs)
            ],
        )

    def _filter(self, lids: list[str] | None, kinds: list[str] | None) -> models.Filter | None:
        must = []
        if lids:
            must.append(models.FieldCondition(key="lid", match=models.MatchAny(any=lids)))
        if kinds:
            must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=kinds)))
        return models.Filter(must=must) if must else None

    def dense(self, query: str, k: int, lids: list[str] | None = None, kinds: list[str] | None = None) -> list[Hit]:
        vec = self.embedder.embed([query])[0].tolist()
        res = self.client.query_points(
            self.COLLECTION, query=vec, using="dense", limit=k, query_filter=self._filter(lids, kinds)
        )
        return [Hit(self.docs[p.payload["doc_id"]], p.score) for p in res.points]

    def hybrid(self, query: str, k: int, lids: list[str] | None = None, kinds: list[str] | None = None) -> list[Hit]:
        vec = self.embedder.embed([query])[0].tolist()
        flt = self._filter(lids, kinds)
        res = self.client.query_points(
            self.COLLECTION,
            prefetch=[
                models.Prefetch(query=vec, using="dense", limit=k * 4, filter=flt),
                models.Prefetch(query=sparse_vector(query), using="bm25", limit=k * 4, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
        )
        return [Hit(self.docs[p.payload["doc_id"]], p.score) for p in res.points]


# Entity linking ------------------------------------------------------------------------


class GaugeLinker:
    """Finds the gauges a question talks about: station id, full name, or the place part of the
    name ("Point of Rocks", "Little Falls") when it is unambiguous."""

    SEPARATORS = (" at ", " near ", " above ", " below ")

    def __init__(self, gauges: list[Gauge]) -> None:
        self.gauges = gauges
        self._names = {g.lid: set(tokens(g.name)) for g in gauges}
        self._rivers = {g.lid: set(tokens(g.river)) for g in gauges}
        self._aliases = {g.lid: self._place_aliases(g.name) for g in gauges}
        self._df = Counter(t for toks in self._names.values() for t in toks)

    @classmethod
    def _place_aliases(cls, name: str) -> list[set[str]]:
        aliases = []
        for paren in re.findall(r"\(([^)]+)\)", name):
            aliases.append(set(tokens(paren)))
        plain = re.sub(r"\([^)]*\)", "", name)
        lower = plain.lower()
        for sep in cls.SEPARATORS:
            idx = lower.find(sep)
            if idx > 0:
                aliases.append(set(tokens(plain[idx + len(sep) :])))
                break
        else:
            if plain.strip() != name.strip():  # "Puente Carretera (Popayán)": the station name alone
                aliases.append(set(tokens(plain)))
        return [a for a in aliases if a]

    def _weight(self, tok: str) -> float:
        return math.log(1 + len(self.gauges) / self._df[tok])

    def link(self, text: str, max_gauges: int = 3) -> list[str]:
        found = [g.lid for g in self.gauges if re.search(rf"\b{g.lid}\b", text)]
        q = set(tokens(text))
        scored = []
        for lid, name_toks in self._names.items():
            if lid in found:
                continue
            total = sum(self._weight(t) for t in name_toks)
            hit = sum(self._weight(t) for t in name_toks & q)
            score = hit / total if total else 0.0
            alias_key: frozenset[str] | None = None
            if score < 0.75:
                matched = [a for a in self._aliases[lid] if a <= q]
                if matched:
                    alias_key = frozenset(max(matched, key=len))
                    score = 0.7 + 0.1 * len(self._rivers[lid] & q) / max(1, len(self._rivers[lid]))
            if score >= 0.7:
                scored.append((score, hit, lid, alias_key))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        # Two gauges named after the same place (Cumberland, Leesburg): keep the better river match.
        best_alias: set[frozenset[str]] = set()
        full_names = [self._names[lid] for score, _, lid, key in scored if key is None] + [
            self._names[lid] for lid in found
        ]
        mentions = list(found)
        kept_matches: list[set[str]] = []
        for _, _, lid, alias_key in scored:
            if alias_key is None:
                # "North Fork Catoctin Creek near Waterford" when the text names the South Fork:
                # its matched words are all explained by a better match already kept.
                matched = self._names[lid] & q
                if any(matched <= prev for prev in kept_matches):
                    continue
                kept_matches.append(matched)
            if alias_key is not None:
                # The place is already part of a gauge named in full ("Opequon Creek near Martinsburg"
                # also contains Martinsburg), so it is not a second mention.
                if alias_key in best_alias or any(alias_key <= name for name in full_names):
                    continue
                best_alias.add(alias_key)
            mentions.append(lid)
        return _in_text_order(mentions[:max_gauges], text, {g.lid: g.name for g in self.gauges}, self._aliases)


def _in_text_order(lids: list[str], text: str, names: dict[str, str], aliases: dict[str, list[set[str]]]) -> list[str]:
    """Order mentions as they appear in the text, so 'from A to B' keeps A first."""
    lower = fold(text)
    words = tokens(text)

    def pos(lid: str) -> int:
        for needle in (lid, fold(names[lid])):
            idx = (text if needle == lid else lower).find(needle)
            if idx >= 0:
                return idx
        for alias in aliases.get(lid, []):
            for i, w in enumerate(words):
                if w in alias:
                    return lower.find(w) if lower.find(w) >= 0 else len(text) + i
        return len(text)

    return sorted(lids, key=pos)


# Graph retrieval -----------------------------------------------------------------------


class GraphRetriever:
    """Resolve gauges, walk the graph for the facts the question needs, then search text in scope."""

    def __init__(self, graph: GaugeGraph, index: HybridIndex, linker: GaugeLinker) -> None:
        self.graph = graph
        self.index = index
        self.linker = linker

    def retrieve(self, query: str, intent: str, k: int, stage_ft: float | None = None) -> list[Hit]:
        lids = self.linker.link(query)
        if not lids:
            return self.index.hybrid(query, k)
        docs: list[Doc] = []
        focus = lids[0]
        if intent == "impact":
            docs.append(self.index.docs[f"stages:{focus}"])
            if stage_ft is not None:
                docs += [self.index.docs[f"impact:{r['doc_id']}"] for r in self.graph.impacts_at(focus, stage_ft, 1)]
        elif intent == "travel" and len(lids) >= 2:
            src, dst = lids[0], lids[1]
            hops = self.graph.downstream(src)
            if dst not in {h.lid for h in hops}:
                src, dst = dst, src
                hops = self.graph.downstream(src)
            prev = src
            for h in hops:
                docs.append(self.index.docs[f"flow:{prev}->{h.lid}"])
                prev = h.lid
                if h.lid == dst:
                    break
        elif intent == "downstream":
            prev = focus
            for h in self.graph.downstream(focus, max_hops=2):
                docs.append(self.index.docs[f"flow:{prev}->{h.lid}"])
                docs.append(self.index.docs[f"stages:{h.lid}"])
                prev = h.lid
        elif intent == "upstream":
            for h in self.graph.upstream(focus, max_hops=1):
                docs.append(self.index.docs[f"flow:{h.lid}->{focus}"])
        elif intent == "stages":
            docs.append(self.index.docs[f"stages:{focus}"])
        seen = {d.doc_id for d in docs}
        hits = [Hit(d, 1.0) for d in docs]
        for hit in self.index.hybrid(query, k, lids=lids):
            if hit.doc.doc_id not in seen and len(hits) < max(k, len(docs)):
                hits.append(hit)
                seen.add(hit.doc.doc_id)
        return hits
