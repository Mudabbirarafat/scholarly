"""The cheap half: embed one short query, search precomputed vectors.

Everything here is sized by one measurement, taken with
`scripts/serving_budget.py` on the development machine and reproducible
anywhere:

    python + numpy                        30.7 MB
    encoder loaded                       203.0 MB
    float16 index beside it (114,581)    288.6 MB   peak 324.6
    search                               291.0 MB   150 ms

So a 512 MB instance holds the encoder and the whole corpus with about
220 MB spare. Two decisions follow from that, and neither is a guess:

**float16 vectors, not float32.** Halving the index saves 84 MiB, which is
the difference between comfortable and tight. The cost is a rounding error in
the scores; `eval --dtype float16` measures what it does to recall rather than
assuming it is free.

**Scored in chunks.** `matrix @ query` on a float16 matrix upcasts the whole
matrix first - a 168 MiB temporary, two thirds of the headroom, for a single
request. A chunk at a time costs `8192 * 384 * 4` bytes, about 12 MiB, and
measured 155 ms against the naive 132 ms. Twenty-three milliseconds is a fair
price for not having a cliff at the far end of concurrency.
"""

import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from scholarly import embeddings, lexical
from scholarly.models import Embedding, Paper

log = logging.getLogger("scholarly.serving")

# 8192 x 384 float32 is about 12 MiB per block. Larger blocks buy very little
# and the point of the chunking is the ceiling, not the speed.
CHUNK = 8192

# Long enough to judge relevance, short enough that twenty of them are not a
# page of text. The full abstract is one click away on OpenAlex.
SNIPPET_CHARS = 320

_WORD = re.compile(r"[a-z0-9]{3,}")


@dataclass(frozen=True)
class Hit:
    """One result, with enough beside it to judge whether it belongs."""

    paper_id: str
    rank: int
    score: float
    title: str
    year: int | None
    snippet: str
    url: str
    # Query terms that actually occur in the abstract. Populated for lexical
    # results, and deliberately empty for dense ones: the model never saw a
    # term, so highlighting words in a dense result would be a decoration
    # pretending to be an explanation.
    matched: list[str] = field(default_factory=list)


@dataclass
class VectorIndex:
    """float16 vectors in memory, searched by chunked dot product."""

    ids: list[str]
    matrix: np.ndarray
    model: str

    @classmethod
    def load(
        cls,
        session: Session,
        *,
        model_name: str | None = None,
        limit: int | None = None,
    ) -> "VectorIndex":
        name = model_name or embeddings.DEFAULT_MODEL
        statement = (
            select(Embedding.paper_id, Embedding.vector)
            .where(Embedding.model == name)
            .order_by(Embedding.paper_id)
        )
        if limit:
            statement = statement.limit(limit)

        ids: list[str] = []
        blocks: list[np.ndarray] = []
        pending: list[np.ndarray] = []
        started = time.perf_counter()

        # Built in blocks rather than one stack of 114,581 arrays: the
        # intermediate list is the peak, not the result.
        for paper_id, blob in session.execute(statement).yield_per(2048):
            ids.append(paper_id)
            pending.append(embeddings.from_blob(blob).astype(np.float16))
            if len(pending) >= CHUNK:
                blocks.append(np.stack(pending))
                pending.clear()
        if pending:
            blocks.append(np.stack(pending))

        if blocks:
            matrix = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
        else:
            matrix = np.zeros((0, embeddings.DIMENSIONS), dtype=np.float16)
        del blocks

        log.info(
            "index loaded",
            extra={
                "vectors": len(ids),
                "mib": round(matrix.nbytes / 1048576, 1),
                "seconds": round(time.perf_counter() - started, 1),
            },
        )
        return cls(ids=ids, matrix=matrix, model=name)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def megabytes(self) -> float:
        return round(self.matrix.nbytes / 1048576, 1)

    def scores(self, vector: np.ndarray) -> np.ndarray:
        """Cosine similarity against every document, a chunk at a time."""
        query = np.asarray(vector, dtype=np.float32)
        out = np.empty(self.matrix.shape[0], dtype=np.float32)
        for start in range(0, self.matrix.shape[0], CHUNK):
            block = self.matrix[start : start + CHUNK]
            out[start : start + CHUNK] = block.astype(np.float32) @ query
        return out

    def search(
        self, vector: np.ndarray, *, k: int = 10, exclude: str | None = None
    ) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        scores = self.scores(vector)
        want = min(k + (1 if exclude else 0), len(self.ids))
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]
        found = [
            (self.ids[position], float(scores[position]))
            for position in top
            if self.ids[position] != exclude
        ]
        return found[:k]


def metadata(session: Session, ids: Sequence[str]) -> dict[str, Paper]:
    """Titles and abstracts for the handful of ids being returned.

    Chunked because SQLite caps bound parameters per statement, and a caller
    asking for a hundred results is not a reason to fail.
    """
    found: dict[str, Paper] = {}
    for start in range(0, len(ids), 400):
        batch = list(ids[start : start + 400])
        for paper in session.scalars(select(Paper).where(Paper.id.in_(batch))):
            found[paper.id] = paper
    return found


def paper_url(paper: Paper) -> str:
    if paper.doi:
        doi = paper.doi
        return doi if doi.startswith("http") else f"https://doi.org/{doi}"
    return f"https://openalex.org/{paper.id}"


def snippet(abstract: str | None) -> str:
    if not abstract:
        return ""
    text = " ".join(abstract.split())
    if len(text) <= SNIPPET_CHARS:
        return text
    cut = text[:SNIPPET_CHARS]
    # Break on a space so the snippet does not end mid-word.
    space = cut.rfind(" ")
    return (cut[:space] if space > SNIPPET_CHARS * 0.6 else cut).rstrip(",;:") + "..."


def matched_terms(query: str, abstract: str | None) -> list[str]:
    """Query terms that occur in the abstract, in query order, deduplicated."""
    if not abstract:
        return []
    haystack = abstract.lower()
    seen: list[str] = []
    for term in _WORD.findall(query.lower()):
        if term in lexical.NOISE or term in seen:
            continue
        if term in haystack:
            seen.append(term)
    return seen


class SearchService:
    """One query in, ranked hits out, from whichever retrievers are asked for.

    Holds the index and the encoder for the life of the process, because
    loading either per request is most of the cost of answering one.
    """

    def __init__(
        self,
        session_factory,
        *,
        index: VectorIndex | None = None,
        quality: dict | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.index = index
        self.quality = quality or {}

    @property
    def retrievers(self) -> list[str]:
        available = ["lexical"]
        if self.index is not None and len(self.index):
            available.append("dense")
        return available

    def search(self, query: str, *, k: int = 10, retriever: str = "both") -> dict:
        query = (query or "").strip()
        if not query:
            return {"query": "", "k": k, "results": {}, "elapsed_ms": {}}

        wanted = ["lexical", "dense"] if retriever == "both" else [retriever]
        results: dict[str, list[dict]] = {}
        elapsed: dict[str, float] = {}

        with self._session_factory() as session:
            for name in wanted:
                if name not in self.retrievers:
                    continue
                started = time.perf_counter()
                if name == "dense":
                    vector = next(embeddings.encode([query]))
                    scored = self.index.search(vector, k=k)
                else:
                    ids = lexical.search(session, query, k=k)
                    # FTS5 returns rank order without a comparable score, and
                    # inventing one would imply the two columns can be
                    # compared number to number. They cannot.
                    scored = [(paper_id, float("nan")) for paper_id in ids]

                papers = metadata(session, [paper_id for paper_id, _ in scored])
                hits = []
                for rank, (paper_id, score) in enumerate(scored, start=1):
                    paper = papers.get(paper_id)
                    if paper is None:
                        continue
                    hits.append(
                        asdict(
                            Hit(
                                paper_id=paper_id,
                                rank=rank,
                                score=None if score != score else round(score, 4),
                                title=paper.title,
                                year=paper.year,
                                snippet=snippet(paper.abstract),
                                url=paper_url(paper),
                                matched=(
                                    matched_terms(query, paper.abstract)
                                    if name == "lexical"
                                    else []
                                ),
                            )
                        )
                    )
                results[name] = hits
                elapsed[name] = round((time.perf_counter() - started) * 1000, 1)

        return {
            "query": query,
            "k": k,
            "retriever": retriever,
            "results": results,
            "elapsed_ms": elapsed,
        }

    def status(self) -> dict:
        return {
            "status": "ok",
            "retrievers": self.retrievers,
            "index": {
                "documents": len(self.index) if self.index else 0,
                "model": self.index.model if self.index else None,
                "megabytes": self.index.megabytes if self.index else 0.0,
                "dtype": "float16",
            },
            "quality": self.quality,
        }


def ensure_bundle(url: str | None, path: str | os.PathLike) -> dict:
    """Fetch the serving database if it is not on disk yet.

    A free instance has no persistent disk: it comes up with the repository
    and nothing else. The committed bundle is small enough to deploy as-is,
    and `BUNDLE_URL` points at a release asset when the whole corpus is
    wanted instead.

    Downloaded to a temporary name and renamed, because a half-written
    database that gets served is indistinguishable from a small one until a
    query hits the missing half.
    """
    target = Path(path)
    if target.exists() and target.stat().st_size > 0:
        return {"bundle": "present", "megabytes": round(target.stat().st_size / 1048576, 1)}
    if not url:
        return {"bundle": "missing", "note": "no BUNDLE_URL and nothing on disk"}

    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.perf_counter()
    written = 0
    log.info("fetching bundle", extra={"url": url})

    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_bytes(1 << 20):
                handle.write(block)
                written += len(block)

    partial.replace(target)
    summary = {
        "bundle": "downloaded",
        "megabytes": round(written / 1048576, 1),
        "seconds": round(time.perf_counter() - started, 1),
    }
    log.info("bundle ready", extra=summary)
    return summary


def load_quality(path: str | os.PathLike | None = None) -> dict:
    """The published metrics, or an empty dict.

    Deliberately not computed at startup. These numbers come from a pipeline
    run against a frozen query set on a complete index; recomputing them in a
    web process would produce a different, unfrozen, unlabelled number that
    looked just as official.
    """
    candidate = Path(path or os.environ.get("QUALITY_FILE") or "data/quality.json")
    if not candidate.exists():
        log.warning("no quality file", extra={"path": str(candidate)})
        return {}
    with candidate.open(encoding="utf-8") as handle:
        return json.load(handle)
