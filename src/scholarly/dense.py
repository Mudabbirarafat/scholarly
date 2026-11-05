"""Dense retrieval: encode the corpus once, then rank by dot product.

Two halves with very different costs, which is the whole reason the pipeline
is split the way it is. `encode_corpus` is the expensive one and belongs in
CI. `DenseIndex` is the cheap one: vectors are already unit length, so
similarity for a whole corpus is a single matrix multiply.

**Loading the index costs memory, measured rather than assumed.** 114,601
vectors at 384 float32 dimensions is 168 MiB resident, plus the encoder at
roughly 400 MB while a query is embedded. On a machine with a gigabyte free
that is the difference between a search and a swap storm, so `DenseIndex`
takes a limit and reports what it loaded instead of quietly eating the
machine.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from scholarly import embeddings
from scholarly.models import Embedding, Paper

log = logging.getLogger("scholarly.dense")


def pending(
    session: Session, model_name: str, *, limit: int | None = None
) -> list[tuple[str, str]]:
    """(id, abstract) for papers this model has not encoded yet.

    Resumable by construction: an encode that dies halfway is restarted by
    running it again, which on a job measured in CPU-hours is the only
    recovery anyone will perform.
    """
    already = select(Embedding.paper_id).where(Embedding.model == model_name)
    statement = (
        select(Paper.id, Paper.abstract)
        .where(Paper.abstract.is_not(None), Paper.id.not_in(already))
        .order_by(Paper.id)
    )
    if limit:
        statement = statement.limit(limit)
    return [(row[0], row[1]) for row in session.execute(statement).all()]


def encode_corpus(
    session: Session,
    *,
    model_name: str | None = None,
    batch_size: int = 32,
    limit: int | None = None,
    commit_every: int = 500,
) -> dict:
    """Encode everything not yet encoded. Safe to stop and restart."""
    name = model_name or embeddings.DEFAULT_MODEL
    todo = pending(session, name, limit=limit)
    if not todo:
        return {"model": name, "encoded": 0, "status": "nothing to do"}

    started = time.perf_counter()
    done = 0
    for index, vector in enumerate(
        embeddings.encode((abstract for _, abstract in todo), batch_size=batch_size)
    ):
        paper_id = todo[index][0]
        session.add(
            Embedding(
                paper_id=paper_id,
                model=name,
                dimensions=int(vector.shape[0]),
                vector=embeddings.to_blob(vector),
            )
        )
        done += 1
        # Flushing periodically keeps the session from holding every vector
        # of a hundred-thousand-document run in memory at once.
        if done % commit_every == 0:
            session.flush()
            log.info("encoded", extra={"done": done, "of": len(todo)})

    elapsed = time.perf_counter() - started
    return {
        "model": name,
        "encoded": done,
        "seconds": round(elapsed, 1),
        "per_second": round(done / elapsed, 1) if elapsed else None,
        "status": "ok",
    }


@dataclass
class DenseIndex:
    ids: list[str]
    matrix: np.ndarray
    model: str

    @classmethod
    def load(
        cls, session: Session, *, model_name: str | None = None, limit: int | None = None
    ) -> "DenseIndex":
        name = model_name or embeddings.DEFAULT_MODEL
        statement = (
            select(Embedding.paper_id, Embedding.vector)
            .where(Embedding.model == name)
            .order_by(Embedding.paper_id)
        )
        if limit:
            statement = statement.limit(limit)
        rows = session.execute(statement).all()
        ids = [row[0] for row in rows]
        matrix = embeddings.stack(row[1] for row in rows)
        log.info(
            "dense index loaded",
            extra={"vectors": len(ids), "mib": round(matrix.nbytes / 1024 / 1024, 1)},
        )
        return cls(ids=ids, matrix=matrix, model=name)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, *, k: int = 10, exclude: str | None = None) -> list[str]:
        """The k nearest papers, best first.

        Vectors are unit length on the way in, so cosine similarity is a dot
        product and the whole corpus is one matmul rather than a loop.
        """
        if not self.ids:
            return []

        vector = next(embeddings.encode([query]))
        scores = self.matrix @ vector

        # argpartition finds the top k without sorting all 114,601 scores,
        # which is most of the per-query cost at this corpus size.
        want = min(k + (1 if exclude else 0), len(self.ids))
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]

        found = [self.ids[position] for position in top if self.ids[position] != exclude]
        return found[:k]


def encoded_count(session: Session, model_name: str | None = None) -> int:
    name = model_name or embeddings.DEFAULT_MODEL
    return (
        session.scalar(
            select(func.count()).select_from(Embedding).where(Embedding.model == name)
        )
        or 0
    )
