"""Encoding the corpus: the expensive half, and the only half that is.

Turning 114,581 abstracts into vectors is CPU-hours and belongs in CI.
Searching them is one matrix multiply and belongs in a request handler, so
the index itself lives in `serving.py` and is used unchanged by both the
deployed instance and the evaluation harness - because an evaluated
retriever that is a separate implementation from the served one publishes a
number about something nobody runs.

**The encoder cost, measured rather than assumed:** 203 MB resident, 12 ms
per query warm, from `scripts/serving_budget.py`. An earlier version of this
docstring guessed 400 MB, which was wrong by a factor of three and would
have ruled out the architecture it was describing.
"""

import hashlib
import logging
import time

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
    unencoded = (Paper.abstract.is_not(None), Paper.id.not_in(already))

    if limit is None:
        rows = session.execute(
            select(Paper.id, Paper.abstract).where(*unencoded).order_by(Paper.id)
        ).all()
        return [(row[0], row[1]) for row in rows]

    # Ordering by id would take a biased slice: OpenAlex ids broadly track
    # when a work was first catalogued, so the first 20,000 are the oldest
    # papers rather than a sample of the corpus. A bounded encode exists to
    # say something indicative about the whole, which it cannot do from one
    # end of it - so the order is a stable hash instead.
    #
    # Only ids are read for that sort. Pulling 114,581 abstracts into memory
    # to discard 94,581 of them is a different mistake from the one being
    # fixed, and on a 512 MB box it is the worse one.
    ids = [row[0] for row in session.execute(select(Paper.id).where(*unencoded)).all()]
    ids.sort(key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    chosen = ids[:limit]

    # Chunked, because SQLite caps how many bound parameters one statement
    # may carry and a 20,000-element IN list is far past it.
    abstracts: dict[str, str] = {}
    for start in range(0, len(chosen), 500):
        batch = chosen[start : start + 500]
        for row in session.execute(
            select(Paper.id, Paper.abstract).where(Paper.id.in_(batch))
        ).all():
            abstracts[row[0]] = row[1]

    return [(paper_id, abstracts[paper_id]) for paper_id in chosen if paper_id in abstracts]


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


def encoded_count(session: Session, model_name: str | None = None) -> int:
    name = model_name or embeddings.DEFAULT_MODEL
    return (
        session.scalar(
            select(func.count()).select_from(Embedding).where(Embedding.model == name)
        )
        or 0
    )
