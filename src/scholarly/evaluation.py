"""The evaluation harness: the part of this project that is the project.

A paper's abstract is the query, and the works it cites are the documents that
should come back. That gives tens of thousands of labelled queries without a
minute of hand-labelling - but it only measures anything if three things are
done carefully, and all three are easy to get wrong in a direction that
flatters the result.

**Qrels are restricted to documents the index actually holds.** Most of what a
paper cites is journal and conference work an arXiv-only corpus cannot contain
- measured here at 85.3%. Counting those as misses caps `recall@10` near 0.15
by construction, and a capped number looks exactly like a bad retriever, so
the natural response is to tune a model that was never the problem. The
discarded fraction is reported rather than hidden.

**The query paper is removed from its own results.** Searching with an
abstract always returns its own paper first. Counting that would add a free
correct answer to every query.

**The set is frozen.** It lives in a table, chosen deterministically from a
seed, because a query set re-derived on each run drifts as the corpus grows -
and two numbers measured a week apart are then incomparable while looking
identical in kind.
"""

import hashlib
import logging
import statistics
import time
from collections.abc import Callable, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, aliased

from scholarly.models import Citation, EvalQuery, Paper

log = logging.getLogger("scholarly.evaluation")

DEFAULT_SEED = "scholarly-v1"


def _order_key(paper_id: str, seed: str) -> str:
    """A stable pseudo-random ordering: same corpus or not, same choice."""
    return hashlib.sha256(f"{seed}:{paper_id}".encode()).hexdigest()


def candidates(session: Session, *, from_year: int, to_year: int) -> list[tuple[str, int]]:
    """(paper_id, relevant_count) for papers usable as evaluation queries.

    Usable means: it has an abstract to search with, and at least one of the
    works it cites is in the index, so a perfect retriever could score 1.0.
    """
    cited = aliased(Paper)
    citing = aliased(Paper)
    rows = session.execute(
        select(Citation.citing_id, func.count())
        .select_from(Citation)
        # The reference has to be in the index, or it is unretrievable and
        # counting it would only lower the ceiling.
        .join(cited, cited.id == Citation.cited_id)
        # The citing paper has to be searchable and inside the query years.
        .join(citing, citing.id == Citation.citing_id)
        .where(
            citing.abstract.is_not(None),
            citing.year.between(from_year, to_year),
        )
        .group_by(Citation.citing_id)
    ).all()
    return [(row[0], row[1]) for row in rows]


def build_query_set(
    session: Session,
    *,
    size: int = 2000,
    dev_fraction: float = 0.25,
    from_year: int = 2018,
    to_year: int = 2021,
    seed: str = DEFAULT_SEED,
    rebuild: bool = False,
) -> dict:
    """Freeze a query set. Refuses to change one that already exists.

    Refusing is the point. Re-rolling the queries after seeing a result is how
    a tuning run quietly becomes a search for a favourable sample.
    """
    existing = session.scalar(select(func.count()).select_from(EvalQuery)) or 0
    if existing and not rebuild:
        return {"status": "already frozen", "queries": existing}

    if rebuild:
        session.execute(text("DELETE FROM eval_queries"))

    pool = candidates(session, from_year=from_year, to_year=to_year)
    pool.sort(key=lambda row: _order_key(row[0], seed))
    chosen = pool[:size]

    dev_cut = int(len(chosen) * dev_fraction)
    for index, (paper_id, relevant) in enumerate(chosen):
        session.add(
            EvalQuery(
                paper_id=paper_id,
                split="dev" if index < dev_cut else "test",
                relevant_count=relevant,
            )
        )

    return {
        "status": "frozen",
        "pool": len(pool),
        "queries": len(chosen),
        "dev": dev_cut,
        "test": len(chosen) - dev_cut,
    }


def qrels(session: Session, paper_id: str) -> set[str]:
    """The relevant documents for a query: its references that are indexed."""
    rows = session.execute(
        select(Citation.cited_id)
        .join(Paper, Paper.id == Citation.cited_id)
        .where(Citation.citing_id == paper_id)
    ).all()
    return {row[0] for row in rows}


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    for position, paper_id in enumerate(retrieved[:k], start=1):
        if paper_id in relevant:
            return 1.0 / position
    return 0.0


def evaluate(
    session: Session,
    search: Callable[[str, int, str], Sequence[str]],
    *,
    split: str = "dev",
    ks: Sequence[int] = (10, 50),
    limit: int | None = None,
) -> dict:
    """Run the query set through a retriever and report what it did.

    `search(query_text, k, exclude_id)` is the only thing a retriever has to
    provide, so lexical, dense and hybrid are all measured by this same code
    rather than each bringing its own definition of recall.
    """
    top_k = max(ks)
    queries = session.execute(
        select(EvalQuery.paper_id).where(EvalQuery.split == split).order_by(EvalQuery.paper_id)
    ).all()
    if limit:
        queries = queries[:limit]

    scores: dict[int, list[float]] = {k: [] for k in ks}
    rr: list[float] = []
    latencies: list[float] = []
    relevant_counts: list[int] = []
    skipped = 0

    for (paper_id,) in queries:
        abstract = session.scalar(select(Paper.abstract).where(Paper.id == paper_id))
        relevant = qrels(session, paper_id)
        if not abstract or not relevant:
            skipped += 1
            continue

        started = time.perf_counter()
        retrieved = search(abstract, top_k, paper_id)
        latencies.append((time.perf_counter() - started) * 1000)

        for k in ks:
            scores[k].append(recall_at_k(retrieved, relevant, k))
        rr.append(reciprocal_rank(retrieved, relevant, top_k))
        relevant_counts.append(len(relevant))

    measured = len(rr)
    if not measured:
        return {"split": split, "queries": 0, "skipped": skipped}

    latencies.sort()
    return {
        "split": split,
        "queries": measured,
        "skipped": skipped,
        **{f"recall@{k}": round(statistics.fmean(scores[k]), 4) for k in ks},
        f"mrr@{top_k}": round(statistics.fmean(rr), 4),
        "relevant_per_query": round(statistics.fmean(relevant_counts), 2),
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p95_ms": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1),
    }


def coverage(session: Session) -> dict:
    """How much of the ground truth the index can actually return.

    The ceiling on every recall number here, and the first thing to quote
    beside one.
    """
    total = session.scalar(select(func.count()).select_from(Citation)) or 0
    inside = (
        session.scalar(
            select(func.count())
            .select_from(Citation)
            .join(Paper, Paper.id == Citation.cited_id)
        )
        or 0
    )
    return {
        "citation_edges": total,
        "retrievable": inside,
        "retrievable_pct": round(100 * inside / total, 1) if total else 0.0,
        "discarded_pct": round(100 * (total - inside) / total, 1) if total else 0.0,
    }
