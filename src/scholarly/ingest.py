"""Walking the corpus into the warehouse.

The shape is: ask OpenAlex for a page, land it raw, parse it, write what
parsed, quarantine what did not, move to the next cursor. Every step is
idempotent, because a run that fails halfway has to be safe to simply run
again - which on a scheduled pipeline is the only recovery anyone will
actually perform.

Paging is by cursor rather than page number. OpenAlex refuses to page past
10,000 results, and the query corpus alone is 68,244.
"""

import logging
from datetime import UTC, datetime

from pydantic import ValidationError

from scholarly import raw
from scholarly.db import insert_citations, session_scope, upsert_papers
from scholarly.http import FetchError, RateLimitedClient
from scholarly.models import IngestRun, Quarantine
from scholarly.sources import openalex

log = logging.getLogger("scholarly.ingest")

SOURCE = "openalex"


def _rows(records: list[dict]) -> tuple[list[dict], list[tuple[str, str]], list[tuple]]:
    """Split a page into papers, citation edges, and what could not be read."""
    papers: list[dict] = []
    edges: list[tuple[str, str]] = []
    rejected: list[tuple] = []

    for record in records:
        try:
            work = openalex.parse_work(record)
        except ValidationError as exc:
            rejected.append((record.get("id"), str(exc)[:400], record))
            continue

        papers.append(
            {
                "id": work.id,
                "title": work.title,
                "year": work.year,
                "doi": work.doi,
                "abstract": work.abstract,
                "reference_count": len(work.referenced_works),
            }
        )
        edges.extend((work.id, cited) for cited in work.referenced_works)

    return papers, edges, rejected


async def ingest_span(
    from_year: int,
    to_year: int,
    *,
    retries: int = 2,
    **kwargs,
) -> dict:
    """Walk a span as monthly slices, retrying any slice that comes up short.

    **Cursor paging against OpenAlex truncates, intermittently.** Measured: a
    2018-2021 query reporting 68,225 records stopped at 33,796; 2018 alone
    then completed exactly at 13,021; and in the very next run 2018 stopped at
    7,590 while 2019 completed. It is upstream and it is not deterministic, so
    there is no slice size that is simply safe.

    What works is making a truncation cheap rather than trying to avoid it: a
    month is about seven pages, so a short slice costs seconds to redo. Slices
    are retried because re-running is safe - every write in the path is
    idempotent - and a retry starts the slice again from its first page.

    One run record per attempt, so the history shows the retries rather than
    hiding them behind a total that looks about right.
    """
    slices = openalex.month_slices(from_year, to_year)
    best: list[tuple[str, dict]] = []

    for first, last in slices:
        attempt_summary = await ingest_corpus(first, last, **kwargs)
        for attempt in range(retries):
            if attempt_summary["status"] == "succeeded":
                break
            log.warning(
                "slice came up short, retrying",
                extra={"slice": first, "attempt": attempt + 1, **attempt_summary},
            )
            retry = await ingest_corpus(first, last, **kwargs)
            # Keep whichever attempt actually collected more; the warehouse
            # holds the union of all of them regardless, because the writes
            # are idempotent.
            if retry["succeeded"] >= attempt_summary["succeeded"]:
                attempt_summary = retry
        best.append((first, attempt_summary))

    short = [first for first, summary in best if summary["status"] != "succeeded"]
    return {
        "slices": len(best),
        "expected": sum(s["expected"] or 0 for _, s in best),
        "succeeded": sum(s["succeeded"] for _, s in best),
        "quarantined": sum(s["quarantined"] for _, s in best),
        # One short slice makes the whole span short. Averaging would hide it.
        "status": "succeeded" if not short else "partial",
        "incomplete_slices": short,
    }


async def ingest_corpus(
    from_date: str,
    to_date: str,
    *,
    require_references: bool = True,
    max_records: int | None = None,
    client: RateLimitedClient | None = None,
    db_url: str | None = None,
    raw_directory: str | None = None,
) -> dict:
    """Ingest one slice of the corpus. Returns a summary of the run."""
    filter_string = openalex.corpus_filter(
        from_date, to_date, require_references=require_references
    )

    with session_scope(db_url) as session:
        run = IngestRun(source=SOURCE)
        session.add(run)
        session.flush()
        run_id = run.id

    log.info(
        "ingest started",
        extra={"run_id": run_id, "filter": filter_string, "max_records": max_records},
    )

    owns_client = client is None
    client = client or RateLimitedClient()

    cursor: str | None = "*"
    page = 0
    seen = succeeded = quarantined = 0
    error: str | None = None
    expected: int | None = None

    try:
        while cursor:
            url = openalex.works_url(filter_string, cursor=cursor)
            try:
                payload = await client.get_json(url, headers=openalex.headers())
            except FetchError as exc:
                # A page that will not load ends the run rather than silently
                # truncating the corpus: a short index that looks complete is
                # worse than a run that says it failed.
                error = f"page {page}: {exc}"
                if exc.status == 401 and not openalex.authenticated():
                    # The failure that will catch every new clone of this
                    # repository, so it says what to do rather than what broke.
                    error = (
                        "OpenAlex requires a free API key to page past the first "
                        "page. Set OPENALEX_API_KEY; see docs/corpus-decisions.md. "
                        f"({exc})"
                    )
                log.error("page fetch failed", extra={"run_id": run_id, "page": page})
                break

            # What the source says exists, read once from the first page. A
            # run has no way to know it was short without it.
            if expected is None:
                expected = (payload.get("meta") or {}).get("count")

            records = payload.get("results") or []
            if not records:
                break

            page += 1
            seen += len(records)
            raw.land(records, run_id=run_id, page=page, directory=raw_directory)

            papers, edges, rejected = _rows(records)

            with session_scope(db_url) as session:
                upsert_papers(session, papers)
                insert_citations(session, edges)
                for identifier, reason, payload_record in rejected:
                    session.add(
                        Quarantine(
                            source=SOURCE,
                            identifier=str(identifier)[:255] if identifier else None,
                            reason=reason,
                            payload=repr(payload_record)[:20000],
                        )
                    )

            succeeded += len(papers)
            quarantined += len(rejected)

            if max_records is not None and seen >= max_records:
                break

            cursor = openalex.next_cursor(payload)
    finally:
        if owns_client:
            await client.aclose()

    # A walk that ends early looks exactly like a walk that finished: the
    # cursor simply stops. Without this check the first full run reported
    # "succeeded" having collected 49.5% of the corpus, and nothing anywhere
    # disagreed. Completeness is the property worth asserting, so it is
    # asserted rather than assumed.
    if error is None and expected and max_records is None and seen < expected:
        missing = expected - seen
        if missing > max(50, int(expected * 0.01)):
            error = (
                f"incomplete: the source reported {expected:,} records and the "
                f"walk ended after {seen:,} ({100 * seen / expected:.1f}%). "
                "The cursor stopped early; ingest in narrower slices."
            )
            log.error("ingest incomplete", extra={"expected": expected, "seen": seen})

    with session_scope(db_url) as session:
        run = session.get(IngestRun, run_id)
        run.requested = seen
        run.succeeded = succeeded
        run.quarantined = quarantined
        run.failed = seen - succeeded - quarantined
        run.finished_at = datetime.now(UTC)
        run.error = error
        run.status = "failed" if error and succeeded == 0 else (
            "partial" if (error or quarantined) else "succeeded"
        )
        status = run.status

    summary = {
        "run_id": run_id,
        "status": status,
        "pages": page,
        "expected": expected,
        "requested": seen,
        "succeeded": succeeded,
        "quarantined": quarantined,
        "error": error,
    }
    log.info("ingest finished", extra=summary)
    return summary
