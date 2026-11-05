"""What a run writes, what it refuses to write, and what it does twice safely."""

from datetime import UTC, datetime

import httpx
from conftest import page, work
from sqlalchemy import func, select

from scholarly import raw
from scholarly.db import session_scope, upsert_papers
from scholarly.ingest import ingest_corpus, ingest_span
from scholarly.models import Citation, IngestRun, Paper, Quarantine


def responder(pages: dict[str, dict]):
    """Serves one payload per cursor, so paging can actually be exercised."""

    def handle(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor", "*")
        if cursor not in pages:
            return httpx.Response(404, json={"error": f"no page for cursor {cursor}"})
        return httpx.Response(200, json=pages[cursor])

    return handle


async def test_a_page_becomes_papers_and_citations(db, mock_client):
    pages = {"*": page([work("W1", refs=["W2", "W3"])])}

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["succeeded"] == 1
    assert summary["status"] == "succeeded"
    with session_scope() as session:
        paper = session.get(Paper, "W1")
        assert paper.title == "A paper"
        assert paper.abstract == "hello world"
        assert paper.reference_count == 2
        cited = session.scalars(select(Citation.cited_id).order_by(Citation.cited_id)).all()
        assert cited == ["W2", "W3"]


async def test_citations_are_kept_even_when_the_cited_paper_is_absent(db, mock_client):
    """The measurement that shaped the schema: 82% of cited works are outside
    this corpus. A foreign key would reject most of the ground truth."""
    pages = {"*": page([work("W1", refs=["W9999999"])])}

    await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Citation)) == 1
        assert session.get(Paper, "W9999999") is None


async def test_the_payload_is_landed_before_it_is_parsed(db, mock_client):
    """So a parser fix replays from disk instead of re-walking the API."""
    pages = {"*": page([work("W1"), work("W2")])}

    await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    landed = list((db / "raw").rglob("page-*.jsonl"))
    assert len(landed) == 1
    replayed = raw.replay(landed[0])
    assert [r["id"] for r in replayed] == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
    ]


async def test_paging_follows_the_cursor_to_the_end(db, mock_client):
    pages = {
        "*": page([work("W1")], next_cursor="c2"),
        "c2": page([work("W2")], next_cursor="c3"),
        "c3": page([work("W3")], next_cursor=None),
    }

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["pages"] == 3
    assert summary["succeeded"] == 3
    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Paper)) == 3


async def test_one_unreadable_record_does_not_cost_the_page(db, mock_client):
    """Live payloads are not a schema. A bad record is kept with its reason."""
    broken = work("W2")
    broken["title"] = "   "
    pages = {"*": page([work("W1"), broken, work("W3")])}

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["succeeded"] == 2
    assert summary["quarantined"] == 1
    assert summary["status"] == "partial"
    with session_scope() as session:
        held = session.scalars(select(Quarantine)).one()
        assert "W2" in (held.identifier or "")
        assert held.payload  # the thing that broke it, for replay


async def test_running_the_same_slice_twice_changes_nothing(db, mock_client):
    """The only recovery anyone performs on a scheduled pipeline is running
    it again, so that has to be safe."""
    pages = {"*": page([work("W1", refs=["W2", "W3"])])}

    await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )
    await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Paper)) == 1
        assert session.scalar(select(func.count()).select_from(Citation)) == 2


def test_re_ingesting_does_not_rewrite_when_a_paper_was_first_seen(db):
    """`first_seen_at` is the one column that is about this warehouse rather
    than about the source. An upsert that clobbers it destroys history on
    every scheduled run, silently."""
    old = datetime(2020, 1, 1, tzinfo=UTC)
    with session_scope() as session:
        session.add(Paper(id="W1", title="first", first_seen_at=old, last_seen_at=old))

    with session_scope() as session:
        upsert_papers(session, [{"id": "W1", "title": "second", "reference_count": 3}])

    with session_scope() as session:
        paper = session.get(Paper, "W1")
        assert paper.title == "second"
        assert paper.reference_count == 3
        assert paper.first_seen_at.replace(tzinfo=UTC) == old
        assert paper.last_seen_at.replace(tzinfo=UTC) > old


async def test_a_limit_stops_the_walk_early(db, mock_client):
    pages = {
        "*": page([work("W1"), work("W2")], next_cursor="c2"),
        "c2": page([work("W3")], next_cursor="c3"),
    }

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        max_records=2,
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["requested"] == 2
    assert summary["pages"] == 1


async def test_a_page_that_will_not_load_fails_the_run_rather_than_truncating(db, mock_client):
    """A short index that reports success is worse than a run that says it
    broke - nothing downstream can tell the difference later."""
    pages = {"*": page([work("W1")], next_cursor="c2")}  # c2 is missing -> 404

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["status"] == "partial"
    assert summary["error"] and "404" in summary["error"]
    with session_scope() as session:
        run = session.scalars(select(IngestRun)).one()
        assert run.error is not None


async def test_a_walk_that_ends_early_is_not_reported_as_a_success(db, mock_client):
    """The first real run collected 49.5% of the corpus and said "succeeded".

    OpenAlex ended the cursor itself after 169 full pages while reporting
    68,225 records available, and a walk that stops early is indistinguishable
    from one that finished - the cursor simply stops either way. Nothing in
    the run, the logs or the database disagreed. Completeness has to be
    checked against what the source claimed, because nothing else will.
    """
    pages = {"*": page([work(f"W{i}") for i in range(200)], count=68225, next_cursor=None)}

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["succeeded"] == 200
    assert summary["expected"] == 68225
    assert summary["status"] == "partial"
    assert "incomplete" in summary["error"]
    assert "68,225" in summary["error"]


async def test_a_complete_walk_is_not_flagged(db, mock_client):
    pages = {"*": page([work("W1"), work("W2")], next_cursor=None)}

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["status"] == "succeeded"
    assert summary["error"] is None


async def test_an_explicit_limit_is_not_an_incomplete_walk(db, mock_client):
    """Asking for 200 of 68,225 and getting 200 is the limit working."""
    pages = {"*": page([work(f"W{i}") for i in range(200)], count=68225, next_cursor="c2")}

    summary = await ingest_corpus(
        "2018-01-01",
        "2018-01-31",
        max_records=200,
        client=mock_client(responder(pages)),
        raw_directory=str(db / "raw"),
    )

    assert summary["status"] == "succeeded"
    assert summary["error"] is None


async def test_a_span_is_walked_as_monthly_slices(db, mock_client):
    """Cursor truncation is intermittent and upstream, so no slice size is
    simply safe. A month is about seven pages: short enough that redoing one
    costs seconds."""
    asked: list[str] = []

    def handle(request):
        asked.append(request.url.params.get("filter", ""))
        return httpx.Response(200, json=page([work(f"W{len(asked)}")], next_cursor=None))

    summary = await ingest_span(
        2018, 2018, client=mock_client(handle), raw_directory=str(db / "raw")
    )

    assert len(asked) == 12
    assert "from_publication_date:2018-01-01" in asked[0]
    assert "to_publication_date:2018-01-31" in asked[0]
    assert "to_publication_date:2018-02-28" in asked[1]
    assert "to_publication_date:2018-12-31" in asked[11]
    assert summary["status"] == "succeeded"
    assert summary["incomplete_slices"] == []


async def test_a_short_slice_is_retried(db, mock_client):
    """Re-running is safe because every write in the path is idempotent, and a
    retry restarts the slice from its first page."""
    attempts: list[int] = []

    def handle(request):
        attempts.append(1)
        # The first attempt at the first slice under-delivers; the retry does not.
        short = len(attempts) == 1
        return httpx.Response(
            200,
            json=page([work(f"W{len(attempts)}")], count=9_000 if short else None,
                      next_cursor=None),
        )

    summary = await ingest_span(
        2018, 2018, retries=2, client=mock_client(handle), raw_directory=str(db / "raw")
    )

    # 12 months plus one retry of the first.
    assert len(attempts) == 13
    assert summary["status"] == "succeeded"


async def test_a_slice_that_stays_short_makes_the_span_partial(db, mock_client):
    def handle(request):
        first = "2018-01-01" in request.url.params.get("filter", "")
        return httpx.Response(
            200,
            json=page([work("W1")], count=9_000 if first else None, next_cursor=None),
        )

    summary = await ingest_span(
        2018, 2018, retries=1, client=mock_client(handle), raw_directory=str(db / "raw")
    )

    assert summary["status"] == "partial"
    assert summary["incomplete_slices"] == ["2018-01-01"]
