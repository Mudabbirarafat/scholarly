"""What a run writes, what it refuses to write, and what it does twice safely."""

from datetime import UTC, datetime

import httpx
from conftest import page, work
from sqlalchemy import func, select

from scholarly import raw
from scholarly.db import session_scope, upsert_papers
from scholarly.ingest import ingest_corpus
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
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
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
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
    )

    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Citation)) == 1
        assert session.get(Paper, "W9999999") is None


async def test_the_payload_is_landed_before_it_is_parsed(db, mock_client):
    """So a parser fix replays from disk instead of re-walking the API."""
    pages = {"*": page([work("W1"), work("W2")])}

    await ingest_corpus(
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
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
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
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
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
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
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
    )
    await ingest_corpus(
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
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
        2018,
        2021,
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
        2018, 2021, client=mock_client(responder(pages)), raw_directory=str(db / "raw")
    )

    assert summary["status"] == "partial"
    assert summary["error"] and "404" in summary["error"]
    with session_scope() as session:
        run = session.scalars(select(IngestRun)).one()
        assert run.error is not None
