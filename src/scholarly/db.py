"""Engine, sessions, and an upsert that works on both dialects.

SQLite locally and in tests, Postgres in anything deployed. The split matters
because `upsert_papers` has to branch on the dialect, and a suite that only
ever runs SQLite never executes the Postgres branch - so CI runs the whole
thing against a real Postgres as well.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from scholarly.models import Base, Citation, Paper

DEFAULT_URL = "sqlite:///scholarly.db"

_engine: Engine | None = None
_Session: sessionmaker | None = None


def _ensure_parent_directory(url: str) -> None:
    """SQLite will not create a missing directory; it just fails to open.

    Worth doing here rather than at the call site, because the failure only
    appears the first time someone points DATABASE_URL at a fresh path - which
    is to say, on a new machine or a fresh deployment.
    """
    if not url.startswith("sqlite:///"):
        return
    path = url.removeprefix("sqlite:///")
    if path and path != ":memory:":
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def init_db(url: str | None = None, *, create: bool = True) -> Engine:
    """Bind the engine, and optionally create the schema."""
    global _engine, _Session
    url = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
    _ensure_parent_directory(url)
    _engine = create_engine(url, future=True)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    if create:
        Base.metadata.create_all(_engine)
    return _engine


def engine() -> Engine:
    if _engine is None:
        init_db()
    assert _engine is not None
    return _engine


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    if _Session is None or url is not None:
        init_db(url)
    assert _Session is not None
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upsert_papers(session: Session, rows: list[dict]) -> int:
    """Insert or update papers, preserving `first_seen_at`.

    Re-ingesting must not rewrite when the corpus first saw a paper - that is
    the one column whose value is about this warehouse rather than about the
    source, and clobbering it would quietly destroy history on every run.
    """
    if not rows:
        return 0

    now = datetime.now(UTC)
    prepared = [{**row, "first_seen_at": now, "last_seen_at": now} for row in rows]

    dialect = session.get_bind().dialect.name
    insert = pg_insert if dialect == "postgresql" else sqlite_insert
    statement = insert(Paper).values(prepared)
    statement = statement.on_conflict_do_update(
        index_elements=[Paper.id],
        set_={
            "title": statement.excluded.title,
            "year": statement.excluded.year,
            "doi": statement.excluded.doi,
            "abstract": statement.excluded.abstract,
            "reference_count": statement.excluded.reference_count,
            "last_seen_at": statement.excluded.last_seen_at,
            # first_seen_at is deliberately absent.
        },
    )
    session.execute(statement)
    return len(prepared)


def insert_citations(session: Session, edges: list[tuple[str, str]]) -> int:
    """Insert citation edges, ignoring ones already recorded.

    Idempotent because every run re-reads papers it has seen before, and an
    edge is a fact about the source rather than about this run - re-stating it
    should cost nothing and change nothing.
    """
    if not edges:
        return 0

    rows = [{"citing_id": citing, "cited_id": cited} for citing, cited in edges]
    dialect = session.get_bind().dialect.name
    insert = pg_insert if dialect == "postgresql" else sqlite_insert
    statement = insert(Citation).values(rows).on_conflict_do_nothing(
        index_elements=[Citation.citing_id, Citation.cited_id]
    )
    session.execute(statement)
    return len(rows)
