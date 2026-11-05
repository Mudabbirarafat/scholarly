"""The warehouse schema.

Four tables, and one of them is shaped by a measurement rather than by taste.

**`citations` is an edge list, not a foreign key.** Only about 18% of a
paper's references are themselves arXiv CS papers - the rest are journal
articles, conference proceedings and books that this corpus will never hold.
A foreign key to `papers` would reject 82% of the ground truth at insert time,
which is the ground truth. Edges are stored whether or not the cited paper is
in the index, and the evaluation restricts them to what is retrievable when it
builds its qrels. See `docs/corpus-decisions.md`.

**`quarantine` keeps a bad record instead of failing a run.** A single
unparseable payload in a page of 200 should cost that record, not the other
199 and the hour of paging that produced them.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    """One searchable document."""

    __tablename__ = "papers"

    # The OpenAlex short id, "W123". One spelling, normalised on the way in -
    # the citations table joins against this and a mismatch would silently
    # produce an empty ground truth.
    id: Mapped[str] = mapped_column(String(32), primary_key=True)

    title: Mapped[str] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    doi: Mapped[str | None] = mapped_column(String(255))
    abstract: Mapped[str | None] = mapped_column(Text)

    # Carried from the payload rather than derived, so re-ingesting a record
    # cannot blank it the way a materialised column would.
    reference_count: Mapped[int] = mapped_column(Integer, default=0)

    # Kept across re-ingests so "when did this corpus first see it" survives.
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Citation(Base):
    """`citing` cites `cited`. The evaluation's ground truth.

    Deliberately NOT `references`: that is a reserved word in Postgres, and a
    table name that needs quoting everywhere is a trap for the first person to
    write a hand query against it.
    """

    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("citing_id", "cited_id", name="uq_citation"),
        Index("ix_citations_cited", "cited_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # A foreign key, because a citing paper is always one we ingested.
    citing_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    # No foreign key: 82% of cited works are outside this corpus by
    # construction, and that is a fact to record rather than an error.
    cited_id: Mapped[str] = mapped_column(String(32))


class IngestRun(Base):
    """What one run asked for and what it got.

    Without this, "the data looks wrong" has no way to become "the run on
    Tuesday quarantined 400 records".
    """

    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="running")

    requested: Mapped[int] = mapped_column(Integer, default=0)
    succeeded: Mapped[int] = mapped_column(Integer, default=0)
    quarantined: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


class Quarantine(Base):
    """A record that could not be parsed, kept with the reason and the payload.

    The payload is the point: a parser fix can be tested against exactly what
    broke it, without going back to the API to reproduce the failure.
    """

    __tablename__ = "quarantine"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    identifier: Mapped[str | None] = mapped_column(String(255))
    reason: Mapped[str] = mapped_column(Text)
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
