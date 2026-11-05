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
    LargeBinary,
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


class Embedding(Base):
    """One vector per paper, in its own table rather than a column on `papers`.

    Deliberately separate. A vector column on `papers` would be rewritten by
    `upsert_papers` on every re-ingest - the derived-column trap, where the
    expensive thing is silently blanked by the cheap thing and nothing errors.
    Keeping it here means a refresh of the registry metadata cannot destroy an
    encode that took an hour of CI to produce.

    `model` is stored with the vector because vectors from different encoders
    are not comparable, and a half-re-encoded index would otherwise look
    exactly like a working one.
    """

    __tablename__ = "embeddings"

    # Composite, because the model is part of a vector's identity: one paper
    # may hold a vector per encoder while a re-encode is in progress, and
    # keying on paper_id alone would make the second model fail to insert
    # rather than coexist.
    paper_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    dimensions: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvalQuery(Base):
    """A paper frozen into the evaluation set, and which split it belongs to.

    Stored rather than computed on demand, because a query set that is
    re-derived each run silently changes as the corpus grows - and then two
    numbers measured a week apart are not comparable, while looking exactly
    as though they are.
    """

    __tablename__ = "eval_queries"

    paper_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    split: Mapped[str] = mapped_column(String(8), index=True)
    relevant_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


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
