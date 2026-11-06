"""The serving bundle.

The property that matters is that the bundle is the same thing, smaller: the
same schema, the same vectors, a working full-text index, and nothing in it
that serving cannot use. A bundle that loads but answers differently from the
warehouse it came from would be indistinguishable from a working one.
"""

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from scholarly import embeddings
from scholarly.bundle import build_bundle
from scholarly.db import session_scope
from scholarly.dense import encode_corpus
from scholarly.lexical import search as lexical_search
from scholarly.models import Citation, Embedding, EvalQuery, Paper
from scholarly.serving import VectorIndex

AXES = ["transformer", "attention", "garden", "protein"]


def fake_vector(text: str) -> np.ndarray:
    raw = np.array([float(text.lower().count(axis)) for axis in AXES], dtype=np.float32)
    if not raw.any():
        raw = np.ones(len(AXES), dtype=np.float32)
    return raw / np.linalg.norm(raw)


@pytest.fixture(autouse=True)
def stub_encoder(monkeypatch):
    monkeypatch.setattr(
        embeddings, "encode", lambda texts, **kw: (fake_vector(t) for t in texts)
    )


@pytest.fixture
def warehouse(db):
    """A warehouse with more in it than serving needs."""
    with session_scope() as session:
        for index in range(12):
            session.add(
                Paper(
                    id=f"W{index:03d}",
                    title=f"Paper {index}",
                    abstract="transformer attention" if index % 2 else "garden soil",
                    year=2018 + (index % 4),
                    doi=f"10.1/{index}",
                )
            )
        # A paper with no abstract never gets a vector, so it has no business
        # being in a bundle.
        session.add(Paper(id="WNONE", title="No abstract", abstract=None))
        session.add(Citation(citing_id="W001", cited_id="W002"))
        session.add(EvalQuery(paper_id="W001", split="dev", relevant_count=1))

    with session_scope() as session:
        encode_corpus(session, model_name=embeddings.DEFAULT_MODEL)
    return db


def opened(path):
    engine = create_engine(f"sqlite:///{path.as_posix()}", future=True)
    return sessionmaker(bind=engine, future=True)()


def test_the_bundle_holds_only_papers_that_have_a_vector(warehouse, tmp_path):
    out = tmp_path / "serving.db"
    with session_scope() as session:
        summary = build_bundle(session, out)

    assert summary["papers"] == 12
    assert summary["vectors"] == 12
    assert "warning" not in summary

    with opened(out) as session:
        assert session.scalar(select(Paper).where(Paper.id == "WNONE")) is None
        assert session.scalar(select(Paper).where(Paper.id == "W001")) is not None


def test_what_serving_cannot_use_is_left_behind(warehouse, tmp_path):
    """Citations are 2.6 million rows the serving side never reads, and the
    frozen query set is the answer key. Neither belongs in a public bundle."""
    out = tmp_path / "serving.db"
    with session_scope() as session:
        build_bundle(session, out)

    with opened(out) as session:
        assert session.scalars(select(Citation)).all() == []
        assert session.scalars(select(EvalQuery)).all() == []


def test_the_vectors_survive_exactly(warehouse, tmp_path):
    out = tmp_path / "serving.db"
    with session_scope() as session:
        build_bundle(session, out)
        original = {
            row.paper_id: embeddings.from_blob(row.vector)
            for row in session.scalars(select(Embedding))
        }

    with opened(out) as session:
        copied = {
            row.paper_id: embeddings.from_blob(row.vector)
            for row in session.scalars(select(Embedding))
        }

    assert set(copied) == set(original)
    for paper_id, vector in original.items():
        assert np.array_equal(copied[paper_id], vector)


def test_both_retrievers_work_against_the_bundle(warehouse, tmp_path):
    """The bundle is what a deployed instance searches, so the test is not
    that it parses - it is that the two retrievers answer from it."""
    out = tmp_path / "serving.db"
    with session_scope() as session:
        build_bundle(session, out)

    with opened(out) as session:
        index = VectorIndex.load(session)
        dense_hits = [paper_id for paper_id, _ in index.search(fake_vector("transformer"), k=3)]
        lexical_hits = lexical_search(session, "transformer attention", k=3)

    assert len(index) == 12
    assert dense_hits, "dense found nothing in the bundle"
    assert lexical_hits, "the bundle's full-text index returned nothing"
    assert all(hit.startswith("W") for hit in lexical_hits)


def test_a_limit_serves_a_smaller_corpus(warehouse, tmp_path):
    out = tmp_path / "small.db"
    with session_scope() as session:
        summary = build_bundle(session, out, limit=5)

    assert summary["papers"] == 5
    assert summary["vectors"] == 5
    with opened(out) as session:
        assert len(VectorIndex.load(session)) == 5


def test_rebuilding_replaces_rather_than_appends(warehouse, tmp_path):
    out = tmp_path / "serving.db"
    with session_scope() as session:
        build_bundle(session, out, limit=5)
        again = build_bundle(session, out, limit=5)

    # Without the unlink, the second build would hit a primary-key collision
    # or silently double the corpus depending on the dialect.
    assert again["papers"] == 5


def test_an_empty_warehouse_makes_an_empty_bundle_not_a_crash(db, tmp_path):
    out = tmp_path / "serving.db"
    with session_scope() as session:
        summary = build_bundle(session, out)

    assert summary["papers"] == 0
    assert summary["vectors"] == 0
    with opened(out) as session:
        assert len(VectorIndex.load(session)) == 0
