"""Encoding and dot-product search, with the encoder stubbed.

The real model is a few hundred megabytes and several minutes; none of that
would test anything here. What is worth testing is everything around it:
resumability, which model a vector belongs to, and the ranking.
"""

import numpy as np
import pytest

from scholarly import embeddings
from scholarly.db import session_scope
from scholarly.dense import encode_corpus, encoded_count, pending
from scholarly.models import Embedding, Paper
from scholarly.serving import VectorIndex

# A toy semantic space: each axis is a topic, so "attention transformer" and
# "transformer sequence" land near each other and far from "tomatoes".
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


def add(session, identifier, abstract):
    session.add(Paper(id=identifier, title=identifier, abstract=abstract))


def test_a_vector_survives_the_round_trip():
    vector = fake_vector("transformer attention")
    restored = embeddings.from_blob(embeddings.to_blob(vector))
    assert np.allclose(vector, restored)


def test_only_unencoded_papers_with_abstracts_are_pending(db):
    with session_scope() as session:
        add(session, "W1", "transformer")
        add(session, "W2", "garden")
        session.add(Paper(id="W3", title="no abstract", abstract=None))

    with session_scope() as session:
        assert {row[0] for row in pending(session, "m")} == {"W1", "W2"}
        encode_corpus(session, model_name="m")

    with session_scope() as session:
        assert pending(session, "m") == []
        # A different model has encoded nothing, so everything is pending again.
        assert len(pending(session, "other")) == 2


def test_encoding_is_resumable(db):
    """An encode measured in CPU-hours will be interrupted, and running it
    again is the only recovery anyone performs."""
    with session_scope() as session:
        for i in range(6):
            add(session, f"W{i}", "transformer attention")

    with session_scope() as session:
        first = encode_corpus(session, model_name="m", limit=4)
    with session_scope() as session:
        second = encode_corpus(session, model_name="m")
    with session_scope() as session:
        third = encode_corpus(session, model_name="m")
        total = encoded_count(session, "m")

    assert first["encoded"] == 4
    assert second["encoded"] == 2
    assert third["encoded"] == 0
    assert total == 6


def test_the_model_is_stored_with_the_vector(db):
    """Vectors from different encoders are not comparable, and a half
    re-encoded index would otherwise look exactly like a working one."""
    with session_scope() as session:
        add(session, "W1", "transformer")

    with session_scope() as session:
        encode_corpus(session, model_name="model-a")
        encode_corpus(session, model_name="model-b")

    with session_scope() as session:
        stored = {row.model for row in session.query(Embedding).all()}
        assert stored == {"model-a", "model-b"}
        assert len(VectorIndex.load(session, model_name="model-a")) == 1


def test_a_bounded_encode_samples_the_corpus_rather_than_one_end_of_it(db):
    """OpenAlex ids broadly track when a work was catalogued, so ordering by
    id takes the oldest papers instead of a sample. A bounded encode exists to
    say something indicative about the whole corpus, which it cannot do from
    one end of it."""
    with session_scope() as session:
        for i in range(100):
            add(session, f"W{i:04d}", "transformer")

    with session_scope() as session:
        picked = [paper_id for paper_id, _ in pending(session, "m", limit=20)]

    assert len(picked) == 20
    # Taken from across the range rather than the first twenty ids.
    assert picked != sorted(picked)[:20]
    assert max(picked) > "W0050"
    # Stable: the same corpus and seed choose the same papers.
    with session_scope() as session:
        assert [p for p, _ in pending(session, "m", limit=20)] == picked


def test_a_large_limit_does_not_blow_the_parameter_cap(db):
    """SQLite caps bound parameters per statement, so fetching the chosen
    subset has to be chunked. 1,200 ids is past the usual 999."""
    with session_scope() as session:
        for i in range(1500):
            add(session, f"W{i:05d}", "transformer")

    with session_scope() as session:
        picked = pending(session, "m", limit=1200)

    assert len(picked) == 1200
    assert all(abstract for _, abstract in picked)
    assert len({paper_id for paper_id, _ in picked}) == 1200
