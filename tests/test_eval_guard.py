"""The guard that refuses to compare retrievers over different corpora.

Written after a local ablation did exactly that: a complete lexical index
against 6,200 vectors, read as though the difference were the retrievers.
The CI gate catches this, but the gate lives in the workflow and `eval` is
what a person runs by hand, so the check has to be here too.

A guard with no test is the thing that quietly stops working, so this drives
the real command through `main()` rather than testing the branch directly.
"""

import numpy as np
import pytest

from scholarly import embeddings
from scholarly.cli import main
from scholarly.db import session_scope
from scholarly.dense import encode_corpus
from scholarly.evaluation import build_query_set
from scholarly.lexical import build_index, corpus_size
from scholarly.models import Citation, Paper

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
def lopsided(db):
    """A corpus the lexical index covers fully and the dense index does not."""
    with session_scope() as session:
        for index in range(8):
            session.add(
                Paper(
                    id=f"W{index:03d}",
                    title=f"Paper {index}",
                    abstract="transformer attention" if index % 2 else "garden soil",
                    year=2019,
                )
            )
        # One citation between two in-corpus papers, so a query exists.
        session.add(Citation(citing_id="W001", cited_id="W002"))

    with session_scope() as session:
        build_index(session, rebuild=True)
        encode_corpus(session, model_name=embeddings.DEFAULT_MODEL, limit=3)
        build_query_set(session, size=2, dev_fraction=0.5, from_year=2019, to_year=2019)

    with session_scope() as session:
        assert corpus_size(session) == 8
    return f"sqlite:///{db / 'test.db'}"


def test_corpus_size_reports_what_the_lexical_index_can_reach(lopsided):
    with session_scope() as session:
        assert corpus_size(session) == 8


def test_corpus_size_is_zero_before_an_index_exists(db):
    with session_scope() as session:
        assert corpus_size(session) == 0


def test_a_dense_run_over_a_smaller_index_is_refused(lopsided, capsys):
    code = main(["--database-url", lopsided, "eval", "--split", "dev", "--retriever", "dense"])
    captured = capsys.readouterr().out

    assert code == 1
    assert "would measure the corpus, not the retriever" in captured
    assert '"dense_index": 3' in captured
    assert '"lexical_index": 8' in captured


def test_hybrid_is_refused_for_the_same_reason(lopsided, capsys):
    # Fusion is the case that misled me: a complete ranking fused with a
    # crippled one looks like evidence against fusion.
    code = main(["--database-url", lopsided, "eval", "--split", "dev", "--retriever", "hybrid"])
    assert code == 1
    assert "would measure the corpus" in capsys.readouterr().out


def test_lexical_alone_is_not_refused(lopsided):
    # Nothing is being compared across corpora, so there is nothing to guard.
    assert main(["--database-url", lopsided, "eval", "--split", "dev"]) == 0


def test_the_opt_out_works_and_still_shows_both_numbers(lopsided, capsys):
    code = main(
        [
            "--database-url",
            lopsided,
            "eval",
            "--split",
            "dev",
            "--retriever",
            "dense",
            "--allow-partial-index",
        ]
    )
    captured = capsys.readouterr().out

    assert code == 0
    # Opting out silences the refusal, not the facts: both coverage numbers
    # stay in the output so the mismatch cannot be read past.
    assert '"lexical_index": 8' in captured
    assert '"index_size": 3' in captured
