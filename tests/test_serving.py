"""The serving path, with the encoder stubbed.

What is worth testing here is not that a transformer works. It is that the
things done for the sake of a 512 MB instance - float16 storage, chunked
scoring - give the same answers as the straightforward version, and that a
result never claims to explain more than it can.
"""

import numpy as np
import pytest

from scholarly import embeddings, serving
from scholarly.db import session_scope
from scholarly.dense import encode_corpus
from scholarly.lexical import build_index
from scholarly.models import Paper
from scholarly.serving import SearchService, VectorIndex

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


def add(session, identifier, abstract, *, title=None, year=2020, doi=None):
    session.add(
        Paper(
            id=identifier,
            title=title or f"Paper {identifier}",
            abstract=abstract,
            year=year,
            doi=doi,
        )
    )


@pytest.fixture
def corpus(db):
    with session_scope() as session:
        add(session, "W1", "transformer attention attention for sequences")
        add(session, "W2", "garden garden tomatoes and soil")
        add(session, "W3", "transformer transformer attention is all you need")
        add(session, "W4", "protein folding with deep learning", doi="10.1/abc")
    with session_scope() as session:
        encode_corpus(session, model_name=embeddings.DEFAULT_MODEL)
        build_index(session, rebuild=True)
    return db


def test_the_index_is_float16_and_ranks_by_similarity(corpus):
    with session_scope() as session:
        index = VectorIndex.load(session)

    assert len(index) == 4
    assert index.matrix.dtype == np.float16
    found = [paper_id for paper_id, _ in index.search(fake_vector("transformer attention"), k=4)]

    # The two on-topic papers come first and the two off-topic ones last.
    # Which of the off-topic pair is last is a genuine tie - both score
    # exactly 0 on this query - so asserting an order there would be
    # asserting an implementation detail of argsort.
    assert set(found[:2]) == {"W1", "W3"}
    assert set(found[2:]) == {"W2", "W4"}


def test_chunked_scoring_agrees_with_the_whole_matrix(corpus, monkeypatch):
    """The chunking exists to avoid a 168 MiB temporary, and it would be a
    silent correctness bug if it ever disagreed with the obvious version."""
    with session_scope() as session:
        index = VectorIndex.load(session)

    query = fake_vector("transformer attention")
    monkeypatch.setattr(serving, "CHUNK", 1)  # one row at a time
    chunked = index.scores(query)
    naive = index.matrix.astype(np.float32) @ query

    assert np.allclose(chunked, naive, atol=1e-3)


def test_an_empty_index_answers_instead_of_failing(db):
    with session_scope() as session:
        index = VectorIndex.load(session)
    assert len(index) == 0
    assert index.megabytes == 0.0
    assert index.search(fake_vector("anything"), k=5) == []


def test_more_neighbours_requested_than_exist(corpus):
    with session_scope() as session:
        index = VectorIndex.load(session)
    assert len(index.search(fake_vector("transformer"), k=500)) == 4


def test_exclude_drops_the_query_paper(corpus):
    with session_scope() as session:
        index = VectorIndex.load(session)
    found = [i for i, _ in index.search(fake_vector("transformer attention"), k=4, exclude="W1")]
    assert "W1" not in found


def test_a_snippet_ends_on_a_word(corpus, monkeypatch):
    monkeypatch.setattr(serving, "SNIPPET_CHARS", 20)
    assert serving.snippet(None) == ""
    assert serving.snippet("short one") == "short one"
    cut = serving.snippet("the quick brown fox jumps over the lazy dog")
    assert cut.endswith("...")
    assert "jump" not in cut or cut.count(" ") >= 3
    assert not cut.removesuffix("...").endswith(" ")


def test_matched_terms_are_only_terms_that_are_really_there(corpus):
    abstract = "transformer attention for protein folding"
    assert serving.matched_terms("transformer protein", abstract) == ["transformer", "protein"]
    # Absent terms are not reported.
    assert serving.matched_terms("garden", abstract) == []
    # Deduplicated, and in query order.
    assert serving.matched_terms("protein protein transformer", abstract) == [
        "protein",
        "transformer",
    ]


def test_the_doi_is_preferred_over_the_openalex_link(corpus):
    with session_scope() as session:
        papers = serving.metadata(session, ["W4", "W1"])
    assert serving.paper_url(papers["W4"]) == "https://doi.org/10.1/abc"
    assert serving.paper_url(papers["W1"]) == "https://openalex.org/W1"


def test_metadata_is_chunked_past_the_parameter_cap(db):
    with session_scope() as session:
        for i in range(1100):
            add(session, f"W{i:05d}", "transformer")
    with session_scope() as session:
        found = serving.metadata(session, [f"W{i:05d}" for i in range(1100)])
    assert len(found) == 1100


def service_for() -> SearchService:
    with session_scope() as session:
        index = VectorIndex.load(session)
    return SearchService(session_scope, index=index, quality={"run": "test"})


def test_both_retrievers_answer_the_same_query(corpus):
    result = service_for().search("transformer attention", k=3, retriever="both")

    assert set(result["results"]) == {"lexical", "dense"}
    assert result["results"]["dense"], "dense returned nothing"
    assert result["results"]["lexical"], "lexical returned nothing"
    assert set(result["elapsed_ms"]) == {"lexical", "dense"}


def test_a_lexical_hit_has_no_invented_score_and_a_dense_hit_no_invented_terms(corpus):
    """FTS5 ranks without a score comparable to cosine similarity, and a dense
    model never saw a term. Filling either in would be a decoration that reads
    as an explanation."""
    result = service_for().search("transformer attention", k=3, retriever="both")

    lexical_hit = result["results"]["lexical"][0]
    dense_hit = result["results"]["dense"][0]

    assert lexical_hit["score"] is None
    assert lexical_hit["matched"], "a lexical hit should say which terms matched"
    assert isinstance(dense_hit["score"], float)
    assert dense_hit["matched"] == []


def test_an_empty_query_is_not_a_search(corpus):
    assert service_for().search("   ", k=5)["results"] == {}


def test_status_reports_what_is_actually_loaded(corpus):
    status = service_for().status()
    assert status["retrievers"] == ["lexical", "dense"]
    assert status["index"]["documents"] == 4
    assert status["index"]["dtype"] == "float16"
    assert status["quality"] == {"run": "test"}


def test_without_an_index_lexical_still_serves(corpus):
    service = SearchService(session_scope, index=None)
    assert service.retrievers == ["lexical"]
    result = service.search("transformer", k=3, retriever="both")
    assert "lexical" in result["results"]
    assert "dense" not in result["results"]


def test_quality_is_read_from_a_file_not_computed(tmp_path):
    assert serving.load_quality(tmp_path / "missing.json") == {}
    path = tmp_path / "quality.json"
    path.write_text('{"recall@10": 0.1182}', encoding="utf-8")
    assert serving.load_quality(path)["recall@10"] == 0.1182
