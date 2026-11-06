"""The HTTP surface.

Mostly about the states a free instance actually spends time in: waking up,
being hammered, and having no published metrics to show. The happy path is
the easy part.
"""

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scholarly import api, embeddings
from scholarly.db import session_scope
from scholarly.dense import encode_corpus
from scholarly.lexical import build_index
from scholarly.models import Paper

AXES = ["transformer", "attention", "garden", "protein"]


def fake_vector(text: str) -> np.ndarray:
    raw = np.array([float(text.lower().count(axis)) for axis in AXES], dtype=np.float32)
    if not raw.any():
        raw = np.ones(len(AXES), dtype=np.float32)
    return raw / np.linalg.norm(raw)


@pytest.fixture(autouse=True)
def stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        embeddings, "encode", lambda texts, **kw: (fake_vector(t) for t in texts)
    )
    # Never pick up a built UI or a real quality file from the working tree.
    monkeypatch.setenv("STATIC_DIR", str(tmp_path / "no-ui"))
    monkeypatch.setenv("QUALITY_FILE", str(tmp_path / "no-quality.json"))


@pytest.fixture
def url(db):
    """A populated warehouse, and the URL the app should open."""
    with session_scope() as session:
        session.add(Paper(id="W1", title="Attention", abstract="transformer attention", year=2017))
        session.add(Paper(id="W2", title="Gardening", abstract="garden soil tomatoes", year=2019))
    with session_scope() as session:
        encode_corpus(session, model_name=embeddings.DEFAULT_MODEL)
        build_index(session, rebuild=True)
    return f"sqlite:///{db / 'test.db'}"


def test_health_is_ready_once_the_index_is_loaded(url):
    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        body = client.get("/api/health").json()

    assert body["warmup"]["state"] == "ready"
    assert body["index"]["documents"] == 2
    assert body["index"]["dtype"] == "float16"
    assert body["retrievers"] == ["lexical", "dense"]


def test_a_search_returns_both_columns(url):
    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        body = client.get("/api/search", params={"q": "transformer attention", "k": 2}).json()

    assert body["results"]["dense"][0]["paper_id"] == "W1"
    assert body["results"]["lexical"][0]["paper_id"] == "W1"
    assert body["results"]["dense"][0]["url"] == "https://openalex.org/W1"
    assert body["results"]["dense"][0]["title"] == "Attention"


def test_the_port_opens_before_the_index_is_ready(url, monkeypatch):
    """A sleeping instance that blocked startup on loading would hand the
    first visitor a gateway timeout. Health has to answer while warming."""
    release = threading.Event()
    original = api.VectorIndex.load

    def slow(session, **kwargs):
        release.wait(timeout=5)
        return original(session, **kwargs)

    monkeypatch.setattr(api.VectorIndex, "load", classmethod(lambda cls, s, **k: slow(s, **k)))

    with TestClient(api.create_app(database_url=url)) as client:
        warming = client.get("/api/health").json()
        assert warming["warmup"]["state"] == "warming"
        assert warming["warmup"]["waiting_seconds"] is not None

        # Dense cannot be served yet, and says so rather than returning
        # nothing and looking like a corpus with no matches.
        refused = client.get("/api/search", params={"q": "transformer", "retriever": "dense"})
        assert refused.status_code == 503
        assert "still loading" in refused.json()["detail"]

        # Lexical does not need the vectors, so it answers immediately.
        lexical = client.get("/api/search", params={"q": "transformer", "retriever": "lexical"})
        assert lexical.status_code == 200
        assert lexical.json()["results"]["lexical"]

        release.set()


def test_a_failed_load_is_reported_not_hidden(url, monkeypatch):
    def explode(cls, session, **kwargs):
        raise RuntimeError("disk went away")

    monkeypatch.setattr(api.VectorIndex, "load", classmethod(explode))

    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        body = client.get("/api/health").json()
        assert body["warmup"]["state"] == "error"
        assert "disk went away" in body["warmup"]["error"]

        refused = client.get("/api/search", params={"q": "transformer", "retriever": "dense"})
        assert refused.status_code == 503
        assert "disk went away" in refused.json()["detail"]


def test_ready_means_the_encoder_is_loaded_too(url, monkeypatch):
    """`ready` is a promise that a dense query will be fast.

    The encoder loads on first use and that first use was measured at 6.7
    seconds. If warming happened after the flag went up, the first visitor in
    that window would wait for it on a page that said it was ready - so the
    warm-up has to be inside the window the flag covers. Making the warm-up
    fail is the cheapest way to prove it runs before `ready` is set.
    """

    def refuse(texts, **kwargs):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(embeddings, "encode", refuse)

    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        body = client.get("/api/health").json()

    assert body["warmup"]["state"] == "error"
    assert "encoder unavailable" in body["warmup"]["error"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": "a"},  # too short to mean anything
        {"q": "transformer", "k": 0},
        {"q": "transformer", "k": 500},  # a whole-corpus dump is not a search
        {"q": "transformer", "retriever": "magic"},
        {},  # no query at all
    ],
)
def test_bad_requests_are_refused(url, params):
    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        assert client.get("/api/search", params=params).status_code == 422


def test_the_rate_limit_bites(url, monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT", 3)
    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        ask = {"q": "transformer", "retriever": "lexical"}
        codes = [client.get("/api/search", params=ask).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_quality_says_it_has_none_rather_than_returning_an_empty_object(url):
    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        response = client.get("/api/quality")
    assert response.status_code == 503
    assert "run the pipeline" in response.json()["detail"]


def test_quality_is_served_when_the_file_is_there(url, tmp_path, monkeypatch):
    path = tmp_path / "quality.json"
    path.write_text('{"run": 42082656593, "dense": {"recall@10": 0.1182}}', encoding="utf-8")
    monkeypatch.setenv("QUALITY_FILE", str(path))

    with TestClient(api.create_app(database_url=url, eager=True)) as client:
        body = client.get("/api/quality").json()

    assert body["dense"]["recall@10"] == 0.1182
