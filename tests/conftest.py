import httpx
import pytest

from scholarly.db import init_db
from scholarly.http import RateLimitedClient


@pytest.fixture
def db(tmp_path):
    """A fresh SQLite warehouse per test, on disk so the dialect is real."""
    init_db(f"sqlite:///{tmp_path / 'test.db'}")
    return tmp_path


@pytest.fixture
def mock_client():
    """A client wired to a handler instead of the network.

    Retries and backoff are turned down so a test that exercises a failure
    path does not spend seconds sleeping between attempts.
    """

    def build(handler, **kwargs):
        kwargs.setdefault("retries", 1)
        kwargs.setdefault("backoff", 0.0)
        return RateLimitedClient(transport=httpx.MockTransport(handler), **kwargs)

    return build


def work(identifier: str, *, title="A paper", year=2020, refs=(), abstract="hello world"):
    """An OpenAlex record in the shape the API actually returns."""
    inverted = None
    if abstract is not None:
        inverted = {}
        for position, token in enumerate(abstract.split()):
            inverted.setdefault(token, []).append(position)
    return {
        "id": f"https://openalex.org/{identifier}",
        "doi": f"https://doi.org/10.48550/arxiv.{identifier}",
        "title": title,
        "publication_year": year,
        "abstract_inverted_index": inverted,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
    }


def page(records, *, next_cursor=None, count=None):
    """`count` overrides meta.count, so a short walk can be simulated."""
    return {
        "meta": {"count": len(records) if count is None else count, "next_cursor": next_cursor},
        "results": records,
    }
