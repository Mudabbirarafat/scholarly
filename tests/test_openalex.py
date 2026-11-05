"""Projecting OpenAlex records: identity, abstracts, and what gets rejected."""

import pytest
from pydantic import ValidationError

from scholarly.sources.openalex import (
    authenticated,
    corpus_filter,
    headers,
    next_cursor,
    parse_work,
    reconstruct_abstract,
    short_id,
    works_url,
)


def test_an_id_has_one_spelling_whatever_came_in():
    assert short_id("https://openalex.org/W2626778328") == "W2626778328"
    assert short_id("W2626778328") == "W2626778328"
    assert short_id("  https://openalex.org/w123  ") == "W123"


def test_something_that_is_not_a_work_id_is_dropped():
    """Better absent than stored as a key nothing can ever match."""
    assert short_id(None) is None
    assert short_id("") is None
    assert short_id("https://doi.org/10.48550/arXiv.1706.03762") is None
    assert short_id("A5023888391") is None  # an author, not a work


def test_documents_and_their_references_end_up_in_the_same_spelling():
    """A regression test for a bug that would produce no error at all.

    The qrels join a paper's references against the documents in the index.
    `referenced_works` arrives as full URLs while `id` reads naturally as the
    bare form - so storing them as they come means the two sides never match,
    every query scores zero relevant documents, and nothing raises. The whole
    evaluation would simply have reported that retrieval was terrible.
    """
    work = parse_work(
        {
            "id": "https://openalex.org/W1",
            "title": "A paper",
            "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        }
    )

    assert work.id == "W1"
    assert work.referenced_works == ["W2", "W3"]
    # The join that the evaluation depends on.
    assert work.id.__class__ is work.referenced_works[0].__class__
    assert all(not ref.startswith("http") for ref in work.referenced_works)


def test_an_abstract_is_rebuilt_in_position_order():
    inverted = {"learning": [1], "Deep": [0], "works": [2]}
    assert reconstruct_abstract(inverted) == "Deep learning works"


def test_a_token_appearing_several_times_lands_in_every_position():
    inverted = {"the": [0, 2], "cat": [1], "hat": [3]}
    assert reconstruct_abstract(inverted) == "the cat the hat"


def test_a_missing_abstract_is_none_rather_than_empty():
    """Absent and empty mean different things, and only one is worth a retry."""
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None
    assert reconstruct_abstract("not a dict") is None


def test_a_malformed_position_list_does_not_lose_the_rest():
    """Live data is not a schema. One bad entry should cost one token."""
    inverted = {"good": [0], "bad": "nonsense", "also": [1]}
    assert reconstruct_abstract(inverted) == "good also"


def test_a_whole_record_is_projected():
    work = parse_work(
        {
            "id": "https://openalex.org/W2626778328",
            "doi": "https://doi.org/10.48550/arxiv.1706.03762",
            "title": "Attention Is All You Need",
            "publication_year": 2017,
            "abstract_inverted_index": {"We": [0], "propose": [1]},
            "referenced_works": ["https://openalex.org/W1632114991"],
        }
    )

    assert work.id == "W2626778328"
    assert work.year == 2017
    assert work.abstract == "We propose"
    assert work.referenced_works == ["W1632114991"]


def test_a_record_with_no_abstract_is_still_a_record():
    """Searchable needs an abstract; being a valid row does not."""
    work = parse_work({"id": "https://openalex.org/W1", "title": "A paper"})
    assert work.abstract is None
    assert work.referenced_works == []


def test_an_unusable_record_is_rejected_rather_than_stored_blank():
    with pytest.raises(ValidationError):
        parse_work({"id": "https://openalex.org/W1", "title": "   "})
    with pytest.raises(ValidationError):
        parse_work({"id": "not-an-id", "title": "A paper"})


def test_documents_and_evaluation_queries_are_filtered_differently():
    """A document only has to be searchable. A query is worthless without the
    reference list that is its ground truth."""
    documents = corpus_filter(2010, 2021, require_references=False)
    queries = corpus_filter(2018, 2021, require_references=True)

    assert "referenced_works_count:>0" not in documents
    assert "referenced_works_count:>0" in queries
    assert "has_abstract:true" in documents
    assert "publication_year:2010-2021" in documents


def test_paging_uses_a_cursor():
    """OpenAlex refuses deep paging past 10,000 results and the corpus is
    several times that, so `page` would stop partway through."""
    first = works_url("x", cursor="*")
    assert "cursor=*" in first and "page=" not in first.replace("per-page=", "")

    nxt = next_cursor({"meta": {"next_cursor": "IlsxNj"}})
    assert nxt == "IlsxNj"
    assert "cursor=IlsxNj" in works_url("x", cursor=nxt)


def test_the_end_of_the_corpus_is_a_missing_cursor():
    assert next_cursor({"meta": {"next_cursor": None}}) is None
    assert next_cursor({"meta": {}}) is None
    assert next_cursor({}) is None


def test_a_key_is_sent_as_a_bearer_token_never_in_the_url(monkeypatch):
    """Both forms work; only one of them stays out of the logs.

    httpx logs full request URLs, and CI output on a public repository is
    public - so a key in the query string is a published key.
    """
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-key-value")

    assert authenticated() is True
    assert headers()["Authorization"] == "Bearer secret-key-value"
    assert "secret-key-value" not in works_url(corpus_filter(2018, 2021, require_references=True))


def test_without_a_key_nothing_is_sent_and_the_caller_can_tell(monkeypatch):
    """Keyless still returns the first page, so the run has to know the
    difference between "no key" and "the key was refused"."""
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    assert authenticated() is False
    assert "Authorization" not in headers()


def test_a_contact_address_is_sent_only_when_configured(monkeypatch):
    """It buys a faster pool, it is deployment configuration, and it must not
    be committed - so an unset variable has to be an ordinary working case."""
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    assert "mailto" not in headers()["User-Agent"]

    monkeypatch.setenv("OPENALEX_MAILTO", "someone@example.com")
    assert "mailto:someone@example.com" in headers()["User-Agent"]
