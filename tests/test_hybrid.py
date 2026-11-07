"""Reciprocal rank fusion.

Pure, so it runs without an index, a model or a database - which is what
makes it worth testing properly rather than only observing the final recall
number and hoping the fusion is the reason it moved.
"""

import pytest

from scholarly.hybrid import DEFAULT_K, fuse_search, reciprocal_rank_fusion


def test_agreement_wins():
    """A document both retrievers like beats one that only one of them
    ranked first - which is the entire reason to fuse."""
    lexical = ["a", "shared", "b"]
    dense = ["c", "shared", "d"]

    assert reciprocal_rank_fusion([lexical, dense])[0] == "shared"


def test_a_document_only_one_retriever_found_still_ranks():
    # Absent from a list contributes nothing rather than counting as last
    # place. The lists are different lengths, and penalising absence would
    # punish the dense retriever for everything BM25 never returned.
    fused = reciprocal_rank_fusion([["only-lexical"], ["only-dense"]])
    assert set(fused) == {"only-lexical", "only-dense"}


def test_rank_matters_more_than_list_order():
    lexical = ["first", "second"]
    dense = ["second", "first"]
    # Perfectly symmetric input: the tiebreak is the document id, so the
    # result is deterministic rather than dependent on dict ordering.
    assert reciprocal_rank_fusion([lexical, dense]) == ["first", "second"]
    assert reciprocal_rank_fusion([dense, lexical]) == ["first", "second"]


def test_k_decides_whether_agreement_beats_a_single_first_place():
    """`k` is the only knob, and this is what it does.

    "alone" is first in one list and absent from the other. "both" is fourth
    in each. Small k makes first place overwhelming; large k flattens the
    positions until appearing twice is worth more.
    """
    lists = [
        ["alone", "x", "y", "both"],
        ["other", "p", "q", "both"],
    ]

    # k=1: alone scores 1/2, both scores 2/5.
    assert reciprocal_rank_fusion(lists, k=1)[0] == "alone"
    # k=1000: alone scores 1/1001, both scores 2/1004.
    assert reciprocal_rank_fusion(lists, k=1000)[0] == "both"


def test_a_duplicate_inside_one_ranking_counts_once():
    # Scoring it twice would let a buggy retriever promote a document by
    # repeating it.
    once = reciprocal_rank_fusion([["a", "b"], ["c"]])
    twice = reciprocal_rank_fusion([["a", "b", "a"], ["c"]])
    assert once == twice


def test_empty_input_is_an_empty_ranking():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_must_be_sensible():
    with pytest.raises(ValueError, match="must be >= 1"):
        reciprocal_rank_fusion([["a"]], k=0)


def test_default_k_is_the_published_one():
    # Not tuned against the dev set: tuning it there and reporting the best
    # result would be fitting to the thing being measured.
    assert DEFAULT_K == 60


def test_fuse_search_asks_each_retriever_for_more_than_it_returns():
    """Fusion can only promote a document somebody returned, so each
    retriever is asked to greater depth than the caller wants."""
    asked: list[int] = []

    def retriever(name: str):
        def search(query: str, k: int, exclude: str) -> list[str]:
            asked.append(k)
            return [f"{name}-{i}" for i in range(k)]

        return search

    fused = fuse_search([retriever("l"), retriever("d")], depth=50)
    found = fused("anything", 5, "")

    assert asked == [50, 50]
    assert len(found) == 5


def test_fuse_search_honours_a_request_deeper_than_the_fusion_depth():
    asked: list[int] = []

    def search(query: str, k: int, exclude: str) -> list[str]:
        asked.append(k)
        return [f"d{i}" for i in range(k)]

    fuse_search([search], depth=10)("q", 80, "")
    assert asked == [80]


def test_fuse_search_passes_the_exclusion_through():
    """The query's own paper must stay excluded, or every metric gains a
    free correct answer it did not earn."""
    seen: list[str] = []

    def search(query: str, k: int, exclude: str) -> list[str]:
        seen.append(exclude)
        return ["a"]

    fuse_search([search, search])("q", 5, "W123")
    assert seen == ["W123", "W123"]
