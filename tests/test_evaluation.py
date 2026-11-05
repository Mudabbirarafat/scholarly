"""The harness: what counts as relevant, what is frozen, and what is measured."""

from sqlalchemy import select

from scholarly.db import session_scope
from scholarly.evaluation import (
    build_query_set,
    coverage,
    evaluate,
    qrels,
    recall_at_k,
    reciprocal_rank,
)
from scholarly.models import Citation, EvalQuery, Paper


def corpus(session, papers, edges):
    for identifier, year in papers:
        session.add(
            Paper(id=identifier, title=identifier, abstract=f"abstract for {identifier}", year=year)
        )
    for citing, cited in edges:
        session.add(Citation(citing_id=citing, cited_id=cited))


def test_recall_and_reciprocal_rank_are_what_they_say():
    relevant = {"a", "b", "c", "d"}
    retrieved = ["x", "a", "y", "c"]

    assert recall_at_k(retrieved, relevant, 10) == 0.5  # found a and c of four
    assert recall_at_k(retrieved, relevant, 1) == 0.0  # x is not relevant
    assert reciprocal_rank(retrieved, relevant, 10) == 0.5  # first hit at rank 2
    assert reciprocal_rank(["z"], relevant, 10) == 0.0


def test_relevance_counts_only_references_the_index_can_return(db):
    """85% of what a paper cites is outside an arXiv-only corpus. Counting
    those as misses caps recall by construction, and a capped number looks
    exactly like a bad retriever."""
    with session_scope() as session:
        corpus(
            session,
            [("W1", 2020), ("W2", 2015), ("W3", 2016)],
            [("W1", "W2"), ("W1", "W3"), ("W1", "W_outside")],
        )

    with session_scope() as session:
        assert qrels(session, "W1") == {"W2", "W3"}


def test_coverage_reports_the_ceiling(db):
    with session_scope() as session:
        corpus(session, [("W1", 2020), ("W2", 2015)], [("W1", "W2"), ("W1", "W_out")])

    with session_scope() as session:
        found = coverage(session)

    assert found["citation_edges"] == 2
    assert found["retrievable"] == 1
    assert found["retrievable_pct"] == 50.0
    assert found["discarded_pct"] == 50.0


def test_the_same_seed_chooses_the_same_queries(db):
    """A query set re-derived each run drifts as the corpus grows, and two
    numbers measured a week apart are then incomparable while looking
    identical in kind."""
    with session_scope() as session:
        corpus(
            session,
            [(f"W{i}", 2020) for i in range(1, 21)] + [("W100", 2015)],
            [(f"W{i}", "W100") for i in range(1, 21)],
        )

    with session_scope() as session:
        build_query_set(session, size=5, from_year=2018, to_year=2021)
        first = [r[0] for r in session.execute(select(EvalQuery.paper_id)).all()]

    with session_scope() as session:
        build_query_set(session, size=5, from_year=2018, to_year=2021, rebuild=True)
        second = [r[0] for r in session.execute(select(EvalQuery.paper_id)).all()]

    assert sorted(first) == sorted(second)
    assert len(first) == 5


def test_a_frozen_set_is_not_silently_re_rolled(db):
    """Re-rolling the queries after seeing a result is how a tuning run
    quietly becomes a search for a favourable sample."""
    with session_scope() as session:
        corpus(
            session,
            [(f"W{i}", 2020) for i in range(1, 11)] + [("W100", 2015)],
            [(f"W{i}", "W100") for i in range(1, 11)],
        )

    with session_scope() as session:
        first = build_query_set(session, size=3, from_year=2018, to_year=2021)
        again = build_query_set(session, size=9, from_year=2018, to_year=2021)

    assert first["status"] == "frozen"
    assert again["status"] == "already frozen"
    assert again["queries"] == 3


def test_a_perfect_retriever_scores_one_and_a_useless_one_scores_zero(db):
    with session_scope() as session:
        corpus(
            session,
            [("W1", 2020), ("W2", 2015), ("W3", 2016), ("W9", 2015)],
            [("W1", "W2"), ("W1", "W3")],
        )
        build_query_set(session, size=10, dev_fraction=1.0, from_year=2018, to_year=2021)

    with session_scope() as session:
        perfect = evaluate(session, lambda q, k, x: ["W2", "W3"], split="dev")
        useless = evaluate(session, lambda q, k, x: ["W9"], split="dev")

    assert perfect["queries"] == 1
    assert perfect["recall@10"] == 1.0
    assert perfect["mrr@50"] == 1.0
    assert useless["recall@10"] == 0.0
    assert useless["mrr@50"] == 0.0


def test_the_retriever_is_told_which_paper_to_exclude(db):
    """The harness owns the rule, so every retriever obeys it identically."""
    with session_scope() as session:
        corpus(session, [("W1", 2020), ("W2", 2015)], [("W1", "W2")])
        build_query_set(session, size=10, dev_fraction=1.0, from_year=2018, to_year=2021)

    seen: list[str] = []

    with session_scope() as session:
        evaluate(session, lambda q, k, x: seen.append(x) or ["W2"], split="dev")

    assert seen == ["W1"]
