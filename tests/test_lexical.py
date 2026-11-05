"""Turning an abstract into a query, and getting ranked papers back."""

from scholarly.db import session_scope
from scholarly.lexical import build_index, search, to_match_query
from scholarly.models import Paper


def add(session, identifier, title, abstract):
    session.add(Paper(id=identifier, title=title, abstract=abstract))


def test_prose_becomes_a_valid_quoted_query():
    built = to_match_query("Transformer models for machine translation")
    assert '"transformer"' in built
    assert " OR " in built
    # Terms carrying no signal in this corpus are dropped.
    assert '"for"' not in built


def test_query_syntax_in_the_abstract_cannot_break_the_query():
    """FTS5 MATCH takes a query language, not prose. A bare AND, NEAR or quote
    in an abstract is syntax, and pasting prose in raw is a syntax error at
    best and a silently wrong query at worst."""
    built = to_match_query('retrieval AND ranking NEAR "quoted" OR (grouped)')
    assert built.count('"') % 2 == 0
    for term in built.split(" OR "):
        assert term.startswith('"') and term.endswith('"')


def test_an_unusable_query_returns_nothing_rather_than_everything(db):
    with session_scope() as session:
        add(session, "W1", "A paper", "transformers for translation")
        build_index(session, rebuild=True)
        assert search(session, "", k=5) == []
        assert search(session, "a an the of", k=5) == []


def test_the_best_match_comes_first(db):
    with session_scope() as session:
        add(session, "W1", "Attention", "attention transformer sequence modelling translation")
        add(session, "W2", "Gardening", "tomatoes soil compost watering greenhouse")
        add(session, "W3", "Transformers", "transformer attention heads translation sequence")
        build_index(session, rebuild=True)

        found = search(session, "transformer attention translation sequence", k=3)

    assert set(found[:2]) == {"W1", "W3"}
    assert "W2" not in found


def test_the_query_paper_is_excluded_from_its_own_results(db):
    """An abstract always retrieves its own paper first. Counting that would
    hand every query a free correct answer it did not earn."""
    with session_scope() as session:
        add(session, "W1", "Attention", "attention transformer sequence translation")
        add(session, "W2", "Related", "attention transformer sequence translation methods")
        build_index(session, rebuild=True)

        with_self = search(session, "attention transformer sequence translation", k=5)
        without = search(
            session, "attention transformer sequence translation", k=5, exclude="W1"
        )

    assert "W1" in with_self
    assert "W1" not in without
    assert "W2" in without


def test_building_the_index_twice_does_not_duplicate_it(db):
    with session_scope() as session:
        add(session, "W1", "A paper", "transformers for translation")
        first = build_index(session, rebuild=True)
        second = build_index(session)
    assert first == second == 1


def test_papers_without_an_abstract_are_not_indexed(db):
    with session_scope() as session:
        add(session, "W1", "Has one", "transformers for translation")
        session.add(Paper(id="W2", title="Has none", abstract=None))
        rows = build_index(session, rebuild=True)
    assert rows == 1
