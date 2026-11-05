"""BM25 over abstracts, using SQLite's own full-text index.

The first retrieval baseline, and deliberately the dullest one available. It
needs no model, no extra service and no memory beyond what SQLite already
uses, so the evaluation harness can produce an honest number before anything
clever exists to compare against. A dense retriever that cannot beat this is
not worth deploying, and without this there is nothing to say that with.

**An abstract cannot be handed to FTS5 as a query.** MATCH takes a query
language, not prose: punctuation, hyphens and bare words like `AND` or `NEAR`
are syntax, and a 200-word abstract pasted in raw is a syntax error at best
and a silently wrong query at worst. `to_match_query` reduces the abstract to
quoted terms joined by OR, which is both valid and closer to what the
retrieval is actually asking - find documents sharing this vocabulary.
"""

import re
from collections import Counter

from sqlalchemy import text
from sqlalchemy.orm import Session

FTS_TABLE = "papers_fts"

# Terms carrying no retrieval signal in this corpus. Not a general stopword
# list - "paper", "results" and "approach" appear in almost every abstract
# here, so they cost a query slot and return everything.
NOISE = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the",
    "their", "then", "these", "this", "to", "we", "which", "with", "paper", "papers",
    "approach", "results", "result", "show", "shows", "method", "methods", "propose",
    "proposed", "using", "used", "use", "based", "new", "also", "however", "such",
    "more", "most", "both", "but", "not", "than", "they", "when", "where", "while",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+-]{2,}")


def to_match_query(prose: str, *, max_terms: int = 40) -> str:
    """Reduce prose to an FTS5 MATCH expression.

    Terms are quoted so that nothing in the text can be read as query syntax,
    and capped because an unbounded OR over a long abstract matches most of
    the corpus and takes proportionally longer to do it.
    """
    counts = Counter(
        word.lower() for word in _WORD.findall(prose or "") if word.lower() not in NOISE
    )
    terms = [term for term, _ in counts.most_common(max_terms)]
    # Doubling an inner quote is FTS5's own escape; a term like `"` would
    # otherwise end the quoted string and corrupt the rest of the query.
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def build_index(session: Session, *, rebuild: bool = False) -> int:
    """Create and fill the full-text index. Returns the number of rows."""
    if rebuild:
        session.execute(text(f"DROP TABLE IF EXISTS {FTS_TABLE}"))

    session.execute(
        text(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5("
            "  paper_id UNINDEXED, title, abstract,"
            # Porter stemming so "learning" and "learned" meet, which matters
            # more here than exact-form precision.
            "  tokenize='porter unicode61'"
            ")"
        )
    )
    existing = session.execute(text(f"SELECT COUNT(*) FROM {FTS_TABLE}")).scalar() or 0
    if existing and not rebuild:
        return existing

    session.execute(
        text(
            f"INSERT INTO {FTS_TABLE}(paper_id, title, abstract) "
            "SELECT id, title, abstract FROM papers WHERE abstract IS NOT NULL"
        )
    )
    return session.execute(text(f"SELECT COUNT(*) FROM {FTS_TABLE}")).scalar() or 0


def search(session: Session, query: str, *, k: int = 10, exclude: str | None = None) -> list[str]:
    """The k best matching paper ids, best first.

    `exclude` drops the query's own paper. When an abstract is the query, the
    paper it came from is always the top hit, and counting that as a success
    would inflate every metric by a free correct answer it did not earn.
    """
    match = to_match_query(query)
    if not match:
        return []

    rows = session.execute(
        text(
            f"SELECT paper_id FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :q "
            # bm25() is negative in FTS5, and more negative is a better match.
            f"ORDER BY bm25({FTS_TABLE}) LIMIT :k"
        ),
        {"q": match, "k": k + (1 if exclude else 0)},
    ).all()

    found = [row[0] for row in rows if row[0] != exclude]
    return found[:k]
