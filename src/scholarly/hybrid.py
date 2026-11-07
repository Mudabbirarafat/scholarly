"""Combining the two retrievers, and finding out whether it is worth it.

BM25 and the dense retriever fail differently. BM25 needs the words and is
helpless when the query says "overparameterised networks" and the paper says
"over-parameterization"; the dense retriever handles that and then loses to
BM25 on a query written in the corpus's own vocabulary, where the exact
string is the right signal. Both failures are visible in the two worked
examples in the README.

Two retrievers with uncorrelated errors are the textbook case for fusing
them. Whether it actually helps *here* is a measurement, which is the point
of this module existing rather than the fusion simply being switched on.

**Reciprocal rank fusion rather than score addition.** The two retrievers do
not produce comparable numbers: FTS5's `bm25()` is an unbounded negative
score, cosine similarity is bounded in [-1, 1]. Adding or averaging them
means inventing a conversion between two scales with no shared meaning, and
the weight chosen would silently decide every tie. RRF reads only the
*rank*, which both retrievers genuinely have, so it needs no such invention.

    score(d) = sum over retrievers of 1 / (k + rank(d))

`k` damps the top of each list: with k=60, first place is worth 1/61 and
second 1/62, a 1.6% difference, so one retriever being confidently wrong
cannot drag a document to the top on its own. It takes agreement, or a very
high rank in one list and presence in the other.
"""

from collections.abc import Callable, Iterable, Sequence

# 60 is the value from the original RRF paper (Cormack et al., 2009) and the
# one every later comparison uses as the baseline. Kept because it is the
# convention, not because it was tuned here - tuning it against the dev set
# and reporting the best result would be fitting to the thing being measured.
DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]],
    *,
    k: int = DEFAULT_K,
) -> list[str]:
    """Fuse ranked lists into one, best first.

    Documents are scored by their position in each list they appear in, and
    absent from a list simply contributes nothing - which is the behaviour
    that matters, because the lists are different lengths and a document
    BM25 never returned should not be penalised as though it were ranked
    last.

    Ties break on the document id, so two replicas fusing the same lists
    produce the same order rather than one that depends on dict iteration.
    """
    if k < 1:
        raise ValueError(f"k damps the head of each list and must be >= 1, got {k}")

    scores: dict[str, float] = {}

    for ranking in rankings:
        counted: set[str] = set()
        for position, document in enumerate(ranking, start=1):
            # A document appearing twice in one list counts once, at its best
            # position. A duplicate within a ranking is a retriever bug, not
            # extra evidence, and letting it score twice would reward it.
            if document in counted:
                continue
            counted.add(document)
            scores[document] = scores.get(document, 0.0) + 1.0 / (k + position)

    return sorted(scores, key=lambda document: (-scores[document], document))


def fuse_search(
    searches: Sequence[Callable[[str, int, str], Sequence[str]]],
    *,
    depth: int = 50,
    k: int = DEFAULT_K,
) -> Callable[[str, int, str], list[str]]:
    """Build one retriever from several.

    Each underlying retriever is asked for `depth` results rather than the
    `k` the caller wants, because fusion can only promote a document that
    somebody returned. Asking both for ten and fusing to ten would mostly
    reproduce whichever list was already better; the value is in a document
    ranked 30th by one retriever and 3rd by the other.
    """

    def search(query: str, want: int, exclude: str) -> list[str]:
        rankings = [list(retrieve(query, max(depth, want), exclude)) for retrieve in searches]
        return reciprocal_rank_fusion(rankings, k=k)[:want]

    return search
