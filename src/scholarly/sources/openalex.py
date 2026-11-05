"""OpenAlex as the single source of papers, abstracts and ground truth.

One record carries everything the pipeline needs: the identifier, the title,
the abstract (as an inverted index) and the works it cites. The original plan
had arXiv OAI-PMH supplying abstracts and OpenAlex supplying references, joined
on an identifier - see `docs/corpus-decisions.md` for why that join turned out
to be unnecessary and awkward.

Two things here are less obvious than they look.

**Abstracts arrive inverted.** OpenAlex stores `{token: [positions]}` rather
than text, for copyright reasons. Reconstruction is a sort, and it is lossy at
the edges: punctuation spacing is approximate and the original line breaks are
gone. That is fine for embedding and for display as a snippet; it is not the
publisher's abstract character for character, and the README should say so.

**Identity has one spelling, chosen here.** `referenced_works` gives full URLs
(`https://openalex.org/W123`) while almost everything else is more readable as
the bare id. The qrels join documents to references, so if the two sides are
stored differently they never match and every query scores zero relevant
documents - with nothing raising an error. `short_id` is applied on the way in,
to both sides, and is the only spelling that reaches the database.
"""

import os
import re
from typing import Any

from pydantic import BaseModel, field_validator

API = "https://api.openalex.org"

# arXiv, as a source. Measured: 1,082,364 works.
ARXIV_SOURCE = "S4306400194"

# OpenAlex's field id for Computer Science.
CS_FIELD = "fields/17"

# Only these fields are fetched. The full record is large and most of it is
# irrelevant; at tens of thousands of records the saving is the difference
# between a workflow that finishes and one that times out.
SELECT = "id,doi,title,publication_year,abstract_inverted_index,referenced_works"

_ID = re.compile(r"(W\d+)\s*$", re.IGNORECASE)


class Work(BaseModel):
    """The projection of an OpenAlex work that the pipeline can use."""

    id: str
    title: str
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    referenced_works: list[str] = []

    model_config = {"extra": "ignore"}

    @field_validator("id", "title")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # An unusable document is rejected here so the caller can record it,
        # rather than landing a blank row that fails much later and further
        # from the cause.
        if not value or not value.strip():
            raise ValueError("empty id or title")
        return value.strip()


def short_id(value: str | None) -> str | None:
    """`https://openalex.org/W123` -> `W123`. The one spelling of an id.

    Returns None for anything that is not an OpenAlex work id, so a malformed
    reference is dropped rather than stored as a key that can never match.
    """
    if not isinstance(value, str):
        return None
    found = _ID.search(value.strip())
    return found.group(1).upper() if found else None


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Rebuild text from OpenAlex's `{token: [positions]}` form.

    None rather than "" when there is nothing to rebuild: an absent abstract
    and an empty one mean different things downstream, and only one of them is
    worth retrying later.
    """
    if not isinstance(inverted, dict) or not inverted:
        return None

    positioned: list[tuple[int, str]] = []
    for token, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, token))

    if not positioned:
        return None

    positioned.sort(key=lambda pair: pair[0])
    text = " ".join(token for _, token in positioned).strip()
    return text or None


def parse_work(document: dict) -> Work:
    """Project one API record. Raises ValidationError if it is unusable."""
    references = [short_id(ref) for ref in document.get("referenced_works") or []]
    return Work(
        id=short_id(document.get("id")) or "",
        title=document.get("title") or "",
        year=document.get("publication_year"),
        doi=document.get("doi"),
        abstract=reconstruct_abstract(document.get("abstract_inverted_index")),
        referenced_works=[ref for ref in references if ref],
    )


def corpus_filter(from_date: str, to_date: str, *, require_references: bool) -> str:
    """The filter string for one slice of the corpus, bounded by dates.

    Dates rather than years because the slice has to be able to get small.
    Cursor paging truncates unpredictably against a large result set - see
    `docs/corpus-decisions.md` - and a month is short enough that a truncated
    slice costs seven pages to redo instead of several hundred.

    `require_references` separates the two populations deliberately: documents
    only need an abstract to be searchable, while a paper used as an evaluation
    query is worthless without the reference list that forms its ground truth.
    """
    parts = [
        f"primary_location.source.id:{ARXIV_SOURCE}",
        f"primary_topic.field.id:{CS_FIELD}",
        f"from_publication_date:{from_date}",
        f"to_publication_date:{to_date}",
        "has_abstract:true",
    ]
    if require_references:
        parts.append("referenced_works_count:>0")
    return ",".join(parts)


def month_slices(from_year: int, to_year: int) -> list[tuple[str, str]]:
    """(first, last) dates for every month in the span, inclusive."""
    from calendar import monthrange

    slices = []
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            last = monthrange(year, month)[1]
            slices.append((f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"))
    return slices


def works_url(filter_string: str, *, cursor: str = "*", per_page: int = 200) -> str:
    """A cursor-paginated page of works.

    Cursor rather than `page`: OpenAlex refuses deep paging past 10,000
    results, and the corpus is several times that.
    """
    return (
        f"{API}/works?filter={filter_string}"
        f"&select={SELECT}&per-page={per_page}&cursor={cursor}"
    )


def authenticated() -> bool:
    return bool(os.environ.get("OPENALEX_API_KEY", "").strip())


def headers() -> dict[str, str]:
    """Identify the client, and authenticate when a key is configured.

    **A key is not optional for this pipeline.** Keyless callers get the first
    page of a cursor-paged query and a 401 on the second, so walking a corpus
    of 68,244 records needs one. It is free.

    The key is sent as a bearer token rather than as the `?api_key=` query
    parameter that the documentation lists first. Both work identically, but a
    key in the URL is a key in every client log line and in CI output - and
    for a public repository, CI output is public.

    `OPENALEX_MAILTO` is separate and still worth setting: it identifies the
    caller in the User-Agent, which is ordinary API manners.
    """
    sent = {"User-Agent": "scholarly/0.1 (https://github.com/Mudabbirarafat/scholarly)"}

    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    if mailto:
        sent["User-Agent"] += f" mailto:{mailto}"

    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if key:
        sent["Authorization"] = f"Bearer {key}"
    return sent


def next_cursor(payload: dict) -> str | None:
    """The cursor for the following page, or None at the end of the corpus."""
    meta: Any = payload.get("meta") or {}
    cursor = meta.get("next_cursor")
    return cursor if isinstance(cursor, str) and cursor else None
