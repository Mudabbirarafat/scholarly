# Corpus and ground truth: what was measured, and what it changed

Everything here was measured against the live OpenAlex API before any code was
written. Three of the four findings contradict the original project spec, and
each one would have surfaced in week three — after the pipeline was built and
the evaluation harness had nothing to chew on.

## 1. arXiv reference coverage collapses in 2022

The spec said to take `cs.*` papers "from the last few years". Reference lists
are the ground truth, so a paper without them is useless as a query.

| arXiv CS papers | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|
| with references | 84.0% | 87.6% | 88.5% | 79.3% | **1.2%** | **1.0%** | **0.2%** |

"The last few years" would have yielded 65 usable queries from 2024.

**It is not OpenAlex indexing lag.** Coverage across *all* Computer Science
works is slightly higher in 2024 (62.9%) than in 2020 (58.2%) — the collapse is
specific to arXiv-sourced records, and lands exactly where arXiv began minting
its own DOIs. Before 2022, an arXiv paper generally reached OpenAlex with a DOI
only once a publisher deposited it, and publishers deposit reference lists with
it. After 2022 the record is a bare preprint registration with no references.

**Decision: the labelled query set comes from 2018–2021.** That is 68,837
papers with references, of which 68,244 also carry an abstract — 99.1%.

Recent papers can still be *indexed* as searchable documents. They cannot be
*evaluated* against. Say so rather than quietly restricting the corpus.

## 2. One source, not two

The spec had arXiv OAI-PMH supplying abstracts and OpenAlex supplying
references, joined on an identifier. The join is unnecessary: a single OpenAlex
record carries the id, DOI, title, `abstract_inverted_index` and
`referenced_works` together.

Dropping it removes an ingestion path, a join key that would have been awkward
for pre-2022 papers (they predate the arXiv DOI scheme), and OAI-PMH's
three-second politeness delay.

arXiv OAI-PMH still works and needs no auth, if a cleaner abstract is ever
wanted. It is not needed to start.

## 3. Only 17.7% of references can ever be retrieved

Measured over 20 query papers from 2021 carrying 1,282 references between them:

| Document corpus | References inside it |
|---|---|
| arXiv CS, 2018–2021 | 12.0% |
| arXiv CS, any year | 17.7% |
| arXiv, any field | 18.6% |

The rest are journal articles, conference proceedings and books that an
arXiv-only index will never contain.

**This is the finding that would have wasted the most time.** Scored naively,
`recall@10` is capped at roughly 0.18 by construction — and the natural
reaction to a 0.1 recall is to go and tune the retriever, which would have done
nothing, because the ceiling is in the labels rather than the model.

**Decision: qrels are restricted to references present in the index**, and the
README reports how many were discarded. That leaves about 11 relevant
documents per query, which is a normal size for an IR test set.

**Corollary: the document corpus must reach further back than the query set.**
References point backwards in time, so queries drawn from 2021 need documents
from well before 2021 to have anything to match.

## 4. A free API key is required, not optional

Found by running the ingest for real rather than against fixtures. The first
page of a cursor-paged query works unauthenticated; the second returns:

```
HTTP 401  "Paging past the first page with cursor= needs a free API key."
```

The corpus is 342 pages, so there is no keyless path to it. The key is free
(an account, or an agent can request one against a verified email address) and
raises the rate limit from 10 to 30 requests a second.

**It travels in an `Authorization: Bearer` header, not the `?api_key=` query
parameter that the documentation lists first.** Both work identically, but
httpx logs full request URLs and CI output on a public repository is public,
so a key in a query string is a published key. `OPENALEX_API_KEY` is read from
the environment and never committed.

`OPENALEX_MAILTO` is separate and still worth setting: it identifies the
caller in the User-Agent, which is ordinary API manners.

## 5. The cursor stops early on large result sets

Found by walking the corpus for real, and then only because the run was
checked against what the source claimed.

| Query | Reported | Collected | |
|---|---|---|---|
| 2018–2021 | 68,225 | 33,796 | 49.5%, status "succeeded" |
| 2018 alone | 13,021 | 13,021 | 100% |

The four-year walk served 168 full pages, a 169th with 196 records, and then
returned a cursor with nothing behind it. No error, no 429 — `$0.9827` of the
`$1` daily budget was still unspent. The ceiling sits somewhere above 33,800
and is not documented.

**A walk that stops early is indistinguishable from one that finished**, since
stopping is what finishing looks like. Nothing in the run, the logs or the
database disagreed with "succeeded".

Two consequences, both now in the code:

- **The span is ingested one year at a time.** Each slice is small enough to
  complete, and each is its own run, so a short slice stays visible instead of
  averaging into a total that looks about right.
- **Completeness is asserted, never assumed.** The count the source reports on
  the first page is compared against what arrived; short by more than 1% and
  the run is `partial` with the arithmetic in the error. An explicit `--limit`
  is exempt.

The general lesson is worth more than the workaround: an index quietly holding
half a corpus would have been embedded, evaluated and reported on as though it
were the whole thing — and the numbers would have looked fine, because they
would have been internally consistent.

## What this means for the schedule

The spec's week three — "build the eval harness" — stays exactly where it is.
These findings make it cheaper rather than later: the query set is a filtered
OpenAlex query, the qrels come from the same records, and the in-corpus
restriction is a join against documents already ingested.
