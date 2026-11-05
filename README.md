# scholarly

Semantic search over computer-science paper abstracts, where the interesting
part is not the search — it is that the search publishes how often it is right,
and a change that makes it worse does not ship.

> **Status:** week one. The corpus and ground-truth decisions are measured and
> written down, and OpenAlex records project cleanly into the shape the
> pipeline needs — verified against the live API, not only against fixtures.
> Ingestion, embedding, retrieval and the evaluation harness are next.

## The idea

Describe what you are looking for in plain English and get back the papers that
match, each with the passage that matched and why. Then open the quality tab
and see the current `recall@10`, the ablation that produced it, and when the
index was last rebuilt.

Three things separate this from a document-chat demo, and all three are the
project:

1. **It is evaluated** — a held-out query set, `recall@k` and `MRR`, and an
   ablation showing which lever moved them and what it cost in latency.
2. **The evaluation gates the deploy** — a rebuild that drops recall fails CI
   and does not publish.
3. **Every result explains itself** — which passage matched, the lexical and
   dense scores, and what the reranker did to the order.

## Ground truth without hand-labelling

A paper's abstract is the query; the works it cites are the documents that
should come back. OpenAlex serves both, CC0 and without authentication.

That idea survives contact with the data only after three corrections, all
measured before any code was written and all recorded in
[`docs/corpus-decisions.md`](docs/corpus-decisions.md):

- **arXiv reference coverage collapses from 88% to 1% at 2022**, when arXiv
  began minting its own DOIs. "The last few years" would have yielded 65 usable
  queries. The query set is drawn from 2018–2021 instead: **68,244 papers** with
  both an abstract and a reference list.
- **One source, not two.** A single OpenAlex record carries the id, abstract
  and references together, so the planned arXiv OAI-PMH join is unnecessary.
- **Only 17.7% of a paper's references are themselves arXiv CS papers.** Scored
  naively, `recall@10` is capped near 0.18 by construction — a ceiling in the
  labels that looks exactly like a broken retriever. Qrels are restricted to
  references present in the index, and the README will report how many were
  discarded.

## Why the pipeline exists

Serving tiers are small and CI runners are not. Embedding a corpus needs memory
that a free instance does not have, while GitHub Actions gives 16 GB and 4
cores for nothing on a public repository. So the expensive half — ingest,
embed, index, evaluate — runs on a schedule in CI, and the serving side only
embeds the query and searches precomputed vectors. That split is what makes
this a pipeline rather than an inference call inside a request handler.

## Layout

```
src/scholarly/sources/openalex.py   records, abstracts, identity, paging
docs/corpus-decisions.md            what was measured, and what it changed
tests/                              14 tests
```

## Running the tests

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

`OPENALEX_MAILTO` is optional: setting it to a contact address joins OpenAlex's
faster pool. It is deployment configuration and is never committed.
