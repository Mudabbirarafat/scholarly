# scholarly

[![CI](https://github.com/Mudabbirarafat/scholarly/actions/workflows/ci.yml/badge.svg)](https://github.com/Mudabbirarafat/scholarly/actions/workflows/ci.yml)

Semantic search over computer-science paper abstracts, where the interesting
part is not the search — it is that the search publishes how often it is right,
and a change that makes it worse does not ship.

> **Status:** the warehouse is full and the ground truth is real.
>
> | | |
> |---|---|
> | documents | **114,601** arXiv CS papers, 2012–2021, every one with an abstract |
> | citation edges | **2,645,782** |
> | evaluation queries | **61,018**, averaging 6.4 relevant documents each |
> | quarantined | 0 |
>
> Ingestion is complete, idempotent, and refuses to call a short walk a
> success. Embedding, retrieval and the evaluation harness are next.

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
  queries. The query set is drawn from 2018–2021 instead: **68,181 papers** with
  both an abstract and a reference list.
- **One source, not two.** A single OpenAlex record carries the id, abstract
  and references together, so the planned arXiv OAI-PMH join is unnecessary.
- **Most of a paper's references are not retrievable**, because they are
  journal and conference publications an arXiv-only index cannot hold. Scored
  naively that caps `recall@10` by construction — a ceiling in the labels that
  looks exactly like a broken retriever. Qrels are restricted to references
  present in the index.
- **The document corpus therefore reaches back further than the query set.**
  References point backwards in time, so 2012–2017 is ingested as documents
  only, with no reference requirement. Measured effect: in-corpus citations
  rose from **9.2% to 14.7%**, which is the ceiling on every recall number this
  project will ever publish.
- **Cursor paging truncates, intermittently and without error.** The corpus is
  walked in monthly slices with a retry, and every walk is checked against the
  count the source reports. 120 slices, 0 incomplete.

## First measured result

BM25 over 114,601 abstracts, 500 held-out dev queries, no model involved:

| | |
|---|---|
| `recall@10` | **0.106** |
| `recall@50` | **0.193** |
| `MRR@50` | **0.176** |
| relevant documents per query | 7.48 |
| latency | p50 **396 ms**, p95 579 ms |

**Quote the ceiling beside the number or the number is meaningless.** Only
14.7% of citation edges point at documents this index holds; the other 85.3%
are journal and conference publications an arXiv-only corpus cannot contain,
and the qrels exclude them. Recall is therefore measured against what a
perfect retriever *could* return, not against every work the paper cited.

Two things this baseline already says:

- **It is a real floor, not a strawman.** Finding one in ten retrievable
  references from an abstract, with no model, is the bar a dense retriever has
  to clear to justify its existence. Without this number, "the embeddings work"
  would be an opinion.
- **Latency is the problem, not recall.** 396 ms at the median is well past
  the 300 ms this is supposed to serve in, and it comes from running a
  40-term OR across the whole index. That is a finding about the baseline,
  and it belongs in the ablation rather than being quietly tuned away.

The test split — 1,500 queries — has not been looked at and will not be until
there is something worth deciding with it.

## Why the pipeline exists

Serving tiers are small and CI runners are not. Embedding a corpus needs memory
that a free instance does not have, while GitHub Actions gives 16 GB and 4
cores for nothing on a public repository. So the expensive half — ingest,
embed, index, evaluate — runs on a schedule in CI, and the serving side only
embeds the query and searches precomputed vectors. That split is what makes
this a pipeline rather than an inference call inside a request handler.

## Layout

```
src/scholarly/sources/openalex.py   records, abstracts, identity, paging, slices
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
