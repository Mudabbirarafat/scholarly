# scholarly

[![CI](https://github.com/Mudabbirarafat/scholarly/actions/workflows/ci.yml/badge.svg)](https://github.com/Mudabbirarafat/scholarly/actions/workflows/ci.yml)

Semantic search over computer-science paper abstracts, where the interesting
part is not the search — it is that the search publishes how often it is right,
and a change that makes it worse fails the build.

## The result

Dense retrieval beats BM25 on the same corpus, the same frozen queries and the
same ground truth. 500 held-out dev queries, 7.48 relevant documents each,
none skipped:

| | BM25 (FTS5) | dense (bge-small) | |
|---|---|---|---|
| `recall@10` | 0.1059 | **0.1182** | +11.6% |
| `recall@50` | 0.1930 | **0.2422** | +25.5% |
| `MRR@50` | 0.1753 | **0.2214** | +26.3% |
| latency p50 | 194.1 ms | **133.3 ms** | faster |
| latency p95 | 236.8 ms | **169.6 ms** | faster |
| documents searched | 114,581 | 114,581 | equal footing |

Every number on this page comes from one run —
[pipeline #3](https://github.com/Mudabbirarafat/scholarly/actions/runs/42082656593),
commit `27748ed`, 4h 51m 50s end to end — and its database is downloadable as
a 376 MB artifact. OpenAlex is live, so counts drift by a few dozen between
runs; tagging numbers with the run that produced them is cheaper than
pretending they are constants.

Two readings of that table matter more than the headline:

**The gap widens with depth.** +11.6% at k=10, +25.5% at k=50, MRR up 26%.
That is the shape of a retriever finding papers BM25 misses entirely, not one
reshuffling BM25's top ten. Vocabulary mismatch is exactly where lexical
search fails, and that is what the gap is made of.

**Quote the ceiling beside the number or the number is meaningless.** Only
14.7% of this corpus's 2,644,616 citation edges point at a document the index
holds; the other 85.3% are journal and conference publications an arXiv-only
corpus cannot contain, and the qrels exclude them. `recall@10` of 0.118 is
measured against what a perfect retriever *could* return. Scored against all
2.6M edges the same system would report a smaller, more flattering-looking
number that meant nothing.

The test split — 1,500 queries — has not been looked at and will not be until
there is something worth deciding with it.

### A correction to the earlier baseline

An earlier version of this file reported BM25 at `p50 396 ms` and concluded
that latency, not recall, was the baseline's problem. That was measured on a
development laptop. On the CI runner the identical query set runs at 194 ms,
and dense at 133 ms. The finding was a property of the machine, not of the
retriever, and it is corrected here rather than quietly overwritten.

## The idea

A paper's abstract is the query; the works it cites are the documents that
should come back. OpenAlex serves both, CC0 and without authentication, so the
ground truth costs nothing and nobody hand-labels anything.

Three things separate this from a document-chat demo:

1. **It is evaluated** — a frozen held-out query set, `recall@k` and `MRR`,
   and a coverage figure published beside every result.
2. **The evaluation gates the deploy** — a rebuild whose dense retriever does
   not beat the lexical floor exits non-zero and does not publish.
3. **The guards are the engineering** — see below. Every significant bug in
   this project was caught by an assertion, not by someone noticing.

## Ground truth without hand-labelling

The idea above survives contact with the data only after five corrections, all
measured before the code that depends on them, all recorded in
[`docs/corpus-decisions.md`](docs/corpus-decisions.md):

- **arXiv reference coverage collapses from 88% to 1% at 2022**, when arXiv
  began minting its own DOIs. "The last few years" would have yielded 65
  usable queries. The query population is 2018–2021 instead: **68,204** papers
  with both an abstract and a reference list.
- **One source, not two.** A single OpenAlex record carries the id, abstract
  and references together, so the planned arXiv OAI-PMH join is unnecessary.
- **Most of a paper's references are not retrievable.** Scored naively that
  caps `recall@10` by construction — a ceiling in the labels that looks exactly
  like a broken retriever. Qrels are restricted to references present in the
  index.
- **The document corpus reaches further back than the query set.** References
  point backwards in time, so 2012–2017 is ingested as documents only, with no
  reference requirement: **46,377** more papers. Measured effect: in-corpus
  citations rose from **9.2% to 14.7%**, which is the ceiling on every recall
  number this project will ever publish.
- **Cursor paging truncates, intermittently and without error.** The corpus is
  walked in monthly slices with a retry, and every walk is checked against the
  count the source reports. 120 slices, 0 incomplete, 0 quarantined.

**The evaluation set is frozen.** 2,000 queries, drawn by a seeded hash from
the 47,813 papers that have both an abstract and at least one in-corpus
reference, split 500 dev / 1,500 test and stored. `freeze` refuses to re-roll a
set that already exists unless explicitly told to, because re-rolling after
seeing a result is not sampling — it is searching for a favourable sample.

## What the guards caught

Three bugs, none of which announced itself. Each one looked like a green build.

**A walk that stopped at 49.5% reported `succeeded`.** A paging walk that ends
early is byte-identical to one that finished — same status, same absence of
errors, just fewer rows. Ingest now compares what it stored against the count
the source reports and fails the step on a short walk. That check then caught
the real cause, which was not the size ceiling first diagnosed but intermittent
upstream truncation, and it is why the corpus is walked in months with a retry.

**Two runs encoded a fifth of the corpus and still went green.** A partial
index searches fewer documents than BM25 does, so dense loses for a reason that
has nothing to do with retrieval quality. The gate now refuses to deliver a
verdict unless `index_size` covers the corpus — it prints both numbers, says
plainly that the comparison is indicative, and stands down. In this run it
judged for the first time, because the index was complete.

**"Encode everything" was impossible to ask for.** GitHub substitutes a
`workflow_dispatch` input's declared default whenever the box is left blank, so
a default of `20000` re-requested 20,000 every time the field was cleared — and
no value could have meant *all*, since any non-empty string is truthy in the
expression that built the flag. 56 minutes of encoding, twice, before the
coverage guard above made it visible. The default is now the empty string.

The pattern is the point: a system that measures itself needs the measurement
to be falsifiable, or a bug in the measurement reads as a result.

## Why the pipeline exists

PyTorch alone is about 800 MB and a free serving instance has 512 MB, so a
transformer cannot be loaded at request time. GitHub Actions gives 16 GB and
4 cores for nothing on a public repository. So the expensive half — ingest,
index, freeze, encode, evaluate — runs on a schedule in CI, and the serving
side only embeds the one short query it was asked about and searches
precomputed vectors.

Measured, so the split is a decision rather than a hunch: encoding 114,581
abstracts took **4h 32m** at 7.15 documents a second on a CI runner, against
8.3 hours on the development laptop. `fastembed` runs `BAAI/bge-small-en-v1.5`
through ONNX Runtime — 384 dimensions, no torch, no CUDA, about 130 MB on
disk. The vectors are 168 MiB, and because they are unit length, similarity
over the whole corpus is one matrix multiply.

## The interface

Both retrievers, side by side, answering the same query. That is the whole
design: the claim this project makes is a comparison, so the page shows the
comparison rather than asking anyone to take it on faith.

Each column says what it can and cannot tell you. BM25 highlights the terms it
matched and shows no score, because FTS5's rank is not comparable to a cosine
similarity. The dense column shows the similarity and highlights nothing,
because the model never saw a term — highlighting words there would be a
decoration that reads as an explanation.

Underneath, the page reports what it is actually searching. When the deployed
instance holds fewer documents than the published measurement was taken on, it
says so in a banner, which is the same discipline the CI gate applies when it
refuses to compare a partial index against BM25.

Serving is sized by measurement rather than estimate
(`scripts/serving_budget.py`, reproducible anywhere):

```
python + numpy                        30.7 MB
encoder loaded                       203.0 MB
float16 index beside it (114,581)    288.6 MB   peak 324.6
search                               291.0 MB   150 ms
```

291 MB against a 512 MB instance, which decides three things: vectors are
float16, scores are computed in chunks rather than upcasting the whole matrix
into a 168 MiB temporary, and the service runs one worker — two would each
load their own encoder and index, and 580 MB on a 512 MB box is an
out-of-memory kill rather than a slow page.

[`DEPLOY.md`](DEPLOY.md) has the rest: UI on Vercel, API on Render, and why
that split rather than putting both in one place.

## What is not built yet

Stated plainly, because the claims above are the ones this repository can
currently support:

- **Not deployed yet.** It runs locally, end to end, and the deployment
  configuration is written and explained — but no URL exists to hand anyone.
- **No ablation.** Reciprocal-rank fusion over both retrievers, a cross-encoder
  reranker and float16 vectors are one row each in a table that does not exist
  yet. The harness takes any `search_fn`, so each is a measurement rather than
  a rewrite.
- **float16 is served but measured in float32.** The published recall was
  computed on exact vectors while the instance searches halved ones. The
  difference is almost certainly a rounding error, and "almost certainly" is
  the sort of phrase this project exists to replace with a number.
- **No migrations.** Schema comes from `create_all`; the warehouse is rebuilt
  from source rather than migrated.

## Layout

```
src/scholarly/sources/openalex.py   records, abstracts, identity, paging, slices
src/scholarly/ingest.py             the walk, the retry, the completeness check
src/scholarly/models.py             papers, citations, vectors, frozen queries
src/scholarly/lexical.py            SQLite FTS5 and BM25
src/scholarly/embeddings.py         fastembed over ONNX, no torch
src/scholarly/dense.py              encode once, resumably; rank by dot product
src/scholarly/evaluation.py         frozen queries, qrels, recall/MRR, coverage
src/scholarly/serving.py            float16 index, chunked scoring, one query in
src/scholarly/bundle.py             the small database a deployed instance reads
src/scholarly/api.py                health, search, and warming up honestly
src/scholarly/cli.py                one command per pipeline step
web/                                React UI: both retrievers, side by side
scripts/serving_budget.py           what serving costs in memory, measured
.github/workflows/pipeline.yml      ingest → index → freeze → encode → eval → gate
docs/corpus-decisions.md            what was measured, and what it changed
tests/                              91 tests
```

## Running it

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

The pipeline is the same commands CI runs, bounded so they finish on a laptop:

```bash
python -m scholarly.cli ingest --from-year 2020 --to-year 2020 --limit 2000
python -m scholarly.cli index --rebuild
python -m scholarly.cli freeze --size 200
python -m scholarly.cli embed --limit 2000 --batch-size 32
python -m scholarly.cli eval --split dev --retriever dense
```

Then the interface, which needs both halves running:

```bash
python -m scholarly.cli bundle --out data/serving.db --limit 5000
DATABASE_URL=sqlite:///data/serving.db python -m uvicorn scholarly.api:app --port 8000
cd web && npm install && npm run dev
```

`OPENALEX_API_KEY` and `OPENALEX_MAILTO` are optional and join a faster pool;
the key travels in a header rather than the query string, because `httpx` logs
full URLs and CI output on a public repository is public. Both are deployment
configuration and neither is committed.
