"""What does halving the index cost in retrieval quality?

The deployed instance stores vectors as float16, which is what makes a
114,581-document index fit beside the encoder on a 512 MB box. The published
recall was computed on float32. That gap has been sitting in the README as
"almost certainly a rounding error", which is exactly the sort of phrase this
project exists to replace with a number.

So: the same frozen query set, the same index, the same code, one difference.

    python scripts/float16_cost.py
    python scripts/float16_cost.py --limit 200

**A partial index does not invalidate this comparison, though it would
invalidate dense-against-BM25.** The gate refuses that one because a smaller
index means fewer documents to find, which has nothing to do with retrieval
quality. Here both arms search exactly the same documents, so whatever
differs is the dtype and nothing else. The absolute recall will be low on a
small index; the delta is the measurement.
"""

import argparse
import json
import sys
import time

import numpy as np
from sqlalchemy import select

from scholarly import embeddings
from scholarly.db import init_db, session_scope
from scholarly.evaluation import evaluate, qrels
from scholarly.models import EvalQuery, Paper
from scholarly.serving import VectorIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None, help="queries to use")
    parser.add_argument("--k", type=int, default=10, help="depth for the agreement check")
    args = parser.parse_args()

    init_db(args.database_url)

    with session_scope() as session:
        started = time.perf_counter()
        half = VectorIndex.load(session, dtype=np.float16)
        full = VectorIndex.load(session, dtype=np.float32)
        if not len(half):
            print("nothing encoded; run `embed` first")
            return 1
        print(
            f"index: {len(half):,} vectors | "
            f"float16 {half.megabytes} MiB, float32 {full.megabytes} MiB | "
            f"loaded in {time.perf_counter() - started:.1f}s\n"
        )

        # One embedding per query, shared by both arms. Encoding twice would
        # add noise from nothing: the query vector is float32 either way, and
        # only the stored document vectors differ.
        cache: dict[str, np.ndarray] = {}

        def vector_for(query: str) -> np.ndarray:
            if query not in cache:
                cache[query] = next(embeddings.encode([query]))
            return cache[query]

        def retriever(index: VectorIndex):
            def search(query: str, k: int, exclude: str) -> list[str]:
                return [
                    paper_id
                    for paper_id, _ in index.search(vector_for(query), k=k, exclude=exclude)
                ]

            return search

        # Warm the cache before either arm runs. The first version of this
        # script did not, so the arm that ran first paid for embedding every
        # query and the second one did not - which made float16 look fourteen
        # times faster than float32 when the only difference measured was a
        # cold cache. Both arms now time the search and nothing else.
        queries_for_warmup = [
            row[0]
            for row in session.execute(
                select(EvalQuery.paper_id)
                .where(EvalQuery.split == args.split)
                .order_by(EvalQuery.paper_id)
            ).all()
        ]
        if args.limit:
            queries_for_warmup = queries_for_warmup[: args.limit]
        warm_started = time.perf_counter()
        for paper_id in queries_for_warmup:
            abstract = session.scalar(select(Paper.abstract).where(Paper.id == paper_id))
            if abstract:
                vector_for(abstract)
        print(
            f"embedded {len(cache):,} queries in "
            f"{time.perf_counter() - warm_started:.1f}s, before timing either arm\n"
        )

        scores = {}
        for label, index in (("float32", full), ("float16", half)):
            began = time.perf_counter()
            scores[label] = evaluate(
                session, retriever(index), split=args.split, limit=args.limit
            )
            scores[label]["seconds"] = round(time.perf_counter() - began, 1)

        # Recall is a blunt instrument for this: it only moves when a changed
        # ranking crosses the k boundary on a relevant document. Comparing the
        # rankings directly says how often anything changed at all.
        queries = queries_for_warmup

        identical = 0
        overlaps: list[float] = []
        worst_gap = 0.0
        compared = 0
        for paper_id in queries:
            abstract = session.scalar(select(Paper.abstract).where(Paper.id == paper_id))
            if not abstract:
                continue
            vector = vector_for(abstract)
            a = full.search(vector, k=args.k, exclude=paper_id)
            b = half.search(vector, k=args.k, exclude=paper_id)
            compared += 1
            if [i for i, _ in a] == [i for i, _ in b]:
                identical += 1
            top_a, top_b = {i for i, _ in a}, {i for i, _ in b}
            overlaps.append(len(top_a & top_b) / max(len(top_a | top_b), 1))
            by_id = dict(b)
            for paper, score in a:
                if paper in by_id:
                    worst_gap = max(worst_gap, abs(score - by_id[paper]))
            qrels(session, paper_id)  # touch the answer key the same way eval does

    report = {
        "queries": compared,
        "k": args.k,
        "float32": {k: v for k, v in scores["float32"].items() if k != "coverage"},
        "float16": {k: v for k, v in scores["float16"].items() if k != "coverage"},
        "identical_top_k": round(identical / compared, 4) if compared else None,
        "mean_jaccard": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
        "largest_score_difference": round(worst_gap, 6),
        "index_megabytes": {"float32": full.megabytes, "float16": half.megabytes},
    }
    print(json.dumps(report, indent=2))

    delta = scores["float16"]["recall@10"] - scores["float32"]["recall@10"]
    print(
        f"\nrecall@10: float32 {scores['float32']['recall@10']} -> "
        f"float16 {scores['float16']['recall@10']}  ({delta:+.4f})"
    )
    print(
        f"{report['identical_top_k']:.1%} of top-{args.k} rankings identical, "
        f"largest score difference {worst_gap:.6f}, "
        f"index {full.megabytes} MiB -> {half.megabytes} MiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
