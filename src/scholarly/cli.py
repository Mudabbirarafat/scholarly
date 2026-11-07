"""Command line entry points, so a pipeline step is a command and not a notebook."""

import argparse
import asyncio
import json
import logging
import sys

from sqlalchemy import func, select

from scholarly import embeddings
from scholarly.bundle import build_bundle
from scholarly.db import init_db, session_scope
from scholarly.dense import encode_corpus, encoded_count
from scholarly.evaluation import build_query_set, coverage, evaluate
from scholarly.hybrid import DEFAULT_K, fuse_search
from scholarly.ingest import ingest_span
from scholarly.lexical import build_index, search
from scholarly.lexical import corpus_size as lexical_corpus_size
from scholarly.models import Citation, IngestRun, Paper
from scholarly.serving import VectorIndex


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    # httpx logs the full request URL at INFO. The API key travels in a header
    # rather than the query string precisely so it cannot leak that way, but a
    # URL log line is also where a key would surface if anyone ever moved it
    # back into the query string - and CI output on a public repository is
    # public. Quiet by default, visible under --verbose.
    logging.getLogger("httpx").setLevel(logging.DEBUG if verbose else logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scholarly")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="walk a slice of the corpus into the warehouse")
    ingest.add_argument("--from-year", type=int, default=2018)
    ingest.add_argument("--to-year", type=int, default=2021)
    ingest.add_argument(
        "--documents-only",
        action="store_true",
        help="do not require references: searchable documents rather than evaluation queries",
    )
    ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after roughly N records per year slice",
    )

    sub.add_parser("status", help="what is in the warehouse")

    index = sub.add_parser("index", help="build the full-text index over abstracts")
    index.add_argument("--rebuild", action="store_true")

    freeze = sub.add_parser("freeze", help="choose and freeze the evaluation query set")
    freeze.add_argument("--size", type=int, default=2000)
    freeze.add_argument("--dev-fraction", type=float, default=0.25)
    freeze.add_argument(
        "--rebuild",
        action="store_true",
        help="re-roll a frozen set; only honest before any result has been seen",
    )

    embed = sub.add_parser("embed", help="encode abstracts into vectors (resumable)")
    embed.add_argument("--limit", type=int, default=None)
    embed.add_argument("--batch-size", type=int, default=32)

    bundle = sub.add_parser("bundle", help="write a compact database for serving")
    bundle.add_argument("--out", default="data/serving.db")
    bundle.add_argument(
        "--limit",
        type=int,
        default=None,
        help="serve this many documents instead of every encoded one",
    )

    run_eval = sub.add_parser("eval", help="measure a retriever against the frozen set")
    run_eval.add_argument("--split", choices=("dev", "test"), default="dev")
    run_eval.add_argument("--limit", type=int, default=None)
    run_eval.add_argument(
        "--retriever", choices=("lexical", "dense", "hybrid"), default="lexical"
    )
    run_eval.add_argument(
        "--fusion-depth",
        type=int,
        default=50,
        help="how deep each retriever is read before fusing (hybrid only)",
    )
    run_eval.add_argument(
        "--allow-partial-index",
        action="store_true",
        help=(
            "compare anyway when the retrievers cover different corpora; "
            "the number is then not a verdict"
        ),
    )
    run_eval.add_argument(
        "--fusion-k",
        type=int,
        default=DEFAULT_K,
        help="RRF damping; 60 is the published default and is not tuned here",
    )
    run_eval.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="vector precision; float16 is what the deployed instance searches",
    )

    args = parser.parse_args(argv)
    _logging(args.verbose)
    init_db(args.database_url)

    if args.command == "ingest":
        summary = asyncio.run(
            ingest_span(
                args.from_year,
                args.to_year,
                require_references=not args.documents_only,
                max_records=args.limit,
            )
        )
        print(json.dumps(summary, indent=2))
        # Anything short of "succeeded" is a non-zero exit, so a pipeline step
        # cannot go on to spend an hour encoding half a corpus. An explicit
        # --limit stays "succeeded", because asking for less and getting it is
        # the limit working rather than the walk breaking.
        return 0 if summary["status"] == "succeeded" else 1

    if args.command == "index":
        with session_scope() as session:
            rows = build_index(session, rebuild=args.rebuild)
        print(json.dumps({"indexed": rows}, indent=2))
        return 0

    if args.command == "freeze":
        with session_scope() as session:
            result = build_query_set(
                session,
                size=args.size,
                dev_fraction=args.dev_fraction,
                rebuild=args.rebuild,
            )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "embed":
        with session_scope() as session:
            result = encode_corpus(session, limit=args.limit, batch_size=args.batch_size)
            result["total_encoded"] = encoded_count(session)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "bundle":
        with session_scope() as session:
            summary = build_bundle(session, args.out, limit=args.limit)
        print(json.dumps(summary, indent=2))
        # An inconsistent bundle is worse than no bundle: it serves, and it
        # serves a corpus whose two retrievers can reach different documents.
        return 1 if summary.get("warning") else 0

    if args.command == "eval":
        with session_scope() as session:
            index = None

            def lexical_retriever(query: str, k: int, exclude: str) -> list[str]:
                return search(session, query, k=k, exclude=exclude)

            def dense_retriever(query: str, k: int, exclude: str) -> list[str]:
                # The query is embedded here rather than inside the index, so
                # the index does one job and the harness measures the same
                # call the request handler makes.
                vector = next(embeddings.encode([query]))
                return [
                    paper_id for paper_id, _ in index.search(vector, k=k, exclude=exclude)
                ]

            if args.retriever in ("dense", "hybrid"):
                index = VectorIndex.load(session, dtype=args.dtype)
                if not len(index):
                    print(json.dumps({"error": "nothing encoded; run `embed` first"}))
                    return 1

                # The two retrievers must cover the same documents or the
                # comparison measures the corpus rather than the retriever.
                # The CI gate already refuses this, but the gate is in the
                # workflow and this command is what people run by hand - so
                # the check belongs here too. It was written after a local
                # ablation compared a complete lexical index against 6,200
                # vectors and read the difference as a result.
                searchable = lexical_corpus_size(session)
                if len(index) < searchable and not args.allow_partial_index:
                    print(
                        json.dumps(
                            {
                                "error": (
                                    f"the dense index holds {len(index):,} of the "
                                    f"{searchable:,} documents the lexical index can reach, "
                                    "so a comparison between them would measure the corpus, "
                                    "not the retriever. Encode the rest, or pass "
                                    "--allow-partial-index if you know the number is not a verdict."
                                ),
                                "dense_index": len(index),
                                "lexical_index": searchable,
                            },
                            indent=2,
                        )
                    )
                    return 1

            if args.retriever == "dense":
                retriever = dense_retriever
            elif args.retriever == "lexical":
                retriever = lexical_retriever
            else:
                # Both arms run against the same index and the same frozen
                # queries, so any difference is the fusion rather than the
                # setup.
                retriever = fuse_search(
                    [lexical_retriever, dense_retriever],
                    depth=args.fusion_depth,
                    k=args.fusion_k,
                )

            result = {
                "retriever": args.retriever,
                "coverage": coverage(session),
                **evaluate(session, retriever, split=args.split, limit=args.limit),
            }
            if index is not None:
                result["lexical_index"] = lexical_corpus_size(session)
                result["index_size"] = len(index)
                result["dtype"] = index.dtype.name
                result["index_mib"] = index.megabytes
            if args.retriever == "hybrid":
                result["fusion"] = {"depth": args.fusion_depth, "k": args.fusion_k}
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "status":
        with session_scope() as session:
            papers = session.scalar(select(func.count()).select_from(Paper))
            citations = session.scalar(select(func.count()).select_from(Citation))
            with_abstract = session.scalar(
                select(func.count()).select_from(Paper).where(Paper.abstract.is_not(None))
            )
            last = session.scalars(
                select(IngestRun).order_by(IngestRun.id.desc()).limit(1)
            ).first()
        print(
            json.dumps(
                {
                    "papers": papers,
                    "with_abstract": with_abstract,
                    "citations": citations,
                    "last_run": None
                    if last is None
                    else {
                        "id": last.id,
                        "status": last.status,
                        "requested": last.requested,
                        "succeeded": last.succeeded,
                        "quarantined": last.quarantined,
                        "error": last.error,
                    },
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
