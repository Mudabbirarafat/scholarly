"""Command line entry points, so a pipeline step is a command and not a notebook."""

import argparse
import asyncio
import json
import logging
import sys

from sqlalchemy import func, select

from scholarly.db import init_db, session_scope
from scholarly.dense import DenseIndex, encode_corpus, encoded_count
from scholarly.evaluation import build_query_set, coverage, evaluate
from scholarly.ingest import ingest_span
from scholarly.lexical import build_index, search
from scholarly.models import Citation, IngestRun, Paper


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

    run_eval = sub.add_parser("eval", help="measure a retriever against the frozen set")
    run_eval.add_argument("--split", choices=("dev", "test"), default="dev")
    run_eval.add_argument("--limit", type=int, default=None)
    run_eval.add_argument("--retriever", choices=("lexical", "dense"), default="lexical")

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

    if args.command == "eval":
        with session_scope() as session:
            if args.retriever == "dense":
                index = DenseIndex.load(session)
                if not len(index):
                    print(json.dumps({"error": "nothing encoded; run `embed` first"}))
                    return 1

                def retriever(query: str, k: int, exclude: str) -> list[str]:
                    return index.search(query, k=k, exclude=exclude)
            else:

                def retriever(query: str, k: int, exclude: str) -> list[str]:
                    return search(session, query, k=k, exclude=exclude)

            result = {
                "retriever": args.retriever,
                "coverage": coverage(session),
                **evaluate(session, retriever, split=args.split, limit=args.limit),
            }
            if args.retriever == "dense":
                result["index_size"] = len(index)
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
