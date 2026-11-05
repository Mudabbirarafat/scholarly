"""Command line entry points, so a pipeline step is a command and not a notebook."""

import argparse
import asyncio
import json
import logging
import sys

from sqlalchemy import func, select

from scholarly.db import init_db, session_scope
from scholarly.ingest import ingest_corpus
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
    ingest.add_argument("--limit", type=int, default=None, help="stop after roughly N records")

    sub.add_parser("status", help="what is in the warehouse")

    args = parser.parse_args(argv)
    _logging(args.verbose)
    init_db(args.database_url)

    if args.command == "ingest":
        summary = asyncio.run(
            ingest_corpus(
                args.from_year,
                args.to_year,
                require_references=not args.documents_only,
                max_records=args.limit,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] != "failed" else 1

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
