"""Landing the payload before anything parses it.

Every page is written to disk exactly as it arrived, before validation. The
reason is narrow and practical: when a parser turns out to be wrong - and it
will, because live payloads are not a schema - the fix can be replayed against
what actually broke it, rather than re-walking tens of thousands of records
through someone else's free API to reproduce the failure.

It also separates two failures that look identical from the database side: "the
source did not give us this" and "we failed to read what the source gave us".
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DIR = "data/raw"


def raw_dir(override: str | None = None) -> Path:
    return Path(override or os.environ.get("RAW_DIR") or DEFAULT_DIR).expanduser()


def land(records: list[dict], *, run_id: int, page: int, directory: str | None = None) -> Path:
    """Write one page of records as JSONL. Returns the file written."""
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    target = raw_dir(directory) / f"{stamp}" / f"run-{run_id:06d}"
    target.mkdir(parents=True, exist_ok=True)

    path = target / f"page-{page:05d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def replay(path: str | Path) -> list[dict]:
    """Read a landed page back. The point of landing it in the first place."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
