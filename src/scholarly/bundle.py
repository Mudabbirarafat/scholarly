"""A small warehouse that only has to answer queries.

The pipeline's database is 700 MB and holds things serving has no use for:
2.6 million citation edges, the frozen evaluation set, ingest history. A
serving instance needs papers, vectors and a full-text index, and nothing
else - so this writes exactly that into a second SQLite file.

**It is the same schema, not a new format.** The bundle is readable by the
same models, the same `VectorIndex`, the same `lexical.search`. A bespoke
binary layout would be smaller and would also mean the thing being served is
no longer the thing that was tested.

**Vectors stay float32 on disk and become float16 in memory.** Halving them
on disk would mean two spellings of a stored vector, and `from_blob` reading
the wrong one is a bug that produces plausible nonsense rather than an error.
Memory is where the saving matters, and `VectorIndex` makes it there.

**Abstracts are not truncated.** BM25 was measured over whole abstracts; an
index built over the first 320 characters would answer differently from the
retriever whose recall this project publishes, while looking identical.
"""

import logging
import time
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from scholarly import embeddings
from scholarly.lexical import build_index
from scholarly.models import Base, Embedding, Paper

log = logging.getLogger("scholarly.bundle")

BATCH = 2000


def build_bundle(
    source: Session,
    destination: str | Path,
    *,
    model_name: str | None = None,
    limit: int | None = None,
) -> dict:
    """Write a serving database, and report what went into it.

    `limit` takes the documents with the lowest ids, which is a biased slice -
    but a serving subset is not a sample for measurement, it is a demo, and
    the interface says how many documents it is searching. The published
    numbers stay attached to the complete index they were measured on.
    """
    model = model_name or embeddings.DEFAULT_MODEL
    target = Path(destination)
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{target.as_posix()}", future=True)
    Base.metadata.create_all(engine)
    out_sessions = sessionmaker(bind=engine, future=True, expire_on_commit=False)

    started = time.perf_counter()
    # Only papers that have a vector: a document the dense retriever cannot
    # reach is one the two columns would disagree about for a reason that has
    # nothing to do with retrieval.
    wanted = (
        select(Embedding.paper_id)
        .where(Embedding.model == model)
        .order_by(Embedding.paper_id)
    )
    if limit:
        wanted = wanted.limit(limit)
    ids = [row[0] for row in source.execute(wanted).all()]

    written = 0
    with out_sessions() as out:
        for start in range(0, len(ids), BATCH):
            batch = ids[start : start + BATCH]
            papers = source.scalars(select(Paper).where(Paper.id.in_(batch))).all()
            vectors = source.scalars(
                select(Embedding).where(
                    Embedding.model == model, Embedding.paper_id.in_(batch)
                )
            ).all()

            for paper in papers:
                out.add(
                    Paper(
                        id=paper.id,
                        title=paper.title,
                        year=paper.year,
                        doi=paper.doi,
                        abstract=paper.abstract,
                        reference_count=paper.reference_count,
                    )
                )
            for vector in vectors:
                out.add(
                    Embedding(
                        paper_id=vector.paper_id,
                        model=vector.model,
                        dimensions=vector.dimensions,
                        vector=vector.vector,
                    )
                )
            out.flush()
            written += len(papers)
            if start and start % (BATCH * 10) == 0:
                log.info("bundling", extra={"written": written, "of": len(ids)})
        out.commit()

    with out_sessions() as out:
        rows = build_index(out, rebuild=True)
        out.commit()
        papers = out.scalar(select(func.count()).select_from(Paper))
        vectors = out.scalar(select(func.count()).select_from(Embedding))

    # This file is committed and downloaded, so its size is a feature rather
    # than an implementation detail. Building it leaves free pages behind -
    # inserts, then a dropped and rebuilt full-text index - and VACUUM gives
    # them back. It cannot run inside a transaction, hence the autocommit.
    before = target.stat().st_size
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.exec_driver_sql("VACUUM")

    engine.dispose()
    size = target.stat().st_size
    if before != size:
        log.info(
            "vacuumed",
            extra={"before_mib": round(before / 1048576, 1), "after_mib": round(size / 1048576, 1)},
        )

    summary = {
        "path": str(target),
        "papers": papers,
        "vectors": vectors,
        "indexed": rows,
        "model": model,
        "megabytes": round(size / 1048576, 1),
        "seconds": round(time.perf_counter() - started, 1),
    }
    log.info("bundle written", extra=summary)

    if papers != vectors:
        # Every paper in the bundle was chosen because it has a vector, so a
        # mismatch means a half-written file rather than a smaller one.
        summary["warning"] = (
            f"{papers:,} papers but {vectors:,} vectors; the bundle is inconsistent"
        )
    return summary
