"""The HTTP surface. Thin on purpose: load once, then answer.

Two things here are shaped by the deployment rather than by taste.

**The port opens before the index is loaded.** A free instance sleeps, and
waking it means reading 114,581 vectors out of SQLite - seconds, not
milliseconds. If startup blocked on that, the first visitor would get a
gateway timeout and conclude the site is broken. Instead the index loads on a
background thread, `/api/health` reports `warming`, and the UI can say so.

**Nothing is computed here that should be measured.** The quality numbers are
read from a file produced by a pipeline run against a frozen query set. A web
process recomputing recall on whatever happens to be in its database would
produce a number that looks just as official and means nothing.
"""

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from scholarly import embeddings
from scholarly.db import init_db, session_scope
from scholarly.serving import SearchService, VectorIndex, ensure_bundle, load_quality

log = logging.getLogger("scholarly.api")

MAX_QUERY_CHARS = 2000
MAX_K = 50
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))


class Warmup:
    """Index loading state, readable from a request handler while it runs."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.ready = False
        self.error: str | None = None
        self.seconds: float | None = None

    @property
    def state(self) -> str:
        if self.error:
            return "error"
        return "ready" if self.ready else "warming"

    def describe(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "load_seconds": self.seconds,
            "waiting_seconds": None if self.ready else round(time.time() - self.started_at, 1),
        }


class Limiter:
    """A small fixed-window limiter, per client address.

    Not a substitute for a real one at the edge. It exists because a public
    endpoint that runs a 150 ms matrix multiply per call should not be a free
    CPU faucet, and because the alternative on this tier is the instance
    falling over.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, who: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.time()
        with self._lock:
            seen = self._hits.setdefault(who, deque())
            while seen and now - seen[0] > 60:
                seen.popleft()
            if len(seen) >= self.per_minute:
                return False
            seen.append(now)
            # Addresses that stopped calling should not accumulate for ever.
            if len(self._hits) > 4096:
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
            return True


@contextmanager
def _session() -> Iterator:
    with session_scope() as session:
        yield session


def create_app(*, database_url: str | None = None, eager: bool = False) -> FastAPI:
    warmup = Warmup()
    limiter = Limiter(RATE_LIMIT)
    service = SearchService(_session, index=None, quality=load_quality())

    def load_index() -> None:
        started = time.perf_counter()
        try:
            with session_scope() as session:
                index = VectorIndex.load(session)

            # The encoder loads on first use, and measured here that first use
            # costs 6.7 seconds. Paying it before `ready` is set is the whole
            # point of the flag: an earlier version warmed the model after
            # announcing readiness, so the first visitor to arrive in that gap
            # waited almost seven seconds for a page that said it was ready.
            if len(index):
                next(embeddings.encode(["warm"]))

            service.index = index
            warmup.seconds = round(time.perf_counter() - started, 1)
            warmup.ready = True
            log.info(
                "ready",
                extra={"documents": len(index), "mib": index.megabytes},
            )
        except Exception as problem:  # noqa: BLE001 - reported, not swallowed
            warmup.error = f"{type(problem).__name__}: {problem}"
            log.exception("index load failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        url = database_url or os.environ.get("DATABASE_URL")
        if url and url.startswith("sqlite:///"):
            # A free instance comes up with the repository and no disk, so the
            # bundle may have to be fetched before anything can be opened.
            fetched = ensure_bundle(os.environ.get("BUNDLE_URL"), url.removeprefix("sqlite:///"))
            log.info("bundle", extra=fetched)
        init_db(database_url)
        if eager:
            load_index()
        else:
            threading.Thread(target=load_index, name="index-load", daemon=True).start()
        yield

    app = FastAPI(
        title="scholarly",
        summary="Semantic search over CS abstracts, with its own retrieval accuracy attached.",
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = [
        origin.strip()
        for origin in os.environ.get(
            "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {**service.status(), "warmup": warmup.describe()}

    @app.get("/api/quality")
    def quality() -> dict:
        if not service.quality:
            raise HTTPException(
                status_code=503,
                detail="no published metrics on this instance; run the pipeline",
            )
        return service.quality

    @app.get("/api/search")
    def search(
        request: Request,
        q: str = Query(..., min_length=2, max_length=MAX_QUERY_CHARS),
        k: int = Query(10, ge=1, le=MAX_K),
        retriever: str = Query("both", pattern="^(both|dense|lexical)$"),
    ) -> dict:
        who = request.client.host if request.client else "unknown"
        if not limiter.allow(who):
            raise HTTPException(status_code=429, detail="too many requests; 30 a minute")

        if retriever in ("both", "dense") and not warmup.ready:
            if warmup.error:
                raise HTTPException(status_code=503, detail=f"index unavailable: {warmup.error}")
            if retriever == "dense":
                raise HTTPException(
                    status_code=503,
                    detail="the index is still loading; lexical search works meanwhile",
                )

        result = service.search(q, k=k, retriever=retriever)
        result["warmup"] = warmup.describe()
        return result

    @app.exception_handler(500)
    def unhandled(_: Request, problem: Exception) -> JSONResponse:
        log.exception("unhandled", exc_info=problem)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    # Serving the built UI from the same process is optional: on Vercel the
    # static side is someone else's job, but one container that serves both is
    # the cheapest way to run this anywhere else.
    static = Path(os.environ.get("STATIC_DIR", "web/dist"))
    if (static / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
        log.info("serving static ui", extra={"dir": str(static)})

    return app


app = create_app()
