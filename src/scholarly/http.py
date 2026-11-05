"""An HTTP client that behaves itself against a free public API.

OpenAlex is open, unauthenticated and run on someone else's budget. Hammering
it concurrently is both rude and counterproductive - the corpus takes tens of
thousands of records to walk, and a client that gets itself throttled finishes
later than one that paces.

Three behaviours, all of which exist because of how real APIs fail:

* **`Retry-After` is obeyed when given.** Guessing a backoff when the server
  has stated one is just a slower way of being wrong.
* **404 is not retried.** It is an answer, not a failure, and retrying it
  three times turns a missing record into four requests.
* **Backoff is jittered.** Without jitter, every retry from a parallel batch
  lands at the same instant and reproduces the burst that caused the throttle.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import httpx

USER_AGENT = "scholarly/0.1"


class FetchError(Exception):
    """A request that did not produce a usable response."""

    def __init__(self, message: str, *, status: int | None = None, headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.headers = headers or {}


@dataclass
class Fetched:
    payload: Any
    headers: dict


class RateLimitedClient:
    """Bounded concurrency, retries with jittered backoff, Retry-After honoured."""

    def __init__(
        self,
        *,
        concurrency: int = 4,
        retries: int = 3,
        backoff: float = 1.0,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._retries = retries
        self._backoff = backoff
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, url: str, headers: dict[str, str] | None = None) -> Fetched:
        async with self._semaphore:
            return await self._get(url, headers or {})

    async def get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        return (await self.get(url, headers)).payload

    async def _get(self, url: str, headers: dict[str, str]) -> Fetched:
        last: Exception | None = None

        for attempt in range(self._retries + 1):
            try:
                response = await self._client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last = FetchError(f"request failed: {exc}")
            else:
                if response.status_code == 200:
                    try:
                        return Fetched(response.json(), dict(response.headers))
                    except ValueError as exc:
                        # A 200 that is not JSON is usually an error page from
                        # something in front of the API. Not worth retrying.
                        raise FetchError(f"response was not JSON: {exc}", status=200) from exc

                error = FetchError(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    status=response.status_code,
                    headers=dict(response.headers),
                )
                # 404 and the other client errors are answers, not failures.
                # Retrying a missing record three times turns one honest "no"
                # into four requests against someone else's free API.
                retryable = response.status_code >= 500 or response.status_code in (408, 429)
                if not retryable:
                    raise error
                last = error

                wait = self._retry_after(response)
                if wait is not None and attempt < self._retries:
                    await asyncio.sleep(wait)
                    continue

            if attempt < self._retries:
                await asyncio.sleep(self._backoff * (2**attempt) + random.random() * 0.3)

        raise last or FetchError("exhausted retries")

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None
