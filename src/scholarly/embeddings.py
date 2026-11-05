"""Turning abstracts into vectors, on a CPU, without PyTorch.

**This is the step the architecture is built around.** PyTorch alone is about
800 MB and a free serving instance has 512 MB, so a transformer cannot be
loaded at request time. GitHub Actions gives 16 GB and 4 cores for nothing on
a public repository. So encoding happens in CI and the serving side only ever
embeds the one short query it was asked about.

`fastembed` runs a small sentence model through ONNX Runtime: no torch, no
CUDA, a few hundred megabytes resident. The default here is 384-dimensional,
which is the number the storage arithmetic in `docs/corpus-decisions.md`
assumes - 114,601 abstracts at 384 float32 dimensions is 176 MB, and half
that stored as float16.

**One vector per abstract, not per chunk.** Abstracts are short enough to
embed whole, and chunking one into three triples the index for a document
whose whole point is that it is already a summary. Whether that is the right
call is itself an ablation row rather than an assumption.
"""

import logging
import os
from collections.abc import Iterable, Iterator

import numpy as np

log = logging.getLogger("scholarly.embeddings")

# Small, English, 384 dimensions, and the quantised ONNX build is about
# 130 MB - which matters because this also has to run inside a CI job.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384

# Vectors are stored float32 and compared in float32. float16 halves the
# index and is worth measuring, but it belongs in the ablation with a recall
# number beside it rather than being adopted because it sounds thrifty.
DTYPE = np.float32

_model = None


def model(name: str | None = None):
    """Load the encoder once per process. Loading is the expensive part."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # imported late: it pulls ONNX in

        chosen = name or os.environ.get("EMBEDDING_MODEL") or DEFAULT_MODEL
        log.info("loading encoder", extra={"model": chosen})
        _model = TextEmbedding(model_name=chosen)
    return _model


def encode(texts: Iterable[str], *, batch_size: int = 32) -> Iterator[np.ndarray]:
    """Encode texts to unit-length vectors, in batches.

    Normalised on the way out so that similarity is a dot product rather than
    a cosine computed per comparison - which, over a hundred thousand
    documents and a hundred queries, is the difference between a matrix
    multiply and a loop.
    """
    for vector in model().embed(list(texts), batch_size=batch_size):
        array = np.asarray(vector, dtype=DTYPE)
        norm = np.linalg.norm(array)
        yield array / norm if norm else array


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


def stack(blobs: Iterable[bytes]) -> np.ndarray:
    """A (n, dimensions) matrix from stored vectors, for one matmul per query."""
    rows = [from_blob(blob) for blob in blobs]
    if not rows:
        return np.zeros((0, DIMENSIONS), dtype=DTYPE)
    return np.vstack(rows)
