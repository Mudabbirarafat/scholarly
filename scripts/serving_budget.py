"""What does serving this actually need in memory?

The architecture rests on one number: a free serving instance has 512 MB, and
the whole reason encoding happens in CI is that a transformer does not fit
beside the data. That claim has so far been an estimate written in a
docstring, which is not the same as a measurement - so this measures it.

Three stages, each runnable alone so the costs can be attributed:

    python scripts/serving_budget.py encoder
    python scripts/serving_budget.py index --documents 114581
    python scripts/serving_budget.py both  --documents 114581

Reported as resident set size, which is what an instance is killed for, not
as Python heap size - most of the cost here is ONNX Runtime's arenas and
numpy buffers, and `tracemalloc` cannot see either.
"""

import argparse
import ctypes
import gc
import sys
import time

import numpy as np

DIMENSIONS = 384
QUERY = "contrastive pretraining for sentence embeddings without labels"


def _rss_peak_bytes() -> tuple[int, int]:
    """(current, peak) resident bytes.

    A measurement tool that returns 0 when it cannot read the counter is worse
    than one that fails: the first version of this reported `0.0 MB` for the
    encoder and looked like very good news. Every path here either returns a
    real reading or raises.
    """
    try:
        import os

        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        current = info.rss
        peak = getattr(info, "peak_wset", current)
        if current <= 0:
            raise RuntimeError("psutil reported a non-positive RSS")
        return current, peak
    except ImportError:
        pass

    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.kernel32.K32GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            ctypes.WinDLL("psapi").GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
        if not counters.WorkingSetSize:
            raise RuntimeError("GetProcessMemoryInfo returned an empty counter block")
        return counters.WorkingSetSize, counters.PeakWorkingSetSize

    current = peak = 0
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) * 1024
    if not current:
        raise RuntimeError("could not read a resident-set size on this platform")
    return current, peak


def report(label: str) -> float:
    current, peak = _rss_peak_bytes()
    print(f"{label:<44} {current / 1048576:7.1f} MB   (peak {peak / 1048576:.1f})")
    return current / 1048576


def stage_encoder() -> None:
    report("python + numpy")
    from fastembed import TextEmbedding

    report("fastembed imported")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    report("encoder loaded")

    started = time.perf_counter()
    vector = np.array(next(iter(model.embed([QUERY]))), dtype=np.float32)
    first = time.perf_counter() - started
    report("one query embedded")

    started = time.perf_counter()
    for _ in range(20):
        next(iter(model.embed([QUERY])))
    warm = (time.perf_counter() - started) / 20
    report("after 20 more queries")
    print(f"\nquery embedding: first {first * 1000:.0f} ms, warm {warm * 1000:.0f} ms")
    print(f"vector: {vector.shape[0]} dimensions, norm {np.linalg.norm(vector):.4f}")


def chunked_scores(matrix: np.ndarray, query: np.ndarray, chunk: int) -> np.ndarray:
    """Dot product against a float16 index without materialising it as float32.

    `matrix @ query` on a float16 matrix upcasts the whole matrix, which for
    this corpus is a 168 MiB temporary on an instance with 512 MB. A chunk at
    a time costs `chunk * 384 * 4` bytes instead.
    """
    scores = np.empty(matrix.shape[0], dtype=np.float32)
    for start in range(0, matrix.shape[0], chunk):
        block = matrix[start : start + chunk]
        scores[start : start + chunk] = block.astype(np.float32) @ query
    return scores


def stage_index(documents: int, *, naive: bool = False) -> None:
    report("python + numpy")
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((documents, DIMENSIONS), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    report(f"float32 index built ({documents:,} x {DIMENSIONS})")

    half = matrix.astype(np.float16)
    del matrix
    gc.collect()
    report("float16 index, float32 freed")

    query = rng.standard_normal(DIMENSIONS).astype(np.float32)
    query /= np.linalg.norm(query)

    started = time.perf_counter()
    scores = chunked_scores(half, query, chunk=8192)
    chunked_ms = (time.perf_counter() - started) * 1000
    report("chunked search done")

    top = np.argpartition(-scores, 9)[:10]
    report("top-10 selected")

    if naive:
        started = time.perf_counter()
        naive_scores = half.astype(np.float32) @ query
        naive_ms = (time.perf_counter() - started) * 1000
        report("naive whole-matrix upcast")
        print(f"\nagreement: {np.allclose(scores, naive_scores, atol=1e-3)}")
        print(f"naive {naive_ms:.0f} ms vs chunked {chunked_ms:.0f} ms")
    else:
        print(f"\nchunked search {chunked_ms:.0f} ms over {documents:,} documents")
    print(f"best score {scores[top].max():.4f}")


def stage_both(documents: int) -> None:
    report("python + numpy")
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    report("encoder loaded")

    rng = np.random.default_rng(0)
    half = np.empty((documents, DIMENSIONS), dtype=np.float16)
    for start in range(0, documents, 16384):
        block = rng.standard_normal((min(16384, documents - start), DIMENSIONS), dtype=np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        half[start : start + block.shape[0]] = block.astype(np.float16)
    del block
    gc.collect()
    report("float16 index loaded beside encoder")

    query = np.array(next(iter(model.embed([QUERY]))), dtype=np.float32)
    report("query embedded")

    started = time.perf_counter()
    scores = chunked_scores(half, query, chunk=8192)
    elapsed = (time.perf_counter() - started) * 1000
    top = np.argpartition(-scores, 9)[:10]
    current = report("search complete")

    print(f"\nsearch {elapsed:.0f} ms, best score {scores[top].max():.4f}")
    budget = 512
    verdict = "fits" if current < budget * 0.9 else "does not fit"
    print(f"{current:.0f} MB resident against a {budget} MB instance: {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encoder", "index", "both"))
    parser.add_argument("--documents", type=int, default=114581)
    parser.add_argument(
        "--naive",
        action="store_true",
        help="also time the whole-matrix upcast, to show what it costs",
    )
    args = parser.parse_args()

    print(f"{args.stage} stage, {sys.platform}, numpy {np.__version__}\n")
    if args.stage == "encoder":
        stage_encoder()
    elif args.stage == "index":
        stage_index(args.documents, naive=args.naive)
    else:
        stage_both(args.documents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
