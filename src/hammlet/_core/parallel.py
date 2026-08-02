"""Bounded process parallelism for expensive direct-map candidate work.

The FFT search is internally batched, whereas direct alpha rescoring and
Powell refinement are independent per candidate.  ``bounded_process_map`` is
the small common primitive for the latter: callers pass a module-level,
picklable worker and candidate payloads, and get results in candidate order.
"""

from __future__ import annotations

from contextlib import contextmanager
import multiprocessing as mp
import os
from typing import Any, Callable, Iterator, Literal, Sequence, TypeVar


Input = TypeVar("Input")
Output = TypeVar("Output")

DEFAULT_CANDIDATE_WORKERS = 10
"""Default process budget for direct alpha rescoring and refinement."""

# These are understood by the common BLAS/OpenMP implementations used by the
# numerical stack.  They are deliberately set in child processes only (and
# temporarily inherited at process creation) so a library caller's process is
# not permanently reconfigured.
_ONE_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
}


def _initialize_worker(
    initializer: Callable[..., None] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Constrain a child, then install caller-owned read-only worker state."""
    os.environ.update(_ONE_THREAD_ENV)
    if initializer is not None:
        initializer(*initargs)


@contextmanager
def _inherited_one_thread_environment() -> Iterator[None]:
    """Temporarily provide the one-thread settings to newly spawned children."""
    previous = {name: os.environ.get(name) for name in _ONE_THREAD_ENV}
    os.environ.update(_ONE_THREAD_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def bounded_process_map(
    worker: Callable[[Input], Output],
    items: Sequence[Input],
    *,
    workers: int = DEFAULT_CANDIDATE_WORKERS,
    chunksize: int = 1,
    start_method: Literal["spawn", "fork", "forkserver"] = "spawn",
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[Output]:
    """Run independent, picklable candidate jobs with a fixed process budget.

    ``worker`` must be a module-level picklable callable.  Results retain the
    exact order of ``items``.  A worker exception is re-raised in the caller
    with multiprocessing's remote traceback; no failed candidate is silently
    replaced or dropped.  ``workers=1`` is intentionally serial, which is
    useful for deterministic debugging and avoids process/pickle overhead.

    With ``workers > 1`` the pool is created with exactly ``workers`` child
    processes, even when there are fewer jobs.  Every child is restricted to
    one BLAS/OpenMP-style computation thread, preventing CPU oversubscription
    when ten independent refinements run together.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    if start_method not in {"spawn", "fork", "forkserver"}:
        raise ValueError(f"unsupported multiprocessing start method: {start_method}")

    # Materializing admits generators at runtime while making the empty-input
    # path cheap and avoiding the surprising creation of idle processes.
    payloads = list(items)
    if not payloads:
        return []
    if workers == 1:
        return [worker(item) for item in payloads]

    context = mp.get_context(start_method)
    # Supplying the settings at creation matters for libraries initialized at
    # import time; the initializer repeats them before user work for clarity.
    with _inherited_one_thread_environment():
        with context.Pool(
            processes=workers,
            initializer=_initialize_worker,
            initargs=(initializer, initargs),
        ) as pool:
            return pool.map(worker, payloads, chunksize=chunksize)
