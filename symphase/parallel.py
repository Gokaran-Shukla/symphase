# -*- coding: utf-8 -*-
"""
symphase.parallel
=================

Parallel map used by the hull code.

Backends
  * "process" (default) - multiprocessing on one node, any number of cores.
      Large shared objects (the PhaseDiagram, the entry list) are placed in
      module globals *before* the pool forks, so on Linux they are inherited
      by copy-on-write and never pickled.  On macOS/Windows ("spawn") they
      are sent once per worker through the pool initializer.
  * "mpi"    - mpi4py, any number of nodes (``srun python -m symphase ...``).
      Every rank holds the shared objects; tasks are distributed
      round-robin and gathered on all ranks.
  * "serial" - no parallelism (also used automatically for small jobs).

Core count:  nproc=None/0/-1  ->  all cores this process may use
(SLURM_CPUS_PER_TASK, then sched_getaffinity, then cpu_count).
"""

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

_SHARED = {}          # filled before forking; read by workers
_MIN_PARALLEL = 24    # below this number of tasks run serially


def available_cores():
    for var in ("SYMPHASE_NPROC", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS_SYMPHASE"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        return os.cpu_count() or 1


def resolve_nproc(nproc):
    if nproc is None or nproc <= 0:
        return available_cores()
    return int(nproc)


def mpi_comm():
    """MPI communicator if running under mpirun/srun with >1 rank, else None"""
    if not any(k in os.environ for k in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK",
                                         "MPI_LOCALNRANKS", "SLURM_NTASKS", "SYMPHASE_MPI")):
        return None
    try:
        from mpi4py import MPI
    except Exception:
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


def mpi_rank():
    c = mpi_comm()
    return c.Get_rank() if c is not None else 0


def is_root():
    return mpi_rank() == 0


def set_shared(**objs):
    _SHARED.clear()
    _SHARED.update(objs)


def get_shared(key):
    return _SHARED[key]


def _init_worker(shared):
    _SHARED.clear()
    _SHARED.update(shared)
    # one BLAS thread per process: avoids oversubscription
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _run_chunk(args):
    func, items = args
    return [func(x) for x in items]


def pmap(func, items, nproc=None, backend="auto", shared=None, chunks_per_worker=4,
         min_items=_MIN_PARALLEL, progress=False):
    """parallel ``[func(x) for x in items]`` preserving order.

    func must be a module-level function; it can read objects passed in
    ``shared`` through ``get_shared(key)``.
    """
    items = list(items)
    n = len(items)
    shared = shared or {}
    set_shared(**shared)

    comm = mpi_comm() if backend in ("auto", "mpi") else None
    if comm is not None:
        rank, size = comm.Get_rank(), comm.Get_size()
        mine = [(i, func(items[i])) for i in range(rank, n, size)]
        allres = comm.allgather(mine)
        out = [None] * n
        for part in allres:
            for i, r in part:
                out[i] = r
        return out

    nproc = resolve_nproc(nproc)
    if backend == "serial" or nproc <= 1 or n < min_items:
        return [func(x) for x in items]

    nproc = min(nproc, n)
    nchunk = min(n, nproc * chunks_per_worker)
    bounds = [round(k * n / nchunk) for k in range(nchunk + 1)]
    tasks = [(func, items[bounds[k]:bounds[k + 1]]) for k in range(nchunk)]

    method = "fork" if sys.platform.startswith("linux") else None
    ctx = mp.get_context(method) if method else mp.get_context()
    # with "fork" the initargs are inherited by reference (no pickling);
    # with "spawn" they are pickled once per worker
    initargs = (dict(_SHARED),)
    out = []
    with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx,
                             initializer=_init_worker, initargs=initargs) as ex:
        for k, res in enumerate(ex.map(_run_chunk, tasks)):
            out.extend(res)
            if progress:
                print(f"\r  {len(out)}/{n} done", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    return out
