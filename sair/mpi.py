# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from .utils import tqdm

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1
is_main_rank = rank == 0


def split(a, n):
    """Job splitter

    :param a: job list
    :param n: number of chunks (processors)
    :returns: splitted jobs
    :rtype: list
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def run_with_mpi(func, objects, title=None):
    """MPI helper

    :param func: function to call
    :type func: function
    :param objects: objects to pass to function
    :type objects: list
    :returns: results
    :rtype: dict
    """
    if rank == 0:  # server
        jobs = split(objects, size)
    else:
        jobs = []
    jobs = comm.scatter(jobs, root=0)

    results = {}
    if rank == 0:
        pbar = tqdm(total=len(jobs), desc=title)
    for _i, job in enumerate(jobs, 1):
        results.update(func(job))
        if rank == 0:
            pbar.update()

    if rank != 0:
        comm.send(results, dest=0, tag=rank)

    if rank == 0:
        for i in range(size):
            if i != 0:
                result = comm.recv(source=i, tag=i)
                results.update(result)

    comm.barrier()
    return results


def run_on_main_rank(func, *args, **kwargs):
    comm.barrier()
    if is_main_rank:
        func(*args, **kwargs)
        for i in range(1, size):
            comm.send(0, dest=i)
    else:
        # https://stackoverflow.com/questions/29170492/mpi4py-substantial-slowdown-by-idle-cores
        import time
        while not comm.Iprobe(source=0):
            time.sleep(1)
        comm.recv(source=0)
