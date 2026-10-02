import numpy as np


def allgather_numpy(comm, local_arr, global_size):
    """Allgatherv a rank-local float64 array into a global array on every rank.

    Each rank contributes a (possibly different-length) slice of the global
    array.  The slices are ordered by rank: rank 0 occupies [0, len_0),
    rank 1 occupies [len_0, len_0+len_1), etc.  This ordering matches the
    contiguous DOF blocks produced by ``assign_global_dof_offset``.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
    local_arr : array-like, float64, 1-D
        The local contribution on this rank.  Cast to C-contiguous float64
        before sending.
    global_size : int
        Total number of float64 elements summed across all ranks.  Must be
        the same value on every rank.

    Returns
    -------
    np.ndarray, float64, shape (global_size,)
        The gathered array, identical on every rank.
    """
    from mpi4py import MPI
    send = np.ascontiguousarray(local_arr, dtype=np.float64)

    # Serial fast-path: skip MPI overhead entirely
    if comm.Get_size() == 1:
        recv = np.empty(global_size, dtype=np.float64)
        recv[:len(send)] = send
        return recv

    recv       = np.zeros(global_size, dtype=np.float64)
    sendcounts = np.array(comm.allgather(len(send)), dtype=np.int32)
    displs     = np.concatenate([[0], np.cumsum(sendcounts[:-1])]).astype(np.int32)
    comm.Allgatherv(
        sendbuf=send,
        recvbuf=[recv, sendcounts, displs, MPI.DOUBLE],
    )
    return recv


def assign_global_dof_offset(comm, local_count):
    """Compute the global DOF offset for this rank via exclusive prefix-sum.

    Assigns a contiguous block of global indices to each rank:
        rank 0 → [0,                    local_count_0)
        rank 1 → [local_count_0,        local_count_0 + local_count_1)
        ...

    Implementation uses MPI ``Exscan`` (exclusive scan with SUM) to
    compute the offset for each rank in a single collective, then
    ``Allreduce`` on a scalar to obtain the global total.  This avoids
    allocating an N-rank array and is the canonical MPI pattern for
    assigning contiguous global index ranges to processes.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
    local_count : int
        Number of local DOFs on this rank.

    Returns
    -------
    offset : int
        Index of the first global DOF owned by this rank.
    global_total : int
        Total DOF count across all ranks.
    """
    from mpi4py import MPI

    # Exclusive scan: offset[r] = sum of local_count for ranks 0 .. r-1
    # local_count_arr = np.array(local_count, dtype=np.int64)
    # offset_arr      = np.zeros(1, dtype=np.int64)
    # offset_init = 0
    # Exscan is undefined on rank 0 — result stays 0, which is correct.assign_global_dof_offset
    offset = comm.exscan(local_count, op=MPI.SUM)
    if offset is None:
        # correct the offset on rank 0
        offset = 0
    # offset = offset_init

    # Global total: reduce the local count to a single sum on all ranks
    # global_total_arr = np.array(local_count, dtype=np.int64)
    global_total = comm.allreduce(local_count, op=MPI.SUM)
    # global_total = int(global_total_arr[0])

    return offset, global_total
