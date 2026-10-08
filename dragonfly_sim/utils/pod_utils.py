"""
Helpers for core/pod.py: PETSc dense-matrix plumbing for the distributed
snapshot SVD, and the compact-support cubic weight function of the
local_weighted basis.

Storage convention (see core/pod.py:SnapshotMatrix):

  - Q (the basis), distributed MATDENSE with the row layout of the state
    vector; its column count (the rank) is the same on every rank.
  - R, Sigma, VT, small PETSc objects on PETSc.COMM_SELF, one identical copy
    per rank. Their size scales with the number of snapshots, not with the
    mesh, so there is nothing to gain from distributing them.

Rank-sized quantities such as Q^T W x are formed as a local dense product on
each rank's row block followed by one Allreduce, which keeps them available
as replicated numpy arrays.
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


# ----------------------------------------------------------------------
# PETSc dense plumbing
# ----------------------------------------------------------------------
def petsc_append_dense_column(comm, old_mat, new_col_local):
    """
    A new distributed MATDENSE with one column more than `old_mat` (None for
    a fresh single-column matrix): its data, then `new_col_local` (this
    rank's slice of the new column) as the last column.
    """
    m_local = new_col_local.shape[0]
    M_global = comm.allreduce(m_local, op=MPI.SUM)
    r_old = 0 if old_mat is None else old_mat.getSize()[1]
    mat = PETSc.Mat().createDense(size=((m_local, M_global), (PETSc.DECIDE, r_old + 1)), comm=comm)
    mat.setUp()
    arr = mat.getDenseArray()
    if old_mat is not None:
        arr[:, :r_old] = old_mat.getDenseArray()
    arr[:, r_old] = new_col_local
    mat.assemble()
    return mat


def petsc_dense_from_local_block(comm, local_block, M_global=None):
    """A distributed MATDENSE whose local row block is `local_block`
    (shape (m_local, ncols))."""
    m_local, ncols = local_block.shape
    if M_global is None:
        M_global = comm.allreduce(m_local, op=MPI.SUM)
    mat = PETSc.Mat().createDense(size=((m_local, M_global), (PETSc.DECIDE, ncols)), comm=comm)
    mat.setUp()
    mat.getDenseArray()[:, :] = local_block
    mat.assemble()
    return mat


def petsc_small_mat_from_numpy(arr):
    """A small numpy array as a PETSc.COMM_SELF MATDENSE (the caller builds
    `arr` identically on every rank)."""
    mat = PETSc.Mat().createDense(size=arr.shape, comm=PETSc.COMM_SELF)
    mat.setUp()
    if arr.size > 0:
        mat.getDenseArray()[:, :] = arr
    mat.assemble()
    return mat


def petsc_small_vec_from_numpy(arr):
    """A small 1D numpy array as a PETSc.COMM_SELF Vec (replicated)."""
    vec = PETSc.Vec().createSeq(arr.shape[0], comm=PETSc.COMM_SELF)
    if arr.size > 0:
        vec.setArray(arr)
    vec.assemble()
    return vec


def petsc_small_mat_to_numpy(mat):
    if mat is None:
        return np.zeros((0, 0))
    return mat.getDenseArray().copy()


def replicated_transpose_product(comm, V, x_local):
    """V^T x as a numpy array on every rank, for a distributed MATDENSE V
    and this rank's owned entries x_local of a vector in V's row layout."""
    return comm.allreduce(V.getDenseArray().T @ x_local, op=MPI.SUM)


class DiagonalWeight:
    """
    A diagonal inner-product weight W = diag(w) with the `.mult(x, y)` that
    core/pod.py:SnapshotMatrix calls, w a Vec in the state layout.
    """

    def __init__(self, w):
        self.w = w

    def mult(self, x, y):
        y.pointwiseMult(self.w, x)

    def destroy(self):
        self.w.destroy()


# ----------------------------------------------------------------------
# Cubic weight function of the local_weighted basis
# ----------------------------------------------------------------------
def cubic_dist_hat(parameter_distnorm, min_dist, max_dist, n_nonzero_weights,
                   tail_eps_scale=1e-3, tie_rtol=1e-9):
    """
    The cubic weight's support radius dist_hat such that exactly the
    k = min(n_nonzero_weights, n_snapshots) snapshots nearest the query point
    get a nonzero weight and the (k+1)-th nearest gets zero.

    k >= n_snapshots: there is no (k+1)-th neighbour to cut at, so dist_hat
    is pushed a small relative distance beyond max_dist (tail_eps_scale keeps
    the farthest snapshot's weight above the 1e-10 pruning threshold in
    PODBasis.construct_basis).

    k or more snapshots tied at min_dist (e.g. a symmetric pair of design
    perturbations) would put dist_hat at min_dist; k is walked forward past
    the tie first. Ties exactly at the (k+1)-th boundary need no handling:
    both snapshots land on dist_hat and both get zero.
    """
    n_snapshots = parameter_distnorm.shape[0]
    k = max(0, min(int(n_nonzero_weights), n_snapshots))

    if k < n_snapshots:
        sorted_dist = np.sort(parameter_distnorm)
        tol = tie_rtol * (max_dist - min_dist)
        while k < n_snapshots and (sorted_dist[k] - min_dist) <= tol:
            k += 1
        if k < n_snapshots:
            return sorted_dist[k]

    return max_dist + tail_eps_scale * (max_dist - min_dist)


def cubic_dist_hat_fixed(min_dist, max_dist, c):
    """
    The cubic weight's support radius from a user-set c in [0, 1):

        dist_hat = c min_dist + (1 - c) max_dist.

    c = 0 puts dist_hat at the farthest snapshot (only it gets zero weight);
    larger c narrows the support towards the nearest snapshot.
    """
    if not 0.0 <= c < 1.0:
        raise ValueError("the cubic cutoff c must lie in [0, 1), got {!r}".format(c))
    return c * min_dist + (1.0 - c) * max_dist


def basis_size_floor(accepted_distnorm, rb_size, spread, margin_scale=1e-3):
    """
    The smallest cubic support radius that leaves the rb_size snapshots
    nearest the query point (among `accepted_distnorm`, the snapshots that
    carry a basis column) with a nonzero weight.

    The cubic weight vanishes AT its support radius, so the radius has to lie
    strictly beyond d_k, the k-th smallest distance (k = min(rb_size, number
    of snapshots)): it is the next larger distance (that snapshot gets zero
    weight), but at least d_k + margin_scale * spread, with `spread` the
    spread of all distances (max - min). The margin keeps the k-th weight
    clear of the 1e-10 pruning threshold when the next distance is barely
    larger, and places the radius beyond d_k when no larger distance exists.
    Snapshots tied with d_k all stay inside.
    """
    d = np.sort(np.asarray(accepted_distnorm, dtype=float))
    k = max(1, min(int(rb_size), d.shape[0]))
    d_k = d[k - 1]
    floor = d_k + margin_scale * spread
    larger = d[d > d_k]
    if larger.size:
        floor = max(floor, larger[0])
    return floor


def cubic_weights(parameter_distnorm, min_dist, dist_hat):
    """
    The cubic Hermite weight with w(min_dist) = 1, w(dist_hat) = 0 and zero
    slope at both ends, zero beyond dist_hat:

        w = 2 t^3 - 3 t^2 + 1,   t = (d - min_dist) / (dist_hat - min_dist).

    This closed form is the exact solution of the 4x4 monomial interpolation
    system, without that system's ill-conditioning at large absolute
    distances. The zeroing beyond dist_hat is required: the polynomial turns
    back up for t > 1.
    """
    t = (parameter_distnorm - min_dist) / (dist_hat - min_dist)
    snapshot_weights = 2.0 * np.power(t, 3) - 3.0 * np.power(t, 2) + 1.0
    snapshot_weights[parameter_distnorm > dist_hat] = 0.
    return snapshot_weights
