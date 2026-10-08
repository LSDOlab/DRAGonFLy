"""
Proper orthogonal decomposition (POD) of solution snapshots, distributed over
MPI with PETSc.

SnapshotMatrix keeps the snapshot matrix S = [x_1, ..., x_n] only as its
incrementally updated, optionally truncated SVD, in a weighted inner product
(x, y)_W = x^T W y. PODBasis turns that SVD into a reduced basis for the
current design point::

  global          the leading rb_size left-singular vectors of S;
  local_weighted  the leading left-singular vectors of S diag(w), with
                  partition-of-unity weights w that decay with the distance
                  between each snapshot's parameter vector and the current
                  one (compact-support cubic by default).

Both bases are W-orthonormal (V^T W V = I). core/reduced_order_model.py
builds W and drives the reduced solves.
"""
from copy import deepcopy
from time import perf_counter

import numpy as np
import scipy.linalg as sp_la
from scipy.spatial.distance import cdist
from mpi4py import MPI
from petsc4py import PETSc

from dragonfly_sim.utils.pod_utils import (petsc_append_dense_column, petsc_dense_from_local_block,
                                           petsc_small_mat_from_numpy, petsc_small_vec_from_numpy,
                                           petsc_small_mat_to_numpy, cubic_dist_hat,
                                           cubic_dist_hat_fixed, basis_size_floor, cubic_weights)


class SnapshotMatrix:
    """
    Incremental weighted QR and SVD of the snapshot matrix.

    Each snapshot x_k (a PETSc Vec in the state layout) updates a weighted QR
    decomposition S = Q R with Q^T W Q = I, then the SVD of the small R; Q is
    rotated into R's left-singular basis every update (Q <- Q U_R), so Q IS
    the SVD's U and only one dense n_dofs x rank matrix is stored. The
    rotation is also what makes truncation to max_rank the standard truncated
    incremental SVD.

    Storage: Q distributed (MATDENSE, rows in the state layout); R, Sigma,
    VT small and replicated on PETSc.COMM_SELF. W is any object with
    W.mult(x, y) (a PETSc Mat or utils/pod_utils.DiagonalWeight), or None
    for the Euclidean inner product.

    A snapshot is rejected (no new column) when its parameter vector
    duplicates an earlier one; its location and the weights reconstructing it
    from the existing snapshots are kept, and the local_weighted basis
    redistributes its weight onto those. A rank-deficient snapshot -- one in
    the span of Q up to a weighted residual ||x - Q Q^T W x||_W below
    qr_rank_tol ||x||_W, a test independent of the snapshot's scale -- adds a
    column to R/VT but not to Q, so row i of the parameter array matches
    column i of R/VT, never of Q.
    """

    def __init__(self, comm, n_dofs=None, W=None, qr_rank_tol=1e-10, reorthogonalize=True,
                 max_rank=None):
        self.comm = comm
        self.n_dofs = n_dofs              # expected global size, checked on the first snapshot
        self.W = W
        self.qr_rank_tol = qr_rank_tol    # relative weighted residual needed for a new basis direction
        self.reorthogonalize = reorthogonalize
        self.max_rank = max_rank          # None: Q gains a column per accepted snapshot, unbounded

        self.Q = None      # n_dofs x r, distributed, W-orthonormal columns (== U)
        self.R = None      # r x n
        self.Sigma = None  # r
        self.VT = None     # r x n

        self.rank = 0
        self.n_snapshots = 0

        self.snapshot_parameter_array = np.array([])
        self.snapshot_list_parameters = []
        self.rejected_snapshot_parameters = []
        # rejected_snapshot_reconstruction_weights[j] has the length n_snapshots
        # had when rejection j happened; consumers zero-pad it. Valid because
        # columns of R are only ever appended, never reordered or dropped
        # (truncation trims rows).
        self.rejected_snapshot_reconstruction_weights = []

    # ------------------------------------------------------------------
    # Parameter bookkeeping
    # ------------------------------------------------------------------
    def construct_snapshot_parameter_array(self):
        self.snapshot_parameter_array = np.vstack(self.snapshot_list_parameters)

    def _is_duplicate_parameter(self, par_vec):
        if len(self.snapshot_list_parameters) == 0:
            return False
        self.construct_snapshot_parameter_array()
        return cdist(par_vec[None, :], self.snapshot_parameter_array).min() <= 1e-10

    # ------------------------------------------------------------------
    # Snapshot update
    # ------------------------------------------------------------------
    def _apply_W(self, x):
        """(W x, owned): W x as a fresh Vec, or x itself when unweighted --
        `owned` says whether the caller must destroy it."""
        if self.W is None:
            return x, False
        Wx = x.duplicate()
        self.W.mult(x, Wx)
        return Wx, True

    def process_and_append_to_snapshot_lists(self, x_k, par_vec):
        """
        Incorporate snapshot `x_k` (not modified) with parameter vector
        `par_vec`. Returns True if it added a new basis direction, False if
        it was a duplicate parameter or rank-deficient within qr_rank_tol.
        """
        if self._is_duplicate_parameter(par_vec):
            PETSc.Sys.Print("Snapshot parameter vector already present; keeping its location and "
                            "reconstruction weights, adding no column.")
            Wx, owns_Wx = self._apply_W(x_k)
            r_vec_np, _, rho = self._project_onto_Q(x_k, Wx)
            x_norm = self._weighted_norm(x_k, Wx)
            if owns_Wx:
                Wx.destroy()
            d = np.linalg.lstsq(petsc_small_mat_to_numpy(self.R), r_vec_np, rcond=None)[0]
            if rho > self.qr_rank_tol * x_norm:
                PETSc.Sys.Print("WARNING: the snapshot at a duplicate parameter location is not well "
                                "reconstructed by the existing basis (relative weighted residual "
                                "{:.6e} > qr_rank_tol {:.6e})".format(rho / x_norm, self.qr_rank_tol))
            self.rejected_snapshot_parameters += [deepcopy(par_vec)]
            self.rejected_snapshot_reconstruction_weights += [d]
            return False

        if self.n_dofs is not None and self.Q is None:
            assert x_k.getSize() == self.n_dofs, (
                "snapshot size {} does not match n_dofs {}".format(x_k.getSize(), self.n_dofs))

        t0 = perf_counter()
        new_direction_found = self._update_weighted_QR_and_SVD(x_k)
        self.snapshot_list_parameters += [deepcopy(par_vec)]
        PETSc.Sys.Print("Snapshot append time: {}".format(perf_counter() - t0))

        assert len(self.snapshot_list_parameters) == self.n_snapshots
        if not new_direction_found:
            PETSc.Sys.Print("Snapshot added no new basis direction (relative weighted residual below "
                            "{}); rank remains {}.".format(self.qr_rank_tol, self.rank))
        return new_direction_found

    def _project_onto_Q(self, x_k, Wx):
        """
        W-orthogonal projection of x_k onto Q: (r, e_local, rho) with r =
        Q^T W x_k (replicated), e_local this rank's slice of x_k - Q r, and
        rho = sqrt(e^T W e). `Wx` is W x_k (or x_k when unweighted).

        With reorthogonalize, a second projection pass (CGS2) removes the
        round-off a single pass leaves when x_k is nearly in span(Q), as
        snapshots from nearby design points are; without it orthogonality
        degrades catastrophically over many updates.
        """
        comm = self.comm
        x_local = x_k.getArray(readonly=True)
        Wx_local = Wx.getArray(readonly=True)

        if self.Q is None:
            r_vec_np = np.zeros((0,))
            e_local = np.array(x_local, copy=True)
        else:
            Q_local = self.Q.getDenseArray()
            r_vec_np = comm.allreduce(Q_local.T @ Wx_local, op=MPI.SUM)
            e_local = x_local - Q_local @ r_vec_np
            if self.reorthogonalize:
                e_tmp = x_k.duplicate()
                e_tmp.setArray(e_local)
                We_tmp, owns = self._apply_W(e_tmp)
                r_vec2_np = comm.allreduce(Q_local.T @ We_tmp.getArray(readonly=True), op=MPI.SUM)
                if owns:
                    We_tmp.destroy()
                e_tmp.destroy()
                e_local = e_local - Q_local @ r_vec2_np
                r_vec_np = r_vec_np + r_vec2_np

        e_vec = x_k.duplicate()
        e_vec.setArray(e_local)
        We_vec, owns = self._apply_W(e_vec)
        rho = float(np.sqrt(max(np.real(e_vec.dot(We_vec)), 0.0)))
        if owns:
            We_vec.destroy()
        e_vec.destroy()
        return r_vec_np, e_local, rho

    def _weighted_norm(self, x, Wx):
        """||x||_W = sqrt(x^T W x), given W x. Collective."""
        return float(np.sqrt(max(np.real(x.dot(Wx)), 0.0)))

    def _update_weighted_QR_and_SVD(self, x_k):
        comm = self.comm
        Wx, owns_Wx = self._apply_W(x_k)
        r_vec_np, e_local, rho = self._project_onto_Q(x_k, Wx)
        x_norm = self._weighted_norm(x_k, Wx)
        if owns_Wx:
            Wx.destroy()
        # relative test: the snapshot's part outside span(Q) must exceed
        # qr_rank_tol times its own size (a zero snapshot never qualifies)
        new_direction_found = rho > self.qr_rank_tol * x_norm

        # R (small, replicated)
        R_old_np = petsc_small_mat_to_numpy(self.R)
        if new_direction_found:
            if R_old_np.shape[0] > 0:
                R_new_np = np.block([[R_old_np, r_vec_np[:, None]],
                                     [np.zeros((1, R_old_np.shape[1])), rho]])
            else:
                R_new_np = np.array([[rho]])
        elif R_old_np.shape[0] > 0:
            R_new_np = np.column_stack([R_old_np, r_vec_np])
        else:
            PETSc.Sys.Print("WARNING: first snapshot has weighted norm below tolerance {}; basis "
                            "remains empty.".format(self.qr_rank_tol))
            R_new_np = np.zeros((0, R_old_np.shape[1] + 1))

        # Q extended by the new direction
        Q_ext = petsc_append_dense_column(comm, self.Q, e_local / rho) if new_direction_found else self.Q

        # SVD of the small R, computed redundantly (and identically) on every rank
        if R_new_np.shape[0] > 0 and R_new_np.shape[1] > 0:
            U_R_np, Sigma_np, VT_np = sp_la.svd(R_new_np, full_matrices=False)
        else:
            U_R_np, Sigma_np, VT_np = np.zeros((0, 0)), np.zeros((0,)), np.zeros((0, R_new_np.shape[1]))

        # Rotate Q into R's left-singular basis, truncated to max_rank:
        # Q' = Q U_R[:, :k], R' = diag(Sigma[:k]) VT[:k]. Q' stays W-orthonormal,
        # Q' R' is the rank-k truncation of the snapshot matrix, and the SVD of
        # R' has U = I, so U = Q'.
        if Q_ext is not None:
            k = Sigma_np.shape[0] if self.max_rank is None else min(Sigma_np.shape[0], self.max_rank)
            Q_new = petsc_dense_from_local_block(comm, Q_ext.getDenseArray() @ U_R_np[:, :k],
                                                 Q_ext.getSize()[0])
            Sigma_np = Sigma_np[:k]
            VT_np = VT_np[:k, :]
            R_new_np = Sigma_np[:, None] * VT_np
            Q_ext.destroy()
        else:
            Q_new = None
        if self.Q is not None and self.Q is not Q_ext:
            self.Q.destroy()
        for obj in (self.R, self.Sigma, self.VT):
            if obj is not None:
                obj.destroy()

        self.Q = Q_new
        self.R = petsc_small_mat_from_numpy(R_new_np)
        self.Sigma = petsc_small_vec_from_numpy(Sigma_np)
        self.VT = petsc_small_mat_from_numpy(VT_np)
        self.rank = 0 if self.Q is None else self.Q.getSize()[1]
        self.n_snapshots = self.R.getSize()[1]
        return new_direction_found

    # ------------------------------------------------------------------
    # Read-only surface
    # ------------------------------------------------------------------
    @property
    def U(self):
        """The left-singular vectors: the same object as self.Q."""
        return self.Q

    @property
    def full_SVD_triplet(self):
        return (self.U, self.Sigma, self.VT)

    def print_info(self):
        if self.Q is not None:
            PETSc.Sys.Print("Snapshot SVD: basis {} x {}, R {}, {} snapshots".format(
                self.Q.getSize()[0], self.Q.getSize()[1], self.R.getSize(), self.n_snapshots))
        else:
            PETSc.Sys.Print("No snapshots processed yet.")

    def gather_dense_to_rank0_numpy(self, mat, root=0):
        """A distributed dense Mat as one numpy array on `root` (diagnostics)."""
        if mat is None:
            return None
        gathered = self.comm.gather(mat.getDenseArray().copy(), root=root)
        return np.vstack(gathered) if self.comm.Get_rank() == root else None

    def destroy(self):
        for obj in (self.Q, self.R, self.Sigma, self.VT):
            if obj is not None:
                obj.destroy()
        self.Q = self.R = self.Sigma = self.VT = None
        self.rank = 0


class PODBasis:
    """
    A reduced basis from the snapshots in a SnapshotMatrix (see the module
    docstring for the basis modes). `basis` is the current basis, a
    distributed MATDENSE in the state layout, rebuilt (destroyed and
    reallocated) by every construct_basis call.
    """
    SUPPORTED_BASIS_MODES = ('global', 'local_weighted')
    WEIGHT_FUNCTIONS = ('Cubic', 'Linear', 'Quadratic', 'InverseDistance')

    def __init__(self, comm, n_dofs, basis_mode='global', subtract_mean=False,
                 mean_weighting='snapshot_weights', IP_matrix=None, max_rank=None, qr_rank_tol=1e-10,
                 cubic_cutoff=None):
        """
        cubic_cutoff: the default c in [0, 1) of the Cubic weight's support
        radius for local_weighted bases (see construct_basis); None selects
        the n_nonzero_weights rule instead.
        """
        if basis_mode not in self.SUPPORTED_BASIS_MODES:
            raise ValueError("basis_mode must be one of {}, got {!r}".format(
                self.SUPPORTED_BASIS_MODES, basis_mode))
        if subtract_mean and basis_mode != 'local_weighted':
            # the snapshots exist only as their SVD, so centring composes into
            # the small weighted factor that only local_weighted forms
            raise NotImplementedError("subtract_mean=True needs basis_mode='local_weighted'")
        if mean_weighting not in ('uniform', 'snapshot_weights'):
            raise ValueError("mean_weighting must be 'uniform' or 'snapshot_weights', got {!r}".format(
                mean_weighting))
        self.comm = comm
        self.n_dofs = n_dofs
        self.basis_mode = basis_mode
        self.subtract_mean = subtract_mean
        self.mean_weighting = mean_weighting
        self.IP_matrix = IP_matrix
        if cubic_cutoff is not None:
            cubic_dist_hat_fixed(0.0, 1.0, cubic_cutoff)    # validates c
        self.cubic_cutoff = cubic_cutoff
        self.snapshot_matrix = SnapshotMatrix(comm, n_dofs, W=IP_matrix, qr_rank_tol=qr_rank_tol,
                                              reorthogonalize=True, max_rank=max_rank)
        self.basis = None
        self.mean = None

    def _set_basis_from_local_block(self, basis_local_block):
        """
        Rebuild self.basis from this rank's row block. (MatCreateSubMatrix is
        not used to slice columns of U: for MATDENSE it fails on a width
        change, and under MPI returns replicated columns or corrupts the heap,
        depending on how the column IS is split.)
        """
        if self.basis is not None:
            self.basis.destroy()
        self.basis = petsc_dense_from_local_block(self.comm, np.ascontiguousarray(basis_local_block))
        return self.basis

    def _local_weighted_snapshot_weights(self, local_parametervector, weightfunction,
                                         parameter_weights, n_nonzero_weights, rb_size,
                                         cubic_cutoff=None):
        """
        Partition-of-unity weights over the accepted snapshots (one per column
        of VT), from the distances between their parameter vectors and
        `local_parametervector`. Duplicate-parameter (rejected) snapshots take
        part in the distance weighting like accepted ones, and their weight is
        then redistributed onto the accepted snapshots that reconstruct them.

        Returns (weights, distances of the accepted snapshots).
        """
        sm = self.snapshot_matrix
        sm.construct_snapshot_parameter_array()
        n_snapshots = sm.n_snapshots
        rejected = sm.rejected_snapshot_parameters
        n_rej = len(rejected)
        params = sm.snapshot_parameter_array
        if n_rej > 0:
            params = np.vstack([params, np.vstack(rejected)])

        parameter_distance = params - local_parametervector[None, :]
        if parameter_weights is not None:
            parameter_distance = parameter_distance * np.asarray(parameter_weights)[None, :]
        parameter_distnorm = np.linalg.norm(parameter_distance, ord=2, axis=1)
        min_dist = parameter_distnorm.min()
        max_dist = parameter_distnorm.max()

        # Every weighting divides by the spread of the distances; with none
        # (all snapshots equidistant, e.g. two symmetric perturbations) they
        # all reduce to uniform weights.
        if (max_dist - min_dist) <= 1e-12 * max(max_dist, 1.0):
            PETSc.Sys.Print("All snapshots are equidistant from the query point (distance {:.6e}); "
                            "using uniform snapshot weights.".format(min_dist))
            snapshot_weights = np.ones_like(parameter_distnorm)
        elif weightfunction == "Cubic":
            if cubic_cutoff is not None:
                # c sets the support radius, but never below the radius that
                # gives the rb_size nearest accepted snapshots a nonzero weight:
                # the basis then always has rb_size columns (when the snapshots
                # span that many directions), and c can only widen the set of
                # snapshots that contribute
                dist_hat_c = cubic_dist_hat_fixed(min_dist, max_dist, cubic_cutoff)
                dist_floor = basis_size_floor(parameter_distnorm[:n_snapshots], rb_size,
                                              max_dist - min_dist)
                dist_hat = max(dist_hat_c, dist_floor)
                PETSc.Sys.Print("cubic support radius: c = {} gives {:.6e}, rb_size = {} needs {:.6e}; "
                                "using {:.6e}".format(cubic_cutoff, dist_hat_c, rb_size, dist_floor, dist_hat))
            else:
                k_target = rb_size if n_nonzero_weights is None else n_nonzero_weights
                dist_hat = cubic_dist_hat(parameter_distnorm, min_dist, max_dist, k_target)
            snapshot_weights = cubic_weights(parameter_distnorm, min_dist, dist_hat)
        elif weightfunction == "Linear":
            avg_dist = (3 * min_dist + max_dist) / 4
            snapshot_weights = avg_dist / (avg_dist - min_dist) - parameter_distnorm / (avg_dist - min_dist)
            snapshot_weights[snapshot_weights < 0.] = 0.
        elif weightfunction == "Quadratic":
            denominator = max_dist**2 - min_dist**2
            snapshot_weights = max_dist**2 / denominator - parameter_distnorm**2 / denominator
        elif weightfunction == "InverseDistance":
            inv = 1.0 / parameter_distnorm**2
            snapshot_weights = inv / inv.max()
        else:
            raise ValueError("weightfunction must be one of {}, got {!r}".format(
                self.WEIGHT_FUNCTIONS, weightfunction))

        w_acc = snapshot_weights[:n_snapshots]
        if n_rej > 0:
            D = np.zeros((n_snapshots, n_rej))
            for j, d in enumerate(sm.rejected_snapshot_reconstruction_weights):
                D[:d.shape[0], j] = d
            w_acc = w_acc + D @ snapshot_weights[n_snapshots:]
            # clip after redistribution: negative reconstruction weights can
            # push an accepted snapshot's total below zero
            w_acc[w_acc < 0.] = 0.

        return w_acc / np.sum(w_acc), parameter_distnorm[:n_snapshots]

    def construct_basis(self, rb_size=10, local_parametervector=None, parameter_weights=None,
                        weightfunction='Cubic', n_nonzero_weights=None, cubic_cutoff=None,
                        return_distance_norms=False, layout_vec=None):
        """
        (Re)build self.basis from the current snapshot SVD; with no snapshots
        self.basis becomes None.

        local_weighted only::

          local_parametervector  the current design point
          parameter_weights      per-parameter factors on the distance (None: 1)
          weightfunction         'Cubic' (compact support), 'Linear',
                                 'Quadratic' or 'InverseDistance'
          n_nonzero_weights      Cubic: number of nearest snapshots with a
                                 nonzero weight (default rb_size)
          cubic_cutoff           Cubic: instead of n_nonzero_weights, a
                                 c in [0, 1) (default: the one given at
                                 construction) with support radius
                                 max(c min_dist + (1 - c) max_dist, r_rb),
                                 r_rb the smallest radius that leaves the
                                 rb_size nearest accepted snapshots a nonzero
                                 weight (utils/pod_utils.basis_size_floor)

        `layout_vec` (the state vector) guards the one invariant the reduced
        solve depends on: the basis rows are distributed exactly like it.
        Returns the accepted snapshots' distances with return_distance_norms
        (local_weighted), else None.
        """
        U, Sig_vec, VT = self.snapshot_matrix.full_SVD_triplet
        n_snapshots = self.snapshot_matrix.n_snapshots
        basis_rank = self.snapshot_matrix.rank

        if layout_vec is not None and U is not None and U.getLocalSize()[0] != layout_vec.getLocalSize():
            raise ValueError("POD basis row layout ({} local rows) does not match the state vector "
                             "layout ({} local rows)".format(U.getLocalSize()[0], layout_vec.getLocalSize()))

        if n_snapshots == 0 or U is None:
            if self.basis is not None:
                self.basis.destroy()
            self.basis = None
            self.mean = None
            return None

        parameter_distnorm = None
        if self.basis_mode == 'global' or n_snapshots < 2:
            # bounded by the rank, not n_snapshots: a rank-deficient snapshot
            # adds a column to R/VT but none to U
            self._set_basis_from_local_block(U.getDenseArray()[:, :min(basis_rank, rb_size)])
        else:
            if local_parametervector is None:
                raise ValueError("the local_weighted basis needs local_parametervector")
            weights, parameter_distnorm = self._local_weighted_snapshot_weights(
                local_parametervector, weightfunction, parameter_weights, n_nonzero_weights, rb_size,
                cubic_cutoff=self.cubic_cutoff if cubic_cutoff is None else cubic_cutoff)
            PETSc.Sys.Print("snapshot weights: {}".format(weights))

            # diagonal weight matrix restricted to the columns above 1e-10
            valid_idxs = np.where(weights >= 1e-10)[0]
            num_rows, num_cols = len(weights), len(valid_idxs)
            weight_mat = np.zeros((num_rows, num_cols))
            weight_mat[valid_idxs, np.arange(num_cols)] = weights[valid_idxs]

            Sig_np = np.asarray(Sig_vec.getArray()).ravel()
            VT_np = VT.getDenseArray()
            if self.subtract_mean:
                # centring is the rank-1 right factor H = I - c 1^T: S_c = S H,
                # so H W = W - c (1^T W), and the mean S c = U (Sig * (VT c))
                c_vec = weights.copy() if self.mean_weighting == 'snapshot_weights' \
                    else np.full(num_rows, 1.0 / max(num_rows, 1))
                if self.mean is not None:
                    self.mean.destroy()
                self.mean = U.createVecLeft()
                self.mean.getArray()[:] = U.getDenseArray() @ (Sig_np * (VT_np @ c_vec))
                self.mean.assemble()
                weight_mat = weight_mat - np.outer(c_vec, weight_mat.sum(axis=0))

            # SVD of diag(Sig) VT W: small (rank x num_cols), replicated, so dense scipy
            t0 = perf_counter()
            U_tilde_np, _, _ = sp_la.svd(Sig_np[:, None] * (VT_np @ weight_mat), full_matrices=False)
            t1 = perf_counter()

            # the basis is U U_tilde truncated to rb_size; truncate U_tilde first
            # so the intermediate never exceeds rb_size columns
            k = min(rb_size, U_tilde_np.shape[1])
            self._set_basis_from_local_block(U.getDenseArray() @ U_tilde_np[:, :k])
            PETSc.Sys.Print("local_weighted basis: {} of {} snapshots carry nonzero weight -> {} basis "
                            "vectors (rb_size = {}); small SVD time {}".format(
                                num_cols, n_snapshots, k, rb_size, t1 - t0))

        PETSc.Sys.Print("Computed POD basis: {} rows (global) x {} columns".format(*self.basis.getSize()))
        return parameter_distnorm if return_distance_norms else None

    def destroy(self):
        if self.basis is not None:
            self.basis.destroy()
            self.basis = None
        if self.mean is not None:
            self.mean.destroy()
            self.mean = None
        self.snapshot_matrix.destroy()
