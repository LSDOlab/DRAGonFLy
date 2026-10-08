"""
core/pod.py: the incremental weighted QR/SVD of SnapshotMatrix, the cubic
weight function of the local_weighted basis, and PODBasis.

The SnapshotMatrix tests plant a known SVD -- random orthonormal singular
vectors and positive singular values spanning 5 decades -- feed its columns
one at a time and compare the incrementally maintained triplet with it, for
the Euclidean inner product and for a sparse SPD weight W (with U^T W U = I
and Sigma the W-weighted singular values; see _build_reference).

Everything asserts on replicated quantities and builds its reference data
from a fixed seed on every rank, so the module runs on any number of ranks
(tests/test_parallel.py reruns it on three). Every collective is called by
every rank.
"""
import contextlib

import numpy as np
import pytest
import scipy.linalg as sp_la
import scipy.sparse as sp_sparse
from mpi4py import MPI
from petsc4py import PETSc

from dragonfly_sim.core.pod import SnapshotMatrix, PODBasis
from dragonfly_sim.utils.pod_utils import (cubic_dist_hat, cubic_dist_hat_fixed, cubic_weights,
                                           basis_size_floor, DiagonalWeight)
from dragonfly_sim.utils.petsc_utils import set_petsc_vec_array


def dense_vec(comm, n):
    """A distributed Vec of global size n, PETSc's default layout."""
    v = PETSc.Vec().createMPI((PETSc.DECIDE, n), comm=comm)
    v.set(0.0)
    return v


class _Case:
    """A fed SnapshotMatrix and the replicated reference it is checked against."""

    def __init__(self, snapshot_matrix, comm, X, U_true, sigma_true, VT_true, W_sparse, W_petsc,
                 template_vec, row_range):
        self.snapshot_matrix = snapshot_matrix
        self.comm = comm
        self.X = X
        self.U_true = U_true
        self.sigma_true = sigma_true
        self.VT_true = VT_true
        self.W_sparse = W_sparse
        self.W_petsc = W_petsc
        self.template_vec = template_vec
        self.row_range = row_range

    @property
    def sigma(self):
        return self.snapshot_matrix.Sigma.getArray().copy()

    @property
    def VT(self):
        return self.snapshot_matrix.VT.getDenseArray().copy()

    @property
    def U(self):
        """U gathered to a replicated array. Collective."""
        return np.vstack(self.comm.allgather(self.snapshot_matrix.U.getDenseArray().copy()))

    def apply_W(self, mat):
        return mat if self.W_sparse is None else self.W_sparse @ mat


class TestSnapshotMatrix:
    TOL = 1e-10               # measured worst case ~1e-13 at 1, 3 and 5 ranks
    N_DOFS = 10_000
    N_SNAPSHOTS = 10
    SEED = 20260818
    SIGMA_RANGE = (1e-4, 10.0)
    MIN_SIGMA_RATIO = 1.01    # degeneracy guard: singular vectors of a tied pair are not unique
    W_CONDITION = 100.0
    TRUNCATED_RANK = 4

    @staticmethod
    def _random_orthonormal(rng, n, k):
        Q, R = np.linalg.qr(rng.standard_normal((n, k)))
        return Q * np.sign(np.diag(R))

    def _random_singular_values(self, rng):
        lo, hi = self.SIGMA_RANGE
        for _ in range(100):
            interior = 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), self.N_SNAPSHOTS - 2)
            sigma = np.sort(np.concatenate(([lo, hi], interior)))[::-1]
            if np.min(sigma[:-1] / sigma[1:]) >= self.MIN_SIGMA_RATIO:
                return sigma
        raise RuntimeError("could not draw well-separated singular values")

    def _W_offdiagonal(self):
        """Off-diagonal of a unit-diagonal tridiagonal Toeplitz W with
        condition number W_CONDITION (eigenvalues fill (1 - 2e, 1 + 2e))."""
        c = self.W_CONDITION
        return (c - 1.0) / (2.0 * (c + 1.0))

    def _build_reference(self, weighted, rank_deficient_to=None):
        """(X, U_true, sigma_true, VT_true, W_sparse), replicated. Weighted:
        W = R^T R with R upper bidiagonal, and U_true = R^-1 U_tilde is
        W-orthonormal, so X = U_true diag(sigma) VT has W-singular values
        sigma."""
        rng = np.random.default_rng(self.SEED)
        U_tilde = self._random_orthonormal(rng, self.N_DOFS, self.N_SNAPSHOTS)
        VT_true = self._random_orthonormal(rng, self.N_SNAPSHOTS, self.N_SNAPSHOTS)
        sigma_true = self._random_singular_values(rng)
        if weighted:
            e = self._W_offdiagonal()
            W_sparse = sp_sparse.diags([e, 1.0, e], [-1, 0, 1], shape=(self.N_DOFS, self.N_DOFS),
                                       format="csr")
            ab = np.zeros((2, self.N_DOFS))
            ab[0, 1:] = e
            ab[1, :] = 1.0
            R = sp_la.cholesky_banded(ab, lower=False)
            U_true = sp_la.solve_banded((0, 1), R, U_tilde)
        else:
            W_sparse = None
            U_true = U_tilde
        if rank_deficient_to is not None:
            U_true = U_true[:, :rank_deficient_to]
            sigma_true = sigma_true[:rank_deficient_to]
            VT_true = VT_true[:rank_deficient_to, :]
        X = U_true @ np.diag(sigma_true) @ VT_true
        return X, U_true, sigma_true, VT_true, W_sparse

    def _build_W_petsc(self, comm, rstart, rend):
        """Distributed AIJ copy of W, row and column layout pinned to the
        snapshot vectors'."""
        e = self._W_offdiagonal()
        n_local = rend - rstart
        W = PETSc.Mat().createAIJ(size=((n_local, self.N_DOFS), (n_local, self.N_DOFS)), nnz=3,
                                  comm=comm)
        W.setUp()
        for i in range(rstart, rend):
            cols = [c for c in (i - 1, i, i + 1) if 0 <= c < self.N_DOFS]
            W.setValues([i], cols, [1.0 if c == i else e for c in cols])
        W.assemble()
        return W

    def _feed_snapshot(self, snapshot_matrix, comm, column, par_vec, row_range):
        rstart, rend = row_range
        x_k = dense_vec(comm, self.N_DOFS)
        set_petsc_vec_array(x_k, column[rstart:rend])
        try:
            return snapshot_matrix.process_and_append_to_snapshot_lists(x_k, par_vec)
        finally:
            x_k.destroy()

    @staticmethod
    def _parameter_vector(index):
        return np.array([float(index), 0.5 * float(index)])

    @contextlib.contextmanager
    def _case(self, weighted, max_rank=None, rank_deficient_to=None, columns=None):
        comm = MPI.COMM_WORLD
        X, U_true, sigma_true, VT_true, W_sparse = self._build_reference(
            weighted, rank_deficient_to=rank_deficient_to)
        fed = X if columns is None else columns
        template_vec = dense_vec(comm, self.N_DOFS)
        row_range = template_vec.getOwnershipRange()
        W_petsc = self._build_W_petsc(comm, *row_range) if weighted else None
        snapshot_matrix = SnapshotMatrix(comm, n_dofs=self.N_DOFS, W=W_petsc, qr_rank_tol=1e-10,
                                         reorthogonalize=True, max_rank=max_rank)
        for j in range(fed.shape[1]):
            self._feed_snapshot(snapshot_matrix, comm, fed[:, j], self._parameter_vector(j), row_range)
        case = _Case(snapshot_matrix, comm, X, U_true, sigma_true, VT_true, W_sparse, W_petsc,
                     template_vec, row_range)
        try:
            yield case
        finally:
            snapshot_matrix.destroy()
            if W_petsc is not None:
                W_petsc.destroy()
            template_vec.destroy()

    def assert_close(self, actual, expected):
        # assert_allclose rather than pytest.approx: the latter is slow on
        # these 10,000-row arrays
        np.testing.assert_allclose(actual, expected, rtol=self.TOL, atol=self.TOL)

    weighted_or_not = pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted_SPD"])

    # -- accuracy of the incremental SVD ---------------------------------
    @weighted_or_not
    def test_singular_values_match_known_svd(self, weighted):
        with self._case(weighted) as case:
            self.assert_close(case.sigma, case.sigma_true)

    @weighted_or_not
    def test_reconstructs_snapshot_matrix(self, weighted):
        with self._case(weighted) as case:
            self.assert_close(case.U @ np.diag(case.sigma) @ case.VT, case.X)

    @weighted_or_not
    def test_left_singular_vectors_are_W_orthonormal(self, weighted):
        with self._case(weighted) as case:
            U = case.U
            self.assert_close(U.T @ case.apply_W(U), np.eye(case.snapshot_matrix.rank))

    @weighted_or_not
    def test_left_singular_vectors_match_known_svd(self, weighted):
        with self._case(weighted) as case:
            alignment = np.abs(case.U.T @ case.apply_W(case.U_true))
            self.assert_close(alignment, np.eye(self.N_SNAPSHOTS))

    @weighted_or_not
    def test_right_singular_vectors_are_orthonormal(self, weighted):
        with self._case(weighted) as case:
            VT = case.VT
            self.assert_close(VT @ VT.T, np.eye(case.snapshot_matrix.rank))

    # -- bookkeeping, rank deficiency, truncation -------------------------
    @weighted_or_not
    def test_rank_and_snapshot_bookkeeping(self, weighted):
        with self._case(weighted) as case:
            sm = case.snapshot_matrix
            assert sm.rank == sm.n_snapshots == len(sm.snapshot_list_parameters) == self.N_SNAPSHOTS
            assert sm.U.getSize() == (self.N_DOFS, self.N_SNAPSHOTS)
            assert sm.VT.getSize() == (self.N_SNAPSHOTS, self.N_SNAPSHOTS)
            assert sm.R.getSize() == (self.N_SNAPSHOTS, self.N_SNAPSHOTS)
            assert sm.Sigma.getSize() == self.N_SNAPSHOTS

    def test_full_svd_triplet_is_all_none_before_any_snapshot(self):
        with self._case(weighted=False, columns=np.zeros((self.N_DOFS, 0))) as case:
            assert case.snapshot_matrix.full_SVD_triplet == (None, None, None)
            assert case.snapshot_matrix.n_snapshots == 0

    @weighted_or_not
    def test_rank_deficient_snapshot_is_rejected(self, weighted):
        with self._case(weighted) as case:
            sm = case.snapshot_matrix
            rank_before, n_before = sm.rank, sm.n_snapshots
            accepted = self._feed_snapshot(sm, case.comm, case.X[:, 0] + 2.0 * case.X[:, 1],
                                           self._parameter_vector(self.N_SNAPSHOTS + 1), case.row_range)
            assert accepted is False
            assert sm.rank == rank_before
            assert sm.n_snapshots == n_before + 1
            U = case.U
            self.assert_close(U.T @ case.apply_W(U), np.eye(sm.rank))

    @weighted_or_not
    @pytest.mark.parametrize("scale", [1e-13, 1.0, 1e8])
    def test_rank_test_is_relative_to_the_snapshot_norm(self, weighted, scale):
        """The same decision at every scale: a snapshot in the span adds no
        direction, a genuinely new one does -- also at 1e-13, where its
        absolute residual (~1e-11) is below qr_rank_tol."""
        new = np.random.default_rng(99).standard_normal(self.N_DOFS)
        with self._case(weighted) as case:
            sm = case.snapshot_matrix
            rank = sm.rank
            in_span = scale * (case.X[:, 0] + 2.0 * case.X[:, 1])
            assert self._feed_snapshot(sm, case.comm, in_span, self._parameter_vector(50),
                                       case.row_range) is False
            assert sm.rank == rank
            assert self._feed_snapshot(sm, case.comm, scale * new, self._parameter_vector(51),
                                       case.row_range) is True
            assert sm.rank == rank + 1
            U = case.U
            self.assert_close(U.T @ case.apply_W(U), np.eye(sm.rank))

    @weighted_or_not
    def test_large_near_span_snapshot_adds_no_round_off_direction(self, weighted):
        """In the span up to a relative 1e-13 at scale 1e6: rejected. An
        absolute test would accept it (residual ~1e-6) and add a direction of
        round-off."""
        rng = np.random.default_rng(7)
        with self._case(weighted) as case:
            sm = case.snapshot_matrix
            x = 1e6 * (case.X[:, 0] + 2.0 * case.X[:, 1])
            x = x + 1e-13 * np.linalg.norm(x) / np.sqrt(self.N_DOFS) * rng.standard_normal(self.N_DOFS)
            assert self._feed_snapshot(sm, case.comm, x, self._parameter_vector(60),
                                       case.row_range) is False
            assert sm.rank == self.N_SNAPSHOTS

    @weighted_or_not
    def test_duplicate_parameter_vector_is_skipped(self, weighted):
        with self._case(weighted) as case:
            sm = case.snapshot_matrix
            rank_before, n_before = sm.rank, sm.n_snapshots
            accepted = self._feed_snapshot(sm, case.comm, case.X[:, 0], self._parameter_vector(0),
                                           case.row_range)
            assert accepted is False
            assert (sm.rank, sm.n_snapshots) == (rank_before, n_before)

    @weighted_or_not
    def test_duplicate_parameter_vector_retains_reconstruction_weights(self, weighted):
        """Feeding back snapshot 0 itself reconstructs as d = e_0."""
        with self._case(weighted) as case:
            sm = case.snapshot_matrix
            n_before = sm.n_snapshots
            self._feed_snapshot(sm, case.comm, case.X[:, 0], self._parameter_vector(0), case.row_range)
            assert len(sm.rejected_snapshot_parameters) == 1
            assert np.array_equal(sm.rejected_snapshot_parameters[0], self._parameter_vector(0))
            expected = np.zeros(n_before)
            expected[0] = 1.0
            self.assert_close(sm.rejected_snapshot_reconstruction_weights[0], expected)

    @weighted_or_not
    def test_duplicate_parameter_vector_weights_satisfy_R_old_d_eq_r(self, weighted):
        """With a wide (truncated) R, the stored d solves R_old d = Q^T W x."""
        k = self.TRUNCATED_RANK
        with self._case(weighted, max_rank=k, rank_deficient_to=k) as case:
            sm = case.snapshot_matrix
            n_before = sm.n_snapshots
            R_old_np = sm.R.getDenseArray().copy()
            combo = 0.3 * case.X[:, 0] + 0.7 * case.X[:, 2]
            rstart, rend = case.row_range
            x_k = dense_vec(case.comm, self.N_DOFS)
            set_petsc_vec_array(x_k, combo[rstart:rend])
            Wx, owns_Wx = sm._apply_W(x_k)
            r_vec_np, _, _ = sm._project_onto_Q(x_k, Wx)
            if owns_Wx:
                Wx.destroy()
            x_k.destroy()
            assert self._feed_snapshot(sm, case.comm, combo, self._parameter_vector(0),
                                       case.row_range) is False
            d = sm.rejected_snapshot_reconstruction_weights[-1]
            assert d.shape == (n_before,)
            assert R_old_np.shape == (k, n_before)
            self.assert_close(R_old_np @ d, r_vec_np)

    @weighted_or_not
    def test_rejected_reconstruction_weight_length_is_frozen_at_rejection_time(self, weighted):
        comm = MPI.COMM_WORLD
        X = self._build_reference(weighted)[0]
        template_vec = dense_vec(comm, self.N_DOFS)
        row_range = template_vec.getOwnershipRange()
        W_petsc = self._build_W_petsc(comm, *row_range) if weighted else None
        sm = SnapshotMatrix(comm, n_dofs=self.N_DOFS, W=W_petsc)
        try:
            for j in range(3):
                self._feed_snapshot(sm, comm, X[:, j], self._parameter_vector(j), row_range)
            self._feed_snapshot(sm, comm, X[:, 0], self._parameter_vector(0), row_range)
            assert sm.rejected_snapshot_reconstruction_weights[0].shape == (3,)
            for j in range(3, self.N_SNAPSHOTS):
                self._feed_snapshot(sm, comm, X[:, j], self._parameter_vector(j), row_range)
            assert sm.n_snapshots == self.N_SNAPSHOTS
            assert sm.rejected_snapshot_reconstruction_weights[0].shape == (3,)
        finally:
            sm.destroy()
            if W_petsc is not None:
                W_petsc.destroy()
            template_vec.destroy()

    @weighted_or_not
    def test_max_rank_truncation_is_lossless_for_rank_deficient_data(self, weighted):
        k = self.TRUNCATED_RANK
        with self._case(weighted, max_rank=k, rank_deficient_to=k) as case:
            sm = case.snapshot_matrix
            assert sm.rank == k
            assert sm.n_snapshots == self.N_SNAPSHOTS
            self.assert_close(case.sigma, case.sigma_true)
            self.assert_close(case.U @ np.diag(case.sigma) @ case.VT, case.X)

    # -- parallel behaviour ------------------------------------------------
    @weighted_or_not
    def test_U_row_layout_matches_snapshot_vector_layout(self, weighted):
        with self._case(weighted) as case:
            assert case.snapshot_matrix.U.getLocalSize()[0] == case.template_vec.getLocalSize()

    @weighted_or_not
    def test_svd_triplet_is_identical_on_all_ranks(self, weighted):
        with self._case(weighted) as case:
            all_sigma = case.comm.allgather(case.sigma)
            all_VT = case.comm.allgather(case.VT)
            assert all(np.array_equal(all_sigma[0], s) for s in all_sigma)
            assert all(np.array_equal(all_VT[0], v) for v in all_VT)


class TestDiagonalWeight:
    def test_matches_a_diagonal_matrix(self):
        """DiagonalWeight (pointwise multiply) gives the same SVD as the same
        diagonal W as an AIJ Mat."""
        comm = MPI.COMM_WORLD
        n, m = 200, 5
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n, m))
        w_np = 10.0 ** rng.uniform(-3, 3, n)
        template = dense_vec(comm, n)
        rstart, rend = template.getOwnershipRange()
        w_vec = template.duplicate()
        set_petsc_vec_array(w_vec, w_np[rstart:rend])
        W_mat = PETSc.Mat().createAIJ(size=((rend - rstart, n), (rend - rstart, n)), nnz=1, comm=comm)
        W_mat.setUp()
        for i in range(rstart, rend):
            W_mat.setValue(i, i, w_np[i])
        W_mat.assemble()
        results = []
        for W in (DiagonalWeight(w_vec), W_mat):
            sm = SnapshotMatrix(comm, n_dofs=n, W=W)
            for j in range(m):
                set_petsc_vec_array(template, X[rstart:rend, j])
                sm.process_and_append_to_snapshot_lists(template, np.array([float(j)]))
            U = np.vstack(comm.allgather(sm.U.getDenseArray().copy()))
            results.append((sm.Sigma.getArray().copy(), U))
            assert U.T @ (w_np[:, None] * U) == pytest.approx(np.eye(m), abs=1e-12)
            sm.destroy()
        assert results[0][0] == pytest.approx(results[1][0], rel=1e-12)
        assert results[0][1] == pytest.approx(results[1][1], abs=1e-12)
        W_mat.destroy()
        w_vec.destroy()
        template.destroy()


class TestCubicDistHat:
    def test_normal_case_dist_hat_is_the_kplus1th_smallest(self):
        dist = np.arange(8.0)
        k = 3
        dist_hat = cubic_dist_hat(dist, dist.min(), dist.max(), k)
        assert dist_hat == pytest.approx(3.0)
        w = cubic_weights(dist, dist.min(), dist_hat)
        assert np.count_nonzero(w) == k
        assert (w[:k] > 0.0).all()
        assert w[k] == pytest.approx(0.0, abs=1e-12)
        assert (w[k + 1:] == 0.0).all()
        assert w[0] == pytest.approx(1.0)
        assert np.all(np.diff(w[:k + 1]) <= 1e-12)

    def test_edge_case_all_snapshots_requested(self):
        dist = np.array([0.0, 1.0, 2.0, 5.0])
        dist_hat = cubic_dist_hat(dist, dist.min(), dist.max(), n_nonzero_weights=4)
        assert dist_hat > dist.max()
        w = cubic_weights(dist, dist.min(), dist_hat)
        assert np.all(w > 0.0) and np.all(np.isfinite(w))

    def test_edge_case_oversubscribed_request_clamps(self):
        dist = np.array([0.0, 1.0, 2.0])
        w = cubic_weights(dist, dist.min(), cubic_dist_hat(dist, dist.min(), dist.max(), 1000))
        assert np.all(w > 0.0)

    def test_tied_nearest_neighbours_are_all_included(self):
        dist = np.array([1.0, 1.0, 1.0, 3.0, 4.0, 5.0])
        dist_hat = cubic_dist_hat(dist, dist.min(), dist.max(), n_nonzero_weights=1)
        assert dist_hat == pytest.approx(3.0)
        w = cubic_weights(dist, dist.min(), dist_hat)
        assert np.count_nonzero(w) == 3
        assert w[0] == w[1] == w[2] == pytest.approx(1.0)
        assert (w[3:] == 0.0).all()

    def test_near_tie_is_finite_not_singular(self):
        dist = np.array([1.0, 1.0 + 1e-13, 1.0 + 1e-13, 3.0, 4.0])
        w = cubic_weights(dist, dist.min(), cubic_dist_hat(dist, dist.min(), dist.max(), 1))
        assert np.all(np.isfinite(w))
        assert np.count_nonzero(w) >= 1

    def test_weights_are_well_defined_at_large_absolute_distances(self):
        dist = np.array([1000.0, 1000.5, 1001.0, 1005.0])
        w = cubic_weights(dist, dist.min(), cubic_dist_hat(dist, dist.min(), dist.max(), 2))
        assert np.all(np.isfinite(w))
        assert np.count_nonzero(w) == 2
        assert w[0] == pytest.approx(1.0)


class TestCubicDistHatFixed:
    """The user-set cutoff dist_hat = c min + (1 - c) max."""

    def test_formula(self):
        assert cubic_dist_hat_fixed(1.0, 5.0, 0.0) == pytest.approx(5.0)
        assert cubic_dist_hat_fixed(1.0, 5.0, 0.8) == pytest.approx(1.8)
        assert cubic_dist_hat_fixed(1.0, 5.0, 0.5) == pytest.approx(3.0)

    def test_c_zero_drops_only_the_farthest(self):
        dist = np.array([0.0, 1.0, 2.0, 5.0])
        w = cubic_weights(dist, dist.min(), cubic_dist_hat_fixed(dist.min(), dist.max(), 0.0))
        assert np.all(w[:3] > 0.0)
        assert w[3] == pytest.approx(0.0, abs=1e-14)

    def test_support_narrows_with_c(self):
        dist = np.arange(11.0)
        counts = [np.count_nonzero(cubic_weights(dist, 0.0, cubic_dist_hat_fixed(0.0, 10.0, c)) > 0)
                  for c in (0.0, 0.5, 0.8)]
        assert counts == [10, 5, 2]

    @pytest.mark.parametrize("c", [-0.1, 1.0, 1.5])
    def test_out_of_range_c_raises(self, c):
        with pytest.raises(ValueError):
            cubic_dist_hat_fixed(0.0, 1.0, c)


class TestBasisSizeFloor:
    """The smallest cubic support radius leaving the rb_size nearest
    snapshots a nonzero weight."""

    def test_next_larger_distance(self):
        d = np.array([3.0, 0.0, 2.0, 1.0, 5.0])
        floor = basis_size_floor(d, rb_size=3, spread=5.0)
        assert floor == pytest.approx(3.0)            # 3rd smallest is 2 -> next is 3
        assert np.count_nonzero(cubic_weights(d, 0.0, floor) > 1e-10) == 3

    def test_ties_at_the_boundary_stay_inside(self):
        d = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
        floor = basis_size_floor(d, rb_size=2, spread=2.0)
        assert floor == pytest.approx(2.0)
        assert np.count_nonzero(cubic_weights(d, 0.0, floor) > 1e-10) == 4

    def test_margin_when_the_next_distance_is_barely_larger(self):
        d = np.array([0.0, 1.0, 1.0 + 1e-9, 3.0])
        floor = basis_size_floor(d, rb_size=2, spread=3.0)
        assert floor == pytest.approx(1.0 + 3e-3)
        w = cubic_weights(d, 0.0, floor)
        assert w[1] > 1e-6                            # the 2nd snapshot is not pruned

    def test_fewer_snapshots_than_rb_size(self):
        d = np.array([0.0, 1.0, 2.0])
        floor = basis_size_floor(d, rb_size=5, spread=2.0)
        assert floor == pytest.approx(2.0 + 2e-3)
        assert np.all(cubic_weights(d, 0.0, floor) > 0.0)


class TestPODLocalWeightedBasis:
    """
    PODBasis.construct_basis(basis_mode='local_weighted'). _qr_case feeds
    orthonormal columns at parameters [0], [1], ..., so querying at [0] gives
    distances [0, 1, ...]; _axis_aligned_pod feeds c_i e_i, so Q = I and
    R = diag(c) and the redistribution arithmetic can be derived by hand.
    """
    N_DOFS = 32
    N_SNAPSHOTS = 8

    @contextlib.contextmanager
    def _qr_case(self, n_snapshots, seed=20260918):
        comm = MPI.COMM_WORLD
        columns, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((self.N_DOFS, n_snapshots)))
        pod = PODBasis(comm, self.N_DOFS, basis_mode='local_weighted')
        template = dense_vec(comm, self.N_DOFS)
        rstart, rend = template.getOwnershipRange()
        try:
            for j in range(n_snapshots):
                set_petsc_vec_array(template, columns[rstart:rend, j])
                pod.snapshot_matrix.process_and_append_to_snapshot_lists(template, np.array([float(j)]))
            yield pod, template
        finally:
            pod.destroy()
            template.destroy()

    def test_normal_case_basis_width_matches_n_nonzero_weights(self):
        with self._qr_case(self.N_SNAPSHOTS) as (pod, template):
            pod.construct_basis(rb_size=self.N_SNAPSHOTS, n_nonzero_weights=3,
                                local_parametervector=np.array([0.0]), layout_vec=template)
            assert pod.basis.getSize()[1] == 3

    def test_default_n_nonzero_weights_is_rb_size(self):
        widths = []
        for n_nonzero in (None, 4):
            with self._qr_case(self.N_SNAPSHOTS) as (pod, template):
                pod.construct_basis(rb_size=4, n_nonzero_weights=n_nonzero,
                                    local_parametervector=np.array([0.0]), layout_vec=template)
                widths.append(pod.basis.getSize()[1])
        assert widths == [4, 4]

    def test_n_nonzero_weights_ge_n_snapshots_uses_every_snapshot(self):
        with self._qr_case(self.N_SNAPSHOTS) as (pod, template):
            pod.construct_basis(rb_size=self.N_SNAPSHOTS, n_nonzero_weights=self.N_SNAPSHOTS + 100,
                                local_parametervector=np.array([0.0]), layout_vec=template)
            assert pod.basis.getSize()[1] == self.N_SNAPSHOTS

    def _nonzero_weights(self, pod, rb_size, c):
        w, _ = pod._local_weighted_snapshot_weights(np.array([0.0]), 'Cubic', None, None, rb_size,
                                                    cubic_cutoff=c)
        return int(np.count_nonzero(w >= 1e-10))

    @pytest.mark.parametrize("rb_size, c, n_weights", [
        (2, 0.5, 4),     # c radius 3.5 > floor 2: c widens the support beyond rb_size snapshots
        (6, 0.8, 6),     # c radius 1.4 < floor 6: the floor keeps rb_size snapshots
        (10, 0.9, 8),    # rb_size above the snapshot count: every snapshot
    ])
    def test_cubic_cutoff_never_narrows_the_basis_below_rb_size(self, rb_size, c, n_weights):
        """Distances 0..7. The support radius is max(c radius, rb_size floor):
        the basis always has min(rb_size, n_snapshots) columns, and c can only
        add contributing snapshots. c overrides n_nonzero_weights."""
        with self._qr_case(self.N_SNAPSHOTS) as (pod, template):
            assert self._nonzero_weights(pod, rb_size, c) == n_weights
            pod.construct_basis(rb_size=rb_size, cubic_cutoff=c, n_nonzero_weights=1,
                                local_parametervector=np.array([0.0]), layout_vec=template)
            assert pod.basis.getSize()[1] == min(rb_size, self.N_SNAPSHOTS)

    def test_cubic_cutoff_floor_counts_accepted_snapshots_only(self):
        """A duplicate of snapshot 0 (its parameter and state) adds no column,
        so it must not take one of the rb_size places: rb_size 4 still gives 4
        columns. (Counting it, the floor would sit at distance 3 and leave 3.)
        Its reconstruction weights are e_0, so the redistribution only adds to
        snapshot 0 and cannot widen the support by itself."""
        with self._qr_case(self.N_SNAPSHOTS) as (pod, template):
            columns, _ = np.linalg.qr(np.random.default_rng(20260918).standard_normal(
                (self.N_DOFS, self.N_SNAPSHOTS)))
            x = columns[:, 0]
            rstart, rend = template.getOwnershipRange()
            set_petsc_vec_array(template, x[rstart:rend])
            assert not pod.snapshot_matrix.process_and_append_to_snapshot_lists(template, np.array([0.0]))
            pod.construct_basis(rb_size=4, cubic_cutoff=0.95, local_parametervector=np.array([0.0]),
                                layout_vec=template)
            assert pod.basis.getSize()[1] == 4

    def test_basis_is_W_orthonormal(self):
        with self._qr_case(self.N_SNAPSHOTS) as (pod, template):
            pod.construct_basis(rb_size=5, local_parametervector=np.array([2.5]), layout_vec=template)
            V = np.vstack(MPI.COMM_WORLD.allgather(pod.basis.getDenseArray().copy()))
            assert V.T @ V == pytest.approx(np.eye(V.shape[1]), abs=1e-12)

    def test_global_basis_is_the_leading_singular_vectors(self):
        comm = MPI.COMM_WORLD
        pod = PODBasis(comm, self.N_DOFS, basis_mode='global')
        assert pod.construct_basis(rb_size=3) is None and pod.basis is None
        template = dense_vec(comm, self.N_DOFS)
        rstart, rend = template.getOwnershipRange()
        X = np.random.default_rng(3).standard_normal((self.N_DOFS, 6))
        for j in range(6):
            set_petsc_vec_array(template, X[rstart:rend, j])
            pod.snapshot_matrix.process_and_append_to_snapshot_lists(template, np.array([float(j)]))
        pod.construct_basis(rb_size=3, layout_vec=template)
        V = np.vstack(comm.allgather(pod.basis.getDenseArray().copy()))
        U_ref = np.linalg.svd(X, full_matrices=False)[0][:, :3]
        assert np.abs(V.T @ U_ref) == pytest.approx(np.eye(3), abs=1e-10)
        pod.destroy()
        template.destroy()

    # -- duplicate-parameter retention and redistribution -----------------
    N_AXIS_DOFS = 3
    AXIS_COEFFS = (2.0, 1.0, 0.5)
    AXIS_PARAMS = (0.0, 10.0, 20.0)

    @contextlib.contextmanager
    def _axis_aligned_pod(self):
        comm = MPI.COMM_WORLD
        pod = PODBasis(comm, self.N_AXIS_DOFS, basis_mode='local_weighted')
        template = dense_vec(comm, self.N_AXIS_DOFS)
        try:
            for i, (ci, pi) in enumerate(zip(self.AXIS_COEFFS, self.AXIS_PARAMS)):
                vals = np.zeros(self.N_AXIS_DOFS)
                vals[i] = ci
                assert self._feed_axis(pod, template, vals, pi) is True
            yield pod, template
        finally:
            pod.destroy()
            template.destroy()

    def _feed_axis(self, pod, template, values, par_scalar):
        rstart, rend = template.getOwnershipRange()
        set_petsc_vec_array(template, np.asarray(values)[rstart:rend])
        return pod.snapshot_matrix.process_and_append_to_snapshot_lists(template, np.array([par_scalar]))

    def test_duplicate_weight_is_redistributed_onto_reconstructing_snapshots(self):
        """Duplicate of snapshot 0's parameter with state 0.5 snap0 + 0.5 snap1
        -> d = [0.5, 0.5, 0]. Linear weights at [0]: distances [0, 10, 20, 0]
        -> [1, 0, 0, 1] -> redistributed [1.5, 0.5, 0] -> [0.75, 0.25, 0]."""
        with self._axis_aligned_pod() as (pod, template):
            assert self._feed_axis(pod, template, [1.0, 0.5, 0.0], 0.0) is False
            assert pod.snapshot_matrix.n_snapshots == 3
            d = pod.snapshot_matrix.rejected_snapshot_reconstruction_weights[0]
            assert d == pytest.approx([0.5, 0.5, 0.0], abs=1e-10)
            w, distnorm = pod._local_weighted_snapshot_weights(np.array([0.0]), 'Linear', None, None, 3)
            assert w == pytest.approx([0.75, 0.25, 0.0], abs=1e-10)
            assert distnorm.shape == (3,)

    def test_n_rej_zero_is_unchanged(self):
        with self._axis_aligned_pod() as (pod, template):
            w, distnorm = pod._local_weighted_snapshot_weights(np.array([0.0]), 'Linear', None, None, 3)
            assert w == pytest.approx([1.0, 0.0, 0.0], abs=1e-10)
            assert distnorm.shape == (3,)

    def test_cubic_n_nonzero_weights_counts_rejected_points_too(self):
        """Distances [0, 10, 20, 0], k = 1: the tie at 0 widens k to 2,
        dist_hat = 10 -> [1, 0, 0, 1] -> [0.75, 0.25, 0]; 2 basis columns."""
        with self._axis_aligned_pod() as (pod, template):
            assert self._feed_axis(pod, template, [1.0, 0.5, 0.0], 0.0) is False
            w, _ = pod._local_weighted_snapshot_weights(np.array([0.0]), 'Cubic', None, 1, 3)
            assert w == pytest.approx([0.75, 0.25, 0.0], abs=1e-10)
            pod.construct_basis(rb_size=3, n_nonzero_weights=1, local_parametervector=np.array([0.0]),
                                layout_vec=template)
            assert pod.basis.getSize()[1] == 2

    def test_parameter_weights_scale_the_distance(self):
        """Weighting the second parameter by 0 makes it irrelevant."""
        comm = MPI.COMM_WORLD
        pod = PODBasis(comm, 4, basis_mode='local_weighted')
        template = dense_vec(comm, 4)
        rstart, rend = template.getOwnershipRange()
        for j, par in enumerate(([0.0, 100.0], [1.0, 0.0], [2.0, 0.0])):
            vals = np.zeros(4)
            vals[j] = 1.0
            set_petsc_vec_array(template, vals[rstart:rend])
            pod.snapshot_matrix.process_and_append_to_snapshot_lists(template, np.array(par))
        w, dist = pod._local_weighted_snapshot_weights(np.array([0.0, 0.0]), 'Linear',
                                                       np.array([1.0, 0.0]), None, 3)
        assert dist == pytest.approx([0.0, 1.0, 2.0])
        assert w[0] == pytest.approx(1.0)
        pod.destroy()
        template.destroy()
