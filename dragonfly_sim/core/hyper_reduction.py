"""
Hyper-reduction of the LSPG ROM's Jacobian with DEIM and MDEIM, as comparison
models beside the POD ROM (DG_windtunnel_model(hyper_reduction=...)).

Ported from project-sandbox CompressibleEuler/2D (core/deim.py), for the
weighted LSPG of core/reduced_order_model.py and for every flow model, the
condensed-gradient (MUSCL, RANS) ones included.

What is hyper-reduced. The LSPG Gauss-Newton step needs S r and S A V at
every iteration (S the residual weight, A = dR/dU, V the POD basis). The
residual stays exact (assembled in full): residual DEIM was measured in the
sandbox and fails on these problems, since the residual barely compresses and
interpolating it moves the solution by 5-15%. Only S A V is approximated, from
a sampled assembly of A. S A V is also LSPG's test basis, so the hyper-reduced
solve converges to (S A V)~^T S r = 0 rather than to the ROM's
(S A V)^T S r = 0: its solution differs from the ROM's by as much as the
Jacobian approximation tilts the test space (MDEIM: round-off to 1e-9 on the
test cases; DEIM: far more), besides the cost and convergence of the loop:

  DEIM   a basis U of Jacobian actions S A v_j (snapshots along the POD basis,
         from the full-order Newton iterates) and points P (rows):
             S A V ~= U C,    C = (P^T U)^+ P^T S A V
         with P^T S A V from the sampled rows of A.
             H = C^T (U^T U) C,    g = C^T U^T (S r)
  MDEIM  a basis {Phi_l} of vec(S A) (the scaled Jacobians of the full-order
         Newton iterates, one entry per nonzero) and points P (entries):
             S A ~= sum_l c_l Phi_l,    c = (P^T Phi)^+ P^T vec(S A)
         with the sampled entries of A. Per basis, (S A_l V) = Phi_l V and
         H_lp = (Phi_l V)^T (Phi_p V) are formed once:
             H = sum_lp c_l c_p H_lp,    g = sum_l c_l (Phi_l V)^T (S r)

The snapshots are scaled by S (rows) where the sandbox used A itself: the
LSPG norm is ||S r||, so a basis of S-scaled actions or matrices does not
neglect the rows of small components (with the free-stream state scaling the
energy rows are larger than the SA rows by orders of magnitude), and the
sampling is still exact, since (S A)_ij = S_i A_ij.

Points: the greedy of Chaturantabut & Sorensen (2010), past the basis width
cycling through the modes again (oversampling), partition independent.

Sampled assembly (RestrictedJacobian). A sampled row is a dof of a cell, and
the rows of a cell are exact when the cell and all its facets are integrated
(FormManager.entity_set_for_cells). With a reconstructed gradient,
A = A_UU + A_UG M^-1 B_GU (utils/gradient_condensation.py), and the rows of the
sampled cells C1 read the gradient of C1 and of its facet neighbours: A_UU and
A_UG are integrated over C1, B_GU over C2 = C1 plus one facet ring, and the
product's rows on C1 are exact.

The bases follow the POD ROM's: global when it is, and for a local_weighted
ROM weighted per design with the weights the ROM's basis used (every snapshot
of a design gets that design's weight), see ReducedOrderModel.basis_settings.

With the ROM's hybrid test space (ReducedOrderModel test_space='hybrid') the
solves are those of the same hybrid equations, the state part of the test basis
from the approximate S A V (HyperReducedHybridLSPG).

The adjoint (HyperReducedLSPGAdjoint) is consistent with the forward solve: that
solve's residual is exact, so its solution follows the design through the true
Jacobian A, and with its test basis Phi~ frozen the adjoint is
(Phi~^T A V)^T lam = V^T g, psi = Phi~ lam, linearized about the method's own
solution. Forming Phi~^T A V costs one full Jacobian assembly per design.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import scipy.linalg as sp_la
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial.distance import cdist

from dragonfly_sim.core.pod import PODBasis
from dragonfly_sim.core.reduced_order_model import (LSPGSolver, HybridLSPGSolver, HybridLSPGAdjoint,
                                                     hybrid_test_basis, scaled_jacobian_basis)
from dragonfly_sim.utils.petsc_utils import set_petsc_vec_array
from dragonfly_sim.utils.solver_profiling import PROFILER as _PROF

# the CFFI options of the Newton problems' forms (Euler_model.set_up_solver,
# core/reconstruction.py), so that the restricted kernels are the same code
_JIT = {"cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"]}

# DEIM snapshot keys: the design's parameter vector and a trailing slot, the
# Newton iterate plus column * ACTION_COLUMN_KEY_STEP (column of the POD basis),
# unique per snapshot so that none is taken for a duplicate parameter
ACTION_COLUMN_KEY_STEP = 1e-6

METHODS = ('deim', 'mdeim')


# ----------------------------------------------------------------------
# Point selection
# ----------------------------------------------------------------------
def _global_argmax_abs(comm, values_local, rstart, excluded):
    """(global index, magnitude) of the largest |entry| over all ranks,
    skipping the global indices `excluded`. Ties go to the lowest index, so the
    result does not depend on the partitioning."""
    magnitude = np.abs(np.asarray(values_local, dtype=float))
    if len(excluded):
        local = np.asarray(excluded, dtype=np.int64) - rstart
        local = local[(local >= 0) & (local < magnitude.size)]
        magnitude[local] = -1.0
    if magnitude.size:
        i = int(np.argmax(magnitude))
        local_best = (float(magnitude[i]), int(rstart + i))
    else:
        # a rank can own no rows; sorts last below
        local_best = (-np.inf, np.iinfo(np.int64).max)
    value, index = min(comm.allgather(local_best), key=lambda vi: (-vi[0], vi[1]))
    return int(index), float(value)


def gather_rows(comm, local, rstart, indices):
    """Rows `indices` (global) of a row-distributed array (`local` this rank's
    rows, the first global index rstart), replicated: (len(indices),) for a
    vector, (len(indices), k) for a matrix. Collective."""
    local = np.asarray(local, dtype=float)
    indices = np.asarray(indices, dtype=np.int64)
    out = np.zeros((indices.size,) + local.shape[1:])
    pos = indices - rstart
    mine = (pos >= 0) & (pos < local.shape[0])
    out[mine] = local[pos[mine]]
    return comm.allreduce(out, op=MPI.SUM)


def greedy_deim_indices(comm, basis_local, rstart, n_points=None, n_modes=None):
    """
    DEIM points of a row-distributed basis (this rank's (m_local, m) rows,
    the first global row rstart): global row indices in selection order,
    identical on every rank. Step j fits mode l = j % n_modes by its
    predecessors at the points so far,

        c = lstsq(P_U[:, :l], P_U[:, l]),    r = U[:, l] - U[:, :l] c,

    and takes the next point at max |r| (selected rows excluded). For
    j < n_modes this is the square interpolation of the published algorithm;
    past it the least-squares fit oversamples.
    """
    basis_local = np.asarray(basis_local, dtype=float)
    width = basis_local.shape[1]
    n_modes = width if n_modes is None else min(int(n_modes), width)
    if n_modes < 1:
        raise ValueError("the basis has no columns to select points for")
    n_points = n_modes if n_points is None else int(n_points)
    n_rows_global = comm.allreduce(basis_local.shape[0], op=MPI.SUM)
    if not 1 <= n_points <= n_rows_global:
        raise ValueError("cannot select {} points from {} rows".format(n_points, n_rows_global))

    indices = []
    P_U = np.zeros((0, n_modes))
    for j in range(n_points):
        l = j % n_modes
        if l == 0:
            residual_local = basis_local[:, 0]
        else:
            c = np.linalg.lstsq(P_U[:, :l], P_U[:, l], rcond=None)[0]
            residual_local = basis_local[:, l] - basis_local[:, :l] @ c
        global_index, _ = _global_argmax_abs(comm, residual_local, rstart, indices)
        indices.append(global_index)
        P_U = np.vstack([P_U, gather_rows(comm, basis_local[:, :n_modes], rstart, [global_index])])
    return np.asarray(indices, dtype=np.int64)


def design_weights(settings, snapshot_parameters):
    """
    The weight of each snapshot (rows of `snapshot_parameters`: the design's
    parameter vector and a trailing slot) under the ROM basis `settings`
    (ReducedOrderModel.basis_settings): its design's weight in the ROM's
    local_weighted basis, 0 for a design the ROM does not hold; None for a
    global basis.
    """
    if settings is None or settings['basis_mode'] != 'local_weighted':
        return None
    designs = np.asarray(snapshot_parameters, dtype=float)[:, :-1]
    d = cdist(designs, settings['design_parameters'])
    nearest = d.argmin(axis=1)
    match = d[np.arange(d.shape[0]), nearest] <= 1e-10
    return np.where(match, settings['design_weights'][nearest], 0.0)


# ----------------------------------------------------------------------
# Sampled assembly
# ----------------------------------------------------------------------
class RestrictedJacobian:
    """
    The steady Jacobian dR/dU of a flow model (its F) with its rows exact on
    the cells of a set of sampled rows and not computed elsewhere: assembled
    over those cells and their facets only (see the module docstring for the
    condensed models). bind() sets the rows, assemble() the values at the
    model's current state (with a reconstructed gradient, G must be up to
    date: the full residual assembly of the LSPG step updates it).
    """

    def __init__(self, model):
        self.model = model
        self.condensed = model.has_condensed_jacobian
        forms, u = model.forms, model.u_vec
        if self.condensed:
            rec = model.reconstruction
            rec.set_up()
            G = rec.G
            self._keys = (("hyper_reduction", "A_UU"), ("hyper_reduction", "A_UG"),
                          ("hyper_reduction", "B_GU"))
            forms.ufl_form(self._keys[0], lambda: ufl.derivative(model.F, u), jit_options=_JIT)
            forms.ufl_form(self._keys[1], lambda: ufl.derivative(model.F, G, ufl.TrialFunction(G.function_space)),
                           jit_options=_JIT)
            forms.ufl_form(self._keys[2], lambda: ufl.derivative(rec.gg_rhs, u, ufl.TrialFunction(u.function_space)),
                           jit_options=_JIT)
        else:
            self._keys = (("hyper_reduction", "J"),)
            forms.ufl_form(self._keys[0], lambda: ufl.derivative(model.F, u), jit_options=_JIT)
        self._twin = None              # the sampled cells C1
        self._closure_twin = None      # C2: C1 and its facet neighbours (condensed)
        self._mats = {}
        self.A = None
        self.n_cells = 0
        self.n_closure_cells = 0

    def bind(self, rows_local):
        """Sample the rows `rows_local` (this rank's owned rows, local
        indices). Collective."""
        forms = self.model.forms
        comm = self.model.mesh_obj.mesh.comm
        self._destroy_mats()
        cells = forms.cells_for_dofs(rows_local)
        entity_set = forms.entity_set_for_cells(cells)
        if self._twin is None:
            self._twin = forms.restrict(entity_set=entity_set)
        else:
            self._twin.rebind(entity_set)
        self.n_cells = comm.allreduce(int(cells.size), op=MPI.SUM)
        self.n_closure_cells = self.n_cells
        if self.condensed:
            closure = forms.cell_neighbours(cells)
            closure_set = forms.entity_set_for_cells(closure)
            if self._closure_twin is None:
                self._closure_twin = forms.restrict(entity_set=closure_set)
            else:
                self._closure_twin.rebind(closure_set)
            self.n_closure_cells = comm.allreduce(int(closure.size), op=MPI.SUM)

    def assemble(self):
        """The sampled Jacobian at the current state, a Mat in the state
        layout reused by every call of one binding. Collective."""
        with _PROF.phase("assemble:jacobian_sampled"):
            if not self.condensed:
                self.A = self._twin.assemble_matrix_petsc(self._keys[0], result=self.A)
                return self.A
            m = self._mats
            k_UU, k_UG, k_GU = self._keys
            m['UU'] = self._twin.assemble_matrix_petsc(k_UU, result=m.get('UU'))
            m['UG'] = self._twin.assemble_matrix_petsc(k_UG, result=m.get('UG'))
            m['GU'] = self._closure_twin.assemble_matrix_petsc(k_GU, result=m.get('GU'))
            m['GU'].diagonalScale(L=self.model.reconstruction.inv_mass)
            if 'C' not in m:
                m['C'] = m['UG'].matMult(m['GU'])
            else:
                m['UG'].matMult(m['GU'], result=m['C'])
            if self.A is None:
                self.A = m['UU'].duplicate(copy=True)
                self.A.axpy(1.0, m['C'], structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            else:
                self.A.zeroEntries()
                self.A.axpy(1.0, m['UU'], structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
                self.A.axpy(1.0, m['C'], structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            self.A.assemble()
            return self.A

    def _destroy_mats(self):
        # the sparsity belongs to one binding
        for M in list(self._mats.values()) + [self.A]:
            if M is not None:
                M.destroy()
        self._mats = {}
        self.A = None

    def destroy(self):
        self._destroy_mats()


# ----------------------------------------------------------------------
# DEIM and MDEIM
# ----------------------------------------------------------------------
class DEIM:
    """
    DEIM of the scaled Jacobian action S A V (module docstring): a POD basis U
    of snapshots S A v_j, the points on it, and the reduced operators of one
    solve (prepare). The POD's basis_mode is the ROM's; its weights come from
    the ROM per solve (build_basis).
    """
    name = 'deim'

    def __init__(self, comm, n_modes=100, n_points=None, basis_mode='global', max_rank=None,
                 qr_rank_tol=1e-10, min_snapshot_norm=None):
        self.comm = comm
        self.n_modes = n_modes
        # None: one point per basis vector; more oversample
        self.n_points = n_points
        self.min_snapshot_norm = min_snapshot_norm
        self.pod = PODBasis(comm, None, basis_mode=basis_mode,
                            max_rank=2 * n_modes if max_rank is None else max_rank, qr_rank_tol=qr_rank_tol)
        self.rjac = None
        self._indices = None
        self._pinv = None
        self._V = None
        self._S_local = None

    # -- snapshots and basis -------------------------------------------------
    @property
    def snapshot_matrix(self):
        return self.pod.snapshot_matrix

    @property
    def n_snapshots(self):
        return self.pod.snapshot_matrix.n_snapshots

    @property
    def basis(self):
        return self.pod.basis

    @property
    def n_basis_modes(self):
        return 0 if self.basis is None else self.basis.getSize()[1]

    def add_snapshot(self, vec, key):
        """Add a snapshot (a Vec, not modified) under the parameter key `key`.
        Returns whether it added a basis direction. Collective."""
        norm = vec.norm()
        if not np.isfinite(norm) or norm == 0.0 or (self.min_snapshot_norm is not None
                                                     and norm < self.min_snapshot_norm):
            return False
        return self.snapshot_matrix.process_and_append_to_snapshot_lists(vec, np.asarray(key, dtype=float))

    def build_basis(self, settings, layout_vec=None):
        """The basis of the next solve, built like the ROM's (`settings`,
        ReducedOrderModel.basis_settings). Returns its width. Collective."""
        self._indices = self._pinv = None
        weights = None
        if self.pod.basis_mode == 'local_weighted' and self.n_snapshots >= 2:
            weights = design_weights(settings, self.snapshot_matrix.snapshot_list_parameters)
            if weights is None:
                # the ROM built a global basis (one snapshot): so does this one
                weights = np.ones(self.n_snapshots)
            elif not np.any(weights > 0.0):
                PETSc.Sys.Print("{}: no snapshot from a design the ROM weights; using uniform weights".format(
                    self.name.upper()))
                weights = np.ones(self.n_snapshots)
        self.pod.construct_basis(rb_size=self.n_modes, layout_vec=layout_vec, snapshot_weights=weights)
        return self.n_basis_modes

    def select_indices(self):
        """The points of the current basis (global indices). Collective."""
        basis = self.basis
        rstart, _ = basis.getOwnershipRange()
        width = self.n_basis_modes
        n_points = width if self.n_points is None else int(self.n_points)
        self._indices = greedy_deim_indices(self.comm, basis.getDenseArray(), rstart,
                                            n_points=n_points, n_modes=width)
        self._pinv = None
        return self._indices

    @property
    def indices(self):
        return self._indices

    def interpolation_matrix(self):
        """P^T U, replicated. Collective."""
        rstart, _ = self.basis.getOwnershipRange()
        return gather_rows(self.comm, self.basis.getDenseArray(), rstart, self._indices)

    def pseudo_inverse(self):
        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.interpolation_matrix())
        return self._pinv

    def conditioning(self):
        """1 / sigma_min(P^T U), the DEIM error amplification (inf: singular).
        Collective."""
        s = np.linalg.svd(self.interpolation_matrix(), compute_uv=False)
        return np.inf if s.size == 0 or s[-1] == 0.0 else float(1.0 / s[-1])

    def _owned_rows(self, layout_vec):
        """This rank's owned rows among the points, local indices."""
        rstart, rend = layout_vec.getOwnershipRange()
        sel = self._indices
        return (sel[(sel >= rstart) & (sel < rend)] - rstart).astype(np.int32)

    # -- one solve's operators -----------------------------------------------
    def prepare(self, model, V, S, hybrid=None):
        """
        Bind the sampled Jacobian to the points and build the operators of a
        solve on POD basis V with residual weight S (Vec or None), and with
        `hybrid` = (n_state, Psi) (ReducedOrderModel.hybrid_test_space) those of the
        hybrid test basis [S^2 A V_s, Psi]. Needs at least as many DEIM basis vectors
        as V has columns, else the reduced Jacobian is singular (ValueError).
        Collective.
        """
        k = V.getSize()[1]
        if self.n_basis_modes < k:
            raise ValueError("{} basis vectors for a POD basis of {}: the hyper-reduced Jacobian would be "
                             "singular".format(self.n_basis_modes, k))
        layout = model.u_vec.x.petsc_vec
        if self.rjac is None:
            self.rjac = RestrictedJacobian(model)
        self.rjac.bind(self._owned_rows(layout))
        self._V = V
        self._S_local = None if S is None else S.getArray(readonly=True).copy()
        rstart, _ = layout.getOwnershipRange()
        self._S_P = np.ones(self._indices.size) if S is None else \
            gather_rows(self.comm, self._S_local, rstart, self._indices)
        U_local = self.basis.getDenseArray()
        self._gram = self.comm.allreduce(U_local.T @ U_local, op=MPI.SUM)
        self._set_hybrid(hybrid)
        if self._hybrid is not None:
            # Psi^T S^-1 U: the adjoint rows of the hybrid Phi^T A V ~ Psi^T S^-1 U C
            self._Psi_U = self.comm.allreduce(self._Psi_S.T @ U_local, op=MPI.SUM)
        if getattr(self, "_AV", None) is not None:
            # a product of the previous binding's matrix
            self._AV.destroy()
        self._AV = None

    def jacobian_coefficients(self):
        """C = (P^T U)^+ P^T S A V at the model's current state, from the
        sampled assembly. Collective."""
        A_s = self.rjac.assemble()
        with _PROF.phase("rom:hr_sample"):
            if self._AV is None:
                self._AV = A_s.matMult(self._V)
            else:
                A_s.matMult(self._V, result=self._AV)
            rstart, _ = self._AV.getOwnershipRange()
            PAV = gather_rows(self.comm, self._AV.getDenseArray(), rstart, self._indices)
            return self.pseudo_inverse() @ (self._S_P[:, None] * PAV)

    def reduced_system(self, Sr_local):
        """(H, g) of the hyper-reduced Gauss-Newton step, Sr_local this rank's
        entries of S r. Collective."""
        C = self.jacobian_coefficients()
        with _PROF.phase("rom:hr_project"):
            g = C.T @ self.comm.allreduce(self.basis.getDenseArray().T @ Sr_local, op=MPI.SUM)
            return C.T @ self._gram @ C, g

    def hybrid_reduced_system(self, Sr_local):
        """(Phi^T A V, Phi^T r, (S A V)^T S r) of the hybrid test basis Phi =
        [S^2 A V_s, Psi] (prepare), S A V ~ U C. Collective."""
        C = self.jacobian_coefficients()
        n_state = self._hybrid
        with _PROF.phase("rom:hr_project"):
            merit = C.T @ self.comm.allreduce(self.basis.getDenseArray().T @ Sr_local, op=MPI.SUM)
            H = np.vstack([C[:, :n_state].T @ self._gram @ C, self._Psi_U @ C])
            g = np.concatenate([merit[:n_state], self.comm.allreduce(self._Psi_S.T @ Sr_local, op=MPI.SUM)])
            return H, g, merit

    def _set_hybrid(self, hybrid):
        # n_state, and Psi S^-1 (this rank's rows) for the adjoint rows of the hybrid system
        if hybrid is None:
            self._hybrid = self._Psi = self._Psi_S = None
            return
        self._hybrid, self._Psi = hybrid
        self._Psi_S = self._Psi if self._S_local is None else self._Psi / self._S_local[:, None]

    def scaled_jacobian_basis(self):
        """This rank's rows of the approximation of S A V at the current
        state, U C. Collective."""
        return self.basis.getDenseArray() @ self.jacobian_coefficients()

    def release(self):
        if getattr(self, "_AV", None) is not None:
            self._AV.destroy()
        self._AV = None
        self._V = None

    def destroy(self):
        self.release()
        if self.rjac is not None:
            self.rjac.destroy()
            self.rjac = None
        self.pod.destroy()


class MDEIM(DEIM):
    """
    MDEIM of the scaled Jacobian S A (module docstring): a POD basis of
    vec(S A), one entry per nonzero of the Newton Jacobian's sparsity (fixed:
    the mesh moves, its topology does not; capture_pattern checks it), the
    points (entries) on it and the affine operators of one solve.
    """
    name = 'mdeim'

    def __init__(self, comm, n_modes=10, n_points=None, basis_mode='global', max_rank=15,
                 qr_rank_tol=1e-10, min_snapshot_norm=None):
        super().__init__(comm, n_modes=n_modes, n_points=n_points, basis_mode=basis_mode,
                         max_rank=max_rank, qr_rank_tol=qr_rank_tol, min_snapshot_norm=min_snapshot_norm)
        self._indptr = None
        self._cols = None
        self._row_of_nz = None
        self.template_vec = None
        self._SAV = None
        self._H = None

    # -- sparsity and snapshots ---------------------------------------------
    @property
    def pattern_is_captured(self):
        return self._indptr is not None

    def capture_pattern(self, mat):
        """Record the sparsity of the Jacobian `mat` (this rank's rows, global
        columns), or check that it has not changed. Collective on the first
        call."""
        indptr, cols, _ = mat.getValuesCSR()
        if self._indptr is not None:
            if not (np.array_equal(indptr, self._indptr) and np.array_equal(cols, self._cols)):
                raise RuntimeError("MDEIM: the Jacobian's sparsity changed ({} -> {} local nonzeros); every "
                                   "snapshot must share one".format(self._indptr[-1], indptr[-1]))
            return
        self._indptr, self._cols = indptr.copy(), cols.copy()
        self._row_of_nz = np.repeat(np.arange(indptr.size - 1), np.diff(indptr)).astype(np.int64)
        self.template_vec = PETSc.Vec().createMPI((int(indptr[-1]), PETSc.DECIDE), comm=self.comm)
        PETSc.Sys.Print("MDEIM: Jacobian sparsity of {} nonzeros".format(self.template_vec.getSize()))

    def scaled_values(self, mat, S_local):
        """This rank's entries of vec(S A) for the Jacobian `mat` (in the
        captured sparsity). Collective on the first call."""
        self.capture_pattern(mat)
        values = mat.getValuesCSR()[2]
        return values if S_local is None else values * S_local[self._row_of_nz]

    def add_values(self, values_local, key):
        set_petsc_vec_array(self.template_vec, values_local)
        return self.add_snapshot(self.template_vec, key)

    def _owned_entries(self):
        """(positions in this rank's nonzeros, their order among the points)
        of the points this rank owns."""
        start, _ = self.template_vec.getOwnershipRange()
        sel = self._indices
        order = np.flatnonzero((sel >= start) & (sel < start + self._indptr[-1]))
        return (sel[order] - start).astype(np.int64), order

    # -- one solve's operators -----------------------------------------------
    def prepare(self, model, V, S, hybrid=None):
        """Bind the sampled Jacobian to the rows of the points and build the
        affine operators Phi_l V and H_lp on POD basis V (and, with `hybrid`, those
        of the hybrid test basis, see DEIM.prepare). Collective."""
        k = V.getSize()[1]
        m = self.n_basis_modes
        if m < 1:
            raise ValueError("MDEIM: empty basis")
        layout = model.u_vec.x.petsc_vec
        if layout.getLocalSize() != self._indptr.size - 1:
            raise RuntimeError("MDEIM: the captured sparsity has {} local rows, the state {}".format(
                self._indptr.size - 1, layout.getLocalSize()))
        if self.rjac is None:
            self.rjac = RestrictedJacobian(model)
        positions, self._order = self._owned_entries()
        rows = self._row_of_nz[positions]
        self.rjac.bind(np.unique(rows).astype(np.int32))
        self._S_local = None if S is None else S.getArray(readonly=True).copy()
        self._S_entry = np.ones(positions.size) if S is None else self._S_local[rows]

        # where the points are in the sampled Jacobian's (fixed) sparsity; -1 for an entry
        # outside it, a structural zero of the steady Jacobian
        A_s = self.rjac.assemble()
        indptr, cols, _ = A_s.getValuesCSR()
        target = self._cols[positions]
        self._positions = np.full(positions.size, -1, dtype=np.int64)
        for i, (r, c) in enumerate(zip(rows, target)):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            j = lo + int(np.searchsorted(cols[lo:hi], c))
            if j < hi and cols[j] == c:
                self._positions[i] = j

        with _PROF.phase("rom:mdeim_offline"):
            self._build_affine_operators(V, layout)
            self._set_hybrid(hybrid)
            if self._hybrid is not None:
                # Psi^T S^-1 (S A_l V): the adjoint rows of the hybrid Phi^T A V
                self._Psi_SAV = self.comm.allreduce(np.einsum('ia,lik->lak', self._Psi_S, self._SAV), op=MPI.SUM)
        self._V = V

    def _build_affine_operators(self, V, layout):
        # Phi_l V for every basis matrix Phi_l (rows of this rank) and H_lp = (Phi_l V)^T (Phi_p V)
        n_local, n_global = layout.getLocalSize(), layout.getSize()
        scratch = PETSc.Mat().create(comm=self.comm)
        scratch.setSizes(((n_local, n_global), (n_local, n_global)))
        scratch.setType('aij')
        scratch.setPreallocationCSR((self._indptr, self._cols))
        phi = self.basis.getDenseArray()
        SAV = []
        try:
            for l in range(self.n_basis_modes):
                scratch.zeroEntries()
                scratch.setValuesCSR(self._indptr, self._cols, phi[:, l].copy())
                scratch.assemble()
                AV_l = scratch.matMult(V)
                SAV.append(AV_l.getDenseArray().copy())
                AV_l.destroy()
        finally:
            scratch.destroy()
        self._SAV = np.stack(SAV)                    # (m, n_local, k)
        self._H = self.comm.allreduce(np.einsum('lik,pij->lpkj', self._SAV, self._SAV), op=MPI.SUM)

    def sampled_values(self):
        """P^T vec(S A) at the current state, from the sampled assembly,
        replicated. Collective."""
        A_s = self.rjac.assemble()
        with _PROF.phase("rom:hr_sample"):
            data = A_s.getValuesCSR()[2]
            owned = np.where(self._positions >= 0, data[np.maximum(self._positions, 0)], 0.0) * self._S_entry
            contribution = np.zeros(self._indices.size)
            contribution[self._order] = owned
            return self.comm.allreduce(contribution, op=MPI.SUM)

    def coefficients(self):
        return self.pseudo_inverse() @ self.sampled_values()

    def reduced_system(self, Sr_local):
        c = self.coefficients()
        with _PROF.phase("rom:hr_project"):
            parts = self.comm.allreduce(np.einsum('lik,i->lk', self._SAV, Sr_local), op=MPI.SUM)
            return np.einsum('l,p,lpkj->kj', c, c, self._H), c @ parts

    def hybrid_reduced_system(self, Sr_local):
        """(Phi^T A V, Phi^T r, (S A V)^T S r) of the hybrid test basis, S A V ~
        sum_l c_l S A_l V. Collective."""
        c = self.coefficients()
        n_state = self._hybrid
        with _PROF.phase("rom:hr_project"):
            merit = c @ self.comm.allreduce(np.einsum('lik,i->lk', self._SAV, Sr_local), op=MPI.SUM)
            H_state = np.einsum('l,p,lpkj->kj', c, c, self._H)[:n_state]
            H = np.vstack([H_state, np.einsum('l,lak->ak', c, self._Psi_SAV)])
            g = np.concatenate([merit[:n_state], self.comm.allreduce(self._Psi_S.T @ Sr_local, op=MPI.SUM)])
            return H, g, merit

    def scaled_jacobian_basis(self):
        return np.einsum('l,lik->ik', self.coefficients(), self._SAV)

    def release(self):
        self._SAV = self._H = None
        self._Psi_SAV = None
        self._V = None

    def destroy(self):
        super().destroy()
        if self.template_vec is not None:
            self.template_vec.destroy()
            self.template_vec = None


# ----------------------------------------------------------------------
# Hyper-reduced LSPG solve and adjoint
# ----------------------------------------------------------------------
class HyperReducedLSPG(LSPGSolver):
    """LSPGSolver whose Gauss-Newton system comes from a prepared DEIM or MDEIM
    (`operator`) instead of the full Jacobian; the residual, the step, its
    limiting and every stopping rule are LSPGSolver's."""

    def __init__(self, V, comm, operator=None, **kwargs):
        super().__init__(V, comm, **kwargs)
        self.operator = operator

    def _reduced_system(self, model):
        r, r_norm = self._weighted_residual(model)
        H, g = self.operator.reduced_system(r)
        return H, g, r_norm


class HyperReducedHybridLSPG(HybridLSPGSolver):
    """HybridLSPGSolver (the ROM's test_space='hybrid') whose system comes from a
    prepared DEIM or MDEIM (`operator`, prepared with the hybrid test basis): the
    same equations Phi^T R(V a) = 0 with Phi = [S^2 A V_s, Psi], S A V approximated."""

    def __init__(self, V, comm, operator=None, **kwargs):
        super().__init__(V, comm, **kwargs)
        self.operator = operator

    def _reduced_system(self, model):
        Sr, r_norm = self._weighted_residual(model)
        H, g, self._merit_gradient = self.operator.hybrid_reduced_system(Sr)
        return H, g, r_norm


class HyperReducedLSPGAdjoint(HybridLSPGAdjoint):
    """
    The adjoint consistent with a hyper-reduced forward solve: that solve converges to
    Phi~^T R(V a) = 0, Phi~ its test basis from the hyper-reduced S A V (S^2 A~ V, or
    [S^2 A~ V_s, Psi] in the hybrid case), with the residual R exact. Its solution therefore
    follows the design through the TRUE Jacobian: with Phi~ frozen (the terms proportional to
    R dropped, as in LSPGAdjoint), (Phi~^T A V)^T lam = V^T g, psi = Phi~ lam. Forming
    Phi~^T A V costs one full Jacobian assembly per state (shared by all right-hand sides).
    Evaluate it at the hyper-reduced solution, the state the forward solve linearizes about.
    Owns S, not V or Psi.
    """

    def __init__(self, operator, V, S, comm, parameter_vector, solution, hybrid=None):
        n_state, Psi = (V.getSize()[1], None) if hybrid is None else hybrid
        super().__init__(V, S, comm, parameter_vector, solution, accepted=False, n_state=n_state, Psi=Psi)
        self.operator = operator

    def _factor(self, model):
        # the reconstructed gradient at this state, which the sampled assembly reads
        model.update_derived_fields()
        SAV_hr = self.operator.scaled_jacobian_basis()
        S = None if self.S is None else self.S.getArray(readonly=True)
        Psi = np.zeros((SAV_hr.shape[0], 0)) if self.Psi is None else self.Psi
        self._Phi = hybrid_test_basis(SAV_hr, S, self.n_state, Psi)
        model.set_up_solver()
        with model.steady_residual():
            _, AV = scaled_jacobian_basis(model, self.V, self.S)       # the true Jacobian
            H = self.comm.allreduce(self._Phi.T @ AV.getDenseArray(), op=MPI.SUM)
        AV.destroy()
        self._lu = sp_la.lu_factor(H)
        self._R = np.diag(np.abs(np.diag(self._lu[0])))
        diag = np.diag(self._R)
        if not np.all(np.isfinite(self._lu[0])) or diag.min() <= np.finfo(float).eps * diag.max():
            raise ValueError("hyper-reduced adjoint: Phi^T A V is singular to working precision "
                             "(|U_ii| from {:.3e} to {:.3e})".format(diag.min(), diag.max()))


# ----------------------------------------------------------------------
# The comparison driver
# ----------------------------------------------------------------------
@dataclass
class HyperReducedResult:
    method: str                # 'deim' or 'mdeim'
    result: object             # the ROMResult of the hyper-reduced solve
    offline_walltime: float    # basis, points, sampled assembly set-up and operators
    info: dict = field(default_factory=dict)
    validation: dict = None


class HyperReductionComparison:
    """
    DEIM and/or MDEIM beside a POD ROM (DG_windtunnel_model(hyper_reduction=...)):
    trained on the full-order Jacobians of the FOM solves, they solve every
    design the ROM solves, with the ROM's basis, weights, warm start and
    stopping rules, and the adjoint of every derivative the optimizer takes,
    and the windtunnel model records it all against the ROM and the FOM. The
    optimizer's solution and adjoint are never theirs.

      methods                           any of 'deim', 'mdeim'
      deim_n_modes, mdeim_n_modes       basis widths (DEIM needs at least the
                                        ROM's rb_size)
      deim_n_points, mdeim_n_points     points (None: one per basis vector)
      deim_max_rank, mdeim_max_rank     stored snapshot SVD rank (DEIM: default
                                        2 deim_n_modes)
      deim_max_snapshots_per_solve, mdeim_max_snapshots_per_solve
                                        Newton iterates harvested per FOM solve
                                        (the converged state is always added)
      min_snapshot_norm                 drop snapshots below this norm
      qr_rank_tol                       as ReducedOrderModel's
      validate                          also measure, at the optimizer's
                                        solution, the sampled assembly's
                                        exactness and the approximation error
                                        (full Jacobian assemblies; diagnostic)

    The defaults are those of the project-sandbox windtunnel model.
    """

    def __init__(self, methods=METHODS, deim_n_modes=100, deim_n_points=None, deim_max_rank=None,
                 deim_max_snapshots_per_solve=10, mdeim_n_modes=10, mdeim_n_points=None, mdeim_max_rank=15,
                 mdeim_max_snapshots_per_solve=4, min_snapshot_norm=None, qr_rank_tol=1e-10, validate=False):
        methods = tuple(methods)
        if not methods or any(m not in METHODS for m in methods):
            raise ValueError("methods must be a nonempty subset of {}, got {}".format(METHODS, methods))
        self.methods = methods
        self.deim_n_modes, self.deim_n_points, self.deim_max_rank = deim_n_modes, deim_n_points, deim_max_rank
        self.mdeim_n_modes, self.mdeim_n_points, self.mdeim_max_rank = mdeim_n_modes, mdeim_n_points, mdeim_max_rank
        self.max_snapshots_per_solve = {'deim': deim_max_snapshots_per_solve,
                                        'mdeim': mdeim_max_snapshots_per_solve}
        self.min_snapshot_norm = min_snapshot_norm
        self.qr_rank_tol = qr_rank_tol
        self.validate = validate
        self.models = {}
        self.comm = None
        self._buffers = {m: [] for m in methods}
        self._harvested = 0
        self._harvest_S = None
        self._harvest_basis = None
        # set while the converged state is harvested: it is added whatever the caps
        self._final = False
        # the operators of the last comparison: the design, its POD basis, offset and S
        self.design = None
        self._compared_key = None
        self._V = self._mean = self._S = None
        self._hybrid = None
        self._solutions = {}
        self._adjoints = {}

    def set_up(self, model, rom):
        """Build the DEIM/MDEIM of `model`, following the bases of `rom`."""
        self.destroy()
        self.comm = model.mesh_obj.mesh.comm
        kw = dict(basis_mode=rom.basis_mode, qr_rank_tol=self.qr_rank_tol, min_snapshot_norm=self.min_snapshot_norm)
        if 'deim' in self.methods:
            self.models['deim'] = DEIM(self.comm, n_modes=self.deim_n_modes, n_points=self.deim_n_points,
                                       max_rank=self.deim_max_rank, **kw)
        if 'mdeim' in self.methods:
            self.models['mdeim'] = MDEIM(self.comm, n_modes=self.mdeim_n_modes, n_points=self.mdeim_n_points,
                                         max_rank=self.mdeim_max_rank, **kw)
        PETSc.Sys.Print("Hyper-reduction comparison: {}; bases follow the ROM's ({})".format(
            ", ".join("{} ({} modes)".format(n.upper(), h.n_modes) for n, h in self.models.items()),
            rom.basis_mode))

    # -- harvesting the full-order Jacobians -----------------------------------
    def _sink(self, A):
        """Buffer the snapshots of the steady Jacobian A at the current FOM
        state (every method below its per-solve cap). Collective."""
        slot = self._harvested
        self._harvested += 1
        S = self._harvest_S
        for name, hr in self.models.items():
            cap = self.max_snapshots_per_solve[name]
            if cap is not None and slot >= cap and not self._final:
                continue
            if name == 'mdeim':
                self._buffers[name].append((slot, hr.scaled_values(A, S).copy()))
            elif self._harvest_basis is not None:
                AV = A.matMult(self._harvest_basis)
                SAV = AV.getDenseArray().copy()
                AV.destroy()
                if S is not None:
                    SAV *= S[:, None]
                self._buffers[name].append((slot, SAV))

    @contextmanager
    def harvesting(self, model, rom, basis):
        """Context of a FOM solve whose Newton Jacobians are harvested;
        `basis` is the POD basis of this evaluation's ROM solve (None: none, and
        DEIM, whose snapshots are actions along it, harvests nothing).
        Collective."""
        self.discard_buffers()
        self._harvest_basis = basis
        S = rom.residual_weight(model)
        self._harvest_S = None if S is None else S.getArray(readonly=True).copy()
        if S is not None:
            S.destroy()
        self._final = False
        cap = max(c if c is not None else np.iinfo(np.int32).max for c in self.max_snapshots_per_solve.values())
        with model.record_fom_jacobians(self._sink, max_jacobians=cap):
            yield

    def harvest_converged_state(self, model):
        """Add the steady Jacobian at the model's current (converged FOM)
        state, which the Newton iterates never assemble. Call after
        harvesting(). Collective."""
        model.set_up_solver()
        problem = model.problem
        with _PROF.phase("rom:hr_harvest"):
            with model.steady_residual():
                problem.J(model.u_vec.x.petsc_vec, problem._A)
            self._final = True
            try:
                self._sink(problem._A)
            finally:
                self._final = False

    def flush(self, parameter_vector):
        """Add the buffered snapshots under the design `parameter_vector`
        (after the comparison of this design, so that it is out of sample).
        Returns {method: (offered, added)}. Collective."""
        pv = np.asarray(parameter_vector, dtype=float).ravel()
        counts = {}
        for name, hr in self.models.items():
            offered = added = 0
            for slot, payload in self._buffers[name]:
                if name == 'mdeim':
                    offered += 1
                    added += bool(hr.add_values(payload, np.append(pv, float(slot))))
                    continue
                vec = PETSc.Vec().createMPI((payload.shape[0], PETSc.DECIDE), comm=self.comm)
                for j in range(payload.shape[1]):
                    set_petsc_vec_array(vec, payload[:, j])
                    offered += 1
                    added += bool(hr.add_snapshot(vec, np.append(pv, slot + j * ACTION_COLUMN_KEY_STEP)))
                vec.destroy()
            counts[name] = (offered, added)
            PETSc.Sys.Print("{} snapshots: {} of {} added a direction; {} stored, rank {}".format(
                name.upper(), added, offered, hr.n_snapshots, hr.snapshot_matrix.rank))
        self.discard_buffers()
        return counts

    def discard_buffers(self):
        self._buffers = {m: [] for m in self.models}
        self._harvested = 0
        self._harvest_basis = None

    # -- forward comparison --------------------------------------------------
    def compare(self, model, rom, parameter_vector, warm_start, phase_record):
        """
        Solve the design `parameter_vector` with each method on the basis,
        offset, weights and warm start of `rom`'s last solve (the solve of this
        design); `phase_record(kind)` is a context recording the phases of a
        part ('<method>_offline', '<method>'). The model's u_vec must hold the
        optimizer's solution (the validation is measured there); it holds the
        last method's solution afterwards. A re-run of the design compared
        last, with no snapshot added since, is skipped. Returns a list of
        HyperReducedResult. Collective.
        """
        pv = np.asarray(parameter_vector, dtype=float).ravel()
        key = (pv.tobytes(), rom.n_snapshots, rom.n_adjoint_snapshots) + \
            tuple(h.n_snapshots for h in self.models.values())
        if key == self._compared_key:
            PETSc.Sys.Print("Hyper-reduction: design already compared; keeping its operators")
            return []
        self._release_operators()
        comm = self.comm
        # the residual of the solves is the Newton problem's (freed after the FOM solve with
        # release_solver_memory)
        model.set_up_solver()
        # copies: the ROM rebuilds its basis and offset at its next solve, and the adjoint
        # of this design is taken later
        self._V = rom.basis.duplicate(copy=True)
        self._mean = None if rom.pod.mean is None else rom.pod.mean.copy()
        self._S = rom.residual_weight(model)
        # the ROM's hybrid test basis (test_space='hybrid'), which the hyper-reduced solves mirror
        self._hybrid = rom.hybrid_test_space()
        settings = rom.basis_settings()
        layout = model.u_vec.x.petsc_vec
        reference = layout.getArray(readonly=True).copy()

        results = []
        for name, hr in self.models.items():
            if hr.n_snapshots < 2:
                PETSc.Sys.Print("{}: {} snapshot(s) stored, 2 needed; not compared".format(name.upper(),
                                                                                           hr.n_snapshots))
                continue
            model.set_u_vec(reference)
            t0 = perf_counter()
            try:
                with phase_record(name + "_offline"):
                    with _PROF.phase("rom:" + name + "_basis"):
                        hr.build_basis(settings, layout_vec=hr.template_vec if name == 'mdeim' else layout)
                    if hr.n_basis_modes == 0:
                        raise ValueError("empty basis")
                    with _PROF.phase("rom:" + name + "_select"):
                        hr.select_indices()
                    model.update_derived_fields()
                    hr.prepare(model, self._V, self._S, hybrid=self._hybrid)
                    validation = None
                    if self.validate:
                        from dragonfly_sim.utils.hyper_reduction_validation import validate
                        validation = validate(hr, model, self._V, self._S)
            except ValueError as exc:
                PETSc.Sys.Print("{}: not compared: {}".format(name.upper(), exc))
                hr.release()
                continue
            offline = perf_counter() - t0
            info = dict(n_modes=hr.n_basis_modes, n_points=int(hr.indices.size),
                        sampled_cells=hr.rjac.n_cells, assembled_cells=hr.rjac.n_closure_cells,
                        total_cells=model.mesh_obj.mesh.topology.index_map(
                            model.mesh_obj.mesh.topology.dim).size_global,
                        conditioning=hr.conditioning(), n_snapshots=hr.n_snapshots)
            PETSc.Sys.Print("{} operators: {} modes, {} points on {} of {} cells ({} assembled), conditioning "
                            "{:.3e}; offline wall time {:.3f}".format(
                                name.upper(), info['n_modes'], info['n_points'], info['sampled_cells'],
                                info['total_cells'], info['assembled_cells'], info['conditioning'], offline))
            with phase_record(name):
                model.interpolate_solution_vector(warm_start)
                t0 = perf_counter()
                if self._hybrid is None:
                    solver = rom.make_solver(HyperReducedLSPG, self._V, self._S, self._mean, comm, operator=hr)
                else:
                    solver = rom.make_solver(HyperReducedHybridLSPG, self._V, self._S, self._mean, comm, operator=hr,
                                             n_state=self._hybrid[0], Psi=self._hybrid[1])
                PETSc.Sys.Print("{} hyper-reduced {} solve".format(name.upper(),
                                                                 "LSPG" if self._hybrid is None else "hybrid"))
                result = rom.run_passes(model, solver, t0)
            results.append(HyperReducedResult(name, result, offline, info, validation))
            # the state its adjoint linearizes about
            self._solutions[name] = result.solution
        self.design = pv
        self._compared_key = key
        return results

    # -- adjoint ---------------------------------------------------------------
    def has_operators(self, parameter_vector):
        return self.design is not None and np.array_equal(
            np.asarray(parameter_vector, dtype=float).ravel(), self.design)

    @property
    def prepared_methods(self):
        return [n for n, h in self.models.items() if h._V is not None]

    def solution(self, name):
        """The solution of method `name` at the design compared last (owned entries),
        the state its adjoint linearizes about; None if it was not solved."""
        return self._solutions.get(name)

    def adjoint(self, name, model, parameter_vector, rhs_local):
        """
        The adjoint of method `name`'s forward solve (HyperReducedLSPGAdjoint, consistent
        with it, in the hybrid case with the ROM's hybrid test basis) for the right-hand
        side rhs_local, at the model's current state, which must be that method's solution
        (`solution(name)`) for the adjoint to be the derivative of its solve. Returns
        (psi, wall time). Collective.
        """
        hr = self.models[name]
        u_local = model.u_vec.x.petsc_vec.getArray(readonly=True)
        adj = self._adjoints.get(name)
        if adj is None or not adj.matches(parameter_vector, u_local):
            if adj is not None:
                adj.destroy()
            S = None if self._S is None else self._S.copy()
            adj = HyperReducedLSPGAdjoint(hr, self._V, S, self.comm, parameter_vector, u_local, hybrid=self._hybrid)
            self._adjoints[name] = adj
        t0 = perf_counter()
        psi = adj.solve(model, rhs_local)
        return psi, perf_counter() - t0

    # -- memory ----------------------------------------------------------------
    def _release_operators(self):
        for adj in self._adjoints.values():
            adj.destroy()
        self._adjoints = {}
        for hr in self.models.values():
            hr.release()
        for obj in (self._V, self._mean, self._S):
            if obj is not None:
                obj.destroy()
        self._V = self._mean = self._S = None
        self._hybrid = None
        self._solutions = {}
        self.design = None
        self._compared_key = None

    def destroy(self):
        self._release_operators()
        for hr in self.models.values():
            hr.destroy()
        self.models = {}
        self.discard_buffers()
