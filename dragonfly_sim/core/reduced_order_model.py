"""
POD reduced-order model (ROM) of a steady flow solve: least-squares
Petrov-Galerkin (LSPG) on a POD basis of converged solutions.

The ROM works with any model built on CompressibleEulerModel -- the Euler
model itself and CompressibleRANSModel, which forwards to it -- through this
interface, and nothing model-specific::

    u_vec                     the state (its layout is the basis's row layout)
    set_up_solver(), problem  the Newton problem: problem.F(x, b) and
                              problem.J(x, problem._A) assemble the residual
                              and its exact Jacobian at the current u_vec
                              (for RANS the condensed-gradient problem, which
                              updates the reconstructed gradient on the way)
    steady_residual()         context in which problem assembles the steady
                              residual (pseudo/physical time terms off)
    positivity_step_length    step length keeping density and pressure positive
    state_scales()            per-component magnitudes for the inner product
    _assemble_physical_residual(), relative_residual_norm()

Inner product. Snapshots and basis use (u, v)_W = u^T W v with, by default,
the state-scaled Euclidean weight W = D^-2, D the per-component state scales
s_k: every component on the same footing (in absolute terms the SA variable
rho*nu_tilde is ~1e-4 against rho*E ~ 2, and a truncated SVD would ignore
it). inner_product='l2' adds the (diagonal) DG0 mass matrix, W = M D^-2,
i.e. int sum_k u_k v_k / s_k^2 dx; on the NACA0012 alpha sweeps it was 10-100x
less accurate than the Euclidean weight (M emphasizes the large far-field
cells). W is built once on the undeformed mesh: the incremental SVD keeps
Q^T W Q = I only for a fixed W. The LSPG residual is measured in the matching
norm ||S r||, S = D^-1 (D^-1 M^-1/2 with 'l2', on the current mesh).
state_scaling=None with 'euclidean' is the plain Euclidean POD/LSPG.

Solve. With V the basis (V^T W V = I), u = V a (+ mean), LSPG is the
Gauss-Newton iteration on min ||S R(V a)||:

    (S A V)^T (S A V) da = -(S A V)^T S r,    a <- a + theta da,

theta from positivity_step_length, optionally backtracked (Armijo) on
||S r||^2 / 2. The reduced system is small and solved redundantly on every
rank. Convergence follows project-sandbox benchmark_pod.py (its PODSolver and
outer loop): passes of at most newton_max_it iterations, each stopping on
||g|| = ||(S A V)^T S r|| < max(atol, rtol ||g_0||), with g_0 from the first
iteration of the first pass (not reset by a restart), on a negligible step, or
on a stall of ||g||; up to max_outer_iterations passes, converged when a pass
ends with ||g|| < max(newton_atol, g_conv_limit ||g_0||), g_0 the same reference.

Derivatives. The ROM solves the LSPG stationarity condition, the projected
residual equations

    Phi^T R(V a; x) = 0,    Phi = S^T S A V,

so when its solution is accepted the adjoint is that of these k equations
(ReducedOrderModel.adjoint, an LSPGAdjoint). With V, S and Phi frozen at the
solution (the basis is not differentiated, and the Gauss-Newton terms, which
are proportional to S R, are dropped), for an output f with g = (df/du)^T:

    (S A V)^T (S A V) lam = V^T g,    psi = Phi lam = S^T S A V lam,

and df/dx = df/dx|_u - psi^T dR/dx with the full-order dR/dx: psi takes the
place of the full-order adjoint A^-T g. It is the Petrov-Galerkin solution of
A^T psi = g in the span of Phi, tested with V: V^T (A^T psi - g) = 0. psi is
computed as S Q y with S A V = Q R (a distributed thin QR) and R^T y = V^T g,
without forming the normal matrix, whose condition number is squared.

Adjoint snapshots (adjoint_snapshots). psi lies in span(S^2 A V), directions
built from state snapshots, which need not hold the adjoint field. The
full-order adjoints the optimizer solves (at the designs the FOM solved) can be
added as snapshots (add_adjoint_snapshot), one per output row:

  'augment'   into the state snapshot matrix (tagged, see core/pod.py, and scaled
              to the W-norm of the design's state snapshot, as adjoints have other
              units): the one basis V then holds them, for the primal solve and the
              LSPG adjoint above alike. Not centred.
  'separate'  into a snapshot matrix of their own (inner product W^-1, the dual of
              the state's, and normalized in it), from which every solve builds an
              adjoint basis Psi (adjoint_rb_size, the design weights of the state
              basis). The adjoint is then the least-squares solution of the adjoint
              equations over span[S^2 A V, Psi], which contains the LSPG adjoint's:
                  min || W^-1/2 (A^T psi - g) ||,
              no longer the exact derivative of the ROM's outputs, but closer to the
              full-order one where Psi holds the adjoint.
  'concatenate'
              into a snapshot matrix of their own, as 'separate'; the basis of every
              solve is then the concatenation of a state basis of ceil(rb_size / 2)
              vectors and an adjoint basis Psi of floor(rb_size / 2) (slots Psi cannot
              fill, e.g. before there are adjoint snapshots, go to the state basis), Psi
              orthogonalized against the state basis so that the two blocks are W-
              orthonormal and mutually W-orthogonal: the states keep their own SVD, so the
              adjoints do not crowd them out of the basis as with 'augment', and the primal
              solve and its adjoint both use the concatenated V.

adjoint_solve chooses the reduced adjoint: 'lspg' (the adjoint of the LSPG equations
above, the exact derivative of the ROM's outputs), 'galerkin' (psi = V z in the span of the
basis itself, V^T A^T V z = V^T g: with 'concatenate' the adjoint basis is then in its trial
space, which the LSPG adjoint's span(S^2 A V) does not contain) or 'least_squares' (the
least-squares adjoint over span[S^2 A V, Psi], 'separate' and 'concatenate' only); the
default is 'least_squares' for 'separate' and 'lspg' otherwise. Only 'lspg' is the exact
derivative of the ROM's outputs, consistent with the forward (LSPG) solve.

test_space='hybrid' ('concatenate' only) changes the forward solve itself: Petrov-Galerkin with
the concatenated basis V = [V_s, Psi_perp] as trial basis and Phi = [S^2 A V_s, Psi] as test
basis, the state part LSPG's and the adjoint part the adjoint basis itself (HybridLSPGSolver).
Its adjoint, (Phi^T A V)^T lam = V^T g, psi = Phi lam (HybridLSPGAdjoint), is the derivative of
those equations (as LSPGAdjoint's of LSPG, Phi frozen) and lies in a space that holds Psi, which
the LSPG adjoint's span(S^2 A V) does not: on the test cases its derivatives were 100-10^4 times
closer to the full-order ones than the state basis gave, at a slightly larger ||R|| (the forward
solve no longer minimizes ||S R||).
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as sp_la
from mpi4py import MPI
from petsc4py import PETSc

from dragonfly_sim.core.pod import PODBasis, resolve_centering
from dragonfly_sim.utils.pod_utils import DiagonalWeight, tsqr, petsc_dense_from_local_block
from dragonfly_sim.utils.derived_fields import diagonal_mass_inverse
from dragonfly_sim.utils.solver_profiling import PROFILER as _PROF


def scaled_jacobian_basis(model, V, S, AV=None):
    """
    S A V at the model's current u_vec, A the exact Jacobian assembled by its
    Newton problem (problem.J). Returns (this rank's rows of S A V, the PETSc
    product A V); `AV`, if given, is a product of the same shape to reuse.
    Without S the rows are a view of AV's array.
    """
    problem = model.problem
    problem.J(model.u_vec.x.petsc_vec, problem._A)
    A = problem._A
    if AV is None:
        AV = A.matMult(V)
    else:
        A.matMult(V, result=AV)
    SAV = AV.getDenseArray()
    if S is not None:
        SAV = S.getArray(readonly=True)[:, None] * SAV
    return SAV, AV


def orthonormal_columns(comm, X_local, rtol=1e-10, atol=0.0):
    """An orthonormal basis (this rank's rows) of the span of the row-distributed
    columns X, dropping directions with singular values below max(rtol times the
    largest, atol) (dependent columns). Collective."""
    Q1, R = tsqr(comm, X_local)
    U, sigma, _ = np.linalg.svd(R)
    keep = sigma > max(rtol * sigma.max(), atol) if sigma.size and sigma.max() > 0.0 \
        else np.zeros(0, dtype=bool)
    return Q1 @ U[:, keep]


class LSPGSolver:
    """
    Reduced Newton (Gauss-Newton) loop on a POD basis, driving a model's
    Newton problem (see the module docstring).

    Parameters
    ----------
    V : PETSc.Mat
        The basis, a distributed MATDENSE in the state layout.
    W : PETSc.Vec or None
        The diagonal inner-product weight, for the restriction
        a = V^T W (u - mean); None: Euclidean (V^T V = I).
    S : PETSc.Vec or None
        The diagonal residual weight; None: the plain residual norm.
    mean : PETSc.Vec or None
        Optional affine offset (u = V a + mean).
    """

    def __init__(self, V, comm, W=None, S=None, mean=None, rtol=1e-10, atol=1e-12, max_it=30,
                 stall_window=3, stall_rtol=1e-10, line_search=True, armijo_c=1e-4, max_backtracks=10):
        self.V = V
        self.comm = comm
        self.W = W
        self.S = S
        self.mean = mean
        self.rtol = rtol
        self.atol = atol
        self.max_it = max_it
        self.stall_window = stall_window
        self.stall_rtol = stall_rtol
        self.line_search = line_search
        self.armijo_c = armijo_c
        self.max_backtracks = max_backtracks
        self.converged = False
        self.stalled = False
        self.n_iterations = 0
        self.last_g_norm = None
        # ||g|| of the first iteration of the first pass: the reference of rtol in every
        # pass of this solver, so a restart does not ask for a further relative reduction
        self.g0 = None
        self._AV = None
        self._r = None

    @property
    def rb_size(self):
        return self.V.getSize()[1]

    def destroy(self):
        for name in ("_AV", "_r"):
            obj = getattr(self, name)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)

    # -- maps between the full and reduced spaces --------------------------
    def restrict(self, u_vec):
        """a = V^T W (u - mean), replicated on every rank."""
        with _PROF.phase("rom:expand"):
            x = u_vec.getArray(readonly=True).copy()
            if self.mean is not None:
                x -= self.mean.getArray(readonly=True)
            if self.W is not None:
                x *= self.W.getArray(readonly=True)
            return self.comm.allreduce(self.V.getDenseArray().T @ x, op=MPI.SUM)

    def expand(self, a, u):
        """u <- V a (+ mean), ghosts updated."""
        with _PROF.phase("rom:expand"):
            x = u.x.petsc_vec
            x.array[:] = self.V.getDenseArray() @ a
            if self.mean is not None:
                x.axpy(1.0, self.mean)
            u.x.scatter_forward()

    # -- the reduced Newton loop ---------------------------------------------
    def _weighted_residual(self, model):
        """Assemble r at the current u_vec; returns (S r local entries, ||S r||)."""
        problem = model.problem
        if self._r is None:
            self._r = problem._b.duplicate()
        problem.F(model.u_vec.x.petsc_vec, self._r)
        with _PROF.phase("rom:project"):
            r = self._r.getArray(readonly=True)
            if self.S is not None:
                r = self.S.getArray(readonly=True) * r
            return r, float(np.sqrt(self.comm.allreduce(float(r @ r), op=MPI.SUM)))

    def _reduced_system(self, model):
        """(H, g, ||S r||) with H = (S A V)^T (S A V), g = (S A V)^T S r at the
        current u_vec."""
        r, r_norm = self._weighted_residual(model)
        if self._AV is not None and self._AV.getSize()[1] != self.V.getSize()[1]:
            self._AV.destroy()
            self._AV = None
        with _PROF.phase("rom:project"):
            SAV, self._AV = scaled_jacobian_basis(model, self.V, self.S, self._AV)
            H = self.comm.allreduce(SAV.T @ SAV, op=MPI.SUM)
            g = self.comm.allreduce(SAV.T @ r, op=MPI.SUM)
        return H, g, r_norm

    def solve(self, model):
        """
        One pass of the LSPG Gauss-Newton loop: advance model.u_vec, starting
        from its restriction onto the basis, towards the LSPG solution in the
        span of V. Call inside model.steady_residual() for a steady solve.
        Returns (n_iterations, converged). ReducedOrderModel.solve runs up to
        max_outer_iterations passes.

        Stopping rules (those of the project-sandbox PODSolver, which
        benchmark_pod.py uses), checked at every iteration from the reduced
        right-hand side g = (S A V)^T S r of the current state::

          converged   ||g|| < max(atol, rtol ||g_0||), g_0 from the first iteration of
                      the solver's first pass (kept over restarts); or, after a step, ||da|| < max(atol,
                      rtol max(||a||, 1))
          stalled     ||g|| > (1 - stall_rtol) times its value stall_window
                      iterations earlier
          stop        non-finite ||g||, a singular reduced Jacobian, or max_it

        Step: the Gauss-Newton step da limited to the positivity-preserving
        length, then (line_search) halved until the merit ||S r||^2 / 2
        decreases sufficiently (Armijo), at most max_backtracks times; if no
        halving does, the positivity-limited step is taken as is. A full
        Gauss-Newton step can overshoot when the minimum residual in the
        basis is large.
        """
        u = model.u_vec
        a = self.restrict(u.x.petsc_vec)
        dx = u.x.petsc_vec.duplicate()
        self.converged = self.stalled = False
        history = []
        n = 0
        PETSc.Sys.Print("LSPG ROM solve, basis width {}".format(self.rb_size))
        for n in range(self.max_it):
            self.expand(a, u)
            H, g, r_norm = self._reduced_system(model)
            g_norm = float(np.linalg.norm(g))
            self.last_g_norm = g_norm
            if self.g0 is None:
                self.g0 = g_norm
            if not np.isfinite(g_norm):
                PETSc.Sys.Print("ROM: non-finite reduced residual at iteration {}; stopping".format(n))
                break
            if g_norm < max(self.atol, self.rtol * self.g0):
                self.converged = True
                PETSc.Sys.Print("ROM converged at iteration {}: |g| {:.6e} (from {:.6e})".format(n, g_norm, self.g0))
                break
            history.append(g_norm)
            if len(history) > self.stall_window and \
                    g_norm > (1.0 - self.stall_rtol) * history[-(self.stall_window + 1)]:
                self.stalled = True
                PETSc.Sys.Print("ROM stalled at iteration {}: |g| {:.6e}, vs {:.6e} {} iterations "
                                "earlier".format(n, g_norm, history[-(self.stall_window + 1)],
                                                 self.stall_window))
                break
            try:
                with _PROF.phase("solve:reduced"):
                    da = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                PETSc.Sys.Print("ROM: singular reduced Jacobian at iteration {}".format(n))
                break
            # the full-space update is +V da; positivity_step_length limits x - theta dx
            dx.array[:] = -(self.V.getDenseArray() @ da)
            theta, _ = model.positivity_step_length(u.x.petsc_vec, dx)
            if self.line_search:
                with _PROF.phase("rom:line_search"):
                    theta = self._backtrack(model, a, da, theta, 0.5 * r_norm**2, float(g @ da))
            a = a + theta * da
            da_norm = float(np.linalg.norm(da))
            PETSc.Sys.Print("ROM iter {}: |g| = {:.6e}, ||S r|| = {:.6e}, |da| = {:.6e}, theta = {:.4f}".format(
                n, g_norm, r_norm, da_norm, theta))
            if da_norm < max(self.atol, self.rtol * max(float(np.linalg.norm(a)), 1.0)):
                self.converged = True
                PETSc.Sys.Print("ROM converged at iteration {}: |da| {:.6e}".format(n, da_norm))
                break
        self.expand(a, u)
        dx.destroy()
        if not self.converged:
            PETSc.Sys.Print("[warning] ROM pass stopped after {} iterations without converging{}".format(
                n + 1, " (stalled)" if self.stalled else ""))
        self.n_iterations = n + 1
        return self.n_iterations, self.converged

    def _backtrack(self, model, a, da, theta, phi, slope):
        """Largest theta * 2^-k (k <= max_backtracks) with phi(a + theta da)
        <= phi + armijo_c theta slope; the given theta if there is none."""
        t = theta
        for _ in range(self.max_backtracks + 1):
            self.expand(a + t * da, model.u_vec)
            _, r_trial = self._weighted_residual(model)
            if np.isfinite(r_trial) and 0.5 * r_trial**2 <= phi + self.armijo_c * t * slope:
                return t
            t *= 0.5
        PETSc.Sys.Print("ROM: no Armijo step along da; taking theta = {:.4f}".format(theta))
        return theta


class LSPGAdjoint:
    """
    The adjoint of an accepted LSPG solution (Derivatives in the module
    docstring): psi = S Q y, with S A V = Q R and R^T y = V^T g, for the basis
    V and residual weight S of that solve. The factorization is formed by the
    first solve() and reused for every later right-hand side (CSDL passes one
    per output row). It belongs to one design and state; matches() checks
    both.

    Parameters
    ----------
    V : PETSc.Mat
        The basis of the solve (owned by the POD, not by this object).
    S : PETSc.Vec or None
        The diagonal residual weight of the solve; destroyed with this object.
    parameter_vector, solution
        The design and this rank's owned entries of the accepted solution.
    """

    def __init__(self, V, S, comm, parameter_vector, solution, accepted=True, adjoint_basis=None,
                 dual_weight=None, galerkin=False):
        self.V = V
        self.S = S
        self.comm = comm
        # adjoint_snapshots='separate': the adjoint basis Psi (a Mat, owned by the ROM) and the
        # weight W^-1/2 (this rank's entries, None: 1) of the least-squares adjoint
        self.adjoint_basis = adjoint_basis
        self.dual_weight = dual_weight
        # adjoint_solve='galerkin': psi = V z with V^T A^T V z = V^T g (LU of the k x k matrix)
        self.galerkin = galerkin
        self._T = None
        # whether the solve it belongs to was accepted: a rejected solve's adjoint is not used
        # by the optimizer, only compared with the full-order one (compute_all_adjoint_errors)
        self.accepted = accepted
        self.parameter_vector = np.array(parameter_vector, dtype=float).ravel()
        self.solution = np.array(solution, dtype=float)
        self._Q = None
        self._R = None

    @property
    def rb_size(self):
        return self.V.getSize()[1]

    def matches(self, parameter_vector, u_local):
        """Whether this is the adjoint at design `parameter_vector` and state
        `u_local` (owned entries): both equal to the accepted solve's on every
        rank. Collective."""
        pv = np.asarray(parameter_vector, dtype=float).ravel()
        local = bool(np.array_equal(pv, self.parameter_vector)
                     and np.array_equal(np.asarray(u_local, dtype=float), self.solution))
        return self.comm.allreduce(local, op=MPI.LAND)

    def matches_design(self, parameter_vector):
        """Whether this adjoint's basis was built for design `parameter_vector`, whatever the
        state: the reduced adjoint is defined at any state, which a rejected solve's is
        evaluated at (the full-order solution)."""
        return bool(np.array_equal(np.asarray(parameter_vector, dtype=float).ravel(), self.parameter_vector))

    def _factor(self, model):
        """Q, R of S A V, A the steady Jacobian at the model's current u_vec; with
        an adjoint basis, those of W^-1/2 A^T over the trial space (see solve)."""
        model.set_up_solver()
        with model.steady_residual():
            SAV, AV = scaled_jacobian_basis(model, self.V, self.S)
            if self.galerkin:
                AV_local = AV.getDenseArray()
                self._lu = sp_la.lu_factor(self.comm.allreduce(self.V.getDenseArray().T @ AV_local, op=MPI.SUM))
                self._Q, self._R = None, np.diag(np.abs(np.diag(self._lu[0])))
            elif self.adjoint_basis is None:
                self._Q, self._R = tsqr(self.comm, SAV)
            else:
                self._factor_least_squares(model, SAV)
        AV.destroy()
        diag = np.abs(np.diag(self._R))
        if not np.all(np.isfinite(self._R)) or diag.min() <= np.finfo(float).eps * diag.max():
            raise ValueError("reduced adjoint: S A V is singular to working precision "
                             "(|R_ii| from {:.3e} to {:.3e})".format(diag.min(), diag.max()))

    def _factor_least_squares(self, model, SAV):
        """The adjoint basis variant: T = an orthonormal basis of span[S^2 A V, Psi]
        (self._T) and Q, R of W^-1/2 A^T T, A the Jacobian scaled_jacobian_basis
        just assembled."""
        S = None if self.S is None else self.S.getArray(readonly=True)
        lspg_space = SAV if S is None else S[:, None] * SAV
        self._T = orthonormal_columns(self.comm, np.column_stack([lspg_space, self.adjoint_basis.getDenseArray()]))
        A = model.problem._A
        x, y = A.createVecLeft(), A.createVecRight()
        ATT = np.empty_like(self._T)
        for j in range(self._T.shape[1]):
            x.array[:] = self._T[:, j]
            A.multTranspose(x, y)
            ATT[:, j] = y.array
        x.destroy()
        y.destroy()
        if self.dual_weight is not None:
            ATT *= self.dual_weight[:, None]
        self._Q, self._R = tsqr(self.comm, ATT)

    def _solve_least_squares(self, model, rhs, first):
        """psi = T z, z minimizing ||W^-1/2 (A^T T z - g)||."""
        Tg = rhs if self.dual_weight is None else self.dual_weight * rhs
        z = sp_la.solve_triangular(self._R, self.comm.allreduce(self._Q.T @ Tg, op=MPI.SUM), lower=False)
        psi = self._T @ z
        if first:
            res = self._Q @ (self._R @ z) - Tg
            PETSc.Sys.Print("least-squares adjoint over span[S^2 A V, Psi] ({} directions, {} adjoint basis "
                            "vectors): |W^-1/2 (A^T psi - g)| / |W^-1/2 g| = {:.3e}".format(
                                self._T.shape[1], self.adjoint_basis.getSize()[1],
                                np.sqrt(self.comm.allreduce(float(res @ res), op=MPI.SUM)
                                        / max(self.comm.allreduce(float(Tg @ Tg), op=MPI.SUM), 1e-300))))
        return psi

    def solve(self, model, rhs_local):
        """
        psi (this rank's owned entries) for the right-hand side g, rhs_local
        being its owned entries. The model's u_vec, mesh and angle of attack
        must be those of the accepted solution (the first call assembles the
        Jacobian there). Collective.
        """
        first = self._R is None
        if first:
            self._factor(model)
        rhs = np.asarray(rhs_local, dtype=float)
        if self.galerkin:
            # (V^T A V)^T z = V^T g
            z = sp_la.lu_solve(self._lu, self.comm.allreduce(self.V.getDenseArray().T @ rhs, op=MPI.SUM), trans=1)
            return self.V.getDenseArray() @ z
        if self.adjoint_basis is not None:
            return self._solve_least_squares(model, rhs, first)
        b = self.comm.allreduce(self.V.getDenseArray().T @ rhs, op=MPI.SUM)
        y = sp_la.solve_triangular(self._R, b, trans='T', lower=False)
        psi = self._Q @ y
        if self.S is not None:
            psi = self.S.getArray(readonly=True) * psi
        psi_norm = np.sqrt(self.comm.allreduce(float(psi @ psi), op=MPI.SUM))
        if not np.isfinite(psi_norm):
            raise ValueError("reduced adjoint: non-finite solution")
        if first:
            diag = np.abs(np.diag(self._R))
            PETSc.Sys.Print("reduced (LSPG) adjoint: basis width {}, |R_ii| from {:.3e} to {:.3e}, "
                            "Petrov-Galerkin residual |V^T (A^T psi - g)| / |V^T g| = {:.3e}".format(
                                self.rb_size, diag.min(), diag.max(),
                                self._galerkin_residual(model, psi, rhs, b)))
        else:
            PETSc.Sys.Print("reduced (LSPG) adjoint: reused the factorization, |psi| = {:.6e}".format(psi_norm))
        return psi

    def _galerkin_residual(self, model, psi, rhs, b):
        """|V^T (A^T psi - g)| / |V^T g|, with the Jacobian _factor assembled
        (zero up to round-off)."""
        A = model.problem._A
        x, Atx = A.createVecLeft(), A.createVecRight()
        x.array[:] = psi
        A.multTranspose(x, Atx)
        res = self.comm.allreduce(self.V.getDenseArray().T @ (Atx.array - rhs), op=MPI.SUM)
        x.destroy()
        Atx.destroy()
        return float(np.linalg.norm(res) / max(np.linalg.norm(b), 1e-300))

    def destroy(self):
        if self.S is not None:
            self.S.destroy()
            self.S = None
        self._Q = self._R = None
        self._T = None


class HybridLSPGSolver(LSPGSolver):
    """
    The reduced Newton loop of test_space='hybrid' (adjoint_snapshots='concatenate'):
    Petrov-Galerkin with the concatenated basis V = [V_s, Psi_perp] as trial basis and
    Phi = [S^2 A V_s, Psi] as test basis, Psi the adjoint POD basis as it is (only its
    first k - n_state columns, k the width of V). The state part of Phi is LSPG's, frozen
    per iteration like it; the reduced equations Phi^T R(V a) = 0 give the step
    (Phi^T A V) da = -Phi^T r. The line search keeps LSPG's merit ||S r||^2 / 2, with its
    own slope.
    """

    def __init__(self, V, comm, n_state=None, Psi=None, **kwargs):
        super().__init__(V, comm, **kwargs)
        self.n_state = n_state
        self.Psi = Psi                        # this rank's rows of the adjoint basis
        self._merit_gradient = None

    def _reduced_system(self, model):
        Sr, r_norm = self._weighted_residual(model)
        if self._AV is not None and self._AV.getSize()[1] != self.V.getSize()[1]:
            self._AV.destroy()
            self._AV = None
        with _PROF.phase("rom:project"):
            SAV, self._AV = scaled_jacobian_basis(model, self.V, self.S, self._AV)
            H, g, self._merit_gradient = hybrid_reduced_system(self.comm, SAV, Sr, self.S, self.n_state, self.Psi)
        return H, g, r_norm

    def _backtrack(self, model, a, da, theta, phi, slope):
        # the slope of the merit ||S r||^2 / 2, not of the hybrid right-hand side
        return super()._backtrack(model, a, da, theta, phi, float(self._merit_gradient @ da))


def hybrid_test_basis(SAV, S, n_state, Psi):
    """This rank's rows of Phi = [S^2 A V_s, Psi[:, :k - n_state]], from S A V."""
    k = SAV.shape[1]
    state = SAV[:, :n_state] if S is None else S[:, None] * SAV[:, :n_state]
    return np.column_stack([state, Psi[:, :k - n_state]])


def hybrid_reduced_system(comm, SAV, Sr, S_vec, n_state, Psi):
    """(Phi^T A V, Phi^T r, (S A V)^T S r) of the hybrid test basis, from S A V and S r."""
    S = None if S_vec is None else S_vec.getArray(readonly=True)
    Phi = hybrid_test_basis(SAV, S, n_state, Psi)
    AV = SAV if S is None else SAV / S[:, None]
    r = Sr if S is None else Sr / S
    H = comm.allreduce(Phi.T @ AV, op=MPI.SUM)
    g = comm.allreduce(Phi.T @ r, op=MPI.SUM)
    merit = comm.allreduce(SAV.T @ Sr, op=MPI.SUM)
    return H, g, merit


class HybridLSPGAdjoint(LSPGAdjoint):
    """
    The adjoint of the hybrid reduced equations Phi^T R(V a) = 0 (HybridLSPGSolver), with
    Phi frozen as in LSPGAdjoint (the terms proportional to R are dropped):
    (Phi^T A V)^T lam = V^T g, psi = Phi lam. psi lies in span[S^2 A V_s, Psi], so it holds the
    adjoint basis itself; for Phi = S^2 A V this is LSPGAdjoint. Owns S, not V or Psi.
    """

    def __init__(self, V, S, comm, parameter_vector, solution, accepted=True, n_state=None, Psi=None):
        super().__init__(V, S, comm, parameter_vector, solution, accepted=accepted)
        self.n_state = n_state
        self.Psi = Psi

    def _factor(self, model):
        model.set_up_solver()
        with model.steady_residual():
            SAV, AV = scaled_jacobian_basis(model, self.V, self.S)
            S = None if self.S is None else self.S.getArray(readonly=True)
            self._Phi = hybrid_test_basis(SAV, S, self.n_state, self.Psi)
            H = self.comm.allreduce(self._Phi.T @ AV.getDenseArray(), op=MPI.SUM)
        AV.destroy()
        self._lu = sp_la.lu_factor(H)
        self._R = np.diag(np.abs(np.diag(self._lu[0])))
        diag = np.diag(self._R)
        if not np.all(np.isfinite(self._lu[0])) or diag.min() <= np.finfo(float).eps * diag.max():
            raise ValueError("hybrid reduced adjoint: Phi^T A V is singular to working precision "
                             "(|U_ii| from {:.3e} to {:.3e})".format(diag.min(), diag.max()))

    def solve(self, model, rhs_local):
        first = self._R is None
        if first:
            self._factor(model)
        lam = sp_la.lu_solve(self._lu, self.comm.allreduce(self.V.getDenseArray().T @ np.asarray(rhs_local, dtype=float),
                                                            op=MPI.SUM), trans=1)
        psi = self._Phi @ lam
        if first:
            PETSc.Sys.Print("hybrid reduced adjoint: {} state and {} adjoint test directions".format(
                self.n_state, self._Phi.shape[1] - self.n_state))
        return psi

    def destroy(self):
        super().destroy()
        self._Phi = None


@dataclass
class ROMResult:
    solution: np.ndarray       # this rank's owned entries of the ROM state
    eta: float                 # ||R|| / ||u|| of the full steady residual
    residual_norm: float
    converged: bool            # the reduced Newton loop converged
    accepted: bool             # eta < eta_threshold
    iterations: int
    basis_width: int
    walltime: float


class ReducedOrderModel:
    """
    POD snapshots, basis and LSPG solve for DG_windtunnel_model
    (reduced_order_model=...). One instance per windtunnel model.

    Basis::

      rb_size            basis width
      basis_mode         'global' or 'local_weighted' (core/pod.py)
      max_rank           cap on the stored snapshot SVD rank (default 3 rb_size)
      centering          local_weighted only: the offset c the snapshots are centred on,
                         u = V a + c (core/pod.py): 'warm_start' (the solve's warm
                         start, i.e. the model's u_vec at solve(), projected onto the
                         snapshot span), 'mean' (their weighted mean) or None. The
                         default 'auto' is 'warm_start' for local_weighted and None for
                         global. subtract_mean=True/False is the deprecated form of
                         'mean'/None
      mean_weighting     'snapshot_weights' (the local distance weights, default) or
                         'uniform' (1/n per snapshot); used with centering='mean'
      qr_rank_tol        relative weighted residual (outside the span of the stored
                         snapshots) below which a snapshot adds no direction
      weightfunction, n_nonzero_weights, cubic_cutoff
                         local_weighted weighting (PODBasis.construct_basis);
                         cubic_cutoff = c sets the cubic support radius to
                         c min_dist + (1 - c) max_dist, widened if needed so
                         that the rb_size nearest snapshots keep a weight
      parameter_scales   fixed per-parameter factors on the snapshot distance
      distance_weighting output-sensitivity weights on the snapshot distance, from the
                         total derivatives of the objective and constraints
                         (utils/dv_sensitivity.py): None (off), 'total_derivative'
                         (default; outputs divided by their own magnitude) or
                         'total_derivative_raw'. They are multiplied with
                         parameter_scales. Without a derivative probe feeding
                         set_distance_sensitivities there is no information, and the
                         distance stays unweighted

    Inner product (default: Euclidean with free-stream state scaling)::

      inner_product      'euclidean' or 'l2' (DG0 mass matrix)
      state_scaling      'freestream' (model.state_scales()), None, or one
                         scale per state component

    Solve and acceptance (the defaults are project-sandbox benchmark_pod.py's)::

      newton_max_it, newton_rtol, newton_atol, stall_window, stall_rtol
                         per-pass LSPGSolver stopping rules
      max_outer_iterations
                         LSPG passes, each restarted from the current state
      g_conv_limit       the ROM solve has converged when the last ||g|| of a pass is
                         below max(newton_atol, g_conv_limit ||g_0||), g_0 the first
                         ||g|| of the first pass (the reference of newton_rtol too);
                         0 never accepts. Keep it >= newton_rtol, or the passes stop
                         before this target
      line_search        Armijo damping of the Gauss-Newton step
      eta_threshold      the ROM solution is accepted when ||R||/||u|| is below it
      min_snapshots      snapshots needed before the ROM is tried
      run_fom_every_evaluation
                         also run the FOM when the ROM is accepted (benchmarking;
                         the accepted output stays the ROM's)

    Adjoint snapshots (module docstring; the windtunnel model adds the full-order adjoints the
    optimizer solves)::

      adjoint_snapshots  None (state snapshots only), 'augment', 'separate' or 'concatenate'
      adjoint_rb_size, adjoint_max_rank
                         'separate': adjoint basis width (default rb_size); 'concatenate'
                         uses floor(rb_size / 2). The stored rank defaults to 3 times the width
      adjoint_solve      'lspg', 'galerkin' or 'least_squares' (module docstring)
      test_space         'lspg' (default) or 'hybrid' ('concatenate' only): the test basis
                         [S^2 A V_s, Psi] (HybridLSPGSolver), with the adjoint of those equations

    After a solve that built a basis, `adjoint` is its LSPGAdjoint (None before the first),
    valid until the next solve; `adjoint.accepted` says whether the solve was accepted, and only
    an accepted solve's adjoint is used for the derivative.
    """

    def __init__(self, rb_size=20, basis_mode='global', max_rank=None, qr_rank_tol=1e-10,
                 centering='auto', subtract_mean=None, mean_weighting='snapshot_weights',
                 weightfunction='Cubic', n_nonzero_weights=None, cubic_cutoff=None,
                 parameter_scales=None, distance_weighting='total_derivative',
                 inner_product='euclidean', state_scaling='freestream',
                 newton_max_it=15, newton_rtol=1e-10, newton_atol=1e-12, stall_window=3,
                 stall_rtol=1e-10, max_outer_iterations=3, g_conv_limit=1e-10, line_search=True,
                 eta_threshold=1e-10, min_snapshots=2, run_fom_every_evaluation=False,
                 adjoint_snapshots=None, adjoint_rb_size=None, adjoint_max_rank=None, adjoint_solve=None,
                 test_space='lspg'):
        if inner_product not in ('l2', 'euclidean'):
            raise ValueError("inner_product must be 'l2' or 'euclidean', got {!r}".format(inner_product))
        if not (state_scaling is None or isinstance(state_scaling, str) and state_scaling == 'freestream'
                or np.ndim(state_scaling) == 1):
            raise ValueError("state_scaling must be 'freestream', None or one scale per state component")
        self.rb_size = rb_size
        self.basis_mode = basis_mode
        self.centering = resolve_centering(centering, subtract_mean, basis_mode)
        if mean_weighting not in ('uniform', 'snapshot_weights'):
            raise ValueError("mean_weighting must be 'uniform' or 'snapshot_weights', got {!r}".format(
                mean_weighting))
        self.mean_weighting = mean_weighting
        self.max_rank = 3 * rb_size if max_rank is None else max_rank
        self.qr_rank_tol = qr_rank_tol
        self.weightfunction = weightfunction
        self.n_nonzero_weights = n_nonzero_weights
        self.cubic_cutoff = cubic_cutoff
        self.parameter_scales = parameter_scales
        if distance_weighting not in (None, 'total_derivative', 'total_derivative_raw'):
            raise ValueError("distance_weighting must be None, 'total_derivative' or "
                             "'total_derivative_raw', got {!r}".format(distance_weighting))
        self.distance_weighting = distance_weighting
        # Latest mean-normalized weights, written by set_distance_sensitivities;
        # None until a gradient evaluation has been probed. They lag the basis
        # they weight by one design point.
        self.distance_sensitivities = None
        self._warned_weight_size = False
        self.inner_product = inner_product
        self.state_scaling = state_scaling
        self.newton_max_it = newton_max_it
        self.newton_rtol = newton_rtol
        self.newton_atol = newton_atol
        self.stall_window = stall_window
        self.stall_rtol = stall_rtol
        self.max_outer_iterations = max_outer_iterations
        self.g_conv_limit = g_conv_limit
        self.line_search = line_search
        self.eta_threshold = eta_threshold
        self.min_snapshots = min_snapshots
        self.run_fom_every_evaluation = run_fom_every_evaluation
        if adjoint_snapshots not in (None, 'augment', 'separate', 'concatenate'):
            raise ValueError("adjoint_snapshots must be None, 'augment', 'separate' or 'concatenate', got "
                             "{!r}".format(adjoint_snapshots))
        if adjoint_solve is None:
            adjoint_solve = 'least_squares' if adjoint_snapshots == 'separate' else 'lspg'
        if adjoint_solve not in ('lspg', 'galerkin', 'least_squares'):
            raise ValueError("adjoint_solve must be 'lspg', 'galerkin' or 'least_squares', got {!r}".format(
                adjoint_solve))
        if adjoint_solve == 'least_squares' and adjoint_snapshots not in ('separate', 'concatenate'):
            raise ValueError("adjoint_solve='least_squares' needs an adjoint basis: adjoint_snapshots='separate' "
                             "or 'concatenate'")
        if test_space not in ('lspg', 'hybrid'):
            raise ValueError("test_space must be 'lspg' or 'hybrid', got {!r}".format(test_space))
        if test_space == 'hybrid' and adjoint_snapshots != 'concatenate':
            raise ValueError("test_space='hybrid' needs adjoint_snapshots='concatenate'")
        self.test_space = test_space
        self.adjoint_snapshots = adjoint_snapshots
        self.adjoint_solve = adjoint_solve
        if adjoint_snapshots == 'concatenate':
            adjoint_rb_size = rb_size // 2
        self.adjoint_rb_size = rb_size if adjoint_rb_size is None else adjoint_rb_size
        self.adjoint_max_rank = 3 * self.adjoint_rb_size if adjoint_max_rank is None else adjoint_max_rank
        if self.adjoint_max_rank < 1:
            raise ValueError("adjoint_max_rank must be at least 1, got {}".format(self.adjoint_max_rank))
        self.adjoint_pod = None
        self.W_dual = None
        # 'concatenate': the concatenated basis of the last solve (see `basis`)
        self._concatenated_basis = None
        # test_space='hybrid': {n_state, Psi} of the last solve's test basis (hybrid_test_space)
        self._hybrid = None
        # the W-norm of each design's state snapshot (adjoint_snapshots='augment' scales to it)
        self._state_norms = {}

        self.pod = None
        self.W = None
        self.scales = None
        self.last_solver = None
        self.adjoint = None
        # how the basis of the last solve was built (basis_settings)
        self._basis_settings = None

    # ------------------------------------------------------------------
    # Set-up and weights
    # ------------------------------------------------------------------
    def set_up(self, model):
        """Build the inner-product weight and the POD on the model's
        current (undeformed) mesh. Call once the model's boundary conditions,
        function spaces and state exist. Collective."""
        layout = model.u_vec.x.petsc_vec
        if self.state_scaling is None:
            self.scales = None
        elif isinstance(self.state_scaling, str):
            self.scales = np.asarray(model.state_scales(), dtype=float)
        else:
            self.scales = np.asarray(self.state_scaling, dtype=float)
        if self.scales is not None and self.scales.shape != (model.n_state,):
            raise ValueError("{} state scales for {} state components".format(
                self.scales.shape[0], model.n_state))
        if self.scales is not None and not np.all(self.scales > 0):
            raise ValueError("state scales must be positive, got {}".format(self.scales))

        if self.W is not None:
            self.W.destroy()
        w = self._diagonal(model, mass_power=1.0, scale_power=-2.0)
        self.W = None if w is None else DiagonalWeight(w)
        if self.pod is not None:
            self.pod.destroy()
        self.pod = PODBasis(model.mesh_obj.mesh.comm, layout.getSize(), basis_mode=self.basis_mode,
                            centering=self.centering, mean_weighting=self.mean_weighting,
                            IP_matrix=self.W, max_rank=self.max_rank, qr_rank_tol=self.qr_rank_tol)
        for obj in (self.adjoint_pod, self.W_dual):
            if obj is not None:
                obj.destroy()
        self.adjoint_pod = self.W_dual = None
        self._state_norms = {}
        if self.adjoint_snapshots in ('separate', 'concatenate'):
            if self.W is not None:
                w = self.W.w.duplicate()
                w.array[:] = 1.0 / self.W.w.getArray(readonly=True)
                self.W_dual = DiagonalWeight(w)
            self.adjoint_pod = PODBasis(model.mesh_obj.mesh.comm, layout.getSize(), basis_mode=self.basis_mode,
                                        centering=None, IP_matrix=self.W_dual, max_rank=self.adjoint_max_rank,
                                        qr_rank_tol=self.qr_rank_tol)
        PETSc.Sys.Print("Initialized POD ROM: basis_mode={}, centering={}, rb_size={}, max_rank={}, "
                        "n_dofs={}, inner product {}{}".format(
                            self.basis_mode, self.centering, self.rb_size, self.max_rank, layout.getSize(),
                            self.inner_product,
                            "" if self.scales is None else ", state scales {}".format(self.scales)))

    def set_distance_sensitivities(self, norms):
        """Take the weights of the snapshot distance from the total-derivative
        norms of dv_sensitivity (the `norms` dict its probe passes to
        `on_norms`): ||dF/dp_j|| over the objective and constraints, normalized
        to mean 1. Unusable norms (non-finite, or all zero) keep the previous
        weights."""
        if self.distance_weighting is None:
            return
        weights = norms['column_norms_relative'] if self.distance_weighting == 'total_derivative' \
            else norms['column_norms']
        if weights is None:
            # no output values in the derivative result
            PETSc.Sys.Print("ROM distance weighting: output values unavailable, using the raw "
                            "total-derivative norms for this evaluation")
            weights = norms['column_norms']
        weights = np.asarray(weights, dtype=float).ravel()
        if not np.all(np.isfinite(weights)) or not np.any(weights > 0.):
            PETSc.Sys.Print("ROM distance weighting: the total-derivative norms are all zero or "
                            "not finite; keeping the previous weights")
            return
        self.distance_sensitivities = weights / weights.mean()
        inert = "" if self.basis_mode == 'local_weighted' else \
            " -- NOT IN USE, basis_mode is '{}'".format(self.basis_mode)
        PETSc.Sys.Print("ROM distance weights ({}, {} parameters, mean-normalized): min {:.4e}, "
                        "median {:.4e}, max {:.4e}{}".format(
                            self.distance_weighting, weights.shape[0],
                            self.distance_sensitivities.min(), np.median(self.distance_sensitivities),
                            self.distance_sensitivities.max(), inert))

    def _distance_weights(self, n_parameters):
        """The per-parameter factors on the snapshot distance: parameter_scales
        times the output-sensitivity weights. None where there is neither, and
        the sensitivity part is dropped when it does not fit the parameter
        vector (e.g. alpha is not a design variable)."""
        scales = None if self.parameter_scales is None else np.asarray(self.parameter_scales, dtype=float)
        weights = self.distance_sensitivities
        if self.distance_weighting is None or weights is None:
            return scales
        if weights.shape[0] != n_parameters:
            if not self._warned_weight_size:
                PETSc.Sys.Print("ROM distance weighting: {} weights do not fit the {}-entry parameter "
                                "vector; using the distance without them. They come from the design "
                                "variables of the CSDL problem, which must be the snapshot "
                                "parameters.".format(weights.shape[0], n_parameters))
                self._warned_weight_size = True
            return scales
        return weights if scales is None else scales * weights

    def _diagonal(self, model, mass_power, scale_power):
        """diag(M^mass_power D^scale_power) as a Vec in the state layout, on
        the model's current mesh (None when it is the identity)."""
        if self.inner_product == 'euclidean' and self.scales is None:
            return None
        V = model.functionspaces["V"]
        if self.inner_product == 'l2':
            d = diagonal_mass_inverse(V, model.forms.dx)
            arr = d.array
            arr[:] = arr ** (-mass_power)
        else:
            d = model.u_vec.x.petsc_vec.duplicate()
            d.set(1.0)
            arr = d.array
        if self.scales is not None:
            # owned entries come in blocks of n_state, one per cell
            arr *= np.tile(self.scales ** scale_power, arr.shape[0] // model.n_state)
        return d

    @property
    def n_snapshots(self):
        """State snapshots (adjoint snapshots are counted by n_adjoint_snapshots)."""
        return 0 if self.pod is None else int(self.pod.snapshot_matrix.state_mask.sum())

    @property
    def basis(self):
        """The basis of the last solve: the state POD basis, or with 'concatenate' the
        concatenated state and adjoint basis. Valid until the next solve."""
        if self._concatenated_basis is not None:
            return self._concatenated_basis
        return None if self.pod is None else self.pod.basis

    def hybrid_test_space(self):
        """(n_state, Psi): the state width and this rank's rows of the adjoint part of the
        last solve's hybrid test basis [S^2 A V_s, Psi] (a copy, its first k - n_state columns,
        k the width of `basis`), or None when that solve was not hybrid (core/hyper_reduction.py
        mirrors it)."""
        if self._hybrid is None:
            return None
        n_state = self._hybrid['n_state']
        k = self.basis.getSize()[1]
        return n_state, np.array(self._hybrid['Psi'][:, :k - n_state], copy=True)

    @property
    def n_adjoint_snapshots(self):
        if self.adjoint_snapshots in ('separate', 'concatenate'):
            return 0 if self.adjoint_pod is None else self.adjoint_pod.snapshot_matrix.n_snapshots
        return 0 if self.pod is None else self.pod.snapshot_matrix.n_snapshots - self.n_snapshots

    def snapshot_singular_values(self):
        sigma = None if self.pod is None else self.pod.snapshot_matrix.Sigma
        return None if sigma is None else sigma.getArray().copy()

    # ------------------------------------------------------------------
    # Snapshots and solve
    # ------------------------------------------------------------------
    def _check_parameter_size(self, parameter_vector):
        stored = self.pod.snapshot_matrix.snapshot_list_parameters
        if len(stored) and np.size(stored[0]) != np.size(parameter_vector):
            raise ValueError(
                "the snapshots hold {} parameters, but this design has {}: every solve must use the "
                "same parameter vector (shape parameters, then alpha). A solve_forward() without "
                "shape parameters mixed into a shape optimization does this; pass snapshot=False to "
                "it.".format(np.size(stored[0]), np.size(parameter_vector)))

    def add_snapshot(self, model, parameter_vector):
        """Add the model's current u_vec as the snapshot at parameter_vector.
        Returns whether it added a basis direction. Collective."""
        self._check_parameter_size(parameter_vector)
        added = self.pod.snapshot_matrix.process_and_append_to_snapshot_lists(
            model.u_vec.x.petsc_vec, np.asarray(parameter_vector, dtype=float))
        self.pod.snapshot_matrix.print_info()
        if self.adjoint_snapshots == 'augment':
            self._state_norms[np.asarray(parameter_vector, dtype=float).tobytes()] = \
                self._weighted_norm(model.u_vec.x.petsc_vec.getArray(readonly=True), self.W, model)
        return added

    def _weighted_norm(self, x_local, weight, model):
        w = None if weight is None else weight.w.getArray(readonly=True)
        local = float(x_local @ x_local) if w is None else float(x_local @ (w * x_local))
        return float(np.sqrt(model.mesh_obj.mesh.comm.allreduce(local, op=MPI.SUM)))

    def add_adjoint_snapshot(self, model, parameter_vector, row, psi_local):
        """
        Add the full-order adjoint psi (this rank's owned entries) of output row `row` at
        design parameter_vector as a snapshot (adjoint_snapshots; a no-op without), scaled as
        the module docstring says. Returns whether it added a basis direction. Collective.
        """
        if self.adjoint_snapshots is None:
            return False
        self._check_parameter_size(parameter_vector)
        pv = np.asarray(parameter_vector, dtype=float)
        psi = np.asarray(psi_local, dtype=float)
        if self.adjoint_snapshots == 'augment':
            norm = self._weighted_norm(psi, self.W, model)
            target = self._state_norms.get(pv.tobytes())
            if target is None and self._state_norms:
                target = float(np.mean(list(self._state_norms.values())))
            snapshots = self.pod.snapshot_matrix
        else:
            norm, target = self._weighted_norm(psi, self.W_dual, model), 1.0
            snapshots = self.adjoint_pod.snapshot_matrix
        if not (np.isfinite(norm) and norm > 0.0) or target is None:
            return False
        vec = model.u_vec.x.petsc_vec.duplicate()
        vec.array[:] = psi * (target / norm)
        try:
            added = snapshots.process_and_append_to_snapshot_lists(vec, pv, tag=('adjoint', int(row)))
        finally:
            vec.destroy()
        PETSc.Sys.Print("ROM adjoint snapshot (output row {}, {}): {} direction; {} adjoint snapshots".format(
            row, self.adjoint_snapshots, "new" if added else "no new", self.n_adjoint_snapshots))
        return added

    def _adjoint_snapshot_weights(self):
        """The local_weighted weights of the adjoint snapshots: their design's weight in the
        state basis of the last solve (uniform where that was global), None for a global
        adjoint basis."""
        sm = self.adjoint_pod.snapshot_matrix
        if self.basis_mode != 'local_weighted' or sm.n_snapshots < 2:
            return None
        settings = self._basis_settings
        if settings is None or settings['basis_mode'] != 'local_weighted':
            return np.ones(sm.n_snapshots)
        from scipy.spatial.distance import cdist
        d = cdist(np.vstack(sm.snapshot_list_parameters), settings['design_parameters'])
        nearest = d.argmin(axis=1)
        weights = np.where(d[np.arange(d.shape[0]), nearest] <= 1e-10, settings['design_weights'][nearest], 0.0)
        return weights if np.any(weights > 0.0) else np.ones(sm.n_snapshots)

    def discard_adjoint(self):
        """Drop the adjoint of the last accepted solve. Collective."""
        if self.adjoint is not None:
            self.adjoint.destroy()
            self.adjoint = None

    def solve(self, model, parameter_vector):
        """
        The ROM solution at parameter_vector, starting from the model's
        current u_vec (projected onto the basis); u_vec holds it afterwards.
        Returns a ROMResult, or None when there are fewer than min_snapshots
        snapshots. When the solution is accepted, `adjoint` is its adjoint;
        any earlier one is discarded first. Collective.
        """
        self.discard_adjoint()
        self._check_parameter_size(parameter_vector)
        if self.n_snapshots < self.min_snapshots:
            PETSc.Sys.Print("ROM: {} snapshot(s) stored, {} needed; skipping the ROM solve".format(
                self.n_snapshots, self.min_snapshots))
            return None
        t0 = perf_counter()
        comm = model.mesh_obj.mesh.comm
        concatenate = self.adjoint_snapshots == 'concatenate' and self.adjoint_pod.snapshot_matrix.n_snapshots > 0
        n_adjoint = self.rb_size // 2 if concatenate else 0
        self._build_state_basis(model, parameter_vector, self.rb_size - n_adjoint)
        if self.pod.basis is None or self.pod.basis.getSize()[1] == 0:
            PETSc.Sys.Print("ROM: empty POD basis; skipping the ROM solve")
            return None
        self._basis_settings = self._settings_of_last_basis()

        adjoint_basis = None
        if self.adjoint_snapshots in ('separate', 'concatenate') and self.adjoint_pod.snapshot_matrix.n_snapshots > 0:
            with _PROF.phase("rom:adjoint_basis"):
                self.adjoint_pod.construct_basis(rb_size=self.adjoint_rb_size, layout_vec=model.u_vec.x.petsc_vec,
                                                 snapshot_weights=self._adjoint_snapshot_weights())
            adjoint_basis = self.adjoint_pod.basis
        if self._concatenated_basis is not None:
            self._concatenated_basis.destroy()
            self._concatenated_basis = None
        if concatenate:
            width = 0 if adjoint_basis is None else adjoint_basis.getSize()[1]
            if width < n_adjoint:
                # the slots the adjoint basis cannot fill go to the state basis
                self._build_state_basis(model, parameter_vector, self.rb_size - width)
                self._basis_settings = self._settings_of_last_basis()
            with _PROF.phase("rom:basis"):
                self._concatenated_basis = self._concatenate(comm, self.pod.basis, adjoint_basis)
            PETSc.Sys.Print("ROM basis: {} state + {} adjoint vectors -> {} columns".format(
                self.pod.basis.getSize()[1], width, self._concatenated_basis.getSize()[1]))
        V = self.basis

        model.set_up_solver()
        with _PROF.phase("rom:weights"):
            S = self.residual_weight(model)
        hybrid = self.test_space == 'hybrid' and concatenate and V.getSize()[1] > self.pod.basis.getSize()[1]
        self._hybrid = None
        if hybrid:
            hybrid_kw = dict(n_state=self.pod.basis.getSize()[1], Psi=adjoint_basis.getDenseArray())
            self._hybrid = hybrid_kw
            solver = self.make_solver(HybridLSPGSolver, V, S, self.pod.mean, comm, **hybrid_kw)
        else:
            solver = self.make_solver(LSPGSolver, V, S, self.pod.mean, comm)
        try:
            result = self.run_passes(model, solver, t0)
        except BaseException:
            if S is not None:
                S.destroy()
            raise
        self.last_solver = solver
        # Kept for every solve that built a basis, accepted or not: a rejected solve's is not used
        # for the derivative, but can be evaluated at the full-order solution and compared with
        # the full-order adjoint. The basis stays valid until the next solve rebuilds it, which
        # discards this adjoint first (the windtunnel model also keeps it across the derivative
        # re-run of a rejected design, which skips the ROM).
        if self.adjoint_solve != 'least_squares':
            adjoint_basis = None
        dual_weight = None if adjoint_basis is None or self.W is None else \
            1.0 / np.sqrt(self.W.w.getArray(readonly=True))
        if hybrid:
            self.adjoint = HybridLSPGAdjoint(V, S, comm, parameter_vector, result.solution,
                                             accepted=result.accepted, **hybrid_kw)
            return result
        self.adjoint = LSPGAdjoint(V, S, comm, parameter_vector, result.solution,
                                   accepted=result.accepted, adjoint_basis=adjoint_basis, dual_weight=dual_weight,
                                   galerkin=self.adjoint_solve == 'galerkin')
        return result

    def _build_state_basis(self, model, parameter_vector, rb_size):
        with _PROF.phase("rom:basis"):
            self.pod.construct_basis(rb_size=rb_size,
                                     local_parametervector=np.asarray(parameter_vector, dtype=float),
                                     parameter_weights=self._distance_weights(len(parameter_vector)),
                                     weightfunction=self.weightfunction,
                                     n_nonzero_weights=self.n_nonzero_weights,
                                     cubic_cutoff=self.cubic_cutoff,
                                     layout_vec=model.u_vec.x.petsc_vec,
                                     warm_start=model.u_vec.x.petsc_vec)

    # an adjoint direction is dropped from the concatenated basis when less than this fraction of
    # it (in the W-norm) lies outside the span of the state basis
    CONCATENATION_RTOL = 1e-8

    def _concatenate(self, comm, V_state, V_adjoint):
        """
        [V_state, Psi_perp]: the state basis as it is (W-orthonormal, from the POD) and the
        adjoint basis orthogonalized against it in the W inner product (two classical
        Gram-Schmidt passes, for round-off) and then W-orthonormalized, so that both blocks
        are orthonormal and mutually orthogonal (V^T W V = I, which the restriction
        a = V^T W (u - mean) needs). Adjoint directions with no part outside the state basis
        (CONCATENATION_RTOL) are dropped.
        """
        Vs = V_state.getDenseArray()
        if V_adjoint is None:
            return petsc_dense_from_local_block(comm, Vs.copy())
        w = np.ones(Vs.shape[0]) if self.W is None else self.W.w.getArray(readonly=True)
        sqrt_w = np.sqrt(w)
        P = V_adjoint.getDenseArray().copy()
        # unit W-norm columns, so that what remains after the projection is measured against 1
        P /= np.maximum(np.sqrt(comm.allreduce(np.sum(w[:, None] * P**2, axis=0), op=MPI.SUM)), 1e-300)
        for _ in range(2):
            P -= Vs @ comm.allreduce(Vs.T @ (w[:, None] * P), op=MPI.SUM)
        Q = orthonormal_columns(comm, sqrt_w[:, None] * P, atol=self.CONCATENATION_RTOL) / sqrt_w[:, None]
        return petsc_dense_from_local_block(comm, np.column_stack([Vs, Q]))

    def residual_weight(self, model):
        """S, the diagonal weight of the LSPG residual norm ||S r|| on the
        model's current mesh, as a new Vec (None: unweighted)."""
        return self._diagonal(model, mass_power=-0.5, scale_power=-1.0)

    def make_solver(self, solver_class, V, S, mean, comm, **kwargs):
        """A `solver_class` (LSPGSolver or a subclass) on basis V with this
        ROM's inner product, residual weight S, offset `mean` and Newton
        settings; `kwargs` go to the subclass."""
        return solver_class(V, comm, W=None if self.W is None else self.W.w, S=S, mean=mean,
                            rtol=self.newton_rtol, atol=self.newton_atol, max_it=self.newton_max_it,
                            stall_window=self.stall_window, stall_rtol=self.stall_rtol,
                            line_search=self.line_search, **kwargs)

    def run_passes(self, model, solver, t0):
        """
        The LSPG passes of a solve with `solver` (make_solver), from the
        model's current u_vec, and the verdict: a ROMResult (walltime from
        t0). The model's u_vec holds the solution afterwards. The solver is
        destroyed; its counters stay readable. Shared by solve() and the
        hyper-reduced comparison solves (core/hyper_reduction.py), which
        therefore stop and are accepted by the same rules. Collective.
        """
        # Up to max_outer_iterations passes, each restarted from the current
        # state; converged when the pass's last |g| is below
        # max(newton_atol, g_conv_limit g0), g0 the solver's first |g| (the
        # LSPG stationarity residual -- the projected residual V^T r does not
        # vanish at the LSPG solution, so it is not tested)
        converged = False
        iterations = 0
        residual_norm = np.nan
        try:
            for _ in range(self.max_outer_iterations):
                with model.steady_residual():
                    its, _ = solver.solve(model)
                iterations += its
                residual_norm = model._assemble_physical_residual()
                if not np.isfinite(residual_norm):
                    PETSc.Sys.Print("ROM: non-finite residual; stopping")
                    break
                g_target = max(self.newton_atol, self.g_conv_limit * solver.g0) \
                    if self.g_conv_limit > 0.0 and solver.g0 is not None else None
                if g_target is not None and solver.last_g_norm is not None and solver.last_g_norm < g_target:
                    converged = True
                    PETSc.Sys.Print("LSPG converged: |g| = {:.6e} < {:.6e} ({:.1e} |g0|)".format(
                        solver.last_g_norm, g_target, self.g_conv_limit))
                    break
            else:
                PETSc.Sys.Print("ROM: {} LSPG passes without |g| < {:.1e} |g0| (last |g| {:.6e}, g0 {})".format(
                    self.max_outer_iterations, self.g_conv_limit, solver.last_g_norm, solver.g0))
        finally:
            solver.destroy()
        solver.n_iterations = iterations

        # one verdict line for every way out of the loop
        PETSc.Sys.Print("ROM LSPG solve {}: |g| = {} (g0 {}, limit {:.1e} |g0|), {} iterations".format(
            "CONVERGED" if converged else "NOT CONVERGED", solver.last_g_norm, solver.g0,
            self.g_conv_limit, iterations))

        eta = model.relative_residual_norm(residual_norm)
        model.u_vec.x.scatter_forward()
        result = ROMResult(solution=model.u_vec.x.petsc_vec.getArray().copy(), eta=eta,
                           residual_norm=residual_norm, converged=converged,
                           accepted=bool(np.isfinite(eta) and eta < self.eta_threshold),
                           iterations=iterations, basis_width=solver.rb_size,
                           walltime=perf_counter() - t0)
        PETSc.Sys.Print("ROM relative residual norm eta: {} (||r||: {}); {} iterations, wall time {}".format(
            eta, residual_norm, iterations, result.walltime))
        return result

    def _settings_of_last_basis(self):
        if self.pod.snapshot_weights is None:
            return {'basis_mode': 'global'}
        return {'basis_mode': 'local_weighted',
                'design_parameters': np.vstack(self.pod.snapshot_matrix.snapshot_list_parameters).copy(),
                'design_weights': self.pod.snapshot_weights.copy()}

    def basis_settings(self):
        """
        How the basis of the last solve was built, for the bases of
        hyper-reduction over the same designs, which follow it
        (core/hyper_reduction.py): {'basis_mode': 'global'}, or for a
        local_weighted basis {'basis_mode': 'local_weighted',
        'design_parameters': the accepted snapshots' parameter vectors (rows),
        'design_weights': their weights}. A local_weighted ROM with a single
        snapshot builds its basis like 'global'. None before the first solve
        that built a basis.
        """
        return self._basis_settings

    def destroy(self):
        self.discard_adjoint()
        if self.pod is not None:
            self.pod.destroy()
            self.pod = None
        if self.W is not None:
            self.W.destroy()
            self.W = None
        for obj in (self.adjoint_pod, self.W_dual, self._concatenated_basis):
            if obj is not None:
                obj.destroy()
        self.adjoint_pod = self.W_dual = self._concatenated_basis = None
