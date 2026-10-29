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

adjoint_bases='per_output' ('separate' and 'concatenate') keeps the adjoint snapshots of each output row
in a snapshot matrix of their own: the adjoints of different outputs can occupy different spaces, and one
basis of adjoint_rb_size vectors over all of them favours the outputs with the most energy. Every solve
builds a basis of adjoint_rb_size vectors per row (weighted per design like the state basis) and
orthonormalizes them together in W^-1, dropping shared directions, into Psi, of up to n_outputs *
adjoint_rb_size directions. With 'concatenate' the basis is then the state basis of ceil(rb_size / 2)
vectors and those adjoint (or paired) directions, so it grows with the number of outputs; with
'separate' Psi is the adjoint basis of the reduced adjoint (adjoint_solve='least_squares' or
'dual_corrected').

hybrid_trial='paired' (test_space='hybrid'; recommended) replaces the adjoint directions of the
trial basis by W^-1 A^T Psi, A the steady Jacobian at the solve's warm start: V = [V_s, W^-1 A^T Psi]
(orthogonalized against V_s), Phi = [S^2 A V_s, Psi]. An adjoint A^-T g lies mostly along A's
weakest directions, so with the adjoint basis itself as trial directions (hybrid_trial='adjoint')
Psi^T A V is small in every column and the reduced matrix has about one near-zero singular value per
output; on a transonic RANS case (examples/airfoil_opt_rans_pod.py) its adjoint was 1.0 (relative,
dual norm) from the full-order one although that lay within 1% of its space. W^-1 A^T psi is the trial
direction that maximizes psi^T A v / ||v||_W, so Psi^T A Z is the Gram matrix of A^T Psi in W^-1 and
the adjoint test directions are paired; there the smallest singular value (dual-orthonormal test,
W-orthonormal trial bases) rose from ~1e-8 to ~5e-5, and the adjoint error fell from 1.0 to 4-20%
(2-20 times the distance of the full-order adjoint to its space, as for a minimum-residual solve on the
same space). The equations, their
Newton loop and their adjoint are those of test_space='hybrid'; the adjoint is the exact derivative of
the ROM's outputs (the basis fixed per design).

Z does not enter the adjoint's space span[S^2 A V_s, Psi]; with Z at the Jacobian the adjoint is
evaluated with, the Psi rows of the adjoint equations are the minimum-residual condition for that
block. Two ways to bring Z to the solution's Jacobian instead of the warm start's:

- paired_update='newton': Z is rebuilt at every Newton iteration from the Jacobian the state test
  directions S^2 A V_s are built from, as LSPG freezes its test basis per iteration (HybridLSPGSolver,
  trial_update), for k products with A^T per iteration. Z is in the trial basis, so each step starts
  from the iterate's projection onto the new basis (an inexact Newton step): the solution lies in
  mean + span[V_s, Z(u*)]. The adjoint uses the last Z. Keeping the iterate instead (its parts along
  earlier Z frozen into the offset) made the solution path-dependent, with a residual 10 times larger
  on the RANS test case.
- paired_update='solution': after the solve, Z is rebuilt at the ROM's solution and the ROM solved
  again from it, until Z changes by less than paired_update_rtol (relative, W-norm, worst column) or
  after paired_update_max_passes re-pairings: one reduced space per solve, for a Jacobian assembly, k
  products with A^T and a ROM solve per re-pairing.

An LSPG ROM whose trial basis holds the adjoints' preimages A^-1 S^-2 psi (so that its adjoint's
space span(S^2 A V) holds psi) was tried and removed: exact at the snapshots' Jacobian, but the preimages
do not extrapolate to other designs (docs/src/reduced_order_modeling.md).

adjoint_solve chooses the reduced adjoint: 'lspg' (the adjoint of the LSPG equations
above, the exact derivative of the ROM's outputs), 'galerkin' (psi = V z in the span of the
basis itself, V^T A^T V z = V^T g: with 'concatenate' the adjoint basis is then in its trial
space, which the LSPG adjoint's span(S^2 A V) does not contain) or 'least_squares' (the
least-squares adjoint over span[S^2 A V, Psi], 'separate' and 'concatenate' only); the
default is 'least_squares' for 'separate' and 'lspg' otherwise. Only 'lspg' is the exact
derivative of the ROM's outputs, consistent with the forward (LSPG) solve.

adjoint_solve='dual_corrected' ('separate', test_space='lspg') is the exact derivative of the
corrected outputs instead (the non-conforming dual approach of Keil, Mechelli, Ohlberger, Schindler and
Volkwein, ESAIM: M2AN 2021): a dual ROM on the adjoint basis Psi, psi_d = Psi z minimizing
||W^-1/2 (A^T Psi z - g)||, corrects an output f to f(u_r) - psi_d^T R(u_r), which leaves an error of
(psi_fom - psi_d)^T R(u_r) only (the "squared" effect of the dual-weighted residual). With the basis,
weights and psi_d's dependence on the design frozen and the terms proportional to R(u_r) dropped (as for
'lspg'), the derivative of that output is df/dx|_u - psi^T dR/dx with

    psi = psi_d + S Q R^-T V^T (g - A^T psi_d),    S A V = Q R,

the LSPG adjoint of what the dual ROM leaves of the adjoint equations: with D = S^-1,
D psi = D psi_lspg + (I - Q Q^T) D psi_d, the LSPG adjoint in span(S A V) and the dual ROM's outside it.
The LSPG forward solve does not see Psi. The output correction is not part of the ROM's outputs yet
(the windtunnel model records it: data_store.output_corrections).

test_space='hybrid' ('concatenate' only) changes the forward solve itself: Petrov-Galerkin with
the concatenated basis V = [V_s, Psi_perp] as trial basis and Phi = [S^2 A V_s, Psi] as test
basis, the state part LSPG's and the adjoint part the adjoint basis itself (HybridLSPGSolver).
Its adjoint, (Phi^T A V)^T lam = V^T g, psi = Phi lam (HybridLSPGAdjoint), is the derivative of
those equations (as LSPGAdjoint's of LSPG, Phi frozen) and lies in a space that holds Psi, which
the LSPG adjoint's span(S^2 A V) does not: on the test cases its derivatives were 100-10^4 times
closer to the full-order ones than the state basis gave, at a slightly larger ||R|| (the forward
solve no longer minimizes ||S R||). hybrid_state_test='orthonormal' uses S Q for the state part of
Phi instead, Q an orthonormal basis of span(S A V_s) (S A V_s = Q R_s): the same span, so the same
reduced equations and adjoint in exact arithmetic; the state rows of the reduced system are those of
S^2 A V_s multiplied by R_s^-T, R_s from the Cholesky factorization of their Gram block (also in the
hyper-reduced solves, which hold that block, not S A V_s; orthonormal_state_rows).
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


def scaled_jacobian_basis(model, V, S, AV=None, assemble=True):
    """
    S A V at the model's current u_vec, A the exact Jacobian assembled by its
    Newton problem (problem.J; assemble=False: as already assembled there).
    Returns (this rank's rows of S A V, the PETSc product A V); `AV`, if given,
    is a product of the same shape and basis to reuse. Without S the rows are a
    view of AV's array.
    """
    problem = model.problem
    if assemble:
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


def subspace_distance(comm, X_local, x_local):
    """||x - P x||_2 / ||x||_2, P the orthogonal projection onto the span of the
    row-distributed columns X, and P x (this rank's rows). Collective."""
    x = np.asarray(x_local, dtype=float)
    Q = orthonormal_columns(comm, np.asarray(X_local, dtype=float)) if X_local is not None and \
        np.asarray(X_local).shape[1] else np.zeros((x.shape[0], 0))
    Px = Q @ comm.allreduce(Q.T @ x, op=MPI.SUM)
    norm = np.sqrt(comm.allreduce(float(x @ x), op=MPI.SUM))
    dist = np.sqrt(comm.allreduce(float((x - Px) @ (x - Px)), op=MPI.SUM))
    return (dist / norm if norm > 0.0 else np.nan), Px


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
            a = self._coordinates(a)
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

    def _coordinates(self, a):
        """The reduced coordinates of the current state after _reduced_system, which may have
        changed the basis (HybridLSPGSolver with trial_update); here unchanged."""
        return a

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
                 dual_weight=None, galerkin=False, dual_corrected=False):
        self.V = V
        self.S = S
        self.comm = comm
        # adjoint_solve='dual_corrected': the LSPG adjoint plus the dual ROM's adjoint psi_d = Psi z
        # (minimum residual) outside S A V's directions (module docstring); last_dual is the psi_d of
        # the last solve, for the output correction -psi_d^T R(u_r)
        self.dual_corrected = dual_corrected
        self._Qd = self._Rd = None
        self.last_dual = None
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
                VtAV = self.comm.allreduce(self.V.getDenseArray().T @ AV_local, op=MPI.SUM)
                self.condition_number = float(np.linalg.cond(VtAV))
                self._lu = sp_la.lu_factor(VtAV)
                self._Q, self._R = None, np.diag(np.abs(np.diag(self._lu[0])))
            elif self.adjoint_basis is None or self.dual_corrected:
                self._Q, self._R = tsqr(self.comm, SAV)
                if self.dual_corrected:
                    self._factor_dual(model)
            else:
                self._factor_least_squares(model, SAV)
        AV.destroy()
        if not self.galerkin:
            self.condition_number = float(np.linalg.cond(self._R))
        diag = np.abs(np.diag(self._R))
        if not np.all(np.isfinite(self._R)) or diag.min() <= np.finfo(float).eps * diag.max():
            raise ValueError("reduced adjoint: S A V is singular to working precision "
                             "(|R_ii| from {:.3e} to {:.3e})".format(diag.min(), diag.max()))

    def _transpose_products(self, model, X):
        """A^T X (this rank's rows), A the Jacobian scaled_jacobian_basis just assembled."""
        A = model.problem._A
        x, y = A.createVecLeft(), A.createVecRight()
        ATX = np.empty_like(X)
        for j in range(X.shape[1]):
            x.array[:] = X[:, j]
            A.multTranspose(x, y)
            ATX[:, j] = y.array
        x.destroy()
        y.destroy()
        return ATX

    def _factor_dual(self, model):
        """The dual ROM of adjoint_solve='dual_corrected': Q_d, R_d of W^-1/2 A^T Psi."""
        ATPsi = self._transpose_products(model, self.adjoint_basis.getDenseArray())
        if self.dual_weight is not None:
            ATPsi *= self.dual_weight[:, None]
        self._Qd, self._Rd = tsqr(self.comm, ATPsi)
        self.dual_condition_number = float(np.linalg.cond(self._Rd))

    def _solve_dual_corrected(self, model, rhs, first):
        """psi = psi_lspg + (I - S P S^-1) psi_d (module docstring), P the orthogonal projection onto
        span(Q), S A V = Q R; psi_d = Psi z minimizing ||W^-1/2 (A^T Psi z - g)||."""
        Dg = rhs if self.dual_weight is None else self.dual_weight * rhs
        z = sp_la.solve_triangular(self._Rd, self.comm.allreduce(self._Qd.T @ Dg, op=MPI.SUM), lower=False)
        psi_d = self.adjoint_basis.getDenseArray() @ z
        b = self.comm.allreduce(self.V.getDenseArray().T @ rhs, op=MPI.SUM)
        y = sp_la.solve_triangular(self._R, b, trans='T', lower=False)
        s = np.ones_like(rhs) if self.S is None else self.S.getArray(readonly=True)
        # S Q (y - Q^T S^-1 psi_d): the LSPG adjoint of the dual ROM's remaining right-hand side
        # V^T (g - A^T psi_d), as V^T A^T psi_d = R^T Q^T S^-1 psi_d
        y = y - self.comm.allreduce(self._Q.T @ (psi_d / s), op=MPI.SUM)
        psi = psi_d + s * (self._Q @ y)
        if not np.isfinite(np.sqrt(self.comm.allreduce(float(psi @ psi), op=MPI.SUM))):
            raise ValueError("reduced adjoint: non-finite solution")
        self.last_dual = psi_d
        if first:
            res = self._Qd @ (self._Rd @ z) - Dg
            PETSc.Sys.Print("dual-corrected adjoint: dual ROM over {} adjoint basis vectors, |W^-1/2 (A^T psi_d - g)| / "
                            "|W^-1/2 g| = {:.3e} (condition {:.3e}); LSPG correction over a basis of width {}, "
                            "condition {:.3e}".format(
                                self.adjoint_basis.getSize()[1],
                                np.sqrt(self.comm.allreduce(float(res @ res), op=MPI.SUM)
                                        / max(self.comm.allreduce(float(Dg @ Dg), op=MPI.SUM), 1e-300)),
                                self.dual_condition_number, self.rb_size, float(np.linalg.cond(self._R))))
        return psi

    def _factor_least_squares(self, model, SAV):
        """The adjoint basis variant: T = an orthonormal basis of span[S^2 A V, Psi]
        (self._T) and Q, R of W^-1/2 A^T T, A the Jacobian scaled_jacobian_basis
        just assembled."""
        S = None if self.S is None else self.S.getArray(readonly=True)
        lspg_space = SAV if S is None else S[:, None] * SAV
        self._T = orthonormal_columns(self.comm, np.column_stack([lspg_space, self.adjoint_basis.getDenseArray()]))
        ATT = self._transpose_products(model, self._T)
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
        if self.dual_corrected:
            return self._solve_dual_corrected(model, rhs, first)
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

    def spaces(self):
        """(this rank's rows of a basis of the space every reduced adjoint psi lies in, of the
        adjoint basis it holds or None), after the first solve: S^2 A V (LSPG), V (Galerkin),
        span[S^2 A V, Psi] (least squares, dual-corrected)."""
        if self.galerkin:
            return self.V.getDenseArray(), None
        if self.dual_corrected and self._T is None:
            S = 1.0 if self.S is None else self.S.getArray(readonly=True)[:, None]
            self._T = orthonormal_columns(self.comm, np.column_stack([S * self._Q, self.adjoint_basis.getDenseArray()]))
        if self.adjoint_basis is not None:
            return self._T, self.adjoint_basis.getDenseArray()
        return (self._Q if self.S is None else self.S.getArray(readonly=True)[:, None] * self._Q), None

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
        self._Qd = self._Rd = None
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

    trial_update (paired_update='newton'): a callable that takes the Jacobian and returns the
    new basis [V_s, Z(A)]. It is called at every iteration with the Jacobian the test basis is
    built from, so the trial directions are re-paired like the state test directions. The
    step (Phi^T A V) da = -Phi^T r is the linearization at the iterate u with the new trial
    directions, taken from u's projection onto the new basis (an inexact Newton step whose
    error, the part of u outside the new span, vanishes as Z converges): the solution lies in
    mean + span[V_s, Z(u*)], as with paired_update='solution'.
    """

    def __init__(self, V, comm, n_state=None, Psi=None, orthonormal=False, trial_update=None, **kwargs):
        super().__init__(V, comm, **kwargs)
        self.n_state = n_state
        self.Psi = Psi                        # this rank's rows of the adjoint basis
        self.orthonormal = orthonormal        # hybrid_state_test='orthonormal'
        self.trial_update = trial_update
        self._merit_gradient = None
        self._projected = None                # trial_update: the iterate's coordinates in the new basis

    def _repair(self, model):
        """trial_update: the basis re-paired with the Jacobian just assembled, and the iterate's
        coordinates in it, its W-orthogonal projection (_coordinates)."""
        self.V = self.trial_update(model.problem._A)
        self._projected = self.restrict(model.u_vec.x.petsc_vec)
        if self._AV is not None:                        # a product of the old basis
            self._AV.destroy()
            self._AV = None

    def _coordinates(self, a):
        a, self._projected = (a if self._projected is None else self._projected), None
        return a

    def _reduced_system(self, model):
        Sr, r_norm = self._weighted_residual(model)
        if self.trial_update is not None:
            with _PROF.phase("rom:project"):
                model.problem.J(model.u_vec.x.petsc_vec, model.problem._A)
            with _PROF.phase("rom:basis"):
                self._repair(model)
        if self._AV is not None and self._AV.getSize()[1] != self.V.getSize()[1]:
            self._AV.destroy()
            self._AV = None
        with _PROF.phase("rom:project"):
            SAV, self._AV = scaled_jacobian_basis(model, self.V, self.S, self._AV,
                                                  assemble=self.trial_update is None)
            H, g, self._merit_gradient = hybrid_reduced_system(self.comm, SAV, Sr, self.S, self.n_state, self.Psi)
            if self.orthonormal:
                H, g = orthonormal_state_rows(H, g, self.n_state)
        return H, g, r_norm

    def _backtrack(self, model, a, da, theta, phi, slope):
        # the slope of the merit ||S r||^2 / 2, not of the hybrid right-hand side
        return super()._backtrack(model, a, da, theta, phi, float(self._merit_gradient @ da))


def hybrid_test_basis(SAV, S, n_state, Psi):
    """This rank's rows of Phi = [S^2 A V_s, Psi[:, :k - n_state]], from S A V."""
    k = SAV.shape[1]
    state = SAV[:, :n_state] if S is None else S[:, None] * SAV[:, :n_state]
    return np.column_stack([state, Psi[:, :k - n_state]])


def state_gram_factor(G):
    """R_s, upper triangular with R_s^T R_s = G, the Gram matrix (S A V_s)^T (S A V_s) of the state
    columns: S A V_s = Q R_s with Q orthonormal."""
    return sp_la.cholesky(0.5 * (G + G.T), lower=False)


def orthonormal_state_rows(H, g, n_state, R=None):
    """(H, g) of a hybrid reduced system with the state test basis S^2 A V_s replaced by S Q, Q an
    orthonormal basis of its span: the state rows multiplied by R_s^-T. R_s from the Gram block
    H[:n_state, :n_state] (state_gram_factor) unless given; g may be None."""
    if n_state == 0:
        return H, g
    if R is None:
        R = state_gram_factor(H[:n_state, :n_state])
    H = H.copy()
    H[:n_state] = sp_la.solve_triangular(R, H[:n_state], trans='T', lower=False)
    if g is not None:
        g = np.array(g, dtype=float, copy=True)
        g[:n_state] = sp_la.solve_triangular(R, g[:n_state], trans='T', lower=False)
    return H, g


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

    def __init__(self, V, S, comm, parameter_vector, solution, accepted=True, n_state=None, Psi=None,
                 orthonormal=False):
        super().__init__(V, S, comm, parameter_vector, solution, accepted=accepted)
        self.n_state = n_state
        self.Psi = Psi
        self.orthonormal = orthonormal

    def _orthonormalize(self, SAV_state):
        """With hybrid_state_test='orthonormal': Phi's state columns S^2 A V_s replaced by S Q,
        i.e. multiplied by R_s^-1, R_s from the Gram matrix of SAV_state (this rank's rows of the
        S A V_s the test basis was built from). Collective."""
        n = self.n_state
        if not self.orthonormal or n == 0:
            return
        R = state_gram_factor(self.comm.allreduce(SAV_state.T @ SAV_state, op=MPI.SUM))
        self._Phi = self._Phi.copy()
        self._Phi[:, :n] = sp_la.solve_triangular(R, self._Phi[:, :n].T, trans='T', lower=False).T

    def _report(self, H, SAV):
        """Print the condition number of Phi^T A V and the basis-independent singular values
        (inf-sup): those of X^T S A V, X an orthonormal basis of span(Phi / S), i.e. of the map
        between the test space in the dual norm (W^-1 = D^2) and the trial space in the W-norm
        (V is W-orthonormal); the reduced adjoint's error is at most about
        sigma_max(S A S^-1) / sigma_min times the distance of psi_fom to the test space."""
        self.condition_number = float(np.linalg.cond(H))
        S = None if self.S is None else self.S.getArray(readonly=True)
        X = orthonormal_columns(self.comm, self._Phi if S is None else self._Phi / S[:, None])
        sv = np.linalg.svd(self.comm.allreduce(X.T @ SAV, op=MPI.SUM), compute_uv=False)
        self.inf_sup = sv
        PETSc.Sys.Print("hybrid reduced adjoint: Phi^T A V condition {:.3e}; basis-independent singular values "
                        "(dual-orthonormal test, W-orthonormal trial) {}".format(
                            self.condition_number, np.array2string(sv, precision=2, max_line_width=1000)))

    def _factor(self, model):
        model.set_up_solver()
        with model.steady_residual():
            SAV, AV = scaled_jacobian_basis(model, self.V, self.S)
            S = None if self.S is None else self.S.getArray(readonly=True)
            self._Phi = hybrid_test_basis(SAV, S, self.n_state, self.Psi)
            self._orthonormalize(SAV[:, :self.n_state])
            H = self.comm.allreduce(self._Phi.T @ AV.getDenseArray(), op=MPI.SUM)
            self._report(H, SAV)
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

    def spaces(self):
        return self._Phi, self._Phi[:, self.n_state:]

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
                         'separate': adjoint basis width (default rb_size); 'concatenate': the
                         width of the adjoint basis (default floor(rb_size / 2)) next to a state
                         basis of ceil(rb_size / 2). Per output row with adjoint_bases=
                         'per_output'. The stored rank defaults to 3 times the width
      adjoint_solve      'lspg', 'galerkin', 'least_squares' or 'dual_corrected' (module docstring)
      test_space         'lspg' (default) or 'hybrid' ('concatenate' only): the test basis
                         [S^2 A V_s, Psi] (HybridLSPGSolver), with the adjoint of those equations
      hybrid_state_test  'normal_equations' (default: S^2 A V_s) or 'orthonormal' (S Q, the same
                         span; module docstring) for the state part of the hybrid test basis
      hybrid_trial       'adjoint' (default: the trial basis holds the adjoint basis, orthogonalized
                         against the state basis) or 'paired' (W^-1 A^T Psi instead, at the solve's
                         warm start; test_space='hybrid' only; module docstring)
      adjoint_bases      'shared' (default: one adjoint snapshot matrix and basis for all output rows)
                         or 'per_output' ('separate' and 'concatenate'): a snapshot matrix per output
                         row, each giving its own adjoint basis of adjoint_rb_size vectors; the bases
                         of all rows, orthonormalized together in W^-1, form Psi (module docstring)
      paired_update      'warm_start' (default: Z at the solve's warm start), 'newton' (re-paired at
                         every Newton iteration) or 'solution' (re-paired at the ROM's solution and
                         solved again, with paired_update_rtol (1e-3) and paired_update_max_passes
                         (3)); hybrid_trial='paired' only; module docstring

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
                 test_space='lspg', hybrid_state_test='normal_equations', hybrid_trial='adjoint',
                 paired_update='warm_start', paired_update_rtol=1e-3, paired_update_max_passes=3,
                 adjoint_bases='shared'):
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
        if adjoint_solve not in ('lspg', 'galerkin', 'least_squares', 'dual_corrected'):
            raise ValueError("adjoint_solve must be 'lspg', 'galerkin', 'least_squares' or 'dual_corrected', got "
                             "{!r}".format(adjoint_solve))
        if adjoint_solve == 'dual_corrected' and (adjoint_snapshots != 'separate' or test_space != 'lspg'):
            raise ValueError("adjoint_solve='dual_corrected' needs adjoint_snapshots='separate' and test_space='lspg'")
        if adjoint_solve == 'least_squares' and adjoint_snapshots not in ('separate', 'concatenate'):
            raise ValueError("adjoint_solve='least_squares' needs an adjoint basis: adjoint_snapshots='separate' "
                             "or 'concatenate'")
        if test_space not in ('lspg', 'hybrid'):
            raise ValueError("test_space must be 'lspg' or 'hybrid', got {!r}".format(test_space))
        if test_space == 'hybrid' and adjoint_snapshots != 'concatenate':
            raise ValueError("test_space='hybrid' needs adjoint_snapshots='concatenate'")
        self.test_space = test_space
        if hybrid_state_test not in ('normal_equations', 'orthonormal'):
            raise ValueError("hybrid_state_test must be 'normal_equations' or 'orthonormal', got {!r}".format(
                hybrid_state_test))
        self.hybrid_state_test = hybrid_state_test
        if hybrid_trial not in ('adjoint', 'paired'):
            raise ValueError("hybrid_trial must be 'adjoint' or 'paired', got {!r}".format(hybrid_trial))
        if hybrid_trial == 'paired' and test_space != 'hybrid':
            raise ValueError("hybrid_trial='paired' needs test_space='hybrid'")
        self.hybrid_trial = hybrid_trial
        if paired_update not in ('warm_start', 'newton', 'solution'):
            raise ValueError("paired_update must be 'warm_start', 'newton' or 'solution', got {!r}".format(
                paired_update))
        if paired_update != 'warm_start' and hybrid_trial != 'paired':
            raise ValueError("paired_update={!r} needs hybrid_trial='paired'".format(paired_update))
        self.paired_update = paired_update
        self.paired_update_rtol = paired_update_rtol
        self.paired_update_max_passes = paired_update_max_passes
        # paired_update: the change of Z (relative, W-norm, worst column) at each re-pairing of the last
        # solve; 'solution' records (change, iterations, eta)
        self.paired_updates = []
        self._paired_Z = None
        self.adjoint_snapshots = adjoint_snapshots
        self.adjoint_solve = adjoint_solve
        if adjoint_bases not in ('shared', 'per_output'):
            raise ValueError("adjoint_bases must be 'shared' or 'per_output', got {!r}".format(adjoint_bases))
        if adjoint_bases == 'per_output' and adjoint_snapshots not in ('separate', 'concatenate'):
            raise ValueError("adjoint_bases='per_output' needs an adjoint basis: adjoint_snapshots='separate' "
                             "or 'concatenate'")
        self.adjoint_bases = adjoint_bases
        # adjoint_bases='per_output': the adjoint snapshot matrix (PODBasis) of each output row, and the
        # combined adjoint basis of the last solve
        self.adjoint_pods = {}
        self._combined_adjoint_basis = None
        # 'concatenate': the departures from orthonormality of the last solve's bases (_orthonormality)
        self.orthonormality = None
        if adjoint_snapshots == 'concatenate' and adjoint_rb_size is None:
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
        for obj in [self.adjoint_pod, self.W_dual] + list(self.adjoint_pods.values()):
            if obj is not None:
                obj.destroy()
        self.adjoint_pod = self.W_dual = None
        self.adjoint_pods = {}
        self._state_norms = {}
        if self.adjoint_snapshots in ('separate', 'concatenate'):
            if self.W is not None:
                w = self.W.w.duplicate()
                w.array[:] = 1.0 / self.W.w.getArray(readonly=True)
                self.W_dual = DiagonalWeight(w)
            if self.adjoint_bases == 'shared':
                self.adjoint_pod = self._new_adjoint_pod(model)
        PETSc.Sys.Print("Initialized POD ROM: basis_mode={}, centering={}, rb_size={}, max_rank={}, "
                        "n_dofs={}, inner product {}{}".format(
                            self.basis_mode, self.centering, self.rb_size, self.max_rank, layout.getSize(),
                            self.inner_product,
                            "" if self.scales is None else ", state scales {}".format(self.scales)))

    def _new_adjoint_pod(self, model):
        return PODBasis(model.mesh_obj.mesh.comm, model.u_vec.x.petsc_vec.getSize(), basis_mode=self.basis_mode,
                        centering=None, IP_matrix=self.W_dual, max_rank=self.adjoint_max_rank,
                        qr_rank_tol=self.qr_rank_tol)

    def _adjoint_pod_list(self):
        """The adjoint snapshot matrices (PODBasis): the shared one, or one per output row in row
        order (adjoint_bases)."""
        if self.adjoint_bases == 'per_output':
            return [self.adjoint_pods[row] for row in sorted(self.adjoint_pods)]
        return [] if self.adjoint_pod is None else [self.adjoint_pod]

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
            return sum(pod.snapshot_matrix.n_snapshots for pod in self._adjoint_pod_list())
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
            if self.adjoint_bases == 'per_output' and int(row) not in self.adjoint_pods:
                self.adjoint_pods[int(row)] = self._new_adjoint_pod(model)
            snapshots = (self.adjoint_pods[int(row)] if self.adjoint_bases == 'per_output'
                         else self.adjoint_pod).snapshot_matrix
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

    def _adjoint_snapshot_weights(self, pod=None):
        """The local_weighted weights of the adjoint snapshots of `pod` (default: the shared
        adjoint snapshot matrix): their design's weight in the state basis of the last solve
        (uniform where that was global), None for a global adjoint basis."""
        sm = (self.adjoint_pod if pod is None else pod).snapshot_matrix
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

    def stored_adjoint_subspace(self):
        """This rank's rows of the left-singular vectors the adjoint snapshot matrix keeps
        (adjoint_snapshots 'separate' / 'concatenate', at most adjoint_max_rank of them), or None:
        every adjoint basis lies in their span."""
        Q = [pod.snapshot_matrix.Q.getDenseArray() for pod in self._adjoint_pod_list()
             if pod.snapshot_matrix.Q is not None]
        return np.column_stack(Q) if Q else None

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
        concatenate = self.adjoint_snapshots == 'concatenate' and self.n_adjoint_snapshots > 0
        n_adjoint = self.rb_size // 2 if concatenate else 0
        self._build_state_basis(model, parameter_vector, self.rb_size - n_adjoint)
        if self.pod.basis is None or self.pod.basis.getSize()[1] == 0:
            PETSc.Sys.Print("ROM: empty POD basis; skipping the ROM solve")
            return None
        self._basis_settings = self._settings_of_last_basis()

        adjoint_basis = None
        if self.adjoint_snapshots in ('separate', 'concatenate') and self.n_adjoint_snapshots > 0:
            with _PROF.phase("rom:adjoint_basis"):
                for pod in self._adjoint_pod_list():
                    if pod.snapshot_matrix.n_snapshots > 0:
                        pod.construct_basis(rb_size=self.adjoint_rb_size, layout_vec=model.u_vec.x.petsc_vec,
                                            snapshot_weights=self._adjoint_snapshot_weights(pod))
                adjoint_basis = self._combined_adjoint_basis_of(comm) if self.adjoint_bases == 'per_output' \
                    else self.adjoint_pod.basis
        if self._concatenated_basis is not None:
            self._concatenated_basis.destroy()
            self._concatenated_basis = None
        paired = False
        Z = None
        if concatenate:
            width = 0 if adjoint_basis is None else adjoint_basis.getSize()[1]
            if width < n_adjoint:
                # the slots the adjoint basis cannot fill go to the state basis
                self._build_state_basis(model, parameter_vector, self.rb_size - width)
                self._basis_settings = self._settings_of_last_basis()
            paired = self.hybrid_trial == 'paired' and self.test_space == 'hybrid' and adjoint_basis is not None
            with _PROF.phase("rom:basis"):
                extra = Z = self._paired_trial_directions(model, adjoint_basis) if paired else adjoint_basis
                self._concatenated_basis = self._concatenate(comm, self.pod.basis, extra)
            PETSc.Sys.Print("ROM basis: {} state + {} {} -> {} columns".format(
                self.pod.basis.getSize()[1], width,
                "paired directions W^-1 A^T Psi" if paired else "adjoint vectors",
                self._concatenated_basis.getSize()[1]))
            if adjoint_basis is not None:
                self.orthonormality = self._orthonormality(comm, self._concatenated_basis.getDenseArray(),
                                                           self.pod.basis.getSize()[1],
                                                           adjoint_basis.getDenseArray())
        V = self.basis

        model.set_up_solver()
        with _PROF.phase("rom:weights"):
            S = self.residual_weight(model)
        hybrid = self.test_space == 'hybrid' and concatenate and V.getSize()[1] > self.pod.basis.getSize()[1]
        self._hybrid = None
        if hybrid:
            hybrid_kw = dict(n_state=self.pod.basis.getSize()[1], Psi=adjoint_basis.getDenseArray(),
                             orthonormal=self.hybrid_state_test == 'orthonormal')
            self._hybrid = hybrid_kw
            trial_update = None
            if paired and self.paired_update == 'newton':
                self._paired_Z = Z
                def trial_update(A):
                    return self._repaired_basis(comm, A, adjoint_basis)
            solver = self.make_solver(HybridLSPGSolver, V, S, self.pod.mean, comm, trial_update=trial_update,
                                      **hybrid_kw)
        else:
            solver = self.make_solver(LSPGSolver, V, S, self.pod.mean, comm)
        self.paired_updates = []
        try:
            result = self.run_passes(model, solver, t0)
            if hybrid and paired and self.paired_update == 'solution':
                result, solver = self._repair_at_solution(model, comm, S, adjoint_basis, Z, hybrid_kw, result,
                                                          solver, t0)
            elif hybrid and paired and self.paired_update == 'newton':
                PETSc.Sys.Print("ROM paired update: Z re-paired {} times; last change {:.3e}".format(
                    len(self.paired_updates), self.paired_updates[-1] if self.paired_updates else np.nan))
            V = self.basis
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
        if self.adjoint_solve not in ('least_squares', 'dual_corrected'):
            adjoint_basis = None
        dual_weight = None if adjoint_basis is None or self.W is None else \
            1.0 / np.sqrt(self.W.w.getArray(readonly=True))
        if hybrid:
            self.adjoint = HybridLSPGAdjoint(V, S, comm, parameter_vector, result.solution,
                                             accepted=result.accepted, **hybrid_kw)
            return result
        self.adjoint = LSPGAdjoint(V, S, comm, parameter_vector, result.solution,
                                   accepted=result.accepted, adjoint_basis=adjoint_basis, dual_weight=dual_weight,
                                   galerkin=self.adjoint_solve == 'galerkin',
                                   dual_corrected=self.adjoint_solve == 'dual_corrected' and adjoint_basis is not None)
        return result

    def _orthonormality(self, comm, V, n_state, Psi):
        """
        The departures from orthonormality of the concatenated basis V = [V_s, V_a] in W and of
        the adjoint basis Psi in W^-1 (max abs entries of the Gram matrix minus I): V overall, its
        state block, its adjoint block, the cross block V_s^T W V_a, and Psi. Printed and
        returned as a dict. Collective.
        """
        w = np.ones(V.shape[0]) if self.W is None else self.W.w.getArray(readonly=True)
        G = comm.allreduce(V.T @ (w[:, None] * V), op=MPI.SUM)
        E = np.abs(G - np.eye(G.shape[0]))
        Gp = comm.allreduce(Psi.T @ (Psi / w[:, None]), op=MPI.SUM)
        result = dict(V=float(E.max()), state=float(E[:n_state, :n_state].max()),
                      adjoint=float(E[n_state:, n_state:].max()) if G.shape[0] > n_state else 0.0,
                      cross=float(E[:n_state, n_state:].max()) if G.shape[0] > n_state else 0.0,
                      Psi=float(np.max(np.abs(Gp - np.eye(Gp.shape[0])))))
        PETSc.Sys.Print("ROM orthonormality: max |V^T W V - I| {V:.1e} (state block {state:.1e}, adjoint block "
                        "{adjoint:.1e}, cross |V_s^T W V_a| {cross:.1e}); max |Psi^T W^-1 Psi - I| {Psi:.1e}".format(
                            **result))
        return result

    def _combined_adjoint_basis_of(self, comm):
        """adjoint_bases='per_output': the adjoint bases of all output rows (just built), side by side
        and orthonormalized in the W^-1 inner product, dropping directions they share
        (CONCATENATION_RTOL), as a dense Mat kept until the next solve. Collective."""
        if self._combined_adjoint_basis is not None:
            self._combined_adjoint_basis.destroy()
            self._combined_adjoint_basis = None
        blocks = [pod.basis.getDenseArray() for pod in self._adjoint_pod_list()
                  if pod.basis is not None and pod.basis.getSize()[1] > 0]
        if not blocks:
            return None
        P = np.column_stack(blocks)
        w = np.ones(P.shape[0]) if self.W_dual is None else self.W_dual.w.getArray(readonly=True)
        # how far the rows' bases overlap: the cosines of the principal angles between each pair, and
        # how far each is from W^-1-orthonormal
        G = comm.allreduce(P.T @ (w[:, None] * P), op=MPI.SUM)
        edges = np.cumsum([0] + [b.shape[1] for b in blocks])
        own = max(float(np.max(np.abs(G[i:j, i:j] - np.eye(j - i)))) for i, j in zip(edges[:-1], edges[1:]))
        cosines = {(a, b): np.linalg.svd(G[edges[a]:edges[a + 1], edges[b]:edges[b + 1]], compute_uv=False)
                   for a in range(len(blocks)) for b in range(a + 1, len(blocks))}
        PETSc.Sys.Print("ROM orthonormality: per-row adjoint bases max |Psi_r^T W^-1 Psi_r - I| {:.1e}; "
                        "cosines of the principal angles between rows {}".format(own, "; ".join(
                            "{}-{}: {}".format(a, b, " ".join("{:.3f}".format(c) for c in cs))
                            for (a, b), cs in cosines.items())))
        sqrt_w = np.sqrt(w)
        Q = orthonormal_columns(comm, sqrt_w[:, None] * P, rtol=self.CONCATENATION_RTOL) / sqrt_w[:, None]
        PETSc.Sys.Print("ROM adjoint bases per output row: {} -> {} directions together".format(
            [b.shape[1] for b in blocks], Q.shape[1]))
        self._combined_adjoint_basis = petsc_dense_from_local_block(comm, Q)
        return self._combined_adjoint_basis

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

    def _paired_trial_directions(self, model, Psi):
        """
        hybrid_trial='paired': this rank's rows of Z = W^-1 A^T Psi, A the steady Jacobian at the
        model's current state (the solve's warm start, before it moves), Psi the adjoint basis: the
        trial directions that pair with the adjoint test directions Psi, Psi^T A Z being the Gram
        matrix of A^T Psi in W^-1 (module docstring). Collective.
        """
        if hasattr(model, "update_derived_fields"):
            model.update_derived_fields()
        model.set_up_solver()
        with model.steady_residual():
            model.problem.J(model.u_vec.x.petsc_vec, model.problem._A)
        return self._paired_directions(model.problem._A, Psi.getDenseArray())

    def _paired_directions(self, A, P):
        """This rank's rows of W^-1 A^T P for an assembled Jacobian A. Collective."""
        x, y = A.createVecLeft(), A.createVecRight()
        Z = np.empty_like(P)
        for j in range(P.shape[1]):
            x.array[:] = P[:, j]
            A.multTranspose(x, y)
            Z[:, j] = y.array
        x.destroy()
        y.destroy()
        if self.W is not None:
            Z /= self.W.w.getArray(readonly=True)[:, None]
        return Z

    def _repaired_basis(self, comm, A, Psi):
        """
        paired_update='newton' (HybridLSPGSolver's trial_update): the concatenated basis with Z
        rebuilt from the Jacobian A, which replaces the solve's (and is returned); records the change
        of Z in paired_updates. Collective.
        """
        Z = self._paired_directions(A, Psi.getDenseArray())
        self.paired_updates.append(self._paired_change(comm, self._paired_Z, Z))
        self._paired_Z = Z
        self._concatenated_basis.destroy()
        self._concatenated_basis = self._concatenate(comm, self.pod.basis, Z)
        return self._concatenated_basis

    def _paired_change(self, comm, Z_old, Z):
        """max over the columns of ||Z - Z_old||_W / ||Z_old||_W. Collective."""
        w = np.ones(Z.shape[0]) if self.W is None else self.W.w.getArray(readonly=True)
        norms = comm.allreduce(np.stack([np.sum(w[:, None] * (Z - Z_old)**2, axis=0),
                                         np.sum(w[:, None] * Z_old**2, axis=0)]), op=MPI.SUM)
        return float(np.max(np.sqrt(norms[0] / np.maximum(norms[1], 1e-300))))

    def _repair_at_solution(self, model, comm, S, Psi, Z, hybrid_kw, result, solver, t0):
        """
        paired_update='solution': rebuild Z = W^-1 A^T Psi at the ROM's solution (the model's u_vec)
        and solve again from there, until Z changes by less than paired_update_rtol or after
        paired_update_max_passes re-pairings (module docstring). Replaces the concatenated basis;
        returns the last solve's result (iterations summed) and solver. Collective.
        """
        iterations = result.iterations
        for _ in range(self.paired_update_max_passes):
            if not np.isfinite(result.eta):
                break
            with _PROF.phase("rom:basis"):
                Z_new = self._paired_trial_directions(model, Psi)
            change = self._paired_change(comm, Z, Z_new)
            if change < self.paired_update_rtol:
                PETSc.Sys.Print("ROM paired update: Z changed by {:.3e} at the solution; kept".format(change))
                self.paired_updates.append((change, 0, result.eta))
                break
            Z = Z_new
            with _PROF.phase("rom:basis"):
                self._concatenated_basis.destroy()
                self._concatenated_basis = self._concatenate(comm, self.pod.basis, Z)
            solver = self.make_solver(HybridLSPGSolver, self.basis, S, self.pod.mean, comm, **hybrid_kw)
            result = self.run_passes(model, solver, t0)
            iterations += result.iterations
            self.paired_updates.append((change, result.iterations, result.eta))
            PETSc.Sys.Print("ROM paired update: Z changed by {:.3e}; re-solved in {} iterations, eta {}".format(
                change, result.iterations, result.eta))
        result.iterations = iterations
        return result, solver

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
        P = (V_adjoint if isinstance(V_adjoint, np.ndarray) else V_adjoint.getDenseArray()).copy()
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
        for obj in [self.adjoint_pod, self.W_dual, self._concatenated_basis, self._combined_adjoint_basis] + \
                list(self.adjoint_pods.values()):
            if obj is not None:
                obj.destroy()
        self.adjoint_pod = self.W_dual = self._concatenated_basis = self._combined_adjoint_basis = None
        self.adjoint_pods = {}
