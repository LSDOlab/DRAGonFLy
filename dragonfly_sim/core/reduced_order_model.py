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
||g|| = ||(S A V)^T S r|| < max(atol, rtol ||g_0||), on a negligible step, or on
a stall of ||g||; up to max_outer_iterations passes, converged when a pass
ends with ||g|| < g_conv_limit.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dragonfly_sim.core.pod import PODBasis
from dragonfly_sim.utils.pod_utils import DiagonalWeight
from dragonfly_sim.utils.derived_fields import diagonal_mass_inverse


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
        x = u_vec.getArray(readonly=True).copy()
        if self.mean is not None:
            x -= self.mean.getArray(readonly=True)
        if self.W is not None:
            x *= self.W.getArray(readonly=True)
        return self.comm.allreduce(self.V.getDenseArray().T @ x, op=MPI.SUM)

    def expand(self, a, u):
        """u <- V a (+ mean), ghosts updated."""
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
        r = self._r.getArray(readonly=True)
        if self.S is not None:
            r = self.S.getArray(readonly=True) * r
        return r, float(np.sqrt(self.comm.allreduce(float(r @ r), op=MPI.SUM)))

    def _reduced_system(self, model):
        """(H, g, ||S r||) with H = (S A V)^T (S A V), g = (S A V)^T S r at the
        current u_vec."""
        problem = model.problem
        r, r_norm = self._weighted_residual(model)
        problem.J(model.u_vec.x.petsc_vec, problem._A)
        A = problem._A
        if self._AV is not None and self._AV.getSize()[1] != self.V.getSize()[1]:
            self._AV.destroy()
            self._AV = None
        if self._AV is None:
            self._AV = A.matMult(self.V)
        else:
            A.matMult(self.V, result=self._AV)
        AV = self._AV.getDenseArray()
        if self.S is not None:
            AV = self.S.getArray(readonly=True)[:, None] * AV
        H = self.comm.allreduce(AV.T @ AV, op=MPI.SUM)
        g = self.comm.allreduce(AV.T @ r, op=MPI.SUM)
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

          converged   ||g|| < max(atol, rtol ||g_0||), g_0 from this pass's first
                      iteration; or, after a step, ||da|| < max(atol,
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
        g0 = None
        history = []
        n = 0
        PETSc.Sys.Print("LSPG ROM solve, basis width {}".format(self.rb_size))
        for n in range(self.max_it):
            self.expand(a, u)
            H, g, r_norm = self._reduced_system(model)
            g_norm = float(np.linalg.norm(g))
            self.last_g_norm = g_norm
            g0 = g_norm if g0 is None else g0
            if not np.isfinite(g_norm):
                PETSc.Sys.Print("ROM: non-finite reduced residual at iteration {}; stopping".format(n))
                break
            if g_norm < max(self.atol, self.rtol * g0):
                self.converged = True
                PETSc.Sys.Print("ROM converged at iteration {}: |g| {:.6e} (from {:.6e})".format(n, g_norm, g0))
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
                da = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                PETSc.Sys.Print("ROM: singular reduced Jacobian at iteration {}".format(n))
                break
            # the full-space update is +V da; positivity_step_length limits x - theta dx
            dx.array[:] = -(self.V.getDenseArray() @ da)
            theta, _ = model.positivity_step_length(u.x.petsc_vec, dx)
            if self.line_search:
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
      qr_rank_tol        relative weighted residual (outside the span of the stored
                         snapshots) below which a snapshot adds no direction
      weightfunction, n_nonzero_weights, cubic_cutoff
                         local_weighted weighting (PODBasis.construct_basis);
                         cubic_cutoff = c sets the cubic support radius to
                         c min_dist + (1 - c) max_dist, widened if needed so
                         that the rb_size nearest snapshots keep a weight
      parameter_scales   per-parameter factors on the snapshot distance

    Inner product (default: Euclidean with free-stream state scaling)::

      inner_product      'euclidean' or 'l2' (DG0 mass matrix)
      state_scaling      'freestream' (model.state_scales()), None, or one
                         scale per state component

    Solve and acceptance (the defaults are project-sandbox benchmark_pod.py's)::

      newton_max_it, newton_rtol, newton_atol, stall_window, stall_rtol
                         per-pass LSPGSolver stopping rules
      max_outer_iterations
                         LSPG passes, each restarted from the current state
      g_conv_limit       the ROM solve has converged when the last ||g|| of a
                         pass is below this
      line_search        Armijo damping of the Gauss-Newton step
      eta_threshold      the ROM solution is accepted when ||R||/||u|| is below it
      min_snapshots      snapshots needed before the ROM is tried
      run_fom_every_evaluation
                         also run the FOM when the ROM is accepted (benchmarking;
                         the accepted output stays the ROM's)
    """

    def __init__(self, rb_size=20, basis_mode='global', max_rank=None, qr_rank_tol=1e-10,
                 weightfunction='Cubic', n_nonzero_weights=None, cubic_cutoff=None,
                 parameter_scales=None, inner_product='euclidean', state_scaling='freestream',
                 newton_max_it=15, newton_rtol=1e-10, newton_atol=1e-12, stall_window=3,
                 stall_rtol=1e-10, max_outer_iterations=3, g_conv_limit=1e-12, line_search=True,
                 eta_threshold=1e-10, min_snapshots=2, run_fom_every_evaluation=False):
        if inner_product not in ('l2', 'euclidean'):
            raise ValueError("inner_product must be 'l2' or 'euclidean', got {!r}".format(inner_product))
        if not (state_scaling is None or isinstance(state_scaling, str) and state_scaling == 'freestream'
                or np.ndim(state_scaling) == 1):
            raise ValueError("state_scaling must be 'freestream', None or one scale per state component")
        self.rb_size = rb_size
        self.basis_mode = basis_mode
        self.max_rank = 3 * rb_size if max_rank is None else max_rank
        self.qr_rank_tol = qr_rank_tol
        self.weightfunction = weightfunction
        self.n_nonzero_weights = n_nonzero_weights
        self.cubic_cutoff = cubic_cutoff
        self.parameter_scales = parameter_scales
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

        self.pod = None
        self.W = None
        self.scales = None
        self.last_solver = None

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
                            IP_matrix=self.W, max_rank=self.max_rank, qr_rank_tol=self.qr_rank_tol)
        PETSc.Sys.Print("Initialized POD ROM: basis_mode={}, rb_size={}, max_rank={}, n_dofs={}, "
                        "inner product {}{}".format(
                            self.basis_mode, self.rb_size, self.max_rank, layout.getSize(),
                            self.inner_product,
                            "" if self.scales is None else ", state scales {}".format(self.scales)))

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
        return 0 if self.pod is None else self.pod.snapshot_matrix.n_snapshots

    def snapshot_singular_values(self):
        sigma = None if self.pod is None else self.pod.snapshot_matrix.Sigma
        return None if sigma is None else sigma.getArray().copy()

    # ------------------------------------------------------------------
    # Snapshots and solve
    # ------------------------------------------------------------------
    def add_snapshot(self, model, parameter_vector):
        """Add the model's current u_vec as the snapshot at parameter_vector.
        Returns whether it added a basis direction. Collective."""
        added = self.pod.snapshot_matrix.process_and_append_to_snapshot_lists(
            model.u_vec.x.petsc_vec, np.asarray(parameter_vector, dtype=float))
        self.pod.snapshot_matrix.print_info()
        return added

    def solve(self, model, parameter_vector):
        """
        The ROM solution at parameter_vector, starting from the model's
        current u_vec (projected onto the basis); u_vec holds it afterwards.
        Returns a ROMResult, or None when there are fewer than min_snapshots
        snapshots. Collective.
        """
        if self.n_snapshots < self.min_snapshots:
            PETSc.Sys.Print("ROM: {} snapshot(s) stored, {} needed; skipping the ROM solve".format(
                self.n_snapshots, self.min_snapshots))
            return None
        t0 = perf_counter()
        comm = model.mesh_obj.mesh.comm
        self.pod.construct_basis(rb_size=self.rb_size,
                                 local_parametervector=np.asarray(parameter_vector, dtype=float),
                                 parameter_weights=self.parameter_scales,
                                 weightfunction=self.weightfunction,
                                 n_nonzero_weights=self.n_nonzero_weights,
                                 cubic_cutoff=self.cubic_cutoff,
                                 layout_vec=model.u_vec.x.petsc_vec)
        if self.pod.basis is None or self.pod.basis.getSize()[1] == 0:
            PETSc.Sys.Print("ROM: empty POD basis; skipping the ROM solve")
            return None

        model.set_up_solver()
        S = self._diagonal(model, mass_power=-0.5, scale_power=-1.0)
        solver = LSPGSolver(self.pod.basis, comm, W=None if self.W is None else self.W.w, S=S,
                            mean=self.pod.mean, rtol=self.newton_rtol, atol=self.newton_atol,
                            max_it=self.newton_max_it, stall_window=self.stall_window,
                            stall_rtol=self.stall_rtol, line_search=self.line_search)
        # Up to max_outer_iterations passes, each restarted from the current
        # state; converged when the pass's last |g| is below g_conv_limit (the
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
                if solver.last_g_norm is not None and solver.last_g_norm < self.g_conv_limit:
                    converged = True
                    PETSc.Sys.Print("LSPG converged: |g| = {:.6e} < {:.1e}".format(
                        solver.last_g_norm, self.g_conv_limit))
                    break
            else:
                PETSc.Sys.Print("ROM: {} LSPG passes without |g| < {:.1e} (last |g| {:.6e})".format(
                    self.max_outer_iterations, self.g_conv_limit, solver.last_g_norm))
        finally:
            solver.destroy()
            if S is not None:
                S.destroy()
        self.last_solver = solver
        solver.n_iterations = iterations

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

    def destroy(self):
        if self.pod is not None:
            self.pod.destroy()
            self.pod = None
        if self.W is not None:
            self.W.destroy()
            self.W = None
