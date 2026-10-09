"""
Time integration: pseudo-transient continuation (PTC) for steady solves and
BDF1/BDF2 for unsteady ones, on one residual.

    R*(U) = R(U) + inv_dt (a0 U + a1 U^n + a2 U^{n-1})       physical BDF term
                 + (w_K / CFL) (U - U_old)                    pseudo-time term

R is the model's steady residual (model.F, unchanged, which is what the
adjoint differentiates). w_K = lambda_K / h_K (+ the model's
time_scale_terms, e.g. viscous 2 nu / h^2) is the cell's inverse local time
step at CFL 1, evaluated at U_old. Both terms are lumped DG0 mass terms and
both live only in TimeTerm.augment(), so they never enter model.F.

Modes (all one step loop, TimeIntegrator._newton_step):

  steady PTC   physical term off (inv_dt = 0); one Newton step per pseudo
               step, U_old <- U before each, the CFL law (CFLController)
               adapts the step from the steady |R|. At the start of a
               pseudo step U = U_old, so the residual assembled there is the
               steady one.
  unsteady     physical term on, pseudo term off (1/CFL = 0): Newton per
               physical step to the residual criterion; Delta t ramp and
               halving on failure (StepSizeController).
  dual time    physical term on, and the pseudo term on inside every
               physical step, with the CFL law restarted each step: PTC on
               R* instead of R.

Switching a term off sets its Constant to exactly zero, so the compiled
forms stay valid across modes; a model with neither term enabled never
builds them at all (the plain Newton solve is unchanged).
"""
from dataclasses import dataclass, field

import numpy as np
from petsc4py import PETSc
import ufl
import dolfinx


# (a0, a1, a2) in (a0 U^{n+1} + a1 U^n + a2 U^{n-1}) / dt at a constant dt.
BDF_COEFFS = {1: (1.0, -1.0, 0.0), 2: (1.5, -2.0, 0.5)}


def bdf_coefficients(order, ratio=None):
    """
    (a0, a1, a2) for (a0 U^{n+1} + a1 U^n + a2 U^{n-1}) / dt_n. Variable-step
    BDF2 with ratio w = dt_n / dt_{n-1}:

        a0 = (1 + 2w)/(1 + w),  a1 = -(1 + w),  a2 = w^2/(1 + w),

    which is BDF_COEFFS[2] at w = 1. BDF1 ignores the ratio.
    """
    if order == 1:
        return BDF_COEFFS[1]
    w = 1.0 if ratio is None else float(ratio)
    return ((1.0 + 2.0 * w) / (1.0 + w), -(1.0 + w), w * w / (1.0 + w))


# ----------------------------------------------------------------------
# CFL law
# ----------------------------------------------------------------------
@dataclass
class CFLController:
    """
    Switched-evolution-relaxation CFL law with the safeguards the p = 0 RANS
    runs needed. Pure Python: update() takes the step's outcome and returns
    the decision, so the law is unit-tested without a solver.

    After a step from residual r_prev to r, with a converged linear solve
    (ksp_ok) and positivity step length theta:

      reject       r non-finite or r > reject_factor r_prev: undo the step,
                   CFL *= cut (stop when CFL < 1e-6 cfl0)
      limited      theta < theta_cut: CFL *= cut, down to cfl_min_limited.
                   Without it a limited step leaves |R| flat, the band below
                   holds CFL, and the limiter goes on to zero every step (p=0
                   MUSCL from freestream froze at M 0.9 and 0.95; one
                   leading-edge cell at p = 1e-4 blocked the step at CFL 1 but
                   not at CFL 0.1, hence a floor below cfl_min).
      no Krylov    an inexact direction never grows CFL
      progress     r < (1 - 1e-3) r_prev: grow by max(r_prev/r, min_growth),
                   capped at max_growth
      rise         r > 1.05 r_prev: shrink by at least 1/max_growth
      band         otherwise (|R| flat within -0.1%..+5%): grow by
                   band_growth on a clean step (theta = 1), else hold. Holding
                   kept the STW L3 wing at CFL 3.15 for 118 of 152 steps.

    CFL stays within [floor, cfl_max]; the floor is cfl_min_limited after a
    limited step and min(cfl_min, CFL) otherwise, so a CFL below cfl_min
    regrows by the usual rule instead of jumping back to cfl_min.

    exhausted: CFL fell below 1e-6 cfl0 after rejections, or the last
    max_frozen_steps steps all had theta = 0. A step with theta = 0 leaves the
    state unchanged, so once a cell sits at the positivity floor every later
    step repeats it; without this stop PTC spent its whole step budget frozen
    (OAT15A, Cadence grid level 1: ~950 identical steps at CFL cfl_min_limited).
    """
    cfl0: float = 5.0
    cfl_max: float = 1e12
    max_growth: float = 2.0
    min_growth: float = 1.5
    cut: float = 0.1
    reject_factor: float = 10.0
    cfl_min: float = 1.0
    theta_cut: float = 0.1
    cfl_min_limited: float = 1e-3
    band_growth: float = 1.2
    max_frozen_steps: int = 5
    cfl: float = field(init=False)
    frozen_steps: int = field(init=False, default=0)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.cfl = float(self.cfl0)
        self.frozen_steps = 0

    def update(self, r_prev, r, ksp_ok=True, theta=1.0):
        """Returns (accept, limited). Updates self.cfl."""
        self.frozen_steps = self.frozen_steps + 1 if theta <= 0.0 else 0
        limited = theta < self.theta_cut
        if not np.isfinite(r) or r > self.reject_factor * r_prev:
            self.cfl *= self.cut
            return False, limited
        growth = min(max(r_prev / r, self.cut), self.max_growth) if r > 0 else self.max_growth
        if limited:
            growth = self.cut
        elif not ksp_ok:
            growth = min(growth, 1.0)
        elif r < (1.0 - 1e-3) * r_prev:
            growth = max(growth, self.min_growth)
        elif r > 1.05 * r_prev:
            growth = min(growth, 1.0 / self.max_growth)
        elif theta >= 1.0:
            growth = self.band_growth
        else:
            growth = 1.0
        floor = self.cfl_min_limited if limited else min(self.cfl_min, self.cfl)
        self.cfl = max(min(self.cfl * growth, self.cfl_max), floor)
        return True, limited

    @property
    def exhausted(self):
        return self.cfl < 1e-6 * self.cfl0 or self.frozen_steps >= self.max_frozen_steps


@dataclass
class StepSizeController:
    """Physical time step: ramp from dt_start by `growth` per step up to dt,
    and halve on a failed step, up to max_retries times."""
    dt: float
    dt_start: float = None
    growth: float = 1.5
    max_retries: int = 8

    def first(self, dt_prev):
        if dt_prev is None:
            return self.dt if self.dt_start is None else self.dt_start
        return min(dt_prev * self.growth, self.dt)


# ----------------------------------------------------------------------
# Forms
# ----------------------------------------------------------------------
class TimeTerm:
    """
    The physical and pseudo-time terms of R* as UFL forms over Constants
    and Functions owned here (see the module docstring).
    """

    def __init__(self, model, physical=False, pseudo=False):
        self.model = model
        msh = model.mesh_obj.mesh
        V = model.functionspaces["V"]
        self.physical, self.pseudo = bool(physical), bool(pseudo)
        self.inv_dt = dolfinx.fem.Constant(msh, 0.0)
        self.a = tuple(dolfinx.fem.Constant(msh, c) for c in BDF_COEFFS[1])
        self.inv_cfl = dolfinx.fem.Constant(msh, 0.0)
        self.U_n = dolfinx.fem.Function(V, name="u_n")
        self.U_nm1 = dolfinx.fem.Function(V, name="u_nm1")
        self.U_old = dolfinx.fem.Function(V, name="u_old")
        for f in (self.U_n, self.U_nm1, self.U_old):
            f.x.array[:] = model.u_vec.x.array

    def local_inverse_time_scale(self, U):
        """w_K = lambda_max / h_min plus the model's time_scale_terms."""
        m = self.model
        w = m._lambda_max_expr(U) / m.mesh_obj.h_min
        for term in m.time_scale_terms:
            w = w + term(U)
        return w

    def augment(self, F):
        m = self.model
        u, v, dx = m.u_vec, m.v_vec, m.forms.dx
        if self.pseudo:
            w = self.local_inverse_time_scale(self.U_old)
            F = F + w * self.inv_cfl * ufl.inner(u - self.U_old, v) * dx
        if self.physical:
            a0, a1, a2 = self.a
            F = F + self.inv_dt * ufl.inner(a0 * u + a1 * self.U_n + a2 * self.U_nm1, v) * dx
        return F

    def set_bdf(self, order, dt, dt_prev=None):
        ratio = None if (order == 1 or dt_prev is None) else dt / dt_prev
        for c, val in zip(self.a, bdf_coefficients(order, ratio)):
            c.value = val
        self.inv_dt.value = 1.0 / dt
        return ratio

    def physical_off(self):
        self.inv_dt.value = 0.0

    def set_cfl(self, cfl):
        self.inv_cfl.value = 0.0 if cfl is None or np.isinf(cfl) else 1.0 / cfl


# ----------------------------------------------------------------------
# Integrator
# ----------------------------------------------------------------------
@dataclass
class StepRecord:
    accepted: bool
    newton_its: int = 0
    ksp_its: int = 0
    res_initial: float = np.nan
    res_final: float = np.nan
    dt: float = np.nan
    order: int = 0
    retries: int = 0


class TimeIntegrator:
    """
    Drives a CompressibleEulerModel (or a model hosted on one) through
    steady PTC, unsteady BDF or dual-time steps. Built by the model
    (set_up_solver / enable_time_stepping); `term` is its TimeTerm.

    Time-dependent boundary conditions (core/farfield.py:TransverseFarfield)
    register through `boundary_updaters`: objects with prepare(order) before a
    physical step's Newton solve and commit() after it is accepted.
    """

    def __init__(self, model, term):
        self.model = model
        self.term = term
        self.boundary_updaters = []
        self.time = 0.0
        self.step_index = 0
        self.dt_prev = None
        self.verbose = True

    # -- helpers ---------------------------------------------------------
    def _print(self, *args):
        if self.verbose:
            PETSc.Sys.Print(*args)

    def _copy(self, dst, src):
        dst.x.array[:] = src.x.array
        dst.x.scatter_forward()

    # -- steady PTC ------------------------------------------------------
    def solve_steady(self, controller=None, max_steps=None, tol=None):
        """
        Pseudo-transient continuation to the steady state from the current
        u_vec. Returns the model's physical residual Vec.
        """
        m = self.model
        if self.term.physical:
            self.term.physical_off()
        if not self.term.pseudo:
            raise RuntimeError("steady PTC needs the pseudo-time term (model.use_ptc = True "
                               "before set_up_solver)")
        if self.boundary_updaters:
            raise RuntimeError("time-dependent boundary conditions (riemann2) have no steady state")
        ctrl = controller if controller is not None else CFLController(**m.ptc_settings)
        tol = m.residual_conv_limit if tol is None else tol
        max_steps = int(m.max_newton_iterations if max_steps is None else max_steps)
        m._ensure_residual_vector()
        # one Newton step per pseudo step
        max_it_saved, m.solver.max_it = m.solver.max_it, 1
        r_prev = m._assemble_physical_residual()
        self._print("[ptc] initial |R| = {:.6e}".format(r_prev))
        m.solver_converged = r_prev < tol
        it = 0
        while not m.solver_converged and it < max_steps:
            it += 1
            cfl = ctrl.cfl
            self.term.set_cfl(cfl)
            self._copy(self.term.U_old, m.u_vec)
            m.last_step_theta = 1.0
            m.solver.solve(m.u_vec)
            m.u_vec.x.scatter_forward()
            r = m._assemble_physical_residual()
            ksp_ok = m.solver.krylov_solver.getConvergedReason() > 0
            theta = m.last_step_theta
            accept, limited = ctrl.update(r_prev, r, ksp_ok, theta)
            self._print("[ptc] step {:3d}  CFL {:.3e}  |R| = {:.6e}{}{}".format(
                it, cfl, r, "" if ksp_ok else "  (linear solve not converged)",
                "  (limited, theta {:.1e})".format(theta) if limited else ""))
            if not accept:
                self._copy(m.u_vec, self.term.U_old)
                self._print("[ptc]   rejected; CFL -> {:.3e}".format(ctrl.cfl))
                if ctrl.exhausted:
                    break
                continue
            m.solver_converged = r < tol
            r_prev = r
            if ctrl.exhausted:
                self._print("[ptc]   no update for {} steps (positivity limiter theta = 0); "
                            "stopping".format(ctrl.frozen_steps))
                break
        m.solver.max_it = max_it_saved
        if not m.solver_converged:
            self._print("WARNING: PTC stopped after {} steps at |R| = {:.6e}".format(it, r_prev))
        self.last_steady_steps = it
        m._assemble_physical_residual()
        m.u_vec.x.scatter_forward()
        return m.residual_vec_physical

    # -- unsteady --------------------------------------------------------
    def start(self, time=0.0, step_index=0, dt_prev=None):
        """Begin (or restart) time stepping from the current u_vec, which is
        taken as U^n (and U^{n-1} unless restored from a checkpoint)."""
        self.time, self.step_index, self.dt_prev = time, step_index, dt_prev
        self._copy(self.term.U_n, self.model.u_vec)
        if dt_prev is None:
            self._copy(self.term.U_nm1, self.model.u_vec)

    def advance(self, step_control, order=2, newton_rtol=1e-8, newton_atol=1e-10,
                newton_max_it=20, dual_time=None):
        """
        One physical time step from U^n. The step size comes from
        step_control (StepSizeController); a failed Newton solve restores
        U^n and halves dt, up to step_control.max_retries times.
        `dual_time` (a CFLController) switches the inner solve to PTC on R*.
        Returns a StepRecord; record.accepted is False if every retry failed.
        """
        m, term = self.model, self.term
        if not term.physical:
            raise RuntimeError("unsteady steps need the physical time term (enable_time_stepping)")
        solver = m.solver
        dt = step_control.first(self.dt_prev)
        for attempt in range(step_control.max_retries + 1):
            k = 1 if (self.dt_prev is None or order == 1) else 2
            ratio = term.set_bdf(k, dt, self.dt_prev)
            for b in self.boundary_updaters:
                b.prepare(k, ratio, dt)
            if dual_time is None:
                term.set_cfl(None)
                ok, rec = self._newton_solve(newton_rtol, newton_atol, newton_max_it)
            else:
                ok, rec = self._dual_time_solve(dual_time, newton_rtol, newton_atol, newton_max_it)
            if ok:
                rec.dt, rec.order, rec.retries = dt, k, attempt
                break
            self._print("step {:4d}  dt={:.3e}: Newton failed (|R| {:.3e} -> {:.3e}); retrying at "
                        "dt={:.3e}".format(self.step_index + 1, dt, rec.res_initial, rec.res_final, 0.5 * dt))
            self._copy(m.u_vec, term.U_n)
            dt *= 0.5
        else:
            return StepRecord(accepted=False, dt=dt, retries=step_control.max_retries)
        for b in self.boundary_updaters:
            b.commit()
        self.time += dt
        self.step_index += 1
        self.dt_prev = dt
        self._copy(term.U_nm1, term.U_n)
        self._copy(term.U_n, m.u_vec)
        return rec

    def _newton_solve(self, rtol, atol, max_it):
        m = self.model
        solver = m.solver
        crit = (solver.convergence_criterion, solver.rtol, solver.atol, solver.max_it)
        solver.convergence_criterion, solver.rtol, solver.atol, solver.max_it = "residual", rtol, atol, max_it
        snes = solver.snes
        snes.setConvergenceHistory(reset=True)
        try:
            n, converged = solver.solve(m.u_vec)
        finally:
            solver.convergence_criterion, solver.rtol, solver.atol, solver.max_it = crit
        hist, _ = snes.getConvergenceHistory()
        r0, r1 = (hist[0], hist[-1]) if len(hist) else (np.nan, np.nan)
        rec = StepRecord(accepted=True, newton_its=n, ksp_its=snes.getLinearSolveIterations(),
                         res_initial=r0, res_final=r1)
        return bool(converged and np.isfinite(r1)), rec

    def _dual_time_solve(self, ctrl, rtol, atol, max_it):
        """PTC on R* inside one physical step; converged when |R*| < max(atol,
        rtol |R*_0|). Each pseudo step is one Newton step."""
        m, term = self.model, self.term
        ctrl.reset()
        b = m.solver.b
        r_prev = self._unsteady_residual_norm(b)
        r0, its, ksp = r_prev, 0, 0
        target = max(atol, rtol * r0)
        while r_prev > target and its < max_it:
            its += 1
            term.set_cfl(ctrl.cfl)
            self._copy(term.U_old, m.u_vec)
            m.last_step_theta = 1.0
            m.solver.solve(m.u_vec)
            ksp += m.solver.snes.getLinearSolveIterations()
            r = self._unsteady_residual_norm(b)
            ok_ksp = m.solver.krylov_solver.getConvergedReason() > 0
            accept, _ = ctrl.update(r_prev, r, ok_ksp, m.last_step_theta)
            if not accept:
                self._copy(m.u_vec, term.U_old)
                if ctrl.exhausted:
                    break
                continue
            r_prev = r
        term.set_cfl(None)
        rec = StepRecord(accepted=True, newton_its=its, ksp_its=ksp, res_initial=r0, res_final=r_prev)
        return bool(r_prev <= target and np.isfinite(r_prev)), rec

    def _unsteady_residual_norm(self, b):
        """|R*| with the pseudo term off (at U = U_old it vanishes anyway)."""
        m = self.model
        cfl_saved = self.term.inv_cfl.value
        self.term.inv_cfl.value = 0.0
        m.problem.F(m.u_vec.x.petsc_vec, b)
        self.term.inv_cfl.value = cfl_saved
        return b.norm()
