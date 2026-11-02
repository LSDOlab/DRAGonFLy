"""Trust-region reduced-basis optimization (TR-RB) of Keil, Mechelli, Ohlberger, Schindler and Volkwein.

T. Keil, L. Mechelli, M. Ohlberger, F. Schindler, S. Volkwein, "A non-conforming dual approach for adaptive
trust-region reduced basis approximation of PDE-constrained parameter optimization", ESAIM: M2AN 55 (2021),
Algorithm 1 (arXiv:2006.09297), after Qian, Grepl, Veroy and Willcox (SISC 2017). The model function of
iteration k is the ROM itself, m_k(mu) = J_r^(k)(mu), whose basis is enriched with the full-order state and
adjoints at every iterate; the trust region is not a ball in the design space but the set where the ROM's
estimated relative error is small,

    q_k(mu) = Delta_J(mu) / |J_r^(k)(mu)| <= delta_k,

which the subproblem (a projected BFGS method on the ROM within the design bounds) enforces by backtracking:
each step mu(j) = P(mu + kappa^j d) is the first that satisfies the Armijo condition

    J_r(mu(j)) - J_r(mu) <= -(kappa_arm / kappa^j) |mu(j) - mu|^2

and q(mu(j)) <= delta. The subproblem stops when its projected gradient is below tau_sub or when it reaches
beta2 delta <= q <= delta (30a-b); its first step, along the negative gradient, is the approximate generalized
Cauchy point mu_AGC. The full-order model is evaluated at the result mu_{k+1} (the enrichment), and the step
is accepted on the sufficient decrease condition J(mu_{k+1}) <= J_r^(k)(mu_AGC) (lines 4-24 of Algorithm 1);
delta grows by 1/beta1 when rho = (J(mu_k) - J(mu_{k+1})) / (J_r(mu_k) - J_r(mu_{k+1})) >= eta_rho and
shrinks by beta1 on a rejection. The iteration stops when the full-order first-order criticality
|mu - P(mu - grad J(mu))| is below tau_FOC (the full-order gradient comes with the enrichment).

Adaptations, all printed in the run log:

- Output constraints. The paper treats bound constraints only (the projection P). The objective and
  inequality constraints c_lower <= c <= c_upper of a modopt problem enter as the Powell-Hestenes-Rockafellar
  augmented Lagrangian (AugmentedLagrangian), whose minimization over the bounds is the TR-RB problem; the
  multipliers and the penalty are updated as in Nocedal and Wright, Numerical Optimization, Algorithm 17.4,
  whenever the full-order criticality at a center is below the current tolerance omega. The augmented
  Lagrangian of the ROM's outputs is the model function; it is exact where the ROM is.
- The error estimate. For RANS there is no rigorous bound. indicator='ncd' uses the shape of the paper's
  bound for the NCD-corrected output (Prop. 2.5), ||r_pr|| ||r_du|| per output, from the dual-corrected ROM
  (DG_windtunnel_model.rom_output_estimates), combined over the outputs with the augmented Lagrangian's
  weights |dl/dy|; indicator='residual' uses the ROM's relative residual eta times |l|. Either is calibrated
  online: at every full-order trial point the observed error |l(mu) - l_r(mu)| of the model is divided by the
  raw estimate there, and the estimate is multiplied by the largest of the last `calibration_window` ratios
  (1 before the first). With an estimate that is not a bound, a "certified" step (line 4 of Algorithm 1) is
  still rejected if the full-order merit increased; the full-order value is computed there anyway.
- ROM warm start. Every ROM evaluation in a subproblem starts from the center's full-order solution, so that
  the model is a function of the design alone (not of the order of the evaluations).
- A ROM with freeze_basis=True builds its basis at the center, by the first evaluation of each subproblem,
  and keeps it through the subproblem.
- A ROM with test_space='frozen' (core/reduced_order_model.py) is frozen at the center by the first
  evaluation of each subproblem: the model is then exact in value and gradient at the center (the frozen
  dual), and its gradient is the exact derivative of its values throughout the subproblem.

Two departures from Algorithm 1, as options:

- acceptance='ratio' accepts a step on rho >= eta_accept (the classical trust-region test) instead of the
  sufficient decrease condition J(mu_{k+1}) <= J_r(mu_AGC), which rejects real decreases that fall short of
  the AGC point's model value; line 11 and the certified acceptance (line 4) are then not used.
- constraints='sqp' replaces the augmented Lagrangian: the subproblem is SLSQP on the ROM's objective and
  constraints, first-order corrected at the center (exact there with the enrichment; the correction keeps
  it so when the basis truncates), within the design bounds and the box |mu - mu_k|_inf <= r_k. Its
  result is then cut back along the step, mu_k + kappa^j s, to the first point where q <= delta: the step,
  if the model's L1 merit phi = f + merit_weight * (constraint violations) there is below the center's.
  SLSQP's result can be above it (at its iteration limit, trading feasibility for the objective
  differently): SLSQP then runs again in a box of beta1 times its step, and after the last such attempt the
  step is the lowest model merit among the subproblem's evaluations within the trust region (not before:
  the lowest of many model values favours the model's largest errors). The full-order model is evaluated (and the
  ROM enriched) at the step, and rho of the L1 merit decides (acceptance='ratio'). delta changes as in
  Algorithm 1 (not without a step: the model was not tested); the box radius r_k doubles with delta when
  the step reached it and becomes beta1 times the step on a rejection. The iteration stops when the model
  predicts a merit decrease below ftol * max(1, |phi|) at a feasible center with SLSQP's converged step
  within the box and the trust region, or when SLSQP converges within the box without a decrease.
"""
import contextlib

import numpy as np
from scipy.optimize import minimize

from dragonfly_sim.utils.trust_region_optimization import TrustRegionModelManagement


class _ModelFailed(Exception):
    """A ROM evaluation in an SLSQP subproblem failed (constraints='sqp')."""


class AugmentedLagrangian:
    """
    The Powell-Hestenes-Rockafellar augmented Lagrangian of a scaled objective f and inequality constraints
    c_lower <= c <= c_upper, each finite bound as g_i(c) >= 0 (g = c - c_lower, c_upper - c):

        l(f, c) = f + sum_i phi(g_i),    phi(g) = -theta g + tau g^2 / 2   if g <= theta / tau,
                                                  -theta^2 / (2 tau)      otherwise,

    continuously differentiable with dl/dg_i = -max(theta_i - tau g_i, 0); the multiplier update is
    theta_i <- max(theta_i - tau g_i, 0) (Nocedal and Wright, Numerical Optimization, 17.4). Without
    constraints l = f.
    """

    def __init__(self, c_lower, c_upper, tau):
        self.c_lower = np.ravel(np.asarray(c_lower, dtype=float))
        self.c_upper = np.ravel(np.asarray(c_upper, dtype=float))
        self._lo = np.flatnonzero(np.isfinite(self.c_lower))
        self._up = np.flatnonzero(np.isfinite(self.c_upper))
        self.tau = float(tau)
        self.theta = np.zeros(self._lo.size + self._up.size)

    @property
    def constrained(self):
        return self.theta.size > 0

    def g(self, c):
        c = np.ravel(c)
        return np.concatenate([c[self._lo] - self.c_lower[self._lo], self.c_upper[self._up] - c[self._up]])

    def _dg_dc(self, n_c):
        E = np.eye(n_c)
        return np.vstack([E[self._lo], -E[self._up]])

    def value(self, f, c):
        g = self.g(c)
        phi = np.where(g <= self.theta / self.tau, -self.theta * g + 0.5 * self.tau * g**2,
                       -self.theta**2 / (2.0 * self.tau))
        return float(f) + float(np.sum(phi))

    def output_weights(self, c):
        """dl/dc, one entry per constraint."""
        c = np.ravel(c)
        return -np.maximum(self.theta - self.tau * self.g(c), 0.0) @ self._dg_dc(c.size)

    def gradient(self, g_f, c, J):
        c, g_f = np.ravel(c), np.ravel(g_f)
        if c.size == 0:
            return g_f.copy()
        return g_f + self.output_weights(c) @ np.asarray(J, dtype=float).reshape(c.size, -1)

    def violation(self, c):
        """The largest constraint violation (scaled)."""
        return float(np.max(np.maximum(-self.g(c), 0.0), initial=0.0))

    def update_multipliers(self, c):
        self.theta = np.maximum(self.theta - self.tau * self.g(c), 0.0)


class TrustRegionReducedBasis(TrustRegionModelManagement):
    """
    TR-RB (module docstring) for a modopt CSDLAlphaProblem whose flow model is a DG_windtunnel_model with a
    ReducedOrderModel; the ROM should hold every snapshot (basis_mode='global' and a basis wider than the
    snapshots: the Lagrangian reduced basis of the paper) and accept a single one (min_snapshots=1).
    Collective: every rank runs the iteration on the same data.

    Parameters
    ----------
    indicator : 'ncd' or 'residual'
        The raw error estimate (module docstring).
    output_names, output_scales : sequences
        indicator='ncd': the windtunnel's names of the objective and the constraints, in the problem's order
        (e.g. ('D', 'L', 'M_pitch')), and the factors that scale them as the problem does.
    delta0, beta1, beta2, eta_rho, kappa, kappa_arm, tau_sub, tau_foc
        Algorithm 1's parameters (the paper's experiments: 0.1, 0.5, 0.95, 0.75, 0.5, 1e-4, 1e-8).
    feasibility_tol : float
        The largest scaled constraint violation at convergence.
    penalty0 : float
        The augmented Lagrangian's initial penalty tau (Nocedal and Wright's mu_0).
    max_subproblem_iterations, max_armijo : int
        BFGS iterations per subproblem and backtracking steps per BFGS iteration.
    max_iterations, max_full_order : int
        Trust-region iterations (subproblems) and full-order solves of new designs, at most.
    delta_min, delta_max : float
        The iteration stops when the trust region falls below delta_min; it does not grow beyond delta_max
        (the paper has no bound; an estimated relative error of 1 bounds nothing useful).
    constraints : 'augmented_lagrangian' or 'sqp'
        How the output constraints enter (module docstring).
    acceptance : 'sufficient_decrease', 'ratio' or None
        The acceptance test (module docstring); None: 'sufficient_decrease' (Algorithm 1) with the augmented
        Lagrangian, 'ratio' with constraints='sqp' (the only one there).
    eta_accept : float
        acceptance='ratio': the smallest rho that accepts a step.
    radius0, radius_max : float
        constraints='sqp': the initial and largest box radius r_k (inf-norm, scaled design variables).
    merit_weight, ftol : float
        constraints='sqp': the L1 merit's constraint weight, and SLSQP's ftol (also the stopping test).
    data_store_file : str or None
        Write the DataStore after every iteration, as TrustRegionModelManagement.
    """

    MAX_AL_UPDATES = 5
    # constraints='sqp': SLSQP solves per subproblem in successively smaller boxes, at most 1 + this
    MAX_BOX_SHRINKS = 3

    def __init__(self, prob, windtunnel, indicator='ncd', output_names=None, output_scales=None, delta0=0.1,
                 beta1=0.5, beta2=0.95, eta_rho=0.75, kappa=0.5, kappa_arm=1e-4, tau_sub=1e-8, tau_foc=1e-4,
                 feasibility_tol=1e-5, penalty0=10.0, max_subproblem_iterations=20, max_armijo=20,
                 max_iterations=40, max_full_order=80, delta_min=1e-8, delta_max=1.0, calibration_window=3,
                 active_eps=1e-3, constraints='augmented_lagrangian', acceptance=None, eta_accept=0.1,
                 radius0=0.25, radius_max=2.0, merit_weight=10.0, ftol=1e-6, data_store_file=None,
                 printer=print):
        super().__init__(prob, windtunnel, model='rom', delta0=delta0, delta_max=delta_max, delta_min=delta_min,
                         eta_accept=eta_accept, merit_weight=merit_weight, subproblem_maxiter=max_subproblem_iterations,
                         ftol=ftol, max_iterations=max_iterations, max_full_order=max_full_order,
                         data_store_file=data_store_file, printer=printer)
        if indicator not in ('ncd', 'residual'):
            raise ValueError("indicator must be 'ncd' or 'residual', got {!r}".format(indicator))
        if indicator == 'ncd' and (output_names is None or output_scales is None):
            raise ValueError("indicator='ncd' needs output_names and output_scales")
        if constraints not in ('augmented_lagrangian', 'sqp'):
            raise ValueError("constraints must be 'augmented_lagrangian' or 'sqp', got {!r}".format(constraints))
        if acceptance is None:
            acceptance = 'ratio' if constraints == 'sqp' else 'sufficient_decrease'
        if acceptance not in ('sufficient_decrease', 'ratio'):
            raise ValueError("acceptance must be 'sufficient_decrease' or 'ratio', got {!r}".format(acceptance))
        if constraints == 'sqp' and acceptance != 'ratio':
            raise ValueError("constraints='sqp' has no AGC point: it needs acceptance='ratio'")
        self.constraints = constraints
        self.acceptance = acceptance
        self.radius0, self.radius_max = radius0, radius_max
        self.indicator = indicator
        self.output_names = None if output_names is None else list(output_names)
        self.output_scales = None if output_scales is None else np.asarray(output_scales, dtype=float)
        self.beta1, self.beta2, self.eta_rho = beta1, beta2, eta_rho
        self.kappa, self.kappa_arm = kappa, kappa_arm
        self.tau_sub, self.tau_foc = tau_sub, tau_foc
        self.feasibility_tol = feasibility_tol
        self.penalty0 = penalty0
        self.max_subproblem_iterations = max_subproblem_iterations
        self.max_armijo = max_armijo
        self.calibration_window = calibration_window
        self.active_eps = active_eps
        p = prob
        constrained = getattr(p, "constrained", False)
        self.al = AugmentedLagrangian(p.c_lower if constrained else [], p.c_upper if constrained else [], penalty0)
        # observed / raw estimate at the full-order trial points (calibration)
        self.ratios = []
        self.effectivity = 1.0
        self._cache = {}
        self._counts = dict(evals=0, full_order=0, failures=0)

    # -- evaluations ---------------------------------------------------------------------------
    def _bounds(self):
        p = self.prob
        n = np.size(p.x0)
        lower = np.where(np.isfinite(p.x_lower), p.x_lower, -np.inf) if p.x_lower is not None else np.full(n, -np.inf)
        upper = np.where(np.isfinite(p.x_upper), p.x_upper, np.inf) if p.x_upper is not None else np.full(n, np.inf)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def _merit(self, out):
        """(l, grad l) of an evaluation under the current augmented Lagrangian (grad None without
        derivatives)."""
        l = self.al.value(out['f'], out['c'])
        grad = self.al.gradient(out['g'], out['c'], out['J']) if 'g' in out else None
        return l, grad

    def _criticality(self, x, grad):
        lower, upper = self._bounds()
        return float(np.linalg.norm(x - np.clip(x - grad, lower, upper)))

    @contextlib.contextmanager
    def _rom_skipped(self):
        """A full-order evaluation does not try the ROM first (it would be rejected anyway)."""
        before = self.rom.min_snapshots
        self.rom.min_snapshots = np.iinfo(np.int64).max
        try:
            yield
        finally:
            self.rom.min_snapshots = before

    def _full_order(self, x, warm_start, required=False):
        """The full-order values and gradients at x, from the full-order solution warm_start (None: the
        windtunnel's): the enrichment of the ROM with the state and adjoint snapshots there. Where the solve
        fails (_full_order_failed) and its adjoints fail with it (e.g. the linear solve at a stalled state),
        the values are nan and there are no gradients, unless required (the start: the error is raised)."""
        if warm_start is not None:
            self.windtunnel.previous_solution = warm_start
        with self._rom_skipped():
            try:
                out = self._evaluate(x, 0.0, derivatives=True)
            except Exception as error:
                if required or not self._full_order_failed():
                    raise
                self.printer("TR-RB: the full-order derivatives at a failed solve failed too ({}: {})".format(
                    type(error).__name__, str(error).splitlines()[0][:160]))
                self._count(x)
                out = dict(f=np.nan, c=np.full(np.size(self.al.c_lower), np.nan), rom=False, solution=None)
        if self._full_order_failed():
            # a failed design solved again (the next subproblem can return it) is another full-order solve
            self._full_order_designs.discard(np.asarray(x, dtype=float).tobytes())
        return out

    def _full_order_failed(self):
        """Whether the last full-order solve failed (not converged, or an invalid mesh): its outputs are the
        windtunnel's 'high drag' placeholders, not the full-order model's."""
        wt = self.windtunnel
        mesh_valid = getattr(getattr(wt, 'mesh', None), 'is_valid', True)
        converged = getattr(getattr(wt, 'sim_model', None), 'last_solve_converged', True)
        return not (mesh_valid and converged)

    def _raw_estimate(self, out, l):
        """The uncalibrated estimate of |l - l_r| at the last (ROM) evaluation, nan without one."""
        wt = self.windtunnel
        if self.indicator == 'residual':
            result = getattr(wt, 'last_rom_result', None)
            return np.nan if result is None else float(result.eta) * abs(l)
        estimates = getattr(wt, 'rom_output_estimates', None)
        if estimates is None:
            return np.nan
        if self.constraints == 'sqp':
            # the model's objective and constraints are all checked against the full-order model's
            weights = np.ones(1 + np.size(out['c']))
        else:
            weights = np.concatenate([[1.0], self.al.output_weights(out['c'])])
        return float(sum(abs(w) * s * estimates[name]
                         for w, s, name in zip(weights, self.output_scales, self.output_names)))

    def _rom_model(self, x, derivatives, center):
        """The model (the ROM's augmented Lagrangian) at x: dict with f, c, l, the raw estimate
        'estimate', and with derivatives g, J, grad; None when the evaluation failed. Every evaluation
        starts from the center's solution; cached per design within a subproblem."""
        key = np.asarray(x, dtype=float).tobytes()
        hit = self._cache.get(key)
        if hit is not None and (not derivatives or hit['grad'] is not None):
            return hit
        if center.get('solution') is not None:
            self.windtunnel.previous_solution = center['solution']
        try:
            out = self._evaluate(x, np.inf, derivatives, bypass_repeat=True)
        except Exception as error:
            self.printer("TR-RB: model evaluation failed ({}: {})".format(
                type(error).__name__, str(error).splitlines()[0][:160]))
            self._counts['failures'] += 1
            return None
        self._counts['evals'] += 1
        if not out['rom']:
            self.printer("TR-RB: no ROM solution at a subproblem design; the full-order model solved it")
            self._counts['full_order'] += 1
        out['l'], out['grad'] = self._merit(out)
        if hit is not None:
            out['estimate'] = hit['estimate']
        else:
            out['estimate'] = self._raw_estimate(out, out['l'])
        out['x'] = np.array(x, dtype=float)
        self._cache[key] = out
        return out

    def _q(self, m):
        """The calibrated relative error estimate of a model evaluation (inf where there is none)."""
        estimate = self.effectivity * m['estimate']
        if not np.isfinite(estimate) or m['l'] == 0.0:
            return np.inf
        return float(estimate / abs(m['l']))

    # -- the subproblem --------------------------------------------------------------------------
    def _subproblem(self, x_k, center, delta):
        """Projected BFGS on the model from x_k within q <= delta (module docstring). Returns a dict with
        x (the result), m (the model there), m0 (at x_k), agc (the model at the AGC point, None if there
        is no step), iterations and message."""
        self._cache = {}
        lower, upper = self._bounds()
        if getattr(self.rom, 'test_space', None) == 'frozen' or getattr(self.rom, 'freeze_basis', False):
            # the ROM (or its basis) is frozen at this center by its first evaluation
            # (ReducedOrderModel.request_freeze)
            self.rom.request_freeze()
        m0 = self._rom_model(x_k, True, center)
        if m0 is None or m0['grad'] is None or not np.all(np.isfinite(m0['grad'])):
            return dict(x=x_k.copy(), m=m0, m0=m0, agc=None, iterations=0,
                        message="the model evaluation at the center failed")
        x, m = x_k.copy(), m0
        H = np.eye(x.size)
        agc = None
        message = "iteration limit"
        iterations = 0
        for it in range(self.max_subproblem_iterations):
            g = m['grad']
            pg = float(np.linalg.norm(x - np.clip(x - g, lower, upper)))
            if it > 0:
                if pg <= self.tau_sub:
                    message = "projected gradient {:.2e} below tau_sub".format(pg)
                    break
                q = self._q(m)
                if self.beta2 * delta <= q <= delta:
                    message = "at the trust-region boundary (q {:.3e} of delta {:.3e})".format(q, delta)
                    break
            # the eps-active bounds take a steepest-descent step, the others the BFGS step (Kelley,
            # Iterative Methods for Optimization, 5.5.3); the first step is the steepest descent (AGC)
            eps = min(self.active_eps, pg)
            active = ((x - lower <= eps) & (g > 0.0)) | ((upper - x <= eps) & (g < 0.0))
            d = -g.copy()
            if it > 0:
                free = ~active
                d[free] = -(H[np.ix_(free, free)] @ g[free])
                if float(g @ d) >= 0.0:
                    d = -g.copy()
            t, found = 1.0, None
            for _ in range(self.max_armijo + 1):
                x_j = np.clip(x + t * d, lower, upper)
                s = x_j - x
                if not np.any(s):
                    break
                m_j = self._rom_model(x_j, False, center)
                if m_j is not None and np.isfinite(m_j['l']) and \
                        m_j['l'] - m['l'] <= -(self.kappa_arm / t) * float(s @ s) and self._q(m_j) <= delta:
                    found = m_j
                    break
                t *= self.kappa
            if found is None:
                message = "no step satisfies the Armijo and trust-region conditions"
                break
            iterations = it + 1
            m_new = self._rom_model(found['x'], True, center)
            if it == 0:
                agc = found['l']
            if m_new is None or m_new['grad'] is None or not np.all(np.isfinite(m_new['grad'])):
                x, m = found['x'], found
                message = "the model gradient failed"
                break
            s, y = m_new['x'] - x, m_new['grad'] - g
            sy = float(s @ y)
            if sy > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
                r = 1.0 / sy
                E = np.eye(x.size) - r * np.outer(s, y)
                H = E @ H @ E.T + r * np.outer(s, s)
            x, m = m_new['x'], m_new
        return dict(x=x, m=m, m0=m0, agc=agc, iterations=iterations, message=message)

    def _sqp_model(self, x, derivatives, center, x_k, correction):
        """constraints='sqp': the ROM's values (and gradients) at x, first-order corrected at x_k, with its L1
        merit 'l' and raw estimate; raises _ModelFailed where the ROM fails."""
        out = self._rom_model(x, derivatives, center)
        if out is None or not (np.isfinite(out['f']) and np.all(np.isfinite(out['c']))) or (
                derivatives and not (np.all(np.isfinite(out['g'])) and np.all(np.isfinite(out['J'])))):
            raise _ModelFailed()
        s = np.asarray(x, dtype=float) - x_k
        m = dict(x=out['x'], f=out['f'] + correction['f0'] + correction['g'] @ s,
                 c=out['c'] + correction['c0'] + correction['J'] @ s, estimate=out['estimate'])
        if derivatives:
            m['g'], m['J'] = out['g'] + correction['g'], out['J'] + correction['J']
        m['l'] = self.merit(m['f'], m['c'])
        return m

    def _sqp_subproblem(self, x_k, center, delta, radius):
        """constraints='sqp' (module docstring): SLSQP on the corrected ROM within the box of the given radius,
        then cut back to q <= delta; the step is that point if its model merit is below the center's, else SLSQP
        runs again in a box of beta1 times its step, at most MAX_BOX_SHRINKS times, and after the last the step is
        the lowest model merit evaluated within the trust region below the center's.
        Returns a dict with x (x_k: no step), m (the model there), iterations (SLSQP's), radius (the last box's),
        uncut (the step is SLSQP's converged result within the box and the trust region), stationary (no step,
        and SLSQP converged within the box: the center is a stationary point of the model) and message."""
        self._cache = {}
        p = self.prob
        lower, upper = self._bounds()
        if getattr(self.rom, 'test_space', None) == 'frozen' or getattr(self.rom, 'freeze_basis', False):
            self.rom.request_freeze()
        at = self._rom_model(x_k, True, center)
        if at is None or not np.all(np.isfinite(at['g'])):
            return dict(x=x_k.copy(), m=None, iterations=0, radius=radius, stationary=False, uncut=False,
                        message="the model evaluation at the center failed")
        correction = dict(f0=center['f'] - at['f'], c0=center['c'] - at['c'], g=center['g'] - at['g'],
                          J=center['J'] - at['J'])
        self.printer("TR-RB: first-order correction at the center: value {:.3e} (objective), {} (constraints); "
                     "gradient {:.3e} of {:.3e} (objective, 2-norm), constraint Jacobian {:.3e} of {:.3e} "
                     "(Frobenius)".format(correction['f0'], np.array2string(correction['c0'], precision=3),
                                          np.linalg.norm(correction['g']), np.linalg.norm(center['g']),
                                          np.linalg.norm(correction['J']), np.linalg.norm(center['J'])))

        def model(x, derivatives):
            return self._sqp_model(x, derivatives, center, x_k, correction)
        try:
            m0 = model(x_k, True)
        except _ModelFailed:
            return dict(x=x_k.copy(), m=None, iterations=0, radius=radius, stationary=False, uncut=False,
                        message="the model evaluation at the center failed")
        constraints = []
        if getattr(p, "constrained", False):
            lo, hi = np.isfinite(p.c_lower), np.isfinite(p.c_upper)
            constraints.append(dict(
                type='ineq',
                fun=lambda x: np.concatenate([(model(x, False)['c'] - p.c_lower)[lo],
                                              (p.c_upper - model(x, False)['c'])[hi]]),
                jac=lambda x: np.vstack([model(x, True)['J'][lo], -model(x, True)['J'][hi]])))
        iterations, message, stationary = 0, "", False
        for attempt in range(self.MAX_BOX_SHRINKS + 1):
            box_lower, box_upper = np.maximum(lower, x_k - radius), np.minimum(upper, x_k + radius)
            success = False
            try:
                result = minimize(lambda x: model(x, False)['f'], x_k, jac=lambda x: model(x, True)['g'],
                                  method='SLSQP', bounds=list(zip(box_lower, box_upper)), constraints=constraints,
                                  options=dict(maxiter=self.subproblem_maxiter, ftol=self.ftol))
                x_s, note = np.clip(result.x, box_lower, box_upper), str(result.message)
                iterations += int(result.nit)
                success = bool(result.success)
            except _ModelFailed:
                # every rank fails alike: the evaluations so far are the candidates
                x_s, note = x_k.copy(), "a model evaluation failed"
            s = x_s - x_k
            step_s = float(np.max(np.abs(s), initial=0.0))
            inside = step_s < radius * (1.0 - 1e-6)
            # the trust region: SLSQP's step cut back to the first point with q <= delta
            cut = None
            if np.any(s):
                t = 1.0
                for _ in range(self.max_armijo + 1):
                    try:
                        m = model(x_k + t * s, False)
                        if self._q(m) <= delta:
                            cut = (t, m)
                            break
                    except _ModelFailed:
                        pass
                    t *= self.kappa
            if cut is not None and cut[1]['l'] < m0['l']:
                uncut = success and inside and cut[0] == 1.0
                if cut[0] < 1.0:
                    note += "; cut to {:.3g} of the step (q {:.3e} of delta {:.3e})".format(cut[0], self._q(cut[1]),
                                                                                         delta)
                return dict(x=cut[1]['x'], m=cut[1], iterations=iterations, radius=radius, stationary=False,
                            uncut=uncut, message=message + note)
            # the cut-back point does not decrease the model merit: a smaller box, unless SLSQP converged within it
            # (the center is then a stationary point of the model)
            stationary = success and inside
            if stationary or step_s == 0.0 or attempt == self.MAX_BOX_SHRINKS:
                message += note
                break
            radius = self.beta1 * step_s
            message += note + "; no decrease within the trust region, box radius -> {:.3e}; ".format(radius)
        if not stationary:
            # last: the lowest model merit of this subproblem's evaluations within the trust region (SLSQP's iterates,
            # the cut-back points), below the center's. Not before: the lowest of many model values favours the
            # model's largest errors
            best = None
            for out in list(self._cache.values()):
                try:
                    m = model(out['x'], False)
                except _ModelFailed:
                    continue
                if m['l'] < m0['l'] and self._q(m) <= delta and (best is None or m['l'] < best['l']):
                    best = m
            if best is not None:
                return dict(x=best['x'], m=best, iterations=iterations, radius=radius, stationary=False, uncut=False,
                            message=message + "; the lowest model merit evaluated within the trust region")
        return dict(x=x_k.copy(), m=m0, iterations=iterations, radius=radius, stationary=stationary, uncut=False,
                    message=message + "; no evaluation within the trust region decreases the model merit")

    def _solve_sqp(self):
        """solve() for constraints='sqp'."""
        x_k = np.array(self.prob.x0, dtype=float)
        center = self._full_order(x_k, None, required=True)
        delta, radius = self.delta0, self.radius0
        self.printer("TR-RB: start, SLSQP subproblems, {} indicator, delta {:.3e}, box radius {:.3e}, merit weight "
                     "{:.3g}".format(self.indicator, delta, radius, self.merit_weight))
        k = 0
        self.reason = "iteration limit"
        while k < self.max_iterations:
            if self.n_full_order >= self.max_full_order:
                self.reason = "full-order budget"
                break
            k += 1
            self._counts = dict(evals=0, full_order=0, failures=0)
            phi_k = self.merit(center['f'], center['c'])
            violation = self.al.violation(center['c'])
            sub = self._sqp_subproblem(x_k, center, delta, radius)
            x_t, m_t, radius = sub['x'], sub['m'], sub['radius']
            step = float(np.max(np.abs(x_t - x_k))) if x_t.size else 0.0
            base = dict(iteration=k, delta=delta, radius=radius, merit=phi_k, violation=violation,
                        subproblem_iterations=sub['iterations'], evaluations=self._counts['evals'],
                        failures=self._counts['failures'], message=sub['message'], step=step,
                        effectivity=self.effectivity)
            pred = phi_k - m_t['l'] if m_t is not None and step > 0.0 else np.nan
            if violation <= self.feasibility_tol and ((sub['uncut'] and pred <= self.ftol * max(1.0, abs(phi_k))) or
                                                      (step == 0.0 and sub['stationary'])):
                self._record_trrb(base, kind="converged", accepted=False)
                self.printer("TR-RB: iteration {}: the model predicts a merit decrease of {:.3e} (step {:.3e}): "
                             "converged".format(k, pred, step))
                self.reason = "converged"
                break
            if not (np.isfinite(pred) and pred > 0.0):
                # the model was not tested against the full-order model: delta stays
                radius *= self.beta1
                self._record_trrb(base, kind="no step", accepted=False)
                self.printer("TR-RB: iteration {}: no step ({}); box radius -> {:.3e}".format(k, sub['message'],
                                                                                            radius))
            else:
                trial = self._full_order(x_t, center['solution'])
                phi_t = self.merit(trial['f'], trial['c'])
                if self._full_order_failed():
                    delta *= self.beta1
                    radius = self.beta1 * step
                    self._record_trrb(base, kind="full-order solve failed", accepted=False, model_merit=m_t['l'],
                                      trial_merit=phi_t)
                    self.printer("TR-RB: iteration {}: the full-order solve at the step ({:.3e}) failed; rejected, "
                                 "delta -> {:.3e}, box radius -> {:.3e}".format(k, step, delta, radius))
                else:
                    error = abs(phi_t - m_t['l'])
                    # the estimate as the subproblem saw it, before this trial calibrates it
                    q_t, estimate_t = self._q(m_t), self.effectivity * m_t['estimate']
                    if np.isfinite(m_t['estimate']) and m_t['estimate'] > 0.0:
                        self.ratios.append(error / m_t['estimate'])
                        self.effectivity = max(self.ratios[-self.calibration_window:])
                    rho = (phi_k - phi_t) / pred
                    accepted = bool(np.isfinite(rho) and rho >= self.eta_accept)
                    if accepted and rho >= self.eta_rho:
                        delta = min(delta / self.beta1, self.delta_max)
                        if step >= 0.99 * radius:
                            radius = min(2.0 * radius, self.radius_max)
                    elif not accepted:
                        delta *= self.beta1
                        radius = self.beta1 * step
                    self._record_trrb(base, kind="ratio", accepted=accepted, model_merit=m_t['l'], trial_merit=phi_t,
                                      rho=rho, model_error=error, estimate=estimate_t, q=q_t,
                                      trial_violation=self.al.violation(trial['c']))
                    self.printer("TR-RB: iteration {}: {} SLSQP iterations, {} model evaluations ({}), step {:.3e}; "
                                 "merit {:.8f} -> {:.8f} (model {:.8f}; model error {:.3e}, estimate {:.3e}; violation "
                                 "{:.3e} -> {:.3e}): rho {:.3f}, {}; delta -> {:.3e}, box radius -> {:.3e}, effectivity "
                                 "{:.3e}".format(k, sub['iterations'], self._counts['evals'], sub['message'], step,
                                                 phi_k, phi_t, m_t['l'], error, estimate_t,
                                                 violation, self.al.violation(trial['c']), rho,
                                                 "accepted" if accepted else "rejected", delta, radius,
                                                 self.effectivity))
                    if accepted:
                        x_k, center = x_t, trial
            if delta < self.delta_min or radius < self.delta_min:
                self.reason = "small trust region"
                break
        self.x, self.fun, self.center, self.delta = x_k, center['f'], center, delta
        self.printer("TR-RB: {} after {} iterations and {} full-order solves; merit {:.8f}, objective {:.8f}, "
                     "violation {:.3e}".format(self.reason, k, self.n_full_order, self.merit(center['f'], center['c']),
                                               center['f'], self.al.violation(center['c'])))
        self.windtunnel.mesh.mesh.comm.Barrier()
        return x_k

    # -- the iteration -------------------------------------------------------------------------
    def solve(self):
        """Run the iteration; returns the final center (scaled design variables), also in self.x."""
        if self.constraints == 'sqp':
            return self._solve_sqp()
        x_k = np.array(self.prob.x0, dtype=float)
        center = self._full_order(x_k, None, required=True)
        delta = self.delta0
        tau = self.al.tau
        omega, eta_feasible = 1.0 / tau, 1.0 / tau**0.1
        self.printer("TR-RB: start, {} indicator, delta {:.3e}{}".format(
            self.indicator, delta, ", augmented Lagrangian penalty {:.3g}".format(tau) if self.al.constrained
            else ""))
        k = 0
        # augmented Lagrangian updates at the current center, at most MAX_AL_UPDATES before a trust-region
        # iteration
        updates = 0
        self.reason = "iteration limit"
        while True:
            l_k, grad_k = self._merit(center)
            foc = self._criticality(x_k, grad_k)
            violation = self.al.violation(center['c'])
            if foc <= self.tau_foc and violation <= self.feasibility_tol:
                self.reason = "converged"
                break
            if self.al.constrained and foc <= max(omega, self.tau_foc) and updates < self.MAX_AL_UPDATES:
                updates += 1
                # the augmented Lagrangian subproblem is solved to omega: update its multipliers, or its
                # penalty (Nocedal and Wright, Algorithm 17.4)
                if violation <= eta_feasible:
                    self.al.update_multipliers(center['c'])
                    eta_feasible /= self.al.tau**0.9
                    omega /= self.al.tau
                    what = "multipliers {}".format(np.array2string(self.al.theta, precision=4))
                else:
                    self.al.tau *= 10.0
                    eta_feasible, omega = 1.0 / self.al.tau**0.1, 1.0 / self.al.tau
                    what = "penalty {:.3g}".format(self.al.tau)
                self.printer("TR-RB: augmented Lagrangian criticality {:.3e}, violation {:.3e}: new {}; "
                             "tolerances omega {:.3e}, eta {:.3e}".format(foc, violation, what, omega, eta_feasible))
                continue
            if k >= self.max_iterations:
                break
            if self.n_full_order >= self.max_full_order:
                self.reason = "full-order budget"
                break
            k += 1
            updates = 0
            self._counts = dict(evals=0, full_order=0, failures=0)
            sub = self._subproblem(x_k, center, delta)
            x_t, m_t, m0, agc = sub['x'], sub['m'], sub['m0'], sub['agc']
            step = float(np.max(np.abs(x_t - x_k))) if x_t.size else 0.0
            base = dict(iteration=k, delta=delta, merit=l_k, criticality=foc, violation=violation,
                        bfgs_iterations=sub['iterations'], evaluations=self._counts['evals'],
                        failures=self._counts['failures'], message=sub['message'], step=step,
                        effectivity=self.effectivity, penalty=self.al.tau, multipliers=self.al.theta.copy())
            if agc is None:
                delta *= self.beta1
                self._record_trrb(base, kind="no step", accepted=False)
                self.printer("TR-RB: iteration {}: no step ({}); delta -> {:.3e}".format(k, sub['message'], delta))
                if delta < self.delta_min:
                    self.reason = "small trust region"
                    break
                continue
            estimate_t = self.effectivity * m_t['estimate']
            l_r_t, l_r_k = m_t['l'], m0['l']
            ratio_test = self.acceptance == 'ratio'
            if not ratio_test and l_r_t - estimate_t > agc:
                # line 11: the model cannot satisfy the sufficient decrease condition
                delta *= self.beta1
                self._record_trrb(base, kind="rejected without a full-order solve", accepted=False,
                                  model_merit=l_r_t, agc=agc)
                self.printer("TR-RB: iteration {}: rejected (model {:.8f} - estimate {:.3e} above the AGC value "
                             "{:.8f}); delta -> {:.3e}".format(k, l_r_t, estimate_t, agc, delta))
                continue
            certified = not ratio_test and l_r_t + estimate_t < agc
            trial = self._full_order(x_t, center['solution'])
            l_t, _ = self._merit(trial)
            error = abs(l_t - l_r_t)
            if self._full_order_failed():
                # no full-order value there: a rejection, which neither calibrates nor enriches
                delta *= self.beta1
                self._record_trrb(base, kind="full-order solve failed", accepted=False, model_merit=l_r_t, agc=agc,
                                  trial_merit=l_t)
                self.printer("TR-RB: iteration {}: the full-order solve at the step ({:.3e}) failed; rejected, "
                             "delta -> {:.3e}".format(k, step, delta))
                if delta < self.delta_min:
                    self.reason = "small trust region"
                    break
                continue
            if np.isfinite(m_t['estimate']) and m_t['estimate'] > 0.0:
                self.ratios.append(error / m_t['estimate'])
                self.effectivity = max(self.ratios[-self.calibration_window:])
            pred = l_r_k - l_r_t
            rho = (l_k - l_t) / pred if pred != 0.0 else np.nan
            if ratio_test:
                accepted = bool(np.isfinite(rho) and pred > 0.0 and rho >= self.eta_accept)
            elif certified:
                # line 4, with an estimate that is not a bound: not if the full-order merit increased
                accepted = bool(l_t < l_k)
            else:
                # lines 13-24: the sufficient decrease condition with the enriched model (exact at mu_{k+1})
                accepted = bool(l_t <= agc)
            if accepted:
                if np.isfinite(rho) and rho >= self.eta_rho:
                    delta = min(delta / self.beta1, self.delta_max)
            else:
                delta *= self.beta1
            kind = "ratio" if ratio_test else "certified" if certified else "checked"
            self._record_trrb(base, kind=kind, accepted=accepted,
                              model_merit=l_r_t, agc=agc, trial_merit=l_t, rho=rho, model_error=error,
                              estimate=estimate_t, q=self._q(m_t))
            self.printer("TR-RB: iteration {}: {} BFGS iterations, {} model evaluations ({}), step {:.3e}; merit "
                         "{:.8f} -> {:.8f} (model {:.8f}, AGC {:.8f}; model error {:.3e}, estimate {:.3e}): rho {:.3f}, "
                         "{} ({}); delta -> {:.3e}, effectivity {:.3e}".format(
                             k, sub['iterations'], self._counts['evals'], sub['message'], step, l_k, l_t, l_r_t, agc,
                             error, estimate_t, rho, "accepted" if accepted else "rejected", kind, delta,
                             self.effectivity))
            if accepted:
                x_k, center = x_t, trial
            if delta < self.delta_min:
                self.reason = "small trust region"
                break
        self.x, self.fun, self.center, self.delta = x_k, center['f'], center, delta
        l_k, grad_k = self._merit(center)
        self.printer("TR-RB: {} after {} iterations and {} full-order solves; merit {:.8f}, objective {:.8f}, "
                     "criticality {:.3e}, violation {:.3e}".format(
                         self.reason, k, self.n_full_order, l_k, center['f'], self._criticality(x_k, grad_k),
                         self.al.violation(center['c'])))
        self.windtunnel.mesh.mesh.comm.Barrier()
        return x_k

    def _record_trrb(self, base, kind, accepted, **values):
        it = dict(base, eval_idx=self.windtunnel.eval_idx - 1, kind=kind, accepted=accepted,
                  full_order_total=self.n_full_order)
        it.update(values)
        self._store_iteration(it)

    def summary(self):
        """One line per iteration."""
        lines = []
        for it in self.iterations:
            lines.append("{:3d}: delta {:.2e} step {:.2e}  merit {:.8f} -> {}  rho {}  {} ({}), {} {}, {} evals, "
                         "{} full-order total".format(
                             it['iteration'], it['delta'], it['step'], it['merit'],
                             "{:.8f}".format(it['trial_merit']) if 'trial_merit' in it else "-",
                             "{:7.3f}".format(it['rho']) if 'rho' in it else "      -",
                             "accepted" if it['accepted'] else "rejected", it['kind'],
                             *((it['bfgs_iterations'], "BFGS") if 'bfgs_iterations' in it else
                               (it['subproblem_iterations'], "SLSQP")),
                             it['evaluations'], it['full_order_total']))
        return "\n".join(lines)
