"""Trust-region model management (TRMM) of a ROM-driven optimization.

The trust-region framework of Alexandrov, Dennis, Lewis and Torczon (Structural Optimization 15,
1998) for optimizing on an approximation model, as POD models use it (Arian, Fahl and Sachs, TRPOD,
ICASE 2000-25): every iteration has a center x_k with the full-order values and gradients (a
full-order solve and adjoints, which the ROM also gets as snapshots), and a model of the objective
and constraints that matches them there to first order,

    m(x) = h_rom(x) + [h(x_k) - h_rom(x_k)] + [grad h(x_k) - grad h_rom(x_k)]^T (x - x_k)

for h the objective and every constraint, h_rom the ROM's output (its solution is accepted at any
finite residual, so the model never falls back to a full-order solve). SLSQP minimizes the model
within the box |x - x_k|_inf <= Delta (and the design bounds); the full-order model is evaluated at
its result x_t, and the ratio of the actual to the predicted decrease of the L1 merit

    phi = f + merit_weight * sum of the constraint violations      (scaled f and c)

decides whether x_t becomes the next center and how Delta changes. Away from the center the ROM's
gradient errors cost efficiency (a smaller rho, a smaller Delta), not the convergence of the
iteration; what the subproblem needs is a model whose gradient is the derivative of its value, as the
ROMs' adjoints are (adjoint_solve 'lspg', 'dual_corrected' with corrected outputs, the hybrid's).

With fewer ROM snapshots than the ROM needs (the first iteration) the model is the first-order Taylor
model at the center (a linear program in the box); with model='fom' it is the full-order model itself
(no correction; the subproblem's evaluations are all full-order), the trust region then only
globalizing SLSQP, as a reference for the cost of the iteration (the windtunnel model then needs no
ROM).
"""
import contextlib

import numpy as np
from scipy.optimize import minimize


class TrustRegionModelManagement:
    """
    TRMM (module docstring) for a modopt CSDLAlphaProblem whose flow model is a DG_windtunnel_model with
    a ReducedOrderModel. Collective: every rank runs the iteration (and SLSQP) on the same data.

    Parameters
    ----------
    prob : modopt CSDLAlphaProblem
        The optimization problem; its x0 is the first center.
    windtunnel : DG_windtunnel_model
        The flow model, with a reduced_order_model unless model='fom'.
    model : 'rom' or 'fom'
        The subproblem's model: the first-order corrected ROM, or the full-order model.
    delta0, delta_max, delta_min : float
        The initial, largest and smallest trust-region radius (inf-norm, scaled design variables).
    eta_accept, eta_shrink, eta_expand : float
        rho >= eta_accept accepts the step; rho < eta_shrink shrinks Delta to a quarter of the step,
        rho > eta_expand with the step on the trust-region boundary doubles it.
    subproblem_maxiter : int
        SLSQP iterations per subproblem.
    ftol : float
        SLSQP's ftol in the subproblems; the iteration stops when the model predicts a merit decrease
        below ftol * max(1, |phi(x_k)|).
    max_iterations, max_full_order : int
        Bounds on the iterations and on the full-order solves (centers, trial points and the
        subproblems' full-order evaluations).
    data_store_file : str or None
        If given, the windtunnel's DataStore is written to Result_dicts/<data_store_file>
        (DataStore.write_store_to_numpy_file) after every iteration, replacing the previous iteration's
        file, so that a run stopped early leaves its data up to the last finished iteration. None (the
        default): the iterations are only recorded in the DataStore.
    """

    def __init__(self, prob, windtunnel, model='rom', delta0=0.25, delta_max=2.0, delta_min=1e-4, eta_accept=0.1,
                 eta_shrink=0.25, eta_expand=0.75, merit_weight=10.0, subproblem_maxiter=20, ftol=1e-6,
                 max_iterations=50, max_full_order=300, data_store_file=None, printer=print):
        if model not in ('rom', 'fom'):
            raise ValueError("model must be 'rom' or 'fom', got {!r}".format(model))
        if model == 'rom' and windtunnel.reduced_order_model is None:
            raise ValueError("model='rom' needs a windtunnel model with a reduced-order model")
        self.prob = prob
        self.windtunnel = windtunnel
        self.rom = windtunnel.reduced_order_model
        self.model = model
        self.delta0 = delta0
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.eta_accept = eta_accept
        self.eta_shrink = eta_shrink
        self.eta_expand = eta_expand
        self.merit_weight = merit_weight
        self.subproblem_maxiter = subproblem_maxiter
        self.ftol = ftol
        self.max_iterations = max_iterations
        self.max_full_order = max_full_order
        self.data_store_file = data_store_file
        self.printer = printer
        # per iteration: dict (see _record)
        self.iterations = []
        # the designs (scaled, as bytes) the full-order model solved: n_full_order counts new designs, not
        # the re-runs of a design CSDL does before its derivatives
        self._full_order_designs = set()
        self.n_full_order = 0
        self.x = None
        self.fun = None
        # the final center's full-order values: dict(f, c, g, J) (scaled)
        self.center = None
        self.reason = None

    # -- evaluations ---------------------------------------------------------------------------
    def merit(self, f, c):
        """The L1 merit of scaled objective f and constraints c."""
        p = self.prob
        if not getattr(p, "constrained", False):
            return float(f)
        c = np.ravel(c)
        violation = np.maximum(p.c_lower - c, 0.0) + np.maximum(c - p.c_upper, 0.0)
        return float(f) + self.merit_weight * float(np.sum(violation))

    def _values(self, derivatives):
        p = self.prob
        out = dict(f=float(np.ravel(p.f_s)[0]), c=np.array(np.ravel(p.c_s), dtype=float) if
                   getattr(p, "constrained", False) else np.zeros(0))
        if derivatives:
            out['g'] = np.array(np.ravel(p.g_s), dtype=float)
            out['J'] = np.array(p.j_s, dtype=float).reshape(out['c'].size, -1) if out['c'].size else \
                np.zeros((0, out['g'].size))
        return out

    def _count(self, x):
        """After a real model evaluation at x: count it if the full-order model solved a new design."""
        key = np.asarray(x, dtype=float).tobytes()
        if not self.windtunnel.last_solve_rom_accepted and key not in self._full_order_designs:
            self._full_order_designs.add(key)
            self.n_full_order += 1
            return 1
        return 0

    @contextlib.contextmanager
    def _threshold(self, eta):
        """The ROM's eta_threshold at eta within the block (nothing without a ROM)."""
        if self.rom is None:
            yield
            return
        eta_before, self.rom.eta_threshold = self.rom.eta_threshold, eta
        try:
            yield
        finally:
            self.rom.eta_threshold = eta_before

    def _evaluate(self, x, eta, derivatives, bypass_repeat=False):
        """The problem's values (and gradients) at x, with the ROM's eta_threshold at eta for this
        evaluation (0: full order; inf: any finite ROM solution); bypass_repeat makes the windtunnel try
        the ROM also at the design the full-order model solved last. Counts full-order solves."""
        if bypass_repeat:
            self.windtunnel._fom_design = None
        evals = self.prob.model_evals
        with self._threshold(eta):
            if derivatives:
                self.prob.check_if_warm_and_compute_derivatives(x, force_rerun=True)
            else:
                self.prob.check_if_warm_and_run_model(x, force_rerun=True)
        if self.prob.model_evals > evals:
            self._count(x)
        out = self._values(derivatives)
        out['rom'] = bool(self.windtunnel.last_solve_rom_accepted)
        solution = getattr(self.windtunnel, 'previous_solution', None)
        out['solution'] = None if solution is None else np.array(solution, copy=True)
        return out

    # -- the subproblem ------------------------------------------------------------------------
    def _subproblem(self, center, x_k, delta):
        """Minimize the model within the trust region from x_k; returns (x_t, the model's merit at x_t,
        its kind, model evaluations, their full-order solves, SLSQP's message)."""
        p = self.prob
        n = x_k.size
        lower = np.where(np.isfinite(p.x_lower), p.x_lower, -np.inf) if p.x_lower is not None else np.full(n, -np.inf)
        upper = np.where(np.isfinite(p.x_upper), p.x_upper, np.inf) if p.x_upper is not None else np.full(n, np.inf)
        bounds = list(zip(np.maximum(lower, x_k - delta), np.minimum(upper, x_k + delta)))

        # warm-start from the center's solution, not the last solve's: after a rejected step that is the
        # trial point's, and a ROM centred on it can land on another solution at the center
        if center.get('solution') is not None:
            self.windtunnel.previous_solution = center['solution']
        kind, correction = self._model_kind(center, x_k)
        counts = dict(evals=0, full_order=0)
        cache = {}

        def model(x, derivatives):
            x = np.asarray(x, dtype=float)
            if kind == 'linear':
                s = x - x_k
                return dict(f=center['f'] + center['g'] @ s, c=center['c'] + center['J'] @ s,
                            g=center['g'], J=center['J'])
            key = x.tobytes()
            hit = cache.get(key)
            if hit is not None and (not derivatives or 'g' in hit):
                return hit
            evals = p.model_evals
            with self._threshold(0.0 if kind == 'fom' else np.inf):
                if derivatives:
                    p.check_if_warm_and_compute_derivatives(x)
                else:
                    p.check_if_warm_and_run_model(x)
            if p.model_evals > evals:
                counts['evals'] += 1
                counts['full_order'] += self._count(x)
            out = self._values(derivatives)
            if correction is not None:
                s = x - x_k
                out['f'] += correction['f0'] + correction['g'] @ s
                out['c'] = out['c'] + correction['c0'] + correction['J'] @ s
                if derivatives:
                    out['g'] = out['g'] + correction['g']
                    out['J'] = out['J'] + correction['J']
            out['x'] = x.copy()
            cache[key] = out
            return out

        constraints = []
        if getattr(p, "constrained", False):
            lo, hi = np.isfinite(p.c_lower), np.isfinite(p.c_upper)
            constraints.append(dict(
                type='ineq',
                fun=lambda x: np.concatenate([(model(x, False)['c'] - p.c_lower)[lo],
                                              (p.c_upper - model(x, False)['c'])[hi]]),
                jac=lambda x: np.vstack([model(x, True)['J'][lo], -model(x, True)['J'][hi]])))
        try:
            result = minimize(lambda x: model(x, False)['f'], x_k, jac=lambda x: model(x, True)['g'], method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options=dict(maxiter=self.subproblem_maxiter, ftol=self.ftol))
        except Exception as error:
            # a model evaluation failed (e.g. the reduced adjoint of a poor ROM solution is singular; every rank
            # raises alike): the subproblem ends at the best design it evaluated, if that beats the center,
            # else without a step (nan: the iteration shrinks the trust region)
            message = "model evaluation failed ({}: {})".format(type(error).__name__, str(error).splitlines()[0][:160])
            self.printer("trust region: " + message)
            phi_k = self.merit(center['f'], center['c'])
            merits = [(self.merit(out['f'], out['c']), i, out) for i, out in enumerate(cache.values())]
            phi_best, _, best = min(merits, default=(np.inf, 0, None), key=lambda t: t[:2])
            if best is None or not phi_best < phi_k:
                return x_k.copy(), np.nan, kind, counts['evals'], counts['full_order'], message
            return best['x'], phi_best, kind, counts['evals'], counts['full_order'], message
        x_t = np.clip(np.asarray(result.x, dtype=float), [b[0] for b in bounds], [b[1] for b in bounds])
        m_t = model(x_t, False)
        return x_t, self.merit(m_t['f'], m_t['c']), kind, counts['evals'], counts['full_order'], str(result.message)

    def _model_kind(self, center, x_k):
        """('fom', None), ('linear', None) without enough ROM snapshots, or ('rom', the first-order
        correction from the ROM's values and gradients at the center)."""
        if self.model == 'fom':
            return 'fom', None
        if self.rom.n_snapshots < self.rom.min_snapshots:
            self.printer("trust region: {} ROM snapshot(s), {} needed; the model is the linear Taylor model".format(
                self.rom.n_snapshots, self.rom.min_snapshots))
            return 'linear', None
        if getattr(self.rom, 'freeze_basis', False):
            # one basis per trust-region iteration: built at the center, by this evaluation
            self.rom.request_freeze()
        try:
            at = self._evaluate(x_k, np.inf, derivatives=True, bypass_repeat=True)
        except Exception as error:
            self.printer("trust region: the ROM evaluation at the center failed ({}: {}); the model is the linear "
                         "Taylor model".format(type(error).__name__, str(error).splitlines()[0][:160]))
            return 'linear', None
        if not at['rom']:
            self.printer("trust region: no ROM solution at the center; the model is the linear Taylor model")
            return 'linear', None
        correction = dict(f0=center['f'] - at['f'], c0=center['c'] - at['c'], g=center['g'] - at['g'],
                          J=center['J'] - at['J'])
        self.printer("trust region: first-order correction at the center: value {:.3e} (objective), {} "
                     "(constraints); gradient {:.3e} of {:.3e} (objective, 2-norm), constraint Jacobian {:.3e} of "
                     "{:.3e} (Frobenius)".format(
                         correction['f0'], np.array2string(correction['c0'], precision=3),
                         np.linalg.norm(correction['g']), np.linalg.norm(center['g']),
                         np.linalg.norm(correction['J']), np.linalg.norm(center['J'])))
        return 'rom', correction

    # -- the iteration -------------------------------------------------------------------------
    def solve(self):
        """Run the iteration; returns the final center (scaled design variables), also in self.x."""
        x_k = np.array(self.prob.x0, dtype=float)
        center = self._evaluate(x_k, 0.0, derivatives=True)
        phi_k = self.merit(center['f'], center['c'])
        delta = self.delta0
        self.printer("trust region: start, merit {:.8f}, Delta {:.3e}".format(phi_k, delta))
        for k in range(self.max_iterations):
            if self.n_full_order >= self.max_full_order:
                self.reason = "full-order budget"
                break
            x_t, phi_model, kind, evals, evals_fom, message = self._subproblem(center, x_k, delta)
            step = float(np.max(np.abs(x_t - x_k))) if x_t.size else 0.0
            pred = phi_k - phi_model
            if not np.isfinite(pred) or pred < 0.0:
                # SLSQP ended above its start (an evaluation failed, e.g. a flow solve, or it traded the
                # objective for feasibility it could not reach): no step; a smaller trust region
                self._record(k, delta, phi_k, phi_model, np.nan, pred, np.nan, np.nan, False, step, kind, evals,
                             evals_fom, message)
                delta *= 0.25
                self.printer("trust region: iteration {}: the {} subproblem ended at a model merit {:.8f} above the "
                             "center's {:.8f} ({}); Delta -> {:.3e}".format(k + 1, kind, phi_model, phi_k, message,
                                                                            delta))
                if delta < self.delta_min:
                    self.reason = "small trust region"
                    break
                continue
            if pred <= self.ftol * max(1.0, abs(phi_k)):
                self._record(k, delta, phi_k, phi_model, np.nan, pred, np.nan, np.nan, False, step, kind, evals,
                             evals_fom, message)
                self.printer("trust region: iteration {}: the {} model predicts a merit decrease of {:.3e} (step "
                             "{:.3e}, Delta {:.3e}): converged".format(k + 1, kind, pred, step, delta))
                self.reason = "converged"
                break
            # the trial's full-order solve starts from the center's solution too: the subproblem's last ROM
            # solution, accepted at any residual, can be far off, and the solve may not recover from it
            if center.get('solution') is not None:
                self.windtunnel.previous_solution = center['solution']
            trial = self._evaluate(x_t, 0.0, derivatives=False)
            phi_t = self.merit(trial['f'], trial['c'])
            ared = phi_k - phi_t
            rho = ared / pred
            accepted = bool(np.isfinite(rho) and rho >= self.eta_accept)
            self._record(k, delta, phi_k, phi_model, phi_t, pred, ared, rho, accepted, step, kind, evals, evals_fom,
                         message)
            self.printer("trust region: iteration {}: {} model, {} evaluations ({} full order), step {:.3e} of Delta "
                         "{:.3e}; merit {:.8f} -> {:.8f} (model {:.8f}): rho {:.3f}, {}".format(
                             k + 1, kind, evals, evals_fom, step, delta, phi_k, phi_t, phi_model, rho,
                             "accepted" if accepted else "rejected"))
            if not np.isfinite(rho) or rho < self.eta_shrink:
                delta = 0.25 * (step if step > 0.0 else delta)
            elif rho > self.eta_expand and step >= 0.99 * delta:
                delta = min(2.0 * delta, self.delta_max)
            if accepted:
                # the full-order solution at x_t is the last solve, so the gradient re-run keeps it and
                # its adjoints are full order (snapshots)
                x_k, center, phi_k = x_t, self._evaluate(x_t, 0.0, derivatives=True), phi_t
            if delta < self.delta_min:
                self.printer("trust region: Delta {:.3e} below {:.1e}; stopping".format(delta, self.delta_min))
                self.reason = "small trust region"
                break
        else:
            self.reason = "iteration limit"
        self.x, self.fun, self.center, self.delta = x_k, center['f'], center, delta
        self.printer("trust region: {} after {} iterations and {} full-order solves; merit {:.8f}".format(
            self.reason, len(self.iterations), self.n_full_order, phi_k))
        self.windtunnel.mesh.mesh.comm.Barrier()
        return x_k

    def _record(self, k, delta, phi_k, phi_model, phi_t, pred, ared, rho, accepted, step, kind, evals, evals_fom,
                message):
        it = dict(iteration=k + 1, eval_idx=self.windtunnel.eval_idx - 1, delta=delta, merit=phi_k,
                  model_merit=phi_model, trial_merit=phi_t, pred=pred, ared=ared, rho=rho, accepted=accepted,
                  step=step, model=kind, evaluations=evals, full_order_evaluations=evals_fom,
                  full_order_total=self.n_full_order, message=message)
        self._store_iteration(it)

    def _store_iteration(self, it):
        """Keep the record `it` of a finished iteration, and add it to the DataStore (rank 0), which is then
        written to data_store_file if one is given, replacing the last iteration's file."""
        self.iterations.append(it)
        if self.windtunnel.mesh.mesh.comm.Get_rank() != 0:
            return
        data_store = self.windtunnel.data_store
        data_store.trust_region_iterations += [dict(it)]
        if self.data_store_file is not None:
            data_store.write_store_to_numpy_file(save_filename=self.data_store_file)

    def summary(self):
        """One line per iteration."""
        return "\n".join("{iteration:3d}: {model:6s} Delta {delta:.2e} step {step:.2e}  merit {merit:.8f} -> "
                         "{trial_merit:.8f} (model {model_merit:.8f})  rho {rho:7.3f}  {acc}  {evaluations:3d} evals "
                         "({full_order_evaluations} full order), {full_order_total} full-order total".format(
                             acc="accepted" if it['accepted'] else "rejected", **it) for it in self.iterations)
