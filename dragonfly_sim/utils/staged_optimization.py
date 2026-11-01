"""ROM-driven SLSQP in stages of decreasing ROM acceptance threshold, with restarts.

A ROM that drives an optimization (DG_windtunnel_model with a ReducedOrderModel whose
eta_threshold accepts its solutions) trades full-order solves for accuracy through eta. A loose
eta early on lets the ROM take most designs, so every full-order solve it does need adds a
snapshot where the ROM was least accurate; near the optimum the steps are small and the outputs
and gradients must be accurate, so eta is tightened there. StagedROMOptimization runs SLSQP with
a schedule of thresholds, loosest first.

The ROM's errors show in SLSQP's line search: a trial point the ROM accepted whose (ROM) merit
does not decrease makes SLSQP shrink the step, and the next trial points are again ROM solutions
near the last iterate, which can stall the optimization (the line search shrinks the step until
the objective change falls below ftol). The driver aborts SLSQP the moment its line search
shrinks the step after a ROM trial point from an iterate the ROM evaluated, and restarts it from
that iterate with the full-order solution and adjoints there, which also become ROM snapshots
where the ROM failed. From an iterate with full-order values and gradient (a run's start, or a
design the ROM rejected) a restart would repeat the same step, so the line search goes on there.
"""
import numpy as np
from mpi4py import MPI

from modopt import SLSQP


class LineSearchStall(Exception):
    """Aborts an SLSQP run from its objective callback: the line search shrank its step after a
    trial point the ROM evaluated. x_restart is the last iterate (scaled design variables)."""

    def __init__(self, x_restart):
        super().__init__("SLSQP's line search shrinks its step after a ROM evaluation")
        self.x_restart = np.array(x_restart, dtype=float)


class StagedROMOptimization:
    """
    SLSQP driven by the ROM of a DG_windtunnel_model, in stages with the ROM's eta_threshold
    taken from eta_schedule in turn (loosest first).

    Every SLSQP run starts at a design evaluated with the full-order model: the driver evaluates
    it with the ROM's eta_threshold at 0, so its solution, and the adjoints of SLSQP's first
    gradient there, are full-order (DG_windtunnel_model keeps the full-order solution of a design
    it re-evaluates right after solving it, see _solve) and become snapshots. A run ends

      - when SLSQP returns (converged, or any other exit mode). The next stage, with the next eta,
        restarts from its result; after the last stage the optimization ends there.
      - when the line search shrinks its step after a trial point the ROM accepted, from an
        iterate the ROM evaluated too: a second trial point since the last gradient, with the one
        before it and the iterate (the design of the last gradient) ROM solutions. The run is
        aborted before that point is evaluated and restarted in the same stage from the iterate. If that restart point is no better than the
        stage's previous restart point (its merit, the scaled objective plus merit_weight times
        the scaled constraint violations, is not lower by more than ftol), or the stage has been
        restarted max_stall_restarts times already, the ROM is not accurate enough at this eta: the
        stage ends at its best full-order evaluated restart point, where the next stage starts (and
        after the last stage the optimization ends).

    SLSQP's quasi-Newton Hessian starts over with every run. maxiter bounds the major iterations
    of all runs together (an aborted run's gradient evaluations count as its iterations),
    max_runs the number of runs. The runs are recorded in the windtunnel's
    data_store.optimization_runs. Collective: every rank runs the driver (and SLSQP) with the same
    data, so every rank takes the same decisions.

    Parameters
    ----------
    prob : modopt CSDLAlphaProblem
        The optimization problem; its x0 is the start of the first run.
    windtunnel : DG_windtunnel_model
        The flow model whose reduced_order_model drives the optimization.
    eta_schedule : sequence of float
        The ROM's eta_threshold per stage, in order.
    """

    def __init__(self, prob, windtunnel, eta_schedule, ftol=1e-6, maxiter=100, max_runs=30, max_stall_restarts=4,
                 merit_weight=10.0, printer=print, turn_off_outputs=False):
        if windtunnel.reduced_order_model is None:
            raise ValueError("StagedROMOptimization needs a windtunnel model with a reduced-order model")
        if len(eta_schedule) == 0:
            raise ValueError("eta_schedule needs at least one threshold")
        self.prob = prob
        self.windtunnel = windtunnel
        self.rom = windtunnel.reduced_order_model
        self.eta_schedule = [float(eta) for eta in eta_schedule]
        self.ftol = ftol
        self.maxiter = maxiter
        self.max_runs = max_runs
        self.max_stall_restarts = max_stall_restarts
        self.merit_weight = merit_weight
        self.printer = printer
        self.turn_off_outputs = turn_off_outputs
        # per run: dict(stage, eta, x0, merit0, eval_idx, reason, iterations, results)
        self.runs = []
        self.x = None
        self.fun = None

    def merit(self):
        """The L1 merit of the problem's last evaluation: scaled objective plus merit_weight times
        the scaled constraint violations."""
        p = self.prob
        f = float(np.ravel(p.f_s)[0])
        if not getattr(p, "constrained", False):
            return f
        c = np.ravel(p.c_s)
        violation = np.maximum(p.c_lower - c, 0.0) + np.maximum(c - p.c_upper, 0.0)
        return f + self.merit_weight * float(np.sum(violation))

    def _full_order_start(self, x):
        """Evaluate x with the full-order model (no ROM solution accepted); its merit."""
        eta, self.rom.eta_threshold = self.rom.eta_threshold, 0.0
        try:
            self.prob.check_if_warm_and_run_model(x, force_rerun=True)
        finally:
            self.rom.eta_threshold = eta
        return self.merit()

    def _run(self, x0, maxiter):
        """One SLSQP run from x0 at the ROM's current eta: (x, objective, reason, iterations, results);
        reason 'stall' (results None) or SLSQP's message."""
        self.prob.x0 = np.array(x0, dtype=float)
        optimizer = SLSQP(self.prob, solver_options={'ftol': self.ftol, 'maxiter': maxiter},
                          turn_off_outputs=self.turn_off_outputs)
        # the iterate of the last gradient, whether the ROM's solution was accepted there, and the
        # trial points evaluated since, with the same
        state = dict(iterate=None, iterate_rom=False, trials=[], gradients=0)
        windtunnel = self.windtunnel

        def evaluate(function):
            def wrapped(x):
                new = state['iterate'] is not None and not np.array_equal(x, state['iterate']) and \
                    not any(np.array_equal(x, t) for t, _ in state['trials'])
                if new and state['iterate_rom'] and state['trials'] and state['trials'][-1][1]:
                    raise LineSearchStall(state['iterate'])
                value = function(x)
                if new:
                    state['trials'].append((np.array(x, dtype=float), bool(windtunnel.last_solve_rom_accepted)))
                return value
            return wrapped

        def differentiate(function):
            def wrapped(x):
                value = function(x)
                if state['iterate'] is None or not np.array_equal(x, state['iterate']):
                    state['gradients'] += 1
                # the gradient's (CSDL evaluates the model at x first) or else that of x's evaluation
                state['iterate'], state['trials'] = np.array(x, dtype=float), []
                state['iterate_rom'] = bool(windtunnel.last_solve_rom_accepted)
                return value
            return wrapped

        optimizer.obj = evaluate(optimizer.obj)
        optimizer.grad = differentiate(optimizer.grad)
        if getattr(self.prob, "constrained", False):
            optimizer.con_in = evaluate(optimizer.con_in)
        try:
            results = optimizer.solve()
        except LineSearchStall as stall:
            # the last iterate's objective is that of its gradient evaluation
            return stall.x_restart, None, "stall", state['gradients'], None
        if self.windtunnel.mesh.mesh.comm.Get_rank() == 0:
            optimizer.print_results()
        return np.array(results['x'], dtype=float), float(results['fun']), str(results['message']), \
            int(results['nit']), results

    def solve(self):
        """Run the stages; returns the final design (scaled design variables), also in self.x."""
        comm = self.windtunnel.mesh.mesh.comm
        x = np.array(self.prob.x0, dtype=float)
        stage, iterations, stage_merit, n_runs, stalls = 0, 0, None, 0, 0
        # (merit, x) of the stage's best restart point (all full-order evaluations)
        best = None
        while True:
            eta = self.eta_schedule[stage]
            self.rom.eta_threshold = eta
            merit0 = self._full_order_start(x)
            eval_idx = self.windtunnel.eval_idx - 1
            if not np.isfinite(merit0):
                self.printer("staged optimization: no finite full-order merit at the restart point; stopping")
                self._record(stage, eta, x, merit0, eval_idx, "non-finite merit", 0)
                break
            restart = self.runs and self.runs[-1]['reason'] == "stall" and self.runs[-1]['stage'] == stage
            if best is None or merit0 < best[0]:
                best = (merit0, np.array(x, dtype=float))
            if restart and (merit0 > stage_merit - self.ftol or stalls > self.max_stall_restarts):
                if merit0 > stage_merit - self.ftol:
                    self.printer("staged optimization: restart point merit {:.8f}, not below the stage's previous "
                                 "restart point's {:.8f} by ftol: the ROM makes no progress at eta {:.1e}".format(
                                     merit0, stage_merit, eta))
                    reason = "no progress"
                else:
                    self.printer("staged optimization: {} restarts at eta {:.1e}: the ROM is not accurate enough "
                                 "there".format(stalls, eta))
                    reason = "restarts"
                self._record(stage, eta, x, merit0, eval_idx, reason, 0)
                if not np.array_equal(best[1], x):
                    self.printer("staged optimization: continuing from the stage's best restart point (merit "
                                 "{:.8f})".format(best[0]))
                    x = best[1]
                    if stage + 1 == len(self.eta_schedule):
                        # the final design's full-order values (self.fun below)
                        self._full_order_start(x)
                if stage + 1 == len(self.eta_schedule):
                    break
                stage, stage_merit, stalls, best = stage + 1, None, 0, None
                continue
            if n_runs >= self.max_runs or iterations >= self.maxiter:
                self._record(stage, eta, x, merit0, eval_idx, "budget", 0)
                self.printer("staged optimization: run or iteration budget used up")
                break
            stage_merit = merit0
            self.printer("staged optimization: run {}, stage {} of {}, eta_threshold {:.1e}, from a full-order "
                         "evaluation (eval {}) with merit {:.8f}; {} major iterations left".format(
                             n_runs + 1, stage + 1, len(self.eta_schedule), eta, eval_idx, merit0,
                             self.maxiter - iterations))
            x_end, fun, reason, nit, results = self._run(x, self.maxiter - iterations)
            iterations += nit
            n_runs += 1
            self._record(stage, eta, x, merit0, eval_idx, reason, nit, results)
            self.printer("staged optimization: run {} ended after {} major iterations: {}".format(
                n_runs, nit, reason))
            x = x_end
            if fun is not None:
                self.fun = fun
            if reason == "stall":
                stalls += 1
                continue
            if stage + 1 == len(self.eta_schedule) or iterations >= self.maxiter:
                break
            stage, stage_merit, stalls, best = stage + 1, None, 0, None
        self.x = x
        # the final design's values: the last run's, or (stopped at a restart point) its full-order ones
        if self.fun is None or self.runs[-1]['reason'] in ("no progress", "restarts", "budget", "non-finite merit"):
            self.fun = float(np.ravel(self.prob.f_s)[0])
        comm.Barrier()
        return x

    def _record(self, stage, eta, x0, merit0, eval_idx, reason, iterations, results=None):
        run = dict(stage=stage, eta=eta, x0=np.array(x0, dtype=float), merit0=merit0, eval_idx=eval_idx,
                   reason=reason, iterations=iterations, results=results)
        self.runs.append(run)
        if self.windtunnel.mesh.mesh.comm.Get_rank() == 0:
            self.windtunnel.data_store.optimization_runs += [
                (eval_idx, stage, eta, float(merit0), reason, int(iterations))]

    def summary(self):
        """One line per run (and per stage that ended at a restart point without a run)."""
        return "\n".join("{}: stage {}, eta {:.1e}, start eval {} (merit {:.8f}), {} iterations, {}".format(
            i + 1, r['stage'] + 1, r['eta'], r['eval_idx'], r['merit0'], r['iterations'], r['reason'])
            for i, r in enumerate(self.runs))
