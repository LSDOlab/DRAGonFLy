"""
The adjoint solves of DG_windtunnel_model reuse work across outputs: the
transposed Jacobian and the preconditioned KSP are built once per design and
shared by the adjoints of all outputs, and freed when the next flow solve starts.
The derivatives must not change.

A graph with three outputs (drag, lift and a moment) is evaluated at a sequence
of angles of attack; the derivatives are compared with central differences of
forward solves, and the number of builds with what reuse implies.
"""
import numpy as np
import pytest

import csdl_alpha as csdl

from test_moments import in_tmp_dir, make_windtunnel, BC  # noqa: F401 (fixture)

MOMENTS = [("pitch", (0.25, 0.), "pitch")]


pytestmark = [pytest.mark.verification]


def graph(wt):
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.add_name('D')
    out.D.set_as_objective()
    out.L.add_name('L')
    out.L.set_as_constraint(lower=-1e6)
    out.M_pitch.add_name('M_pitch')
    out.M_pitch.set_as_constraint(lower=-1e6)
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    return sim, alpha, [out.D, out.L, out.M_pitch]


def run_designs(wt, alphas):
    """Totals of D, L and M_pitch wrt alpha at each alpha, and the number of
    builds of the transposed operators."""
    sim, alpha, outputs = graph(wt)
    builds = []
    if wt.sim_model.has_condensed_jacobian:
        build = wt.sim_model.linearized_solver

        def counting_build(*args, **kwargs):
            builds.append(1)
            return build(*args, **kwargs)
        wt.sim_model.linearized_solver = counting_build
    totals = []
    for a in alphas:
        sim[alpha] = np.array([a])
        sim.run()
        t = sim.compute_totals()
        totals.append([float(np.asarray(t[o, alpha]).ravel()[0]) for o in outputs])
    return totals, builds


@pytest.mark.parametrize("kind, first_order", [("rans", False), ("euler", False), ("rans", True)])
def test_reused_adjoint_gives_the_same_derivatives(kind, first_order, in_tmp_dir):
    kw = {}
    if kind == "rans":
        from dragonfly_sim.core.RANS_model import CompressibleRANSModel
        kw = dict(model_class=CompressibleRANSModel, model_kwargs={'Re': 1e3})
    wt = make_windtunnel(moments=MOMENTS, **kw)
    if kind == "rans":
        wt.max_newton_iterations = 400
    wt.linear_solver_atol = 1e-9
    wt.linear_residual_accept_limit = 1e-8
    wt.adjoint_first_order_guess = first_order
    alphas = np.radians([2.0, 2.05, 2.1])
    totals, builds = run_designs(wt, alphas)

    if kind == "rans":
        # one build of the transposed Jacobian and preconditioner per design, not per output
        assert len(builds) == 3

    if first_order:
        # every adjoint started from a converged first-order solution, and the exact solves
        # (which gave the totals checked below) ran on the restored operators
        guess = wt._last_first_order_guess
        assert guess is not None and guess["used"] and np.isfinite(guess["ratio"]) and guess["iterations"] > 0, guess
    else:
        assert wt._last_first_order_guess is None
    if kind == "rans":
        # the operators of the last design stay cached until the next forward solve
        assert wt._adjoint_cache is not None
        wt.solve_forward(alphas[0])
        assert wt._adjoint_cache is None

    # and the totals are right: central differences of plain forward solves
    h = 1e-5
    for a, tot in zip(alphas, totals):
        Mp, Mm = (wt.solve_forward(a + s*h)[2] for s in (1, -1))
        fd = [(Mp[k] - Mm[k]) / (2*h) for k in ("D", "L")]
        fd.append((Mp["moments"]["pitch"]["M"] - Mm["moments"]["pitch"]["M"]) / (2*h))
        for name, f, d in zip(("D", "L", "M_pitch"), fd, tot):
            assert abs(f - d) < 1e-4 * abs(f), (kind, name, a, f, d)
