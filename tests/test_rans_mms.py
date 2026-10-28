"""
RANS manufactured-solution convergence: the L2 error of the whole
discretization falls at the rate of the basis.

p = 0 (MUSCL inviscid flux, reconstructed viscous facet gradients, SA-neg
sources): rate 1, the order of piecewise constants. Measured rates on
8..32: 0.97-1.02 for laminar NS and SA-neg.

DG p = 1, 2 (symmetric interior penalty): velocity and temperature at about
p + 1 (measured on 8..16: 1.89-1.93 at p = 1, 3.03-3.08 at p = 2, both
closures); the pressure, carried by the upwind inviscid flux alone, at about
p + 1/4 (1.9 and 2.25).
"""
import numpy as np
import pytest

from dragonfly_sim.core.turbulence_models import Laminar, SpalartAllmarasNeg

from cases import mms_rans_model, l2_error


pytestmark = [pytest.mark.verification]


def solve_mms(closure, n, p=0):
    kw = dict(poly_order=p, quadrature_degree=2 * p + 2) if p else {}
    model, U_e = mms_rans_model(closure, n=n, **kw)
    model.interpolate_solution_vector(U_e)
    model.direct_linear_solver = True
    model.use_ptc = False
    model.residual_conv_limit = 1e-11
    model.max_newton_iterations = 20
    model.report_positivity_limiter = False
    model.set_up_solver()
    model.solve_system()
    assert model.solver_converged
    return l2_error(model, U_e)


@pytest.mark.parametrize("closure", [Laminar(), SpalartAllmarasNeg()], ids=["laminar", "sa-neg"])
def test_p0_mms_convergence(closure):
    errs = np.array([solve_mms(closure, n) for n in (8, 16, 32)])
    rates = np.log2(errs[:-1] / errs[1:])
    assert np.all(rates[-1] > 0.9), rates


@pytest.mark.parametrize("p", [1, 2])
def test_dg_mms_convergence(p):
    errs = np.array([solve_mms(Laminar(), n, p) for n in (4, 8, 16)])
    rates = np.log2(errs[-2] / errs[-1])
    assert np.all(rates[1:] > p + 0.6), rates     # velocity, temperature
    assert rates[0] > p + 0.1, rates               # pressure
