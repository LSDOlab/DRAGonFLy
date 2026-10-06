"""
p = 0 RANS manufactured-solution convergence: the L2 error of the whole
discretization (MUSCL inviscid flux, reconstructed viscous facet gradients,
SA-neg sources) falls at rate 1, the order of piecewise constants.
Measured rates on 8..32: 0.97-1.02 for laminar NS and SA-neg.
"""
import numpy as np
import pytest

from dragonfly_sim.core.turbulence_models import Laminar, SpalartAllmarasNeg

from cases import mms_rans_model, l2_error


def solve_mms(closure, n):
    model, U_e = mms_rans_model(closure, n=n)
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
