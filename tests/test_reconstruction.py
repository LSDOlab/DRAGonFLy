"""
The p = 0 Green-Gauss reconstruction and its exact elimination from the
Newton system.
"""
import numpy as np
import pytest
import ufl
import dolfinx
import dolfinx.fem.petsc

from dragonfly_sim.core.turbulence_models import Laminar, SpalartAllmarasNeg

from cases import rans_square_model, mms_rans_model


def test_green_gauss_is_exact_for_a_linear_field():
    # Uniform quads: the facet average of a linear field is its value at the
    # facet midpoint, so the Green-Gauss sum is exact in every cell, boundary
    # cells included (their facets see the exact field as boundary state).
    model = rans_square_model(SpalartAllmarasNeg(), n=5, jitter=0.0)
    x = ufl.SpatialCoordinate(model.mesh_obj.mesh)
    slopes = np.array([[0.3, -0.2], [1.1, 0.4], [-0.5, 0.7], [2.0, -1.3], [0.05, 0.02]])
    U_lin = ufl.as_vector([1.0 + slopes[k, 0] * x[0] + slopes[k, 1] * x[1] for k in range(5)])
    model.add_boundary_condition(1, U_lin, "dirichlet")
    model.interpolate_solution_vector(U_lin)
    model.reconstruction.update_gradient()
    G = model.reconstruction.G.x.array.reshape(-1, 5, 2)
    np.testing.assert_allclose(G, np.broadcast_to(slopes, G.shape), atol=1e-12)


def _perturbed_mms(closure, limiter):
    model, U_e = mms_rans_model(closure, n=5, limiter=limiter, limiter_eps=1e-4)
    model.interpolate_solution_vector(U_e)
    rng = np.random.default_rng(3)
    model.u_vec.x.array[:] *= 1.0 + 0.01 * rng.standard_normal(model.u_vec.x.array.size)
    model.u_vec.x.scatter_forward()
    return model, rng


@pytest.mark.parametrize("closure, limiter", [(SpalartAllmarasNeg(), None),
                                              (SpalartAllmarasNeg(), "vanalbada"),
                                              (Laminar(), None)],
                         ids=["sa-unlimited", "sa-vanalbada", "laminar"])
def test_condensed_jacobian_matches_finite_difference(closure, limiter):
    model, rng = _perturbed_mms(closure, limiter)
    problem = model.reconstruction.make_problem(model.F, model.J)
    x = model.u_vec.x.petsc_vec
    problem.J(x, problem._A)
    A = problem._A
    du = x.duplicate()
    du.setArray(rng.standard_normal(du.getLocalSize()) * np.abs(x.getArray()))
    Jdu = A.createVecLeft()
    A.mult(du, Jdu)

    u0 = model.u_vec.x.array.copy()
    eps = 1e-6
    fd = []
    for s in (1.0, -1.0):
        x.axpy(s * eps, du)
        model.u_vec.x.scatter_forward()
        b = problem._b.duplicate()
        problem.F(x, b)
        fd.append(b)
        model.u_vec.x.array[:] = u0
        model.u_vec.x.scatter_forward()
    fd_vec = fd[0] - fd[1]
    fd_vec.scale(0.5 / eps)
    diff = fd_vec.copy()
    diff.axpy(-1.0, Jdu)
    assert diff.norm() / Jdu.norm() < 1e-6

    # the compact part alone (gradients frozen) is not the Jacobian
    A_uu = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(model.J))
    A_uu.assemble()
    A_uu.mult(du, Jdu)
    fd_vec.axpy(-1.0, Jdu)
    assert fd_vec.norm() / Jdu.norm() > 1e-3
