"""
The Barter-Darmofal artificial viscosity (core/shock_capturing.py): dormant
in a uniform stream, driven by pressure jumps, differentiable (Taylor test of
the exact Jacobian), converging with the flow, and carried exactly through
the derived-field chain rule.
"""
import numpy as np
import pytest
import dolfinx
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc

from cases import euler_channel_model, seed_mean_flow, FREESTREAM


def _residual(model):
    model.update_derived_fields()
    b = dolfinx.fem.petsc.create_vector(dolfinx.fem.extract_function_spaces(model.F_physical_form))
    with b.localForm() as loc:
        loc.set(0.0)
    dolfinx.fem.petsc.assemble_vector(b, model.F_physical_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


def _owned(model, vec):
    V = model.functionspaces["V"]
    return vec.getArray()[:V.dofmap.index_map.size_local * model.n_state].reshape(-1, model.n_state)


@pytest.mark.verification
def test_uniform_stream_leaves_the_viscosity_dormant():
    model = euler_channel_model(n=6, bc_dict=dict(FREESTREAM, alpha=0.0), shock_capturing=True)
    assert model.n_state == 5 and model.aux_variables == ("sigma",)
    r = _residual(model)
    assert r.norm() < 1e-13


@pytest.mark.verification
def test_pressure_jumps_drive_the_viscosity_source():
    model = euler_channel_model(n=8, shock_capturing=True)
    seed_mean_flow(model)                        # smooth pressure variations, sigma = 0
    rows = _owned(model, _residual(model))
    sigma_rows = rows[:, model.shock_capturing.index]
    # at sigma = 0 the sigma equation's residual is minus its (non-negative) source
    assert np.all(sigma_rows <= 1e-15)
    assert MPI.COMM_WORLD.allreduce(float(sigma_rows.min()), op=MPI.MIN) < -1e-4


@pytest.mark.verification
def test_jacobian_passes_a_taylor_test():
    model = euler_channel_model(n=6, shock_capturing=True)
    seed_mean_flow(model)
    V = model.functionspaces["V"]
    k = model.shock_capturing.index
    arr = model.u_vec.x.array.reshape(-1, model.n_state)
    coords = V.tabulate_dof_coordinates()
    arr[:, k] = 0.3 + 0.2 * np.sin(5 * coords[:, 0])     # a nonzero viscosity field
    model.u_vec.x.scatter_forward()
    u0 = model.u_vec.x.array.copy()
    r0 = _residual(model).copy()
    J = model.assemble_dRdu_mat()
    # a smooth, partition-independent direction: HLL is only piecewise smooth
    # (min/max of wave speeds), so a rough one can cross a kink
    w = J.createVecRight()
    c = coords[:w.getLocalSize() // model.n_state]
    w.setArray(np.stack([np.sin(2 * c[:, 0] + 3 * c[:, 1] + k) for k in range(model.n_state)], axis=1).ravel())
    Jw = J.createVecLeft()
    J.mult(w, Jw)
    errors = []
    for eps in (1e-4, 5e-5):
        model.u_vec.x.petsc_vec.waxpy(eps, w, PETSc.Vec().createWithArray(u0[:w.getLocalSize()]))
        model.u_vec.x.scatter_forward()
        r = _residual(model)
        r.axpy(-1.0, r0)
        r.axpy(-eps, Jw)
        errors.append(r.norm())
        model.u_vec.x.array[:] = u0
    # second-order remainder
    assert errors[1] < 0.3 * errors[0], errors


@pytest.mark.verification
def test_ptc_converges_with_the_viscosity():
    model = euler_channel_model(n=10, bc_dict=dict(FREESTREAM, alpha=np.radians(5.0)),
                                shock_capturing=True)
    model.report_positivity_limiter = False
    model.use_ptc, model.fom_newton_inner_max_it = True, 1
    model.max_newton_iterations, model.residual_conv_limit = 80, 1e-11
    model.set_up_solver()
    model.solve_system()
    assert model.solver_converged
    sigma = _owned(model, model.u_vec.x.petsc_vec)[:, model.shock_capturing.index]
    assert np.all(sigma > -1e-12) and np.all(sigma < 1.0)


@pytest.mark.verification
def test_alpha_derivative_through_the_reconstruction_matches_finite_differences():
    """MUSCL Euler with the viscosity: dR/dalpha (total, through the
    Green-Gauss gradient the Ducros switch reads) against central differences."""
    from dragonfly_sim.core.mesh_manager import Mesh
    from dragonfly_sim.core.MUSCL_Euler_model import MUSCLEulerModel
    from cases import jittered_square, channel_tags, FARFIELD_TAG, WALL_TAG
    mesh_obj = Mesh(jittered_square(6), lambda pts: np.zeros(pts.shape[1], dtype=bool))
    ff, wall = channel_tags(mesh_obj)
    mesh_obj.tag_boundary_facets([(ff, FARFIELD_TAG), (wall, WALL_TAG)])
    meta = {'quadrature_degree': 1}
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=meta)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(measure_metadata=meta)
    model = MUSCLEulerModel(mesh_obj, shock_capturing=True)
    alpha = np.radians(3.0)
    model.define_farfield_conditions(dict(FREESTREAM, alpha=alpha))
    model.define_elements_functionspaces()
    model.define_trial_testfunctions()
    model.compute_initial_conditions_from_farfield()
    model.interpolate_solution_vector()
    model.define_farfield_bc(FARFIELD_TAG)
    model.define_slipwall_bc(WALL_TAG)
    model.compute_weakform()
    seed_mean_flow(model)
    arr = model.u_vec.x.array.reshape(-1, model.n_state)
    arr[:, model.shock_capturing.index] = 0.2
    model.u_vec.x.scatter_forward()
    model._ensure_residual_vector()
    dRda = model.assemble_dRdalpha_vec().copy()
    h = 1e-6
    res = []
    for a in (alpha + h, alpha - h):
        model.set_angle_of_attack(a)
        res.append(_residual(model).copy())
    model.set_angle_of_attack(alpha)
    fd = res[0]
    fd.axpy(-1.0, res[1])
    fd.scale(0.5 / h)
    fd.axpy(-1.0, dRda)
    assert fd.norm() < 1e-6 * max(dRda.norm(), 1.0), (fd.norm(), dRda.norm())
