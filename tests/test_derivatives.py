"""
Total derivatives for the adjoint (CSDL): dR/dalpha, dR/dx_mesh and a force
functional's dD/dU and dD/dx, through the derived fields (Green-Gauss
gradient, cell centres, wall distance), against central finite differences
on a small SA-neg case with a no-slip wall and a characteristic far field.
"""
import numpy as np
import pytest
import ufl
import dolfinx
from mpi4py import MPI

from dragonfly_sim.core.turbulence_models import SpalartAllmarasNeg

from cases import rans_square_model, manufactured_state, WALL_TAG

ALPHA = np.radians(3.0)


@pytest.fixture(scope="module")
def case():
    model = rans_square_model(SpalartAllmarasNeg(), n=5, Re=200.0, wall_bottom=True, jitter=0.05)
    model.compute_initial_conditions_from_inlet()
    model.interpolate_solution_vector()
    model.set_angle_of_attack(ALPHA)
    model.define_farfield_bc(1)
    model.define_wall_bc(WALL_TAG)
    model.compute_weakform()
    U_e, _ = manufactured_state(model)
    model.interpolate_solution_vector(U_e)
    rng = np.random.default_rng(5)
    model.u_vec.x.array[:] *= 1.0 + 0.01 * rng.standard_normal(model.u_vec.x.array.size)
    model.u_vec.x.scatter_forward()
    model._ensure_residual_vector()
    mo = model.mesh_obj
    X = mo.mesh.geometry.x[:mo.n_owned_nodes, :2]
    # The x-motion vanishes on the left and right sides: a node there sits
    # straight above the wall's free end vertex, where the distance has a
    # kink (nearest point switching between segment and vertex).
    dX = np.stack([np.sin(3 * X[:, 0] + X[:, 1]) * X[:, 0] * (1 - X[:, 0]),
                   np.cos(2 * X[:, 0] - X[:, 1])], axis=1) * (X[:, 1] * (1 - X[:, 1]) + 0.3)[:, None]
    return model, model.u_vec.x.petsc_vec.getArray().copy(), dX


def _residual(model):
    model._assemble_physical_residual()
    return model.residual_vec_physical.getArray().copy()


def _global_norm(a, comm):
    return np.sqrt(comm.allreduce(float(a @ a), MPI.SUM))


def test_dR_dalpha(case):
    model, u0, _ = case
    comm = model.mesh_obj.mesh.comm
    eps = 1e-6
    model.set_angle_of_attack(ALPHA + eps)
    Rp = _residual(model)
    model.set_angle_of_attack(ALPHA - eps)
    Rm = _residual(model)
    model.set_angle_of_attack(ALPHA)
    fd = (Rp - Rm) / (2 * eps)
    an = model.assemble_dRdalpha_vec().getArray()[:fd.size]
    assert _global_norm(fd - an, comm) / _global_norm(an, comm) < 1e-7


def test_dR_dx(case):
    model, u0, dX = case
    mo = model.mesh_obj
    comm = mo.mesh.comm
    eps = 1e-6
    mo.apply_node_motions(eps * dX)
    Rp = _residual(model)
    mo.apply_node_motions(-eps * dX)
    Rm = _residual(model)
    mo.reset_nodes()
    fd = (Rp - Rm) / (2 * eps)
    D = model.compute_dRdxmesh_mat(u0, mo.coordinate_space)
    v = D.createVecRight()
    v.setArray((mo.dof_to_node_map @ dX).ravel())
    y = D.createVecLeft()
    D.mult(v, y)
    an = y.getArray()[:fd.size]
    assert _global_norm(fd - an, comm) / _global_norm(an, comm) < 1e-7
    # the transpose (what the adjoint uses) is consistent with it
    w = D.createVecLeft()
    w.setArray(np.random.default_rng(1).standard_normal(w.getLocalSize()))
    z = D.createVecRight()
    D.multTranspose(w, z)
    assert abs(w.dot(y) - z.dot(v)) <= 1e-12 * abs(w.dot(y))


def _drag(model):
    n = model.mesh_obj.n
    return ufl.dot(model.wall_force_density(model.u_vec, n), ufl.as_vector([1.0, 0.0])) \
        * model.forms.ds(WALL_TAG)


def _J(model):
    model.update_derived_fields()
    return model.mesh_obj.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(_drag(model))), MPI.SUM)


def test_functional_state_and_mesh_derivatives(case):
    model, u0, dX = case
    mo = model.mesh_obj
    comm = mo.mesh.comm
    g = model.state_derivative(_drag(model)).getArray().copy()
    du = np.random.default_rng(7).standard_normal(u0.size) * np.abs(u0)
    eps = 1e-6
    model.set_u_vec(u0 + eps * du)
    Jp = _J(model)
    model.set_u_vec(u0 - eps * du)
    Jm = _J(model)
    model.set_u_vec(u0)
    an = comm.allreduce(float(g @ du), MPI.SUM)
    assert abs((Jp - Jm) / (2 * eps) - an) < 1e-7 * abs(an)

    gx = model.mesh_derivative(_drag(model)).getArray().copy()
    mo.apply_node_motions(eps * dX)
    Jp = _J(model)
    mo.apply_node_motions(-eps * dX)
    Jm = _J(model)
    mo.reset_nodes()
    an = comm.allreduce(float(gx @ (mo.dof_to_node_map @ dX).ravel()), MPI.SUM)
    assert abs((Jp - Jm) / (2 * eps) - an) < 1e-7 * abs(an)


def test_wall_distance_and_its_mesh_derivative(case):
    model, _, dX = case
    wd = model.wall_distance
    mo = model.mesh_obj
    # bottom wall at y = 0 on the undeformed mesh: d = y
    y = wd.function.function_space.tabulate_dof_coordinates()[:, 1]
    np.testing.assert_allclose(wd.function.x.array, y, atol=1e-14)
    eps = 1e-6
    n_own = wd.function.function_space.dofmap.index_map.size_local
    mo.apply_node_motions(eps * dX)
    dp = wd.function.x.array[:n_own].copy()
    mo.apply_node_motions(-eps * dX)
    dm = wd.function.x.array[:n_own].copy()
    mo.reset_nodes()
    D = wd.mesh_jacobian()
    v = D.createVecRight()
    v.setArray((mo.dof_to_node_map @ dX).ravel())
    out = D.createVecLeft()
    D.mult(v, out)
    # Compare where the nearest wall point is strictly inside a facet: a node
    # straight above a wall vertex (the unjittered boundary nodes) sits on a
    # kink of d, where the derivative is one-sided.
    t = wd._nearest[2][:n_own]
    inside = (t > 1e-3) & (t < 1 - 1e-3)
    assert MPI.COMM_WORLD.allreduce(int(inside.sum()), MPI.SUM) > 10
    np.testing.assert_allclose(out.getArray()[inside], ((dp - dm) / (2 * eps))[inside], rtol=1e-7, atol=1e-10)


def test_functionals_differing_only_in_a_constant_get_their_own_derivative(case):
    """Lift and drag have the same UFL signature (they differ only in the
    direction Constant); the compiled-form cache must not hand one the
    other's derivative."""
    model, u0, _ = case
    msh = model.mesh_obj.mesh
    n = model.mesh_obj.n
    f = model.wall_force_density(model.u_vec, n)
    e_x = dolfinx.fem.Constant(msh, np.array([1.0, 0.0]))
    e_y = dolfinx.fem.Constant(msh, np.array([0.0, 1.0]))
    forms = [ufl.dot(f, e) * model.forms.ds(WALL_TAG) for e in (e_x, e_y)]
    assert forms[0].signature() == forms[1].signature()
    grads = [model.state_derivative(J).getArray().copy() for J in forms]
    comm = msh.comm
    du = np.random.default_rng(11).standard_normal(u0.size) * np.abs(u0)
    eps = 1e-6
    for J, g in zip(forms, grads):
        vals = []
        for s in (1.0, -1.0):
            model.set_u_vec(u0 + s * eps * du)
            model.update_derived_fields()
            vals.append(comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(J)), MPI.SUM))
        model.set_u_vec(u0)
        an = comm.allreduce(float(g @ du), MPI.SUM)
        assert abs((vals[0] - vals[1]) / (2 * eps) - an) < 1e-7 * abs(an)
