"""
The quantities tests/test_euler_regression.py compares against
tests/data/euler_reference.npz. Every array is rank-independent: per-cell
values are ordered by original (input-mesh) cell index, and matrices enter
only through their products with fixed smooth fields and their norms.
"""
import numpy as np
import dolfinx
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc

from cases import euler_channel_model, seed_mean_flow, gather_by_original_cell


def _smooth_state_field(V):
    """A fixed smooth vector field on V (one value per dof)."""
    c = V.tabulate_dof_coordinates()
    bs = V.dofmap.index_map_bs
    x, y = c[:, 0], c[:, 1]
    w = np.stack([np.sin(2 * x + 3 * y + k) for k in range(bs)], axis=1)
    return w.ravel()


def _smooth_mesh_field(Vx):
    c = Vx.tabulate_dof_coordinates()
    bs = Vx.dofmap.index_map_bs
    x, y = c[:, 0], c[:, 1]
    w = np.stack([np.cos(3 * x - y + 0.5 * k) * x * (1 - x) for k in range(bs)], axis=1)
    return w.ravel()


def _vec_by_cell(V, vec):
    bs = V.dofmap.index_map_bs
    n_own = V.dofmap.index_map.size_local
    return gather_by_original_cell(V, vec.getArray()[:n_own * bs].reshape(n_own, bs))


def euler_reference_quantities(comm=None):
    comm = comm if comm is not None else MPI.COMM_WORLD
    model = euler_channel_model(n=8, comm=comm)
    seed_mean_flow(model)
    V = model.functionspaces["V"]
    out = {}

    # residual
    b = dolfinx.fem.petsc.create_vector(dolfinx.fem.extract_function_spaces(model.F_physical_form))
    with b.localForm() as loc:
        loc.set(0.0)
    dolfinx.fem.petsc.assemble_vector(b, model.F_physical_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    out["residual"] = _vec_by_cell(V, b)

    # Jacobian: norm and action on a smooth field
    A = model.assemble_dRdu_mat()
    w = A.createVecRight()
    n_loc = w.getLocalSize()
    w.setArray(_smooth_state_field(V)[:n_loc])
    Aw = A.createVecLeft()
    A.mult(w, Aw)
    out["jacobian_norm"] = np.array([A.norm(PETSc.NormType.FROBENIUS)])
    out["jacobian_action"] = _vec_by_cell(V, Aw)

    # dR/dalpha
    out["dRdalpha"] = _vec_by_cell(V, model.assemble_dRdalpha_vec())

    # dR/dx_mesh: norm and action on a smooth mesh perturbation
    Vx = model.mesh_obj.coordinate_space
    D = model.compute_dRdxmesh_mat(model.u_vec.x.petsc_vec.getArray().copy(), Vx)
    dx = D.createVecRight()
    dx.setArray(_smooth_mesh_field(Vx)[:dx.getLocalSize()])
    Ddx = D.createVecLeft()
    D.mult(dx, Ddx)
    out["dRdx_norm"] = np.array([D.norm(PETSc.NormType.FROBENIUS)])
    out["dRdx_action"] = _vec_by_cell(V, Ddx)

    # three Newton steps of the release solve from a perturbed state
    seed_mean_flow(model, amplitude=0.3)
    model.report_positivity_limiter = False
    model.fom_newton_inner_max_it = 1
    model.max_newton_iterations = 1
    model.residual_conv_limit = 1e-300
    model.set_up_solver()
    hist = []
    for _ in range(3):
        hist.append(model.solve_system().norm())
    out["newton_history"] = np.array(hist)
    out["newton_state"] = gather_by_original_cell(
        V, model.u_vec.x.array[:V.dofmap.index_map.size_local * V.dofmap.index_map_bs].reshape(
            -1, V.dofmap.index_map_bs))
    return out
