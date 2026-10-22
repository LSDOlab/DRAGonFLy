"""
The quantities tests/test_euler_regression.py compares against
tests/data/euler_reference.npz. Every array is rank-independent: per-cell
values are ordered by original (input-mesh) cell index, and matrices enter
only through their products with fixed smooth fields and their norms.

The reference was generated with the former conservative unknowns U = (rho,
rho u, rho E) in the far-field dict's units (rho_inf = p_inf = 1). The model
now solves for V = (p', u, T) scaled with the free stream
(core/materials.py) and weights the energy row, which is a change of
unknowns and a constant row scaling of the same discrete equations:

    R_ref(U(V)) = D_R R(V),   D_R = (u_inf, u_inf^2 (per direction), u_inf^3 c_p)

with u_inf the free-stream speed in the dict's units and c_p the solver's.
So the residual, dR/dalpha and dR/dx (at a fixed state) compare after the
row scaling, and the Jacobian through the chain rule J_ref = D_R J (dV/dU),
with dV/dU by complex step. Everything is mapped onto the reference here.
"""
import numpy as np
import dolfinx
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc

from cases import (euler_channel_model, seed_mean_flow, gather_by_original_cell,
                   primitive_from_physical, legacy_conservative)


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


def row_scaling(model):
    """D_R per state component (module docstring)."""
    u = model.material.dimensional['speed']
    return np.array([u] + [u ** 2] * model.dimensions + [u ** 3 * float(model.material.cp.value)])


def primitive_from_conservative(model, U):
    """V of conservative rows U (dict units); complex-step safe."""
    g = model.material.gamma
    d = model.dimensions
    rho, m, E = U[:, 0], U[:, 1:1 + d], U[:, 1 + d]
    u = m / rho[:, None]
    p = (g - 1.0) * (E - 0.5 * np.sum(m * m, axis=1) / rho)
    return primitive_from_physical(model, rho, u, p)


def dV_dU(model, V_rows):
    """dV/dU per cell, (n, bs, bs), at the states V_rows, by complex step."""
    U = legacy_conservative(model, V_rows)
    n, bs = U.shape
    out = np.empty((n, bs, bs))
    h = 1e-30
    for j in range(bs):
        Uc = U.astype(complex)
        Uc[:, j] += 1j * h
        out[:, :, j] = primitive_from_conservative(model, Uc).imag / h
    return out


def _owned_rows(V, arr):
    bs = V.dofmap.index_map_bs
    return arr[:V.dofmap.index_map.size_local * bs].reshape(-1, bs)


def euler_reference_quantities(comm=None):
    comm = comm if comm is not None else MPI.COMM_WORLD
    model = euler_channel_model(n=8, comm=comm)
    seed_mean_flow(model)
    V = model.functionspaces["V"]
    D_R = row_scaling(model)
    out = {}

    # residual
    b = dolfinx.fem.petsc.create_vector(dolfinx.fem.extract_function_spaces(model.F_physical_form))
    with b.localForm() as loc:
        loc.set(0.0)
    dolfinx.fem.petsc.assemble_vector(b, model.F_physical_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    out["residual"] = _vec_by_cell(V, b) * D_R

    # Jacobian: J_ref = D_R J B with B = dV/dU (block diagonal), its norm and
    # its action on a smooth field
    A = model.assemble_dRdu_mat()
    B_blocks = dV_dU(model, _owned_rows(V, model.u_vec.x.array))
    w = A.createVecRight()
    n_loc = w.getLocalSize()
    w_rows = _smooth_state_field(V)[:n_loc].reshape(-1, model.n_state)
    w.setArray(np.einsum("nij,nj->ni", B_blocks, w_rows).ravel())
    Aw = A.createVecLeft()
    A.mult(w, Aw)
    out["jacobian_action"] = _vec_by_cell(V, Aw) * D_R
    A_aij = A.convert("aij", PETSc.Mat())
    B = A_aij.duplicate()
    B.zeroEntries()
    r0 = A_aij.getOwnershipRange()[0]
    bs = model.n_state
    for k, blk in enumerate(B_blocks):
        rows = np.arange(r0 + k * bs, r0 + (k + 1) * bs, dtype=PETSc.IntType)
        B.setValues(rows, rows, blk.ravel())
    B.assemble()
    J_ref = A_aij.matMult(B)
    scale = J_ref.createVecLeft()
    scale.setArray(np.tile(D_R, scale.getLocalSize() // bs))
    J_ref.diagonalScale(L=scale)
    out["jacobian_norm"] = np.array([J_ref.norm(PETSc.NormType.FROBENIUS)])

    # dR/dalpha
    out["dRdalpha"] = _vec_by_cell(V, model.assemble_dRdalpha_vec()) * D_R

    # dR/dx_mesh: norm and action on a smooth mesh perturbation
    Vx = model.mesh_obj.coordinate_space
    D = model.compute_dRdxmesh_mat(model.u_vec.x.petsc_vec.getArray().copy(), Vx)
    dx = D.createVecRight()
    dx.setArray(_smooth_mesh_field(Vx)[:dx.getLocalSize()])
    Ddx = D.createVecLeft()
    D.mult(dx, Ddx)
    out["dRdx_action"] = _vec_by_cell(V, Ddx) * D_R
    D.diagonalScale(L=scale)
    out["dRdx_norm"] = np.array([D.norm(PETSc.NormType.FROBENIUS)])

    # the converged Newton solution from a perturbed state (the iterates
    # themselves depend on the unknowns; the solution does not)
    seed_mean_flow(model, amplitude=0.3)
    model.report_positivity_limiter = False
    model.fom_newton_inner_max_it = 1
    model.max_newton_iterations = 1
    model.residual_conv_limit = 1e-300
    model.set_up_solver()
    hist = [model.solve_system().norm() for _ in range(12)]
    out["converged_history"] = np.array(hist)
    out["converged_state"] = legacy_conservative(
        model, gather_by_original_cell(V, _owned_rows(V, model.u_vec.x.array)))
    return out
