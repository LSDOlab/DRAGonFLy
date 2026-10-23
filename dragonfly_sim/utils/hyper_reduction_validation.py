"""
Diagnostics of the hyper-reduced Jacobians of core/hyper_reduction.py
(HyperReductionComparison(validate=True)), measured against a full assembly
at the model's current state. The project-sandbox deim_validation.py, for the
Jacobian: the residual is not hyper-reduced.

  restriction_max_error   max |sampled - full| over the sampled rows of A V
                          (DEIM) or the sampled entries of S A (MDEIM),
                          relative to the largest of them: the sampled
                          assembly is exact up to round-off (the kernels are
                          compiled with -ffast-math)
  jacobian_error          ||S A V - approximation||_F / ||S A V||_F
  reduced_jacobian_error  the same for the Gauss-Newton matrix (S A V)^T (S A V)
  projection_error        MDEIM: ||vec(S A) - Phi Phi^T vec(S A)|| / ||vec(S A)||,
                          how well the basis holds this Jacobian

Each costs a full Jacobian assembly and products with it; diagnostics only.
"""
import numpy as np
from mpi4py import MPI

from dragonfly_sim.core.hyper_reduction import gather_rows


def _norm(comm, local):
    return float(np.sqrt(comm.allreduce(float(np.sum(np.asarray(local)**2)), op=MPI.SUM)))


def _full_jacobian(model):
    """The steady Jacobian at the current state, assembled in full into the
    Newton problem's matrix (overwritten)."""
    model.set_up_solver()
    problem = model.problem
    with model.steady_residual():
        problem.J(model.u_vec.x.petsc_vec, problem._A)
    return problem._A


def validate(hr, model, V, S):
    """The diagnostics (module docstring) of a prepared DEIM or MDEIM `hr` at
    the model's current state: a dict. Collective."""
    comm = hr.comm
    S_local = None if S is None else S.getArray(readonly=True)
    A = _full_jacobian(model)
    AV = A.matMult(V)
    SAV = AV.getDenseArray().copy()
    rstart, _ = AV.getOwnershipRange()
    AV.destroy()
    if S_local is not None:
        SAV *= S_local[:, None]
    diag = {}

    if hr.name == 'deim':
        A_s = hr.rjac.assemble()
        AV_s = A_s.matMult(V)
        sampled = gather_rows(comm, AV_s.getDenseArray(), rstart, hr.indices)
        AV_s.destroy()
        full = gather_rows(comm, SAV, rstart, hr.indices)
        S_P = gather_rows(comm, np.ones(SAV.shape[0]) if S_local is None else S_local, rstart, hr.indices)
        sampled = S_P[:, None] * sampled
        approx = hr.scaled_jacobian_basis()
    else:
        values = hr.scaled_values(A, S_local)
        sampled = hr.sampled_values()
        start, _ = hr.template_vec.getOwnershipRange()
        full = gather_rows(comm, values, start, hr.indices)
        phi = hr.basis.getDenseArray()
        coeff = comm.allreduce(phi.T @ values, op=MPI.SUM)
        diag['projection_error'] = _norm(comm, values - phi @ coeff) / max(_norm(comm, values), 1e-300)
        approx = hr.scaled_jacobian_basis()

    scale = max(float(np.max(np.abs(full))) if full.size else 0.0, 1e-300)
    diag['restriction_max_error'] = float(np.max(np.abs(sampled - full))) / scale if full.size else 0.0
    ref = _norm(comm, SAV)
    diag['jacobian_error'] = _norm(comm, SAV - approx) / max(ref, 1e-300)
    H_full = comm.allreduce(SAV.T @ SAV, op=MPI.SUM)
    H_hr = comm.allreduce(approx.T @ approx, op=MPI.SUM)
    diag['reduced_jacobian_error'] = float(np.linalg.norm(H_hr - H_full) / max(np.linalg.norm(H_full), 1e-300))
    return diag
