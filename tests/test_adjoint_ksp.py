"""
The sweep of Krylov methods for the adjoint (examples/benchmark_adjoint_ksp.py) on a small 3D RANS flow:
every method that PETSc has converges to the same adjoint solutions (true residual of every solve), the
options reach the KSP, and a method the PETSc build lacks (HPDDM's GCRO-DR in the conda build) is reported
as unavailable instead of aborting.
"""
import importlib.util
import os

import pytest
from petsc4py import PETSc

from test_adjoint_speedups import solved_windtunnel, in_tmp_dir  # noqa: F401 (fixture)

SPEC = importlib.util.spec_from_file_location(
    "benchmark_adjoint_ksp", os.path.join(os.path.dirname(__file__), os.pardir, "examples", "benchmark_adjoint_ksp.py"))
ksp_sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ksp_sweep)


def test_ksp_variants_solve_the_same_adjoints(in_tmp_dir):
    from dragonfly_sim.core.postprocessor import DG_postprocessor
    wt, u = solved_windtunnel(3)
    # the first variants are PETSc's own; the HPDDM ones may be unavailable
    results = ksp_sweep.sweep(wt, wt.postprocessor, u, matrix_free=False)
    by_name = {r["name"]: r for r in results}
    assert len(by_name) == len(ksp_sweep.VARIANTS)
    available = [r for r in results if "unavailable" not in r]
    assert len(available) >= 6                      # fgmres x3, dgmres x2, lgmres
    n_outputs = 2 + len(wt.postprocessor.moment_specs)
    for r in available:
        assert len(r["solves"]) == n_outputs
        for s in r["solves"]:
            assert s["reason"] > 0, (r["name"], s)
            assert s["true_rel_residual"] < 1e-4, (r["name"], s)
    # a method the PETSc build lacks is reported, not raised
    for r in results:
        if r["name"].startswith("hpddm") and "unavailable" in r:
            assert r["unavailable"].startswith("PETSc error")


def test_ksp_options_reach_the_adjoint_ksp(in_tmp_dir):
    """The options override the model's own: the KSP type, and a restart short enough to matter."""
    wt, u = solved_windtunnel(3)
    wt.sim_model.set_u_vec(u)
    wt.sim_model.update_derived_fields()
    kw = dict(transpose=True, prefix="adj_", rtol=1e-12, atol=1e-9, max_it=1000, release_operands=True)
    iterations = {}
    for name, options in (("fgmres", {}), ("dgmres", {"ksp_type": "dgmres", "ksp_gmres_restart": 10,
                                                      "ksp_dgmres_eigen": 2, "ksp_dgmres_max_eigen": 4,
                                                      "ksp_pc_side": "right", "ksp_norm_type": "unpreconditioned"}),
                          ("fgmres short", {"ksp_gmres_restart": 8})):
        ksp = wt.sim_model.linearized_solver(ksp_options=options, **kw)
        rhs = ksp_sweep.adjoint_rhs(wt, wt.postprocessor, u)[0][1]
        A = ksp.getOperators()[0]
        b, x = A.createVecLeft(), A.createVecRight()
        b.setArray(rhs)
        ksp.solve(b, x)
        assert ksp.getType() == (options.get("ksp_type") or "fgmres")
        assert ksp.getConvergedReason() > 0
        iterations[name] = ksp.getIterationNumber()
        wt.sim_model.destroy_adjoint_operator(ksp)
    # a restart of 8 forces restarts in a solve of ~21 iterations, which changes the iteration count
    assert iterations["fgmres short"] != iterations["fgmres"] or iterations["dgmres"] != iterations["fgmres"], iterations
