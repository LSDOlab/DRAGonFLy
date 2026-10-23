"""
core/hyper_reduction.py: the sampled Jacobian assembly (exact on its rows, for
the plain and the condensed-gradient models), DEIM and MDEIM, the
hyper-reduced LSPG solve and adjoint, and the comparison beside the ROM in
DG_windtunnel_model.

The flow cases are those of test_reduced_order_model.py. Every test is
communicator-agnostic (test_parallel.py runs this module on 3 ranks, where the
sampled cells' facets owned by another rank are the case that can go wrong).
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from dragonfly_sim.core.hyper_reduction import (RestrictedJacobian, DEIM, MDEIM, HyperReducedLSPG,
                                                HyperReducedLSPGAdjoint, HyperReductionComparison,
                                                greedy_deim_indices, design_weights)
from dragonfly_sim.core.reduced_order_model import ReducedOrderModel, LSPGSolver, LSPGAdjoint
from dragonfly_sim.core.MUSCL_Euler_model import MUSCLEulerModel
from dragonfly_sim.utils.pod_utils import petsc_dense_from_local_block

from test_reduced_order_model import in_tmp_dir, make_windtunnel, global_array, ALPHAS  # noqa: F401


def _windtunnel(kind, **kw):
    if kind == "muscl":
        return make_windtunnel("euler", model_class=MUSCLEulerModel, suffix="hr", **kw)
    return make_windtunnel(kind, suffix="hr", **kw)


def _steady_jacobian(m):
    m.set_up_solver()
    with m.steady_residual():
        m.problem.J(m.u_vec.x.petsc_vec, m.problem._A)
    return m.problem._A


def _random_basis(m, k, seed=3):
    """A random k-column dense Mat in the state layout (owned rows), the same on any partitioning."""
    x = m.u_vec.x.petsc_vec
    rstart, rend = x.getOwnershipRange()
    full = np.random.default_rng(seed).standard_normal((x.getSize(), k))
    return petsc_dense_from_local_block(m.mesh_obj.mesh.comm, full[rstart:rend].copy())


def _owned_rows_of_cells(m, every):
    """The owned rows of every `every`-th owned cell (local indices)."""
    msh = m.mesh_obj.mesh
    n_owned = msh.topology.index_map(msh.topology.dim).size_local
    rows = m.forms.dofs_for_cells(np.arange(0, n_owned, every, dtype=np.int32))
    return rows[rows < m.u_vec.x.petsc_vec.getLocalSize()]


def _row_values(A, rows_local):
    """{(local row, global col): value} of the rows `rows_local` of a Mat."""
    indptr, cols, data = A.getValuesCSR()
    out = {}
    for r in rows_local:
        for j in range(indptr[r], indptr[r + 1]):
            out[(int(r), int(cols[j]))] = data[j]
    return out


# ----------------------------------------------------------------------
# Sampled assembly
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["euler", "muscl", "rans"])
@pytest.mark.unit_test
def test_restricted_jacobian_rows_are_exact(in_tmp_dir, kind):
    """The sampled rows of the restricted Jacobian equal those of the Newton problem's steady
    Jacobian, on cells next to the boundary and to other ranks too; with a reconstructed
    gradient through the 2-ring closure. And it is a restriction: most rows are not computed."""
    wt = _windtunnel(kind)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    wt.solve_forward(np.radians(2.0))
    rows = _owned_rows_of_cells(m, 13)
    rj = RestrictedJacobian(m)
    assert rj.condensed == (kind != "euler")
    rj.bind(rows)
    m.update_derived_fields()
    A_s = rj.assemble()
    sampled = _row_values(A_s, rows)
    full = _row_values(_steady_jacobian(m), rows)
    scale = comm.allreduce(max((abs(v) for v in full.values()), default=0.0), op=MPI.MAX)
    worst = max((abs(sampled.get(key, 0.0) - v) for key, v in full.items()), default=0.0)
    worst = max([worst] + [abs(v) for key, v in sampled.items() if key not in full])
    assert comm.allreduce(worst, op=MPI.MAX) <= 1e-12 * scale
    if kind != "euler":
        assert rj.n_closure_cells > rj.n_cells
    # only the rows of the sampled cells' (one or, condensed, two) facet rings are assembled
    msh = m.mesh_obj.mesh
    ring = m.forms.cell_neighbours(m.forms.cells_for_dofs(rows))
    if kind != "euler":
        ring = m.forms.cell_neighbours(ring)
    allowed = m.forms.dofs_for_cells(ring)
    indptr = A_s.getValuesCSR()[0]
    assembled = np.flatnonzero(np.diff(indptr))
    assert np.all(np.isin(assembled, allowed))
    n_cells = msh.topology.index_map(msh.topology.dim).size_global
    assert comm.allreduce(int(ring.size), op=MPI.SUM) < n_cells
    rj.destroy()


@pytest.mark.unit_test
def test_greedy_points_do_not_depend_on_the_partitioning():
    comm = MPI.COMM_WORLD
    n, k = 97, 5
    full = np.random.default_rng(7).standard_normal((n, k))
    sizes = [n // comm.size + (r < n % comm.size) for r in range(comm.size)]
    rstart = sum(sizes[:comm.rank])
    local = full[rstart:rstart + sizes[comm.rank]]
    for n_points in (k, 2 * k + 1):
        distributed = greedy_deim_indices(comm, local, rstart, n_points=n_points)
        serial = greedy_deim_indices(MPI.COMM_SELF, full, 0, n_points=n_points)
        assert np.array_equal(distributed, serial)
        assert len(set(distributed.tolist())) == n_points


@pytest.mark.unit_test
def test_design_weights_follow_the_rom():
    settings = {'basis_mode': 'local_weighted', 'design_parameters': np.array([[0.1, 2.0], [0.3, 2.5]]),
                'design_weights': np.array([0.75, 0.25])}
    keys = np.array([[0.1, 2.0, 0.0], [0.1, 2.0, 1.0 + 3e-6], [0.3, 2.5, 0.0], [0.9, 2.5, 0.0]])
    assert design_weights(settings, keys) == pytest.approx([0.75, 0.75, 0.25, 0.0])
    assert design_weights({'basis_mode': 'global'}, keys) is None


# ----------------------------------------------------------------------
# DEIM, MDEIM and the hyper-reduced LSPG
# ----------------------------------------------------------------------
def _scaled_AV(m, V, S):
    A = _steady_jacobian(m)
    AV = A.matMult(V)
    SAV = AV.getDenseArray() * S.getArray(readonly=True)[:, None]
    AV.destroy()
    return SAV.copy()


@pytest.mark.parametrize("kind", ["euler", "rans"])
@pytest.mark.unit_test
def test_deim_and_mdeim_reproduce_a_jacobian_in_their_basis(in_tmp_dir, kind):
    """With this state's S A V (DEIM) or vec(S A) (MDEIM) in the snapshots, the hyper-reduced
    S A V is exact up to round-off, and so are the Gauss-Newton system and the adjoint, which
    then equal the plain LSPG solver's and LSPGAdjoint's."""
    rom = ReducedOrderModel(rb_size=3)
    wt = _windtunnel(kind, rom=rom)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    wt.solve_forward(np.radians(2.0))
    k = 3
    V = _random_basis(m, k)
    S = rom.residual_weight(m)
    SAV = _scaled_AV(m, V, S)
    ref = np.sqrt(comm.allreduce(float(np.sum(SAV**2)), op=MPI.SUM))

    deim = DEIM(comm, n_modes=k)
    vec = m.u_vec.x.petsc_vec.duplicate()
    for j in range(k):
        vec.array[:] = SAV[:, j]
        assert deim.add_snapshot(vec, [2.0, float(j)])
    mdeim = MDEIM(comm, n_modes=1)
    A = _steady_jacobian(m)
    assert mdeim.add_values(mdeim.scaled_values(A, S.getArray(readonly=True)), [2.0, 0.0])

    r = m.problem._b.duplicate()
    m.problem.F(m.u_vec.x.petsc_vec, r)
    Sr = r.getArray(readonly=True) * S.getArray(readonly=True)
    H_ref = comm.allreduce(SAV.T @ SAV, op=MPI.SUM)
    g_ref = comm.allreduce(SAV.T @ Sr, op=MPI.SUM)
    rhs = np.random.default_rng(comm.rank).standard_normal(SAV.shape[0])
    plain = LSPGAdjoint(V, None if S is None else S.copy(), comm, [2.0], m.u_vec.x.petsc_vec.getArray())
    psi_ref = plain.solve(m, rhs)

    for hr, layout in ((deim, m.u_vec.x.petsc_vec), (mdeim, mdeim.template_vec)):
        hr.build_basis(None, layout_vec=layout)
        hr.select_indices()
        m.update_derived_fields()
        hr.prepare(m, V, S)
        err = np.sqrt(comm.allreduce(float(np.sum((hr.scaled_jacobian_basis() - SAV)**2)), op=MPI.SUM))
        assert err <= 1e-9 * ref, hr.name
        H, g = hr.reduced_system(Sr)
        assert H == pytest.approx(H_ref, rel=1e-8, abs=1e-10 * np.abs(H_ref).max())
        assert g == pytest.approx(g_ref, rel=1e-7, abs=1e-9 * np.abs(g_ref).max())
        adj = HyperReducedLSPGAdjoint(hr, V, None if S is None else S.copy(), comm, [2.0],
                                      m.u_vec.x.petsc_vec.getArray())
        psi = adj.solve(m, rhs)
        diff = np.sqrt(comm.allreduce(float(np.sum((psi - psi_ref)**2)), op=MPI.SUM))
        assert diff <= 1e-7 * np.sqrt(comm.allreduce(float(psi_ref @ psi_ref), op=MPI.SUM)), hr.name
        adj.destroy()
        hr.destroy()
    plain.destroy()
    for obj in (vec, r, S, V):
        obj.destroy()


@pytest.mark.unit_test
def test_hyper_reduced_solver_takes_the_operator_system(in_tmp_dir):
    """HyperReducedLSPG is LSPGSolver with the operator's Gauss-Newton system."""
    class Exact:
        def __init__(self, model, V, S):
            self.m, self.V, self.S = model, V, S

        def reduced_system(self, Sr):
            SAV = _scaled_AV(self.m, self.V, self.S)
            comm = self.m.mesh_obj.mesh.comm
            return comm.allreduce(SAV.T @ SAV, op=MPI.SUM), comm.allreduce(SAV.T @ Sr, op=MPI.SUM)

    rom = ReducedOrderModel(rb_size=3)
    wt = _windtunnel("euler", rom=rom)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    wt.solve_forward(np.radians(2.0))
    V = _random_basis(m, 3)
    S = rom.residual_weight(m)
    plain = rom.make_solver(LSPGSolver, V, S, None, comm)
    hyper = rom.make_solver(HyperReducedLSPG, V, S, None, comm, operator=Exact(m, V, S))
    m.set_up_solver()
    with m.steady_residual():
        H0, g0, r0 = plain._reduced_system(m)
        H1, g1, r1 = hyper._reduced_system(m)
    assert r1 == r0
    assert H1 == pytest.approx(H0, rel=1e-12) and g1 == pytest.approx(g0, rel=1e-12, abs=1e-14)
    plain.destroy()
    hyper.destroy()


# ----------------------------------------------------------------------
# The comparison in DG_windtunnel_model
# ----------------------------------------------------------------------
def _run_sequence(wt, alphas=ALPHAS):
    out = []
    for a in alphas:
        out.append(wt.solve_forward(np.radians(a)))
        wt.eval_idx += 1                 # as solve_residual_equations does
    return out


@pytest.mark.parametrize("kind, eta_threshold", [("euler", 1e-9), ("rans", 1e-7)])
@pytest.mark.verification
def test_comparison_leaves_the_optimizer_unchanged(in_tmp_dir, request, kind, eta_threshold):
    """
    With the comparison, every solution handed back is the one without it, the ROM keeps its
    snapshots, and DEIM and MDEIM record a solve for every design the ROM solved once they hold
    two snapshots, in the documented layouts, with their phases, which assemble the sampled
    Jacobian and never the full one. MDEIM reproduces the ROM's solution closely.
    """
    from dragonfly_sim.utils.solver_profiling import PROFILER
    request.addfinalizer(lambda: (PROFILER.configure(enabled=False), PROFILER.reset()))
    reference = _run_sequence(_windtunnel(kind, rom=ReducedOrderModel(rb_size=5, eta_threshold=eta_threshold)))
    rom = ReducedOrderModel(rb_size=5, eta_threshold=eta_threshold)
    hr = HyperReductionComparison(deim_n_modes=8, mdeim_n_modes=4, validate=True)
    wt = _windtunnel(kind, rom=rom, hyper_reduction=hr, profile_phases=True)
    out = _run_sequence(wt)
    comm = MPI.COMM_WORLD
    for (u0, c0, f0), (u1, c1, f1) in zip(reference, out):
        assert c0 == c1
        u0, u1 = global_array(comm, u0), global_array(comm, u1)
        assert np.linalg.norm(u1 - u0) <= 1e-12 * np.linalg.norm(u0)
    assert rom.n_snapshots == 3
    if comm.rank == 0:
        ds = wt.data_store
        rom_evals = [e for e, _ in ds.rom_relative_residuals]
        for m in ("deim", "mdeim"):
            M = m.upper()
            evals = [e for e, _ in getattr(ds, m + "_relative_residuals")]
            assert evals and set(evals) <= set(rom_evals)
            n = len(evals)
            for name in (M + "_walltime", M + "_force_coefficients", M + "_meets_threshold",
                         M + "_lspg_converged", m + "_iterations", m + "_coefficient_errors",
                         m + "_l2_errors", m + "_operator_info", m + "_validation"):
                assert [r[0] for r in getattr(ds, name)] == evals, name
            for _, online, offline in getattr(ds, M + "_walltime"):
                assert online > 0.0 and offline > 0.0
            for _, info in getattr(ds, m + "_operator_info"):
                assert info["sampled_cells"] <= info["assembled_cells"] < info["total_cells"]
            for _, diag in getattr(ds, m + "_validation"):
                assert diag["restriction_max_error"] <= 1e-12
            assert all(len(v) == n for v in getattr(ds, M + "_outputs").values())
            idx, err = ds.output_errors("c_l", relative=True, kind=M, reference="ROM")
            assert list(idx) == evals and np.all(np.isfinite(err))
            assert [s[0] for s in getattr(ds, m + "_snapshots")] == [0, 1, 2]
            records = [p for _, k, p in ds.phase_times if k == m]
            assert len(records) == n
            for p in records:
                assert "assemble:jacobian_sampled" in p and "assemble:jacobian" not in p
            assert len([k for _, k, _ in ds.phase_times if k == m + "_offline"]) == n
        # MDEIM, trained on the Jacobians of the snapshots, tracks the ROM
        for row in ds.mdeim_l2_errors[1:]:
            assert row[5] < 1e-6


@pytest.mark.verification
def test_comparison_follows_a_local_weighted_rom(in_tmp_dir):
    """With a local_weighted ROM the DEIM and MDEIM bases are local_weighted too, with each
    snapshot weighted by its design's weight in the ROM's basis."""
    rom = ReducedOrderModel(rb_size=3, basis_mode='local_weighted', eta_threshold=1e-9)
    hr = HyperReductionComparison(deim_n_modes=6, mdeim_n_modes=3)
    wt = _windtunnel("euler", rom=rom, hyper_reduction=hr)
    _run_sequence(wt)
    settings = rom.basis_settings()
    assert settings['basis_mode'] == 'local_weighted'
    for name, model in hr.models.items():
        assert model.pod.basis_mode == 'local_weighted'
        w = model.pod.snapshot_weights
        assert w is not None
        expected = design_weights(settings, model.snapshot_matrix.snapshot_list_parameters)
        assert w == pytest.approx(expected / expected.sum(), rel=1e-12), name
    if MPI.COMM_WORLD.rank == 0:
        assert wt.data_store.mdeim_relative_residuals


@pytest.mark.verification
def test_comparison_adjoints(in_tmp_dir, request):
    """
    dD/dalpha through CSDL with the comparison: the derivative is the one without it, and the
    DEIM and MDEIM adjoints of each derivative are recorded against the full-order adjoint and,
    where the optimizer used it, the ROM's: at a rejected design (full-order adjoint) and at an
    accepted one (ROM adjoint, the full-order reference then solved by the comparison).
    """
    import csdl_alpha as csdl
    from dragonfly_sim.utils.solver_profiling import PROFILER
    request.addfinalizer(lambda: (PROFILER.configure(enabled=False), PROFILER.reset()))

    def derivatives(hyper_reduction):
        recorder = csdl.Recorder(inline=False)
        recorder.start()
        rom = ReducedOrderModel(rb_size=5, eta_threshold=1e-9)
        wt = _windtunnel("euler", rom=rom, hyper_reduction=hyper_reduction, profile_phases=True)
        alpha = csdl.Variable(name='alpha', value=np.array([np.radians(ALPHAS[0])]))
        alpha.set_as_design_variable()
        u_vec = wt.evaluate(alpha=alpha)
        out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
        out.D.set_as_objective()
        recorder.stop()
        sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
        dD = []
        for a in ALPHAS[:3]:
            sim[alpha] = np.array([np.radians(a)])
            sim.run()
        dD.append(float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0]))   # rejected: FOM adjoint
        sim[alpha] = np.array([np.radians(ALPHAS[4])])
        sim.run()
        dD.append(float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0]))   # accepted: ROM adjoint
        return wt, dD

    _, dD_ref = derivatives(None)
    wt, dD = derivatives(HyperReductionComparison(deim_n_modes=8, mdeim_n_modes=4))
    assert dD == pytest.approx(dD_ref, rel=1e-9)
    if MPI.COMM_WORLD.rank == 0:
        ds = wt.data_store
        kinds = [k for _, k, _ in ds.adjoint_solves]
        assert kinds[-1] == "rom" and "fom" in kinds
        for m in ("deim", "mdeim"):
            errors = getattr(ds, m + "_adjoint_errors")
            assert errors, m
            for eval_idx, row, absolute, relative, rel_rom, walltime in errors:
                assert row == 0 and np.isfinite(absolute) and np.isfinite(relative) and walltime > 0.0
            # the last derivative took the ROM adjoint: compared with it too
            assert np.isfinite(errors[-1][4])
            assert "adjoint_" + m in [k for _, k, _ in ds.phase_times]
        # MDEIM's adjoint is the ROM's up to the Jacobian approximation (both are reduced
        # adjoints, far from the full-order one on a basis of three snapshots)
        assert ds.mdeim_adjoint_errors[-1][4] < 1e-3
        # ... and what they put into the derivative is close to the full-order adjoint's
        for m in ("deim", "mdeim"):
            vector, derivative = getattr(ds, m + "_adjoint_errors"), getattr(ds, m + "_adjoint_derivative_errors")
            assert [r[:2] for r in derivative] == [r[:2] for r in vector]
            for _, _, errors in derivative:
                assert set(errors) == {"alpha"}
                absolute, relative, reference, vs_rom = errors["alpha"]
                assert reference > 0.0 and relative == pytest.approx(absolute / reference, rel=1e-12)
        _, _, mdeim_last = ds.mdeim_adjoint_derivative_errors[-1]
        assert mdeim_last["alpha"][1] < 1e-4 and mdeim_last["alpha"][3] < 1e-4
        assert ds.mdeim_adjoint_errors[-1][3] > 0.5          # while the vectors are far apart


@pytest.mark.unit_test
def test_derivative_contributions(in_tmp_dir):
    """The derivative contribution of an adjoint is (dR/dinput)^T psi, for alpha and (with mesh
    motions) the mesh coordinates, and its errors are those of the contributions: an adjoint
    1.1 times the full-order one is 10% off for every input, also when the vectors are compared
    with the ROM's."""
    rom = ReducedOrderModel(rb_size=3)
    wt = _windtunnel("euler", rom=rom)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    u, _, _ = wt.solve_forward(np.radians(2.0))
    motions = np.zeros((wt.mesh.mesh.geometry.x.shape[0], wt.mesh.mesh.geometry.dim))
    wt._adjoint_point = (u, np.radians(2.0), motions)
    wt._fom_contribution = None
    wt._contribution_operators = None
    wt._current_adjoint_row = 0
    psi = np.random.default_rng(comm.rank).standard_normal(u.size)
    try:
        c, c2 = wt._derivative_contributions([psi, 2.0 * psi])
        assert set(c) == {"alpha", "mesh_node_motions"}
        dr_dalpha = m.assemble_dRdalpha_vec(u).getArray()[:u.size]
        assert c["alpha"] == pytest.approx(comm.allreduce(float(dr_dalpha @ psi), op=MPI.SUM), rel=1e-12)
        dr_dx = m.compute_dRdxmesh_mat(u, wt.mesh.coordinate_space)
        x, y = dr_dx.createVecLeft(), dr_dx.createVecRight()
        x.array[:] = psi
        dr_dx.multTranspose(x, y)
        assert c["mesh_node_motions"] == pytest.approx(y.getArray(), rel=1e-12, abs=1e-14)
        assert c2["mesh_node_motions"] == pytest.approx(2.0 * c["mesh_node_motions"], rel=1e-12, abs=1e-14)
        errors = wt._contribution_errors(1.1 * psi, psi, psi_rom=1.2 * psi)
        for key in ("alpha", "mesh_node_motions"):
            absolute, relative, reference, vs_rom = errors[key]
            assert relative == pytest.approx(0.1, rel=1e-10)
            assert vs_rom == pytest.approx(0.1 / 1.2, rel=1e-10)
    finally:
        wt._release_contribution_operators()
