"""
core/reduced_order_model.py: the LSPG POD reduced-order model, for the Euler
and the RANS model through the same interface, and its use in
DG_windtunnel_model.

The flow cases are the jittered unit square of test_windtunnel_forward.py
with the bottom side as the body wall: slip for Euler, adiabatic no-slip
(SA-neg RANS, pseudo-transient continuation) for RANS.
"""
import os

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx

from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.reduced_order_model import ReducedOrderModel, LSPGSolver
from dragonfly_sim.core.RANS_model import CompressibleRANSModel
from dragonfly_sim.core.pod import PODBasis

from cases import jittered_square

BC = {'rho': 1.0, 'p': 1.0, 'M': 0.5, 'alpha': np.radians(2.0)}


_WINDTUNNELS = []


def _close_writers(wt):
    """Close a windtunnel's ADIOS2 writers. A writer destroyed later, after the
    working directory changed, cannot find its relative path and ADIOS2
    aborts the process."""
    m, mesh = wt.sim_model, wt.mesh
    writers = [getattr(m, "fom_solution_writer", None), getattr(m, "fom_pressure_writer", None),
               getattr(mesh, "_deformation_writer", None)]
    writers += list((getattr(mesh, "_quality_writers", None) or {}).values())
    for w in writers:
        if w is not None and w.writer is not None:
            w.writer.close()
            w.writer = None


@pytest.fixture
def in_tmp_dir(tmp_path):
    """The windtunnel writes output files into the working directory."""
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    cwd = os.getcwd()
    os.chdir(path)
    yield
    while _WINDTUNNELS:
        _close_writers(_WINDTUNNELS.pop())
    comm.Barrier()
    os.chdir(cwd)


def make_windtunnel(kind, rom=None, Re=200.0, suffix="rom", **extra):
    kw = dict(extra)
    if kind == "rans":
        kw.update(model_class=CompressibleRANSModel, model_kwargs={"Re": Re})
    wt = DG_windtunnel_model(jittered_square(8), BC, moments=[("pitch", (0.25, 0.), "pitch")],
                             mesh_inner_bdry_function=lambda x: x[1] < 1e-8,
                             poly_order=0, filename_suffix="{}_{}".format(suffix, kind),
                             reduced_order_model=rom, **kw)
    wt.euler_residual_conv_limit = 1e-11
    wt.set_up_sim()
    _WINDTUNNELS.append(wt)
    return wt


def global_array(comm, local):
    return np.concatenate(comm.allgather(np.asarray(local)))


# ----------------------------------------------------------------------
# Model interface
# ----------------------------------------------------------------------
@pytest.mark.unit_test
def test_state_scales(in_tmp_dir):
    wt = make_windtunnel("euler")
    m = wt.sim_model
    inlet = m.farfield_conditions
    U = inlet['M'] * inlet['c']
    assert m.state_scales() == pytest.approx([1.0, U, U, 1.0 / 0.4 + 0.5 * U**2], rel=1e-14)


@pytest.mark.unit_test
def test_rans_state_scales_match_the_limiter_scales(in_tmp_dir):
    wt = make_windtunnel("rans", Re=6e6)
    m = wt.sim_model
    tm = m.turbulence_model
    scales = m.state_scales()
    assert len(scales) == m.n_state == 5
    expected_extra = 1e3 * tm.chi_inf * float(m.mu_inf_const.value)
    assert scales[4] == pytest.approx(expected_extra, rel=1e-12)
    # the MUSCL limiter reads the same expressions
    lim = m._limiter_scales()
    assert [float(s) for s in lim[:4]] == pytest.approx(scales[:4], rel=1e-14)


@pytest.mark.unit_test
def test_steady_residual_switches_the_pseudo_time_term_off(in_tmp_dir):
    wt = make_windtunnel("rans")
    m = wt.sim_model
    assert m.use_ptc
    m.set_up_solver()
    term = m.time_integrator.term
    term.set_cfl(3.0)
    U_old = term.U_old
    U_old.x.array[:] = m.u_vec.x.array
    U_old.x.array[0::m.n_state] *= 1.01        # make the pseudo term nonzero
    U_old.x.scatter_forward()

    b_ptc = m.problem._b.duplicate()
    m.problem.F(m.u_vec.x.petsc_vec, b_ptc)
    with m.steady_residual():
        assert float(term.inv_cfl.value) == 0.0 and float(term.inv_dt.value) == 0.0
        b_steady = m.problem._b.duplicate()
        m.problem.F(m.u_vec.x.petsc_vec, b_steady)
    assert float(term.inv_cfl.value) == pytest.approx(1.0 / 3.0)
    m._ensure_residual_vector()
    m._assemble_physical_residual()
    diff = b_steady.copy()
    diff.axpy(-1.0, m.residual_vec_physical)
    assert diff.norm() <= 1e-12 * max(m.residual_vec_physical.norm(), 1.0)
    b_ptc.axpy(-1.0, b_steady)
    assert b_ptc.norm() > 1e-6
    for v in (b_ptc, b_steady, diff):
        v.destroy()


@pytest.mark.unit_test
def test_positivity_step_length_caps_and_limits(in_tmp_dir):
    m = make_windtunnel("euler").sim_model
    x = m.u_vec.x.petsc_vec
    dx = x.duplicate()
    dx.set(0.0)
    theta, cap = m.positivity_step_length(x, dx)
    assert cap == 0.0
    # a step that empties the density everywhere at theta = 1
    dx.array[0::m.n_state] = x.array[0::m.n_state]
    theta, cap = m.positivity_step_length(x, dx)
    assert 0.0 < theta < 1.0 and theta <= cap
    dx.destroy()


# ----------------------------------------------------------------------
# Inner product and basis
# ----------------------------------------------------------------------
@pytest.mark.unit_test
def test_default_W_is_the_state_scaled_euclidean_product(in_tmp_dir):
    rom = ReducedOrderModel(rb_size=4)
    assert (rom.inner_product, rom.state_scaling) == ("euclidean", "freestream")
    wt = make_windtunnel("rans", rom=rom, Re=6e6)
    m = wt.sim_model
    w = rom.W.w.getArray().reshape(-1, m.n_state)
    assert w == pytest.approx(np.tile(1.0 / rom.scales**2, (w.shape[0], 1)), rel=1e-14)


@pytest.mark.unit_test
def test_W_is_the_scaled_L2_inner_product(in_tmp_dir):
    rom = ReducedOrderModel(rb_size=4, inner_product="l2")
    wt = make_windtunnel("rans", rom=rom, Re=6e6)
    m = wt.sim_model
    s = rom.scales
    u = m.u_vec
    form = sum(u[k]**2 / s[k]**2 for k in range(m.n_state)) * m.forms.dx
    comm = m.mesh_obj.mesh.comm
    exact = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)
    x = u.x.petsc_vec
    Wx = x.duplicate()
    rom.W.mult(x, Wx)
    assert x.dot(Wx) == pytest.approx(exact, rel=1e-12)
    Wx.destroy()


@pytest.mark.parametrize("inner_product, scaling", [("l2", "freestream"), ("l2", None), ("euclidean", None)])
@pytest.mark.unit_test
def test_restriction_inverts_expansion(in_tmp_dir, inner_product, scaling):
    rom = ReducedOrderModel(rb_size=3, inner_product=inner_product, state_scaling=scaling)
    wt = make_windtunnel("euler", rom=rom)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    rng = np.random.default_rng(1)
    x = m.u_vec.x.petsc_vec
    for j in range(3):
        x.array[:] = rng.standard_normal(x.getLocalSize())
        rom.pod.snapshot_matrix.process_and_append_to_snapshot_lists(x, np.array([float(j)]))
    rom.pod.construct_basis(rb_size=3, layout_vec=x)
    solver = LSPGSolver(rom.pod.basis, comm, W=None if rom.W is None else rom.W.w)
    a = np.array([0.3, -1.2, 2.0])
    solver.expand(a, m.u_vec)
    assert solver.restrict(x) == pytest.approx(a, rel=1e-12, abs=1e-12)
    assert (rom.W is None) == (inner_product == "euclidean" and scaling is None)


@pytest.mark.unit_test
def test_scaled_L2_keeps_the_turbulence_variable_under_truncation(in_tmp_dir):
    """
    At Re 6e6 rho*nu_tilde is ~1e-4 against rho*E ~ 2. Four snapshots: three
    mean-flow perturbations (relative sizes 1, 1, 1e-3) and one perturbing
    rho*nu_tilde alone by its own scale; max_rank 3 drops one. The Euclidean
    POD drops the rho*nu_tilde direction; with the state scaling (with or
    without the mass matrix) the small mean-flow perturbation goes instead.
    """
    errors = {}
    for inner, scaling in (("euclidean", "freestream"), ("l2", "freestream"), ("euclidean", None)):
        rom = ReducedOrderModel(rb_size=3, max_rank=3, inner_product=inner, state_scaling=scaling)
        wt = make_windtunnel("rans", rom=rom, Re=6e6, suffix="trunc_{}_{}".format(inner, scaling))
        m = wt.sim_model
        comm = m.mesh_obj.mesh.comm
        scales = np.asarray(m.state_scales())
        x = m.u_vec.x.petsc_vec
        n_cells = x.getLocalSize() // m.n_state
        rng = np.random.default_rng(5 + comm.rank)
        snapshots = []
        for k, amp in enumerate((1.0, 1.0, 1e-3)):
            d = np.zeros((n_cells, m.n_state))
            d[:, :4] = amp * 1e-2 * scales[:4] * rng.standard_normal((n_cells, 4))
            snapshots.append(d.ravel())
        d = np.zeros((n_cells, m.n_state))
        d[:, 4] = 1e-2 * scales[4] * rng.standard_normal(n_cells)
        snapshots.append(d.ravel())
        for j, snap in enumerate(snapshots):
            x.array[:] = snap
            rom.pod.snapshot_matrix.process_and_append_to_snapshot_lists(x, np.array([float(j)]))
        rom.pod.construct_basis(rb_size=3, layout_vec=x)
        solver = LSPGSolver(rom.pod.basis, comm, W=None if rom.W is None else rom.W.w)
        # relative W-error of projecting the turbulence snapshot onto the basis
        x.array[:] = snapshots[3]
        solver.expand(solver.restrict(x), m.u_vec)
        proj = m.u_vec.x.petsc_vec.getArray().copy()
        err = comm.allreduce(np.sum(((proj - snapshots[3]).reshape(-1, m.n_state)[:, 4])**2), op=MPI.SUM)
        ref = comm.allreduce(np.sum(snapshots[3].reshape(-1, m.n_state)[:, 4]**2), op=MPI.SUM)
        errors[(inner, scaling)] = np.sqrt(err / ref)
    assert errors[("euclidean", "freestream")] < 1e-8
    assert errors[("l2", "freestream")] < 1e-8
    assert errors[("euclidean", None)] > 0.5


# ----------------------------------------------------------------------
# LSPG solve, Euler and RANS through one interface
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["euler", "rans"])
@pytest.mark.verification
def test_lspg_recovers_a_solution_in_the_basis(in_tmp_dir, kind):
    """With the converged FOM state u* in the snapshot set, the ROM started
    from another state converges back to u* and to FOM-level eta, for each
    inner product. The three FOM solves are shared by the inner products."""
    wt = make_windtunnel(kind)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    states = {}
    for a in (2.0, 2.8, 2.4):
        u, converged, _ = wt.solve_forward(np.radians(a))
        assert converged
        states[a] = u
    m.set_angle_of_attack(np.radians(2.4))
    for inner_product, scaling in (("euclidean", "freestream"), ("l2", "freestream"), ("euclidean", None)):
        rom = ReducedOrderModel(rb_size=5, inner_product=inner_product, state_scaling=scaling,
                                eta_threshold=1e-9)
        rom.set_up(m)
        for a, u in states.items():
            m.set_u_vec(u)
            rom.add_snapshot(m, np.array([np.radians(a)]))
        # start from u*(2.0) at alpha 2.4
        m.set_u_vec(states[2.0])
        if kind == "rans":
            m.time_integrator.term.set_cfl(7.0)
        result = rom.solve(m, np.array([np.radians(2.4)]))
        assert result is not None and result.converged and result.accepted, (inner_product, scaling)
        u_rom, u_fom = global_array(comm, result.solution), global_array(comm, states[2.4])
        assert np.linalg.norm(u_rom - u_fom) <= 1e-8 * np.linalg.norm(u_fom), (inner_product, scaling)
        assert result.eta < 1e-10, (inner_product, scaling)
        if kind == "rans":
            assert float(m.time_integrator.term.inv_cfl.value) == pytest.approx(1.0 / 7.0)
        rom.destroy()


# ----------------------------------------------------------------------
# DG_windtunnel_model
# ----------------------------------------------------------------------
# alpha close to the first snapshot's (2 degrees). The
# ROM interpolating between two snapshots reaches eta ~3e-7 (Euler) and is
# rejected; with three it reaches 1e-10 (Euler) / 5e-9 (RANS at Re 200)
ALPHAS = (2.0, 2.6, 2.3, 2.1, 2.45)


@pytest.mark.parametrize("kind, eta_threshold, run_fom", [("euler", 1e-9, False), ("rans", 1e-7, True)])
@pytest.mark.unit_test
def test_windtunnel_accepts_the_rom_between_snapshots(in_tmp_dir, kind, eta_threshold, run_fom):
    """Two FOM snapshots, a rejected ROM (third snapshot), then two accepted
    ROM solutions. Without run_fom_every_evaluation (Euler) the accepted ROM
    solution is the output and the warm start and no FOM runs; with it (RANS)
    the FOM runs too and the coefficient errors are recorded."""
    rom = ReducedOrderModel(rb_size=5, eta_threshold=eta_threshold, run_fom_every_evaluation=run_fom)
    wt = make_windtunnel(kind, rom=rom)
    out = []
    for a in ALPHAS:
        out.append(wt.solve_forward(np.radians(a)))
        wt.eval_idx += 1                 # as solve_residual_equations does: records match by eval_idx
    assert all(c for _, c, _ in out)
    assert rom.n_snapshots == 3          # the two first evaluations and the rejected third
    if MPI.COMM_WORLD.rank == 0:
        ds = wt.data_store
        assert [acc for _, acc in ds.ROM_meets_threshold] == [False, True, True]
        assert len(ds.ROM_lspg_converged) == 3
        assert all(isinstance(c, (bool, np.bool_)) for _, c in ds.ROM_lspg_converged)
        assert len(ds.global_snapshot_sing_vals_per_iteration) == 3
        assert len(ds.FOM_walltime) == (len(ALPHAS) if run_fom else 3)
        assert len(ds.rom_coefficient_errors) == (3 if run_fom else 1)
        # L2 state errors, over the domain and the wall, wherever the FOM ran after the ROM
        for errors in (ds.absolute_l2_errors_per_iteration, ds.relative_l2_errors_per_iteration,
                       ds.absolute_l2_bdry_errors_per_iteration, ds.relative_l2_bdry_errors_per_iteration):
            assert [e[0] for e in errors] == [e[0] for e in ds.rom_coefficient_errors]
        for _, rel in ds.relative_l2_errors_per_iteration[1:] + ds.relative_l2_bdry_errors_per_iteration[1:]:
            assert 0.0 <= rel < 1e-3
        for _, dcd, dcl, dcm in ds.rom_coefficient_errors[1:]:
            assert max(abs(dcd), abs(dcl), abs(dcm)) < 1e-6
        # every postprocessor output is stored by name for the ROM (and the FOM if it ran)
        names = {"L", "D", "c_l", "c_d", "M_pitch", "c_m_pitch"}
        assert set(ds.output_names) == names
        assert all(len(ds.ROM_outputs[n]) == 3 for n in names)
        for n in names:
            idx, err = ds.output_errors(n, relative=False)
            # an output has an error wherever the FOM ran after the ROM: the same evaluations as rom_coefficient_errors
            assert list(idx) == [e[0] for e in ds.rom_coefficient_errors]
            ref = {"c_d": 1, "c_l": 2, "c_m_pitch": 3}
            if n in ref:
                assert err == pytest.approx([e[ref[n]] for e in ds.rom_coefficient_errors], abs=1e-14)
            if run_fom:
                assert np.all(np.abs(ds.output_errors(n)[1][1:]) < 1e-3)
    u = out[-1][0]
    assert np.array_equal(wt.previous_solution, u)
    if not run_fom:
        # the output is the ROM solution itself
        m = wt.sim_model
        m.set_u_vec(u)
        assert m.relative_residual_norm(m._assemble_physical_residual()) < eta_threshold


@pytest.mark.parametrize("kind, eta_threshold", [("euler", 1e-9), ("rans", 1e-7)])
@pytest.mark.verification
def test_windtunnel_rom_adjoint(in_tmp_dir, kind, eta_threshold):
    """
    dD/dalpha through CSDL with a ROM. At 2.3 degrees the ROM (two snapshots)
    is rejected and the FOM solves; compute_totals re-runs that design, which
    then skips the ROM and takes the full-order adjoint. At 2.45 degrees (three
    snapshots) the ROM is accepted, the derivative takes the reduced adjoint,
    and it matches central differences of the ROM's own solves (accepted at
    both points; the global basis is fixed).
    """
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    rom = ReducedOrderModel(rb_size=5, eta_threshold=eta_threshold)
    wt = make_windtunnel(kind, rom=rom, suffix="adjoint")
    alpha = csdl.Variable(name='alpha', value=np.array([np.radians(ALPHAS[0])]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    root = MPI.COMM_WORLD.rank == 0
    ds = wt.data_store

    for a in ALPHAS[:3]:
        sim[alpha] = np.array([np.radians(a)])
        sim.run()
    # the rejected third solve: its adjoint is kept (for the comparison with the full-order one),
    # but is not the one the derivative uses
    assert rom.n_snapshots == 3 and rom.adjoint is not None and not rom.adjoint.accepted
    sim.compute_totals()
    assert rom.n_snapshots == 3 and rom.adjoint is not None and not rom.adjoint.accepted
    if root:
        assert [acc for _, acc in ds.ROM_meets_threshold] == [False]
        fom_kinds = [k for _, k, _ in ds.adjoint_solves]
        assert fom_kinds and set(fom_kinds) == {"fom"}

    a = np.radians(ALPHAS[4])
    sim[alpha] = np.array([a])
    sim.run()
    dD = float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0])
    assert rom.adjoint is not None and rom.adjoint.accepted
    h = 1e-5
    fd = (wt.solve_forward(a + h)[2]["D"] - wt.solve_forward(a - h)[2]["D"]) / (2 * h)
    PETSc.Sys.Print("ROM-ADJOINT-CHECK {}: dD/dalpha {:.12f}, FD of the ROM solves {:.12f}".format(
        kind, dD, fd))
    if root:
        assert [acc for _, acc in ds.ROM_meets_threshold] == [False] + [True] * 4
        assert set(k for _, k, _ in ds.adjoint_solves[len(fom_kinds):]) == {"rom"}
    assert abs(fd - dD) < 1e-6 * abs(fd), (fd, dD)


@pytest.mark.parametrize("kind, eta_threshold", [("euler", 1e-9), ("rans", 1e-7)])
@pytest.mark.verification
def test_windtunnel_adjoint_errors(in_tmp_dir, kind, eta_threshold):
    """
    compute_all_adjoint_errors records the ROM adjoint's error against the
    full-order adjoint at the same state, and leaves the derivative handed to
    the optimizer, and the record of adjoint solves, as they are without it.
    """
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    rom = ReducedOrderModel(rb_size=5, eta_threshold=eta_threshold)
    wt = make_windtunnel(kind, rom=rom, suffix="adjoint_errors")
    assert wt.compute_all_adjoint_errors is False
    alpha = csdl.Variable(name='alpha', value=np.array([np.radians(ALPHAS[0])]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    root = MPI.COMM_WORLD.rank == 0
    ds = wt.data_store

    for a in ALPHAS[:3]:
        sim[alpha] = np.array([np.radians(a)])
        sim.run()
    sim.compute_totals()                 # the rejected design: full-order adjoint, and with the flag off
                                         # nothing is compared
    sim[alpha] = np.array([np.radians(ALPHAS[4])])
    sim.run()
    dD_off = float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0])
    assert rom.adjoint is not None
    n_solves = len(ds.adjoint_solves)
    assert ds.adjoint_errors == []

    wt.compute_all_adjoint_errors = True
    sim.run()                            # the same design: the ROM again
    dD_on = float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0])
    # the optimizer still gets the ROM adjoint (the re-run ROM solve restarts from its own
    # solution, hence round-off rather than bitwise equality)
    assert dD_on == pytest.approx(dD_off, rel=1e-9)
    if root:
        assert [k for _, k, _ in ds.adjoint_solves[n_solves:]] == ["rom"]
        (eval_idx, row, absolute, relative, fom_norm, accepted, rom_walltime), = ds.adjoint_errors
        assert eval_idx == ds.adjoint_solves[-1][0] and row == 0 and accepted is True
        assert fom_norm > 0.0 and absolute > 0.0 and rom_walltime > 0.0
        assert relative == pytest.approx(absolute / fom_norm, rel=1e-12)
        assert relative < 1.0
        # the error of the ROM adjoint's contribution to dD/dalpha, small where the vectors are far apart
        (d_eval, d_row, d_accepted, contribution), = ds.adjoint_derivative_errors
        assert (d_eval, d_row, d_accepted) == (eval_idx, row, True) and set(contribution) == {"alpha"}
        assert contribution["alpha"][1] < 1e-4


@pytest.mark.parametrize("kind, eta_threshold", [("euler", 1e-9), ("rans", 1e-7)])
@pytest.mark.verification
def test_windtunnel_adjoint_errors_of_a_rejected_rom(in_tmp_dir, request, kind, eta_threshold):
    """At a design whose ROM solve was rejected the optimizer uses the full-order adjoint, and
    compute_all_adjoint_errors compares the reduced adjoint of the rejected solve, evaluated at
    the full-order state, with it: the derivative is unchanged, the adjoint solve record shows
    only the full-order adjoint, and the error is recorded for every output row."""
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    rom = ReducedOrderModel(rb_size=5, eta_threshold=eta_threshold)
    from dragonfly_sim.utils.solver_profiling import PROFILER
    request.addfinalizer(lambda: (PROFILER.configure(enabled=False), PROFILER.reset()))
    wt = make_windtunnel(kind, rom=rom, suffix="adjoint_errors_rejected", profile_phases=True)
    alpha = csdl.Variable(name='alpha', value=np.array([np.radians(ALPHAS[0])]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    root = MPI.COMM_WORLD.rank == 0
    ds = wt.data_store

    for a in ALPHAS[:3]:                 # the third solve: two snapshots, the ROM is rejected
        sim[alpha] = np.array([np.radians(a)])
        sim.run()
    assert rom.adjoint is not None and not rom.adjoint.accepted
    dD_off = float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0])
    n_solves = len(ds.adjoint_solves)
    assert ds.adjoint_errors == []       # flag off: nothing compared

    wt.compute_all_adjoint_errors = True
    sim.run()                            # the same design: skips the ROM, keeps its adjoint
    assert rom.adjoint is not None and not rom.adjoint.accepted
    dD_on = float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0])
    assert dD_on == pytest.approx(dD_off, rel=1e-9)
    if root:
        assert [k for _, k, _ in ds.adjoint_solves[n_solves:]] == ["fom"]
        (eval_idx, row, absolute, relative, fom_norm, accepted, rom_walltime), = ds.adjoint_errors
        assert eval_idx == ds.adjoint_solves[-1][0] and row == 0 and accepted is False
        assert fom_norm > 0.0 and absolute > 0.0 and rom_walltime > 0.0
        assert relative == pytest.approx(absolute / fom_norm, rel=1e-12)
        # the phases of the adjoint the optimizer used, and of the reference solve, are separate
        kinds = [k for _, k, _ in ds.phase_times if k.startswith("adjoint")]
        assert kinds.count("adjoint_fom") == 2 and kinds.count("adjoint_rom_reference") == 1
        assert "adjoint_rom" not in kinds


@pytest.mark.verification
def test_windtunnel_records_phase_times(in_tmp_dir):
    """profile_phases records the phases of every forward evaluation, of its ROM and FOM flow
    solves and of its adjoint in data_store.phase_times; the adjoint is the ROM's here."""
    import csdl_alpha as csdl
    from dragonfly_sim.utils.solver_profiling import PROFILER
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    rom = ReducedOrderModel(rb_size=5, eta_threshold=1e-9)
    wt = make_windtunnel("euler", rom=rom, suffix="phases", profile_phases=True)
    try:
        alpha = csdl.Variable(name='alpha', value=np.array([np.radians(ALPHAS[0])]))
        alpha.set_as_design_variable()
        u_vec = wt.evaluate(alpha=alpha)
        out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
        out.D.set_as_objective()
        recorder.stop()
        sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
        for a in tuple(ALPHAS[:3]) + (ALPHAS[4],):
            sim[alpha] = np.array([np.radians(a)])
            sim.run()
        sim.compute_totals()                       # at ALPHAS[4]: the ROM was accepted
        if MPI.COMM_WORLD.rank == 0:
            ds = wt.data_store
            kinds = [k for _, k, _ in ds.phase_times]
            # compute_totals re-runs the design, hence five forward evaluations
            assert kinds.count("forward") == 5 and "fom" in kinds and "rom" in kinds
            assert kinds[-1] == "adjoint_rom"
            idx, forward = ds.phase_time_series("forward", by_group=True)
            assert list(idx) == [0, 1, 2, 3, 4]
            assert "assemble" in forward and "solve" in forward
            # self times partition a record: the phases of the ROM solve add up to no more than
            # the wall time recorded for it, which also covers the unprofiled remainder
            _, rom_phases = ds.phase_time_series("rom", by_group=True)
            assert sum(v.sum() for v in rom_phases.values()) <= sum(w for _, w in ds.ROM_walltime) + 1e-6
            assert all(v >= 0.0 for _, _, p in ds.phase_times for t in p.values() for v in t[:2])
    finally:
        PROFILER.configure(enabled=False)
        PROFILER.reset()


@pytest.mark.verification
def test_recorded_outputs_are_the_optimizers(in_tmp_dir):
    """With use_postprocessor, the outputs recorded for every solve are those of the
    postprocessor the optimizer evaluates, including a moment the windtunnel model was not
    built with, at the evaluation's own angle of attack (the lift direction follows alpha)."""
    import csdl_alpha as csdl
    from dragonfly_sim.core.postprocessor import DG_postprocessor
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    wt = make_windtunnel("euler", suffix="outputs")
    pp = DG_postprocessor(wt.mesh, wt.sim_model, wt.WALL_TAG, p_inf_dim=101325.,
                          moments=[("pitch", (0.25, 0.), "pitch"), ("nose", (0., 0.), "pitch")])
    wt.use_postprocessor(pp)
    alpha = csdl.Variable(name='alpha', value=np.array([np.radians(ALPHAS[0])]))
    alpha.set_as_design_variable()
    out = pp.evaluate(wt.evaluate(alpha=alpha), alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    compared = ("L", "D", "M_pitch", "M_nose", "c_l")
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False,
                                         additional_outputs=[getattr(out, n) for n in compared])
    root = MPI.COMM_WORLD.rank == 0
    for a in ALPHAS[:3]:
        sim[alpha] = np.array([np.radians(a)])
        sim.run()
        if root:
            recorded = {n: v[-1][1] for n, v in wt.data_store.FOM_outputs.items()}
            assert set(recorded) == {"L", "D", "c_l", "c_d", "M_pitch", "c_m_pitch", "M_nose", "c_m_nose"}
            for n in compared:
                assert recorded[n] == pytest.approx(float(np.asarray(sim[getattr(out, n)]).ravel()[0]),
                                                    rel=1e-12, abs=1e-14), n
    with pytest.raises(ValueError, match="mesh and sim_model"):
        wt.use_postprocessor(DG_postprocessor(wt.mesh, make_windtunnel("euler", suffix="other").sim_model,
                                              wt.WALL_TAG))


@pytest.mark.unit_test
def test_windtunnel_falls_back_to_the_fom(in_tmp_dir):
    """eta_threshold 0 never accepts the ROM: every evaluation is the FOM
    from the same warm start, identical to a windtunnel without a ROM (to the
    last bit serially; in parallel two plain windtunnels already differ by
    ~1e-15 run to run)."""
    rom = ReducedOrderModel(rb_size=5, eta_threshold=0.0)
    wt_rom = make_windtunnel("euler", rom=rom, suffix="fallback")
    wt_fom = make_windtunnel("euler", suffix="plain")
    for a in ALPHAS[:4]:
        u_rom, c_rom, f_rom = wt_rom.solve_forward(np.radians(a))
        u_fom, c_fom, f_fom = wt_fom.solve_forward(np.radians(a))
        assert c_rom and c_fom
        assert u_rom == pytest.approx(u_fom, rel=1e-12, abs=1e-12)
        for name, moment in f_rom.pop("moments").items():
            assert moment == pytest.approx(f_fom["moments"][name], rel=1e-12, abs=1e-12)
        f_fom.pop("moments")
        assert f_rom == pytest.approx(f_fom, rel=1e-12, abs=1e-12)
    assert rom.n_snapshots == 4
    if MPI.COMM_WORLD.rank == 0:
        assert [acc for _, acc in wt_rom.data_store.ROM_meets_threshold] == [False, False]


# ----------------------------------------------------------------------
# LSPG and its stopping rules on a linear stand-in model
# ----------------------------------------------------------------------
class _LinearProblem:
    """R(u) = K u - f with a non-symmetric K, in the problem interface the
    ROM uses (F, J, _A, _b)."""

    def __init__(self, K, f, u):
        self._A, self._b, self.K, self.f, self.u = K, f.duplicate(), K, f, u

    def F(self, x, b):
        self.K.mult(x, b)
        b.axpy(-1.0, self.f)

    def J(self, x, A):
        pass


class _LinearModel:
    """The host interface of core/reduced_order_model.py, for R(u) = K u - f."""

    def __init__(self, comm, n=60, n_state=3, seed=11):
        import contextlib
        from types import SimpleNamespace
        rng = np.random.default_rng(seed)
        self.n_state = n_state
        # blocks of n_state per "cell", like a flow state (the state scaling
        # repeats per block)
        x = PETSc.Vec().createMPI((PETSc.DECIDE, n), bsize=n_state, comm=comm)
        rs, re = x.getOwnershipRange()
        K = PETSc.Mat().createAIJ(size=((re - rs, n), (re - rs, n)), nnz=(3, 3), comm=comm)
        K.setUp()
        for i in range(rs, re):
            K.setValue(i, i, 3.0 + np.sin(i))
            if i + 1 < n:
                K.setValue(i, i + 1, 0.7)          # upper only: non-symmetric
        K.assemble()
        f = x.duplicate()
        f.array[:] = np.random.default_rng(seed + 1).standard_normal(n)[rs:re]
        self.u_vec = SimpleNamespace(x=SimpleNamespace(petsc_vec=x, scatter_forward=lambda: None))
        self.problem = _LinearProblem(K, f, self.u_vec)
        self.mesh_obj = SimpleNamespace(mesh=SimpleNamespace(comm=comm))
        self.functionspaces = {"V": None}          # read, but unused, by the Euclidean weights
        self.steady_residual = contextlib.nullcontext
        self.rng, self.n, self.rs, self.re = rng, n, rs, re

    def set_up_solver(self):
        pass

    def positivity_step_length(self, x, dx):
        return 1.0, 1.0

    def state_scales(self):
        return [1.0] * self.n_state

    def _assemble_physical_residual(self):
        r = self.problem._b
        self.problem.F(self.u_vec.x.petsc_vec, r)
        return r.norm()

    def relative_residual_norm(self, full_norm):
        return full_norm / self.u_vec.x.petsc_vec.norm()


def _linear_rom(model, k=4, state_scaling=None, **kw):
    rom = ReducedOrderModel(rb_size=k, inner_product="euclidean", state_scaling=state_scaling, **kw)
    rom.set_up(model)
    x = model.u_vec.x.petsc_vec
    for j in range(k):
        x.array[:] = model.rng.standard_normal(model.n)[model.rs:model.re]
        rom.add_snapshot(model, np.array([float(j)]))
    x.set(0.0)
    return rom


@pytest.mark.verification
def test_lspg_solves_the_least_squares_normal_equations():
    """The ROM solution satisfies (K V)^T (K V a - f) = 0 (LSPG), not the
    Galerkin V^T (K V a - f) = 0, which differs for a non-symmetric K."""
    comm = MPI.COMM_WORLD
    model = _LinearModel(comm)
    rom = _linear_rom(model)
    result = rom.solve(model, np.array([0.5]))
    assert result.converged
    V = np.vstack(comm.allgather(rom.pod.basis.getDenseArray().copy()))
    K = model.problem.K
    KV = np.column_stack([global_array(comm, _matvec(K, V[:, j], model)) for j in range(V.shape[1])])
    u = global_array(comm, result.solution)
    f = global_array(comm, model.problem.f.getArray())
    r = global_array(comm, _matvec(K, u, model)) - f
    assert np.linalg.norm(KV.T @ r) <= 1e-10 * np.linalg.norm(KV.T @ f)
    assert np.linalg.norm(V.T @ r) > 1e-3 * np.linalg.norm(V.T @ f)


def _matvec(K, v_global, model):
    x = model.u_vec.x.petsc_vec.duplicate()
    x.array[:] = v_global[model.rs:model.re]
    y = x.duplicate()
    K.mult(x, y)
    out = y.getArray().copy()
    x.destroy()
    y.destroy()
    return out


@pytest.mark.unit_test
def test_outer_passes_and_absolute_g_limit():
    """newton_max_it 1 and g_conv_limit 0: max_outer_iterations passes of one
    iteration, never converged (benchmark_pod.py's budget structure)."""
    model = _LinearModel(MPI.COMM_WORLD)
    rom = _linear_rom(model, newton_max_it=1, max_outer_iterations=3, g_conv_limit=0.0)
    result = rom.solve(model, np.array([0.5]))
    assert result.iterations == 3
    assert not result.converged


@pytest.mark.unit_test
def test_relative_g_tolerance_keeps_the_first_pass_reference():
    """g0 is the first iteration of the first pass: a later pass does not
    reset it, so it converges against the same relative target."""
    model = _LinearModel(MPI.COMM_WORLD)
    rom = _linear_rom(model, newton_max_it=1, max_outer_iterations=3, g_conv_limit=0.0)
    rom.solve(model, np.array([0.5]))
    solver = rom.last_solver
    assert solver.g0 is not None and solver.g0 > 0.0
    # the first pass of one iteration started from g0; later passes began lower
    assert solver.last_g_norm < solver.g0


# ----------------------------------------------------------------------
# The reduced (LSPG) adjoint
# ----------------------------------------------------------------------
@pytest.mark.verification
def test_tsqr_matches_numpy_qr():
    """Thin QR of a 7 x 4 matrix split by rows: on 3 ranks every block has
    fewer rows than columns."""
    from dragonfly_sim.utils.pod_utils import tsqr
    comm = MPI.COMM_WORLD
    X = np.random.default_rng(3).standard_normal((7, 4))
    Q_local, R = tsqr(comm, X[np.array_split(np.arange(7), comm.size)[comm.rank]])
    Q = np.vstack(comm.allgather(Q_local))
    np.testing.assert_allclose(Q @ R, X, atol=1e-13)
    np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-13)
    np.testing.assert_array_equal(np.triu(R), R)
    np.testing.assert_allclose(np.abs(R), np.abs(np.linalg.qr(X)[1]), rtol=1e-12, atol=1e-13)


@pytest.mark.verification
def test_reduced_adjoint_is_the_lspg_adjoint():
    """
    The adjoint of an accepted LSPG solution of R(u) = K u - f, with a state
    scaling (S != I). R is linear, so the reduced adjoint psi is the exact
    derivative of the ROM output: with dR/df = -I, the change of g^T u_rom
    under f -> f + df is psi^T df. psi solves the Petrov-Galerkin projection
    V^T (K^T psi - g) = 0, differs from the full-order adjoint K^-T g, and a
    second right-hand side reuses the factorization.
    """
    comm = MPI.COMM_WORLD
    model = _LinearModel(comm)
    n_J = []
    J = model.problem.J
    model.problem.J = lambda x, A: (n_J.append(1), J(x, A))
    rom = _linear_rom(model, state_scaling=[1.0, 10.0, 0.1], eta_threshold=np.inf)
    result = rom.solve(model, np.array([0.5]))
    assert result.accepted and rom.adjoint is not None
    u0 = global_array(comm, result.solution)
    s = global_array(comm, rom.adjoint.S.getArray())
    assert s.max() / s.min() == pytest.approx(100.0)

    g = np.random.default_rng(21).standard_normal(model.n)[model.rs:model.re]
    calls = len(n_J)
    psi = rom.adjoint.solve(model, g)
    assert len(n_J) == calls + 1
    np.testing.assert_allclose(rom.adjoint.solve(model, -2.0 * g), -2.0 * psi, rtol=1e-14, atol=1e-14)
    assert len(n_J) == calls + 1

    K = model.problem.K
    Kd = np.column_stack([global_array(comm, _matvec(K, np.eye(model.n)[:, j], model))
                          for j in range(model.n)])
    V = np.vstack(comm.allgather(rom.pod.basis.getDenseArray().copy()))
    psi, g = global_array(comm, psi), global_array(comm, g)
    assert np.linalg.norm(V.T @ (Kd.T @ psi - g)) <= 1e-12 * np.linalg.norm(V.T @ g)
    psi_fom = np.linalg.solve(Kd.T, g)
    assert np.linalg.norm(psi - psi_fom) > 0.1 * np.linalg.norm(psi_fom)

    # the derivative of the ROM output itself: re-solve with a perturbed f
    df = np.random.default_rng(22).standard_normal(model.n)
    model.problem.f.array[:] += df[model.rs:model.re]
    u1 = global_array(comm, rom.solve(model, np.array([0.5])).solution)
    assert g @ (u1 - u0) == pytest.approx(psi @ df, rel=1e-9)


@pytest.mark.unit_test
def test_solve_forward_without_snapshot_and_parameter_size_guard(in_tmp_dir):
    """A baseline solve with snapshot=False stores nothing; snapshots of a
    different parameter count are refused with a clear error (not a scipy
    cdist failure)."""
    rom = ReducedOrderModel(rb_size=2)
    wt = make_windtunnel("euler", rom=rom)
    wt.solve_forward(np.radians(2.0), snapshot=False)
    assert rom.n_snapshots == 0
    assert wt.reduced_order_model is rom
    wt.solve_forward(np.radians(2.0))
    assert rom.n_snapshots == 1
    with pytest.raises(ValueError, match="same parameter vector"):
        rom.add_snapshot(wt.sim_model, np.zeros(26))


@pytest.mark.parametrize("centering, mean_weighting", [("mean", "snapshot_weights"), ("mean", "uniform"),
                                                       ("warm_start", "snapshot_weights")])
@pytest.mark.verification
def test_windtunnel_rom_adjoint_with_centred_basis(in_tmp_dir, centering, mean_weighting):
    """local_weighted basis centred on the weighted mean or the warm start
    (u = V a + c): the ROM is accepted, the reduced adjoint is used, and
    dD/dalpha matches central differences of the ROM's own solves."""
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    rom = ReducedOrderModel(rb_size=5, eta_threshold=1e-9, basis_mode='local_weighted',
                            centering=centering, mean_weighting=mean_weighting)
    wt = make_windtunnel("euler", rom=rom, suffix="adjoint_mean")
    alpha = csdl.Variable(name='alpha', value=np.array([np.radians(ALPHAS[0])]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    for a in ALPHAS[:3]:
        sim[alpha] = np.array([np.radians(a)])
        sim.run()
    a = np.radians(ALPHAS[4])
    sim[alpha] = np.array([a])
    sim.run()
    dD = float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0])
    assert rom.pod.mean is not None and rom.adjoint is not None
    h = 1e-5
    fd = (wt.solve_forward(a + h)[2]["D"] - wt.solve_forward(a - h)[2]["D"]) / (2 * h)
    assert abs(fd - dD) < 1e-6 * abs(fd), (fd, dD)


@pytest.mark.unit_test
def test_centering_options_are_validated():
    assert ReducedOrderModel(basis_mode='local_weighted').centering == 'warm_start'   # the default
    assert ReducedOrderModel().centering is None                                     # global: none
    assert ReducedOrderModel(basis_mode='local_weighted', centering=None).centering is None
    with pytest.raises(ValueError, match="local_weighted"):
        ReducedOrderModel(centering='mean')
    with pytest.raises(ValueError, match="mean_weighting"):
        ReducedOrderModel(basis_mode='local_weighted', centering='mean', mean_weighting='bogus')
    with pytest.warns(DeprecationWarning):
        assert ReducedOrderModel(basis_mode='local_weighted', subtract_mean=True).centering == 'mean'


@pytest.mark.unit_test
def test_g_conv_limit_is_relative_to_the_first_g():
    """A linear problem converges in one iteration: accepted with a relative
    limit, never with 0."""
    model = _LinearModel(MPI.COMM_WORLD)
    rom = _linear_rom(model, g_conv_limit=1e-10)
    result = rom.solve(model, np.array([0.5]))
    assert result.converged
    assert rom.last_solver.last_g_norm <= max(rom.newton_atol, 1e-10 * rom.last_solver.g0)

