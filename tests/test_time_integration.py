"""
Time integration (core/time_integration.py), checkpoints and the
non-reflecting far fields.
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

from dragonfly_sim.core.time_integration import bdf_coefficients, CFLController, StepSizeController
from dragonfly_sim.core.farfield import TransverseFarfield
from dragonfly_sim.utils.checkpoint import save_checkpoint, load_checkpoint

from cases import euler_channel_model, seed_mean_flow, FREESTREAM, WALL_TAG


# ----------------------------------------------------------------------
# pure Python
# ----------------------------------------------------------------------
@pytest.mark.verification
def test_bdf_coefficients():
    assert bdf_coefficients(1) == (1.0, -1.0, 0.0)
    np.testing.assert_allclose(bdf_coefficients(2), (1.5, -2.0, 0.5))
    np.testing.assert_allclose(bdf_coefficients(2, 1.0), (1.5, -2.0, 0.5))
    for w in (0.5, 1.5, 3.0):
        a = bdf_coefficients(2, w)
        assert abs(sum(a)) < 1e-14                    # consistency: constants are steady
        # exact for linear in time: a0 t1 + a1 t0 + a2 t_{-1} = dt with t1 = 1, t0 = 0, t_{-1} = -1/w
        assert abs(a[0] * 1.0 + a[2] * (-1.0 / w) - 1.0) < 1e-14


@pytest.mark.unit_test
def test_cfl_controller_decisions():
    c = CFLController(cfl0=10.0)
    assert c.update(1.0, 0.5) == (True, False) and c.cfl == 20.0      # progress: x2 (cap)
    c.cfl = 10.0
    c.update(1.0, 0.99)                                                # slow progress: min growth
    assert c.cfl == 15.0
    c.cfl = 10.0
    c.update(1.0, 1.01)                                                # flat band, clean: band growth
    assert c.cfl == pytest.approx(12.0)
    c.cfl = 10.0
    c.update(1.0, 1.01, theta=0.5)                                     # flat band, mildly limited: hold
    assert c.cfl == 10.0
    c.update(1.0, 1.2)                                                 # rise > 5%: shrink
    assert c.cfl == pytest.approx(5.0)
    assert c.update(1.0, 20.0) == (False, False) and c.cfl == pytest.approx(0.5)   # reject
    c.cfl = 2.0
    assert c.update(1.0, 1.0, theta=0.01) == (True, True)             # limited: cut below the floor of 1
    assert c.cfl == pytest.approx(0.2)
    c.update(1.0, 0.5)                                                 # regrows from below the floor
    assert c.cfl == pytest.approx(0.4)
    c.cfl = 10.0
    c.update(1.0, 0.5, ksp_ok=False)                                   # inexact solve: no growth
    assert c.cfl == 10.0


@pytest.mark.unit_test
def test_cfl_controller_milder_cut_on_a_rise():
    # default: any rise above 5% at least halves the CFL
    c = CFLController(cfl0=10.0)
    c.update(1.0, 1.1)
    assert c.cfl == pytest.approx(5.0)
    # milder: the cut is the residual ratio, clipped to [rise_factor_min, rise_factor_max]
    c = CFLController(cfl0=10.0, rise_factor_min=0.7, rise_factor_max=0.9)
    c.update(1.0, 1.1)                                                 # 1/1.1 = 0.909 -> clipped to 0.9
    assert c.cfl == pytest.approx(9.0)
    c.cfl = 10.0
    c.update(1.0, 1.25)                                                # 1/1.25 = 0.8: proportional
    assert c.cfl == pytest.approx(8.0)
    c.cfl = 10.0
    c.update(1.0, 3.0)                                                 # large rise: no cut below 0.7
    assert c.cfl == pytest.approx(7.0)
    c.cfl = 10.0
    assert c.update(1.0, 20.0) == (False, False) and c.cfl == pytest.approx(1.0)   # rejection is unchanged


@pytest.mark.unit_test
def test_cfl_controller_stops_when_frozen():
    # theta = 0 leaves the state unchanged: max_frozen_steps of them in a row exhaust the controller
    c = CFLController(cfl0=5.0, max_frozen_steps=3)
    for _ in range(2):
        assert c.update(1.0, 1.0, theta=0.0) == (True, True)
        assert not c.exhausted
    c.update(1.0, 1.0, theta=1e-3)                                     # any update restarts the count
    assert c.frozen_steps == 0
    for _ in range(3):
        c.update(1.0, 1.0, theta=0.0)
    assert c.exhausted
    c.reset()
    assert not c.exhausted and c.frozen_steps == 0


@pytest.mark.unit_test
def test_step_size_controller():
    s = StepSizeController(dt=0.05, dt_start=1e-3, growth=1.5)
    assert s.first(None) == 1e-3 and s.first(1e-3) == pytest.approx(1.5e-3) and s.first(0.04) == 0.05
    assert StepSizeController(dt=0.05).first(None) == 0.05


# ----------------------------------------------------------------------
# with a model
# ----------------------------------------------------------------------
@pytest.mark.unit_test
def test_checkpoint_round_trip(tmp_path):
    comm = MPI.COMM_WORLD
    model = euler_channel_model(n=6)
    seed_mean_flow(model)
    path = comm.bcast(str(tmp_path / "ckpt.npz") if comm.rank == 0 else None, root=0)
    ref = model.u_vec.x.array.copy()
    save_checkpoint(path, [model.u_vec], {"step": 3, "t": 0.15})
    model.u_vec.x.array[:] = 0.0
    meta = load_checkpoint(path, [model.u_vec])
    assert meta == {"step": 3, "t": 0.15, "state_layout": "primitive"}
    np.testing.assert_array_equal(model.u_vec.x.array, ref)
    # a checkpoint of the former conservative unknowns (no layout tag) is refused
    if comm.rank == 0:
        import json
        data = dict(np.load(path))
        data["meta"] = json.dumps({"step": 3, "t": 0.15})
        np.savez(path, **data)
    comm.Barrier()
    with pytest.raises(ValueError, match="conservative"):
        load_checkpoint(path, [model.u_vec])


def _farfield_square(n=6, transverse=False, beta=0.0):
    """Euler on a square with the characteristic far field all round."""
    from dragonfly_sim.core.mesh_manager import Mesh
    from dragonfly_sim.core.Euler_model import CompressibleEulerModel
    from cases import jittered_square
    msh = jittered_square(n)
    mesh_obj = Mesh(msh, lambda pts: np.zeros(pts.shape[1], dtype=bool))
    mesh_obj.tag_boundary_facets([(np.ones(mesh_obj.bdry_facets.size, dtype=bool), 6)])
    meta = {'quadrature_degree': 1}
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=meta)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(measure_metadata=meta)
    model = CompressibleEulerModel(mesh_obj)
    model.residual_conv_limit, model.max_newton_iterations = 1e-10, 50
    model.report_positivity_limiter = False
    model.define_farfield_conditions(FREESTREAM)
    model.define_elements_functionspaces()
    model.define_trial_testfunctions()
    model.compute_initial_conditions_from_farfield()
    model.interpolate_solution_vector()
    tf = TransverseFarfield(model, 6, beta=beta, relax_length=1.0) if transverse else None
    model.define_farfield_bc(6, transverse=tf)
    model.compute_weakform()
    return model, tf


def _run_steps(model, tf, steps, dt=0.05):
    integrator = model.enable_time_stepping()
    model.set_up_solver()
    if tf is not None:
        tf.attach(integrator)
    integrator.verbose = False
    integrator.start()
    ctrl = StepSizeController(dt=dt)
    for _ in range(steps):
        rec = integrator.advance(ctrl)
        assert rec.accepted
    return integrator


@pytest.mark.verification
def test_unsteady_freestream_is_preserved():
    model, _ = _farfield_square()
    u0 = model.u_vec.x.array.copy()
    _run_steps(model, None, 3)
    np.testing.assert_allclose(model.u_vec.x.array, u0, rtol=0, atol=1e-12)


def _perturb(model):
    # a smooth density/pressure bump at constant temperature (acoustic and
    # entropy waves): the absolute pressure p_ref + p' scaled by 1 + bump
    V = model.functionspaces["V"]
    c = V.tabulate_dof_coordinates()
    bump = 0.05 * np.exp(-30 * ((c[:, 0] - 0.5)**2 + (c[:, 1] - 0.5)**2))
    a = model.u_vec.x.array.reshape(-1, 4)
    a[:, 0] += bump * (model.farfield_conditions['p'] + a[:, 0])
    model.u_vec.x.scatter_forward()


@pytest.mark.verification
def test_transverse_farfield_with_beta_zero_is_riemann():
    states = []
    for transverse, beta in ((False, 0.0), (True, 0.0), (True, 0.5)):
        model, tf = _farfield_square(transverse=transverse, beta=beta)
        _perturb(model)
        _run_steps(model, tf, 4)
        states.append(model.u_vec.x.array.copy())
    # beta = 0: delta relaxes from 0 and stays 0, the state is the riemann one
    np.testing.assert_allclose(states[1], states[0], rtol=0, atol=1e-12)
    # beta > 0 does change the boundary data once waves reach it
    assert MPI.COMM_WORLD.allreduce(float(np.abs(states[2] - states[0]).max()), MPI.MAX) > 1e-8


@pytest.mark.verification
def test_steady_ptc_and_unsteady_euler_reach_the_same_state():
    """PTC from a perturbed state and many large BDF1 steps converge to the
    same steady channel flow as the release Newton solve."""
    ref = euler_channel_model(n=6)
    seed_mean_flow(ref, amplitude=0.3)
    ref.report_positivity_limiter = False
    ref.set_up_solver()
    ref.solve_system()
    assert ref.solver_converged

    ptc = euler_channel_model(n=6)
    seed_mean_flow(ptc, amplitude=0.3)
    ptc.report_positivity_limiter = False
    ptc.use_ptc = True
    ptc.ptc_settings = {"cfl0": 10.0}
    ptc.set_up_solver()
    ptc.time_integrator.verbose = False
    ptc.solve_system()
    assert ptc.solver_converged
    np.testing.assert_allclose(ptc.u_vec.x.array, ref.u_vec.x.array, rtol=0, atol=1e-8)

    bdf = euler_channel_model(n=6)
    seed_mean_flow(bdf, amplitude=0.3)
    bdf.report_positivity_limiter = False
    integrator = bdf.enable_time_stepping()
    bdf.set_up_solver()
    integrator.verbose = False
    integrator.start()
    ctrl = StepSizeController(dt=1e3)
    for _ in range(30):
        assert integrator.advance(ctrl, order=1).accepted
    np.testing.assert_allclose(bdf.u_vec.x.array, ref.u_vec.x.array, rtol=0, atol=1e-7)


@pytest.mark.verification
def test_dual_time_steps_match_newton_steps():
    """Dual time solves the same R* = 0 per physical step, by PTC instead of
    plain Newton: the states agree to the inner tolerance."""
    states = []
    for dual in (False, True):
        model = euler_channel_model(n=6)
        seed_mean_flow(model, amplitude=0.3)
        model.report_positivity_limiter = False
        integrator = model.enable_time_stepping(dual_time=dual)
        model.set_up_solver()
        integrator.verbose = False
        integrator.start()
        ctrl = StepSizeController(dt=0.05)
        inner = CFLController(cfl0=100.0) if dual else None
        for _ in range(3):
            rec = integrator.advance(ctrl, newton_rtol=1e-10, newton_max_it=50, dual_time=inner)
            assert rec.accepted
        states.append(model.u_vec.x.array.copy())
    np.testing.assert_allclose(states[1], states[0], rtol=0, atol=1e-8)


def _ptc_channel(warm_start):
    model = euler_channel_model(n=6)
    seed_mean_flow(model, amplitude=0.3)
    model.report_positivity_limiter = False
    model.use_ptc = True
    model.ptc_settings = {"cfl0": 10.0}
    model.ptc_warm_start = warm_start
    model.set_up_solver()
    model.time_integrator.verbose = False
    return model


@pytest.mark.unit_test
def test_ptc_warm_start_continues_at_the_final_cfl():
    """After a converged steady solve, a solve from that end state at a slightly
    different angle of attack continues at the final CFL instead of ramping up
    from cfl0 again: fewer steps, same converged state."""
    steps, states = {}, {}
    for warm in (False, True):
        model = _ptc_channel(warm)
        integrator = model.time_integrator
        assert integrator._warm_start_cfl() is None          # nothing to continue from yet
        model.solve_system()
        assert model.solver_converged
        cold_steps = integrator.last_steady_steps
        assert cold_steps > 3
        alpha = model.farfield_conditions['alpha']
        model.set_angle_of_attack(alpha + np.radians(0.05))
        model.solve_system()
        assert model.solver_converged
        steps[warm] = integrator.last_steady_steps
        states[warm] = model.u_vec.x.array.copy()
    assert steps[True] < steps[False], steps
    np.testing.assert_allclose(states[True], states[False], rtol=0, atol=1e-7)


@pytest.mark.unit_test
def test_ptc_warm_start_needs_the_end_state_of_the_last_solve():
    model = _ptc_channel(True)
    integrator = model.time_integrator
    model.solve_system()
    assert integrator._warm_start_cfl() is not None
    model.u_vec.x.array[:] *= 1.0 + 1e-9
    model.u_vec.x.scatter_forward()
    assert integrator._warm_start_cfl() is None


# ----------------------------------------------------------------------
# transients in the primitive unknowns
# ----------------------------------------------------------------------
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Newton iterates depend on the partitioning")
@pytest.mark.verification
def test_transient_matches_the_conservative_trajectory():
    """
    The physical BDF term acts on the conserved state U(V), so every time step
    solves the same discrete equations as the former conservative unknowns
    did: the BDF2 trajectory of the released channel matches the one the
    conservative code computed (tests/data/euler_reference_transient.npz, in
    its time unit chord / sqrt(p_inf/rho_inf); here time is in chords per
    free-stream speed, so dt scales with that speed).
    """
    import os
    from cases import legacy_conservative, gather_by_original_cell
    ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                               "euler_reference_transient.npz"))
    model = euler_channel_model(n=8)
    seed_mean_flow(model, amplitude=0.3)
    model.report_positivity_limiter = False
    integrator = model.enable_time_stepping()
    model.set_up_solver()
    integrator.verbose = False
    integrator.start()
    dt = float(ref["dt"]) * model.material.dimensional['speed']
    control = StepSizeController(dt=dt, dt_start=dt)
    V = model.functionspaces["V"]
    bs = V.dofmap.index_map_bs
    for k, U_ref in enumerate(ref["states"]):
        rec = integrator.advance(control, order=2, newton_rtol=1e-13, newton_atol=1e-14, newton_max_it=30)
        assert rec.accepted
        V_rows = gather_by_original_cell(V, model.u_vec.x.array[:V.dofmap.index_map.size_local * bs].reshape(-1, bs))
        np.testing.assert_allclose(legacy_conservative(model, V_rows), U_ref, rtol=0, atol=1e-10,
                                   err_msg="step {}".format(k + 1))


def _box_model(material=None, preconditioning=None, n=10, farfield=None):
    """Euler in a closed box (slip walls all round), seeded with a smooth
    flow in the solver's units. The walls take the interior numerical flux
    against the mirror state, which carries exactly no mass or energy (the
    default slip wall's one-sided flux F(U).n + alpha (U - U_mirror) carries
    rho u.n, which only the steady state drives to zero)."""
    from dragonfly_sim.utils.Euler_utils import slip_wall_state
    from dragonfly_sim.core.mesh_manager import Mesh
    from dragonfly_sim.core.Euler_model import CompressibleEulerModel
    from cases import jittered_square
    mesh_obj = Mesh(jittered_square(n), lambda pts: np.zeros(pts.shape[1], dtype=bool))
    mesh_obj.tag_boundary_facets([(np.ones(mesh_obj.bdry_facets.size, dtype=bool), 2)])
    meta = {'quadrature_degree': 1}
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=meta)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(measure_metadata=meta)
    model = CompressibleEulerModel(mesh_obj, material=material, preconditioning=preconditioning)
    model.report_positivity_limiter = False
    model.define_farfield_conditions(farfield or FREESTREAM)
    model.define_elements_functionspaces()
    model.define_trial_testfunctions()
    mirror = slip_wall_state(model.mean_flow(model.u_vec), model.mesh_obj.n, model.n_mean)
    model.add_boundary_condition(2, mirror, "slip", inviscid_flux="numerical")
    model.compute_weakform()
    c = model.functionspaces["V"].tabulate_dof_coordinates()
    x, y = c[:, 0], c[:, 1]
    a = model.u_vec.x.array.reshape(-1, 4)
    a[:, 0] = 0.1 * np.sin(3 * x) * np.cos(2 * y)
    a[:, 1] = 0.3 * np.cos(2 * x + y)
    a[:, 2] = 0.2 * np.sin(x - 3 * y)
    a[:, 3] = 1.0 + 0.05 * np.cos(x + 2 * y)
    model.u_vec.x.scatter_forward()
    return model


def _totals(model):
    """Integrals of the conserved state over the domain (mass, momentum, energy)."""
    import dolfinx
    U = model.conserved(model.u_vec)
    comm = model.mesh_obj.mesh.comm
    return np.array([comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(U[k] * model.forms.dx)),
                                    op=MPI.SUM) for k in range(model.n_state)])


@pytest.mark.parametrize("fluid", ["air", "air-preconditioned", "air-dual-time", "water"])
@pytest.mark.verification
def test_closed_box_conserves_mass_and_energy_in_time(fluid):
    """With the BDF term on U(V), the total mass and energy of a closed box
    stay constant to round-off over unsteady steps, for every fluid and with
    preconditioning or dual time (no flux crosses a slip wall)."""
    from dragonfly_sim.core.materials import ConstantDensity
    from dragonfly_sim.core.time_integration import CFLController
    if fluid == "water":
        model = _box_model(material=ConstantDensity(),
                           farfield={'U': 5.0, 'T': 293.0, 'p': 101325.0, 'alpha': 0.0})
    else:
        model = _box_model(preconditioning=fluid != "air")
    integrator = model.enable_time_stepping(dual_time=fluid == "air-dual-time")
    model.set_up_solver()
    integrator.verbose = False
    integrator.start()
    before = _totals(model)
    control = StepSizeController(dt=0.05, dt_start=0.05)
    dual = CFLController(cfl0=10.0) if fluid == "air-dual-time" else None
    for _ in range(5):
        assert integrator.advance(control, order=2, newton_rtol=1e-10, newton_atol=1e-12,
                                  newton_max_it=30, dual_time=dual).accepted
    after = _totals(model)
    for k in (0, 3):                          # mass, energy
        assert abs(after[k] - before[k]) <= 1e-11 * abs(before[k]), (k, before[k], after[k])
    assert abs(after[1] - before[1]) > 1e-6   # momentum is not conserved: the walls push


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial comparison of fields")
@pytest.mark.verification
def test_bdf2_converges_at_second_order_in_time():
    """An acoustic pulse in a closed box (fixed mesh): the BDF2 solution at t
    = 0.2 against a dt/16 reference converges at second order in dt."""
    def run(dt, steps):
        model = _box_model(n=12)
        c = model.functionspaces["V"].tabulate_dof_coordinates()
        a = model.u_vec.x.array.reshape(-1, 4)
        a[:, 0] = 0.05 * np.exp(-40 * ((c[:, 0] - 0.5)**2 + (c[:, 1] - 0.5)**2))
        a[:, 1:3] = 0.0
        a[:, 3] = 1.0
        model.u_vec.x.scatter_forward()
        integrator = model.enable_time_stepping()
        model.set_up_solver()
        integrator.verbose = False
        integrator.start()
        control = StepSizeController(dt=dt, dt_start=dt)
        for _ in range(steps):
            assert integrator.advance(control, order=2, newton_rtol=1e-10, newton_atol=1e-12,
                                      newton_max_it=30).accepted
        return model.u_vec.x.array.copy()
    ref = run(0.2 / 160, 160)
    err = [np.linalg.norm(run(0.2 / n, n) - ref) for n in (10, 20, 40)]
    rates = np.log2(np.array(err[:-1]) / np.array(err[1:]))
    assert np.all(rates > 1.8), (err, rates)


@pytest.mark.unit_test
def test_incremental_newton_criterion_and_time_labelled_output(tmp_path):
    """Unsteady steps on the Newton update criterion match the residual
    criterion's state, and a time-dependent solution file takes the time."""
    states = []
    for criterion in ("residual", "incremental"):
        model = _box_model(n=6)
        integrator = model.enable_time_stepping()
        model.set_up_solver()
        integrator.verbose = False
        integrator.start()
        control = StepSizeController(dt=0.05, dt_start=0.05)
        for _ in range(2):
            assert integrator.advance(control, order=2, newton_rtol=1e-12, newton_atol=1e-12,
                                      newton_max_it=30, newton_criterion=criterion).accepted
        states.append(model.u_vec.x.array.copy())
    np.testing.assert_allclose(states[1], states[0], rtol=0, atol=1e-9)
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path / "unsteady") if comm.rank == 0 else None, root=0)
    model.define_sol_export_files(file_name=path, time_dependent=True)
    model.write_solution_output(None, time=0.05)
    model.write_solution_output(None, time=0.10)
    with pytest.raises(ValueError):
        model.write_solution_output(None, time=0.07)       # time must increase
    assert comm.bcast(os.path.isdir(path + ".bp") if comm.rank == 0 else None, root=0)
