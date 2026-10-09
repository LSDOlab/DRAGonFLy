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

from cases import euler_channel_model, seed_mean_flow, FREESTREAM, INLET_TAG, OUTLET_TAG, WALL_TAG


# ----------------------------------------------------------------------
# pure Python
# ----------------------------------------------------------------------
def test_bdf_coefficients():
    assert bdf_coefficients(1) == (1.0, -1.0, 0.0)
    np.testing.assert_allclose(bdf_coefficients(2), (1.5, -2.0, 0.5))
    np.testing.assert_allclose(bdf_coefficients(2, 1.0), (1.5, -2.0, 0.5))
    for w in (0.5, 1.5, 3.0):
        a = bdf_coefficients(2, w)
        assert abs(sum(a)) < 1e-14                    # consistency: constants are steady
        # exact for linear in time: a0 t1 + a1 t0 + a2 t_{-1} = dt with t1 = 1, t0 = 0, t_{-1} = -1/w
        assert abs(a[0] * 1.0 + a[2] * (-1.0 / w) - 1.0) < 1e-14


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


def test_step_size_controller():
    s = StepSizeController(dt=0.05, dt_start=1e-3, growth=1.5)
    assert s.first(None) == 1e-3 and s.first(1e-3) == pytest.approx(1.5e-3) and s.first(0.04) == 0.05
    assert StepSizeController(dt=0.05).first(None) == 0.05


# ----------------------------------------------------------------------
# with a model
# ----------------------------------------------------------------------
def test_checkpoint_round_trip(tmp_path):
    comm = MPI.COMM_WORLD
    model = euler_channel_model(n=6)
    seed_mean_flow(model)
    path = comm.bcast(str(tmp_path / "ckpt.npz") if comm.rank == 0 else None, root=0)
    ref = model.u_vec.x.array.copy()
    save_checkpoint(path, [model.u_vec], {"step": 3, "t": 0.15})
    model.u_vec.x.array[:] = 0.0
    meta = load_checkpoint(path, [model.u_vec])
    assert meta == {"step": 3, "t": 0.15}
    np.testing.assert_array_equal(model.u_vec.x.array, ref)


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
    model.define_inlet_outlet_conditions(FREESTREAM)
    model.define_elements_functionspaces()
    model.define_trial_testfunctions()
    model.compute_initial_conditions_from_inlet()
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


def test_unsteady_freestream_is_preserved():
    model, _ = _farfield_square()
    u0 = model.u_vec.x.array.copy()
    _run_steps(model, None, 3)
    np.testing.assert_allclose(model.u_vec.x.array, u0, rtol=0, atol=1e-12)


def _perturb(model):
    # a smooth density/pressure bump (acoustic and entropy waves)
    V = model.functionspaces["V"]
    c = V.tabulate_dof_coordinates()
    bump = 0.05 * np.exp(-30 * ((c[:, 0] - 0.5)**2 + (c[:, 1] - 0.5)**2))
    a = model.u_vec.x.array.reshape(-1, 4)
    a[:, 0] *= 1 + bump
    a[:, 3] *= 1 + bump
    model.u_vec.x.scatter_forward()


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
