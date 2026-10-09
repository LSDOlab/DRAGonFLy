"""
DG_windtunnel_model and DG_postprocessor without an FFD block or mesh warper:
a forward flow analysis on a fixed mesh, with and without a CSDL graph, and
the adjoint derivative with respect to alpha against finite differences.

The domain is a jittered unit square whose bottom side is the "body" wall;
the rest of the boundary is the far field, split into inflow and outflow.
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

import csdl_alpha as csdl

from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model

from cases import jittered_square

BC = {'inlet': {'rho': 1.0, 'p': 1.0, 'M': 0.5, 'alpha': np.radians(2.0)}, 'outlet': {'p': 1.0}}


def bottom_wall(x):
    return x[1] < 1e-8


@pytest.fixture
def in_tmp_dir(tmp_path):
    """The windtunnel writes output files into the working directory: use one
    temporary directory, shared by all ranks."""
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    cwd = os.getcwd()
    os.chdir(path)
    yield
    comm.Barrier()
    os.chdir(cwd)


def make_windtunnel():
    wt = DG_windtunnel_model(jittered_square(8), BC, mesh_inner_bdry_function=bottom_wall,
                             poly_order=0, filename_suffix="test_forward")
    wt.euler_residual_conv_limit = 1e-12
    wt.log_cm_alpha = False
    wt.set_up_sim()
    wt.postprocessor.log_cm_alpha = False
    return wt


def test_forward_analysis_without_csdl(in_tmp_dir):
    wt = make_windtunnel()
    assert wt.ffd is None and wt.mesh_warper is None
    u, converged, forces = wt.solve_forward()
    assert converged
    assert set(forces) >= {"L", "D", "M", "c_l", "c_d", "c_m"}
    # forces_and_coefficients agrees with the postprocessor's own assemblies
    L, c_l = wt.postprocessor.compute_cl(u)
    assert forces["L"] == L and forces["c_l"] == c_l
    with pytest.raises(RuntimeError):
        wt.wall_geometry()


def test_forward_csdl_graph_alpha_derivative(in_tmp_dir):
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    wt = make_windtunnel()
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    sim.run()
    D0 = float(sim[out.D][0])
    totals = sim.compute_totals()
    dD = float(np.asarray(totals[out.D, alpha]).ravel()[0])

    # finite differences of plain forward solves
    h = 1e-6
    a0 = BC['inlet']['alpha']
    Dp = wt.solve_forward(a0 + h)[2]["D"]
    Dm = wt.solve_forward(a0 - h)[2]["D"]
    fd = (Dp - Dm) / (2 * h)
    assert abs(fd - dD) < 1e-5 * max(abs(fd), 1e-12), (fd, dD)
    assert abs(wt.solve_forward(a0)[2]["D"] - D0) < 1e-10


def rans_alpha_derivative(release, alphas):
    """D and dD/dalpha of a laminar-ish RANS flow at each alpha in turn, in one
    CSDL graph (forward solve, then adjoint, per evaluation)."""
    from dragonfly_sim.core.RANS_model import CompressibleRANSModel
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    wt = DG_windtunnel_model(jittered_square(8), BC, mesh_inner_bdry_function=bottom_wall,
                             poly_order=0, filename_suffix="test_release",
                             model_class=CompressibleRANSModel, model_kwargs={'Re': 1e3})
    wt.euler_residual_conv_limit = 1e-12
    wt.max_newton_iterations = 400
    wt.log_cm_alpha = False
    wt.release_solver_memory = release
    wt.set_up_sim()
    wt.postprocessor.log_cm_alpha = False
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    results = []
    for a in alphas:
        sim[alpha] = np.array([a])
        sim.run()
        totals = sim.compute_totals()
        results.append((float(sim[out.D][0]), float(np.asarray(totals[out.D, alpha]).ravel()[0])))
    # close the output files here: the graph keeps the model alive past the
    # test, and a VTX writer closed after leaving the temporary directory fails
    from dragonfly_sim.utils.filewriter import FileWriter
    for writer in vars(wt.sim_model.flow).values():
        if isinstance(writer, FileWriter) and writer.writer is not None:
            writer.writer.close()
    return wt, results


def test_release_solver_memory_gives_the_same_rans_derivatives(in_tmp_dir):
    """release_solver_memory frees the flow solver after every forward solve and
    the condensed dR/dU after every adjoint; both are rebuilt when next needed,
    so values and adjoint derivatives are unchanged."""
    alphas = [np.radians(2.0), np.radians(2.5)]
    _, kept = rans_alpha_derivative(False, alphas)
    wt, released = rans_alpha_derivative(True, alphas)
    flow = wt.sim_model.flow
    assert not flow.solver_is_set_up and not hasattr(flow, "_dRdu_condensed")
    for (D, dD), (D_r, dD_r) in zip(kept, released):
        assert abs(D_r - D) <= 1e-10 * abs(D), (D, D_r)
        assert abs(dD_r - dD) <= 1e-8 * abs(dD), (dD, dD_r)


@pytest.mark.parametrize("once", [False, True])
def test_solution_file_fields_and_write_once_per_design(once, in_tmp_dir, monkeypatch):
    """The solution file holds the state as separate fields plus velocity and
    pressure; with write_once_per_design, re-evaluating a design (CSDL re-runs
    the model for the derivatives) writes no second frame."""
    from dragonfly_sim.core.RANS_model import CompressibleRANSModel
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    wt = DG_windtunnel_model(jittered_square(6), BC, mesh_inner_bdry_function=bottom_wall,
                             poly_order=0, filename_suffix="test_fields",
                             model_class=CompressibleRANSModel, model_kwargs={'Re': 1e3})
    wt.max_newton_iterations = 400
    wt.log_cm_alpha = False
    wt.write_once_per_design = once
    wt.set_up_sim()
    wt.postprocessor.log_cm_alpha = False
    flow = wt.sim_model.flow
    writer = flow.fom_solution_writer
    assert writer.field_names == ["density", "momentum", "energy", "rho_nu_tilde",
                                  "velocity", "pressure"]
    labels = []
    monkeypatch.setattr(writer, "interpolate_and_write",
                        lambda values, write_counter=None, time=None: labels.append(
                            (write_counter, {k: v.x.array.copy() for k, v in values.items()})))
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    sim.run()
    sim.compute_totals()            # evaluates the same design again
    sim[alpha] = np.array([BC['inlet']['alpha'] + 0.01])
    sim.run()
    assert [k for k, _ in labels] == ([0, 2] if once else [0, 1, 2])

    # the fields of the last frame are the components of the state
    values = labels[-1][1]
    U = flow.u_vec.x.array.reshape(-1, flow.n_state)
    np.testing.assert_allclose(values["density"], U[:, 0], rtol=1e-14)
    np.testing.assert_allclose(values["momentum"].reshape(-1, 2), U[:, 1:3], rtol=1e-14)
    np.testing.assert_allclose(values["energy"], U[:, 3], rtol=1e-14)
    np.testing.assert_allclose(values["rho_nu_tilde"], U[:, 4], rtol=1e-14)
    np.testing.assert_allclose(values["velocity"].reshape(-1, 2), U[:, 1:3] / U[:, :1], rtol=1e-13)
    p = 0.4 * (U[:, 3] - 0.5 * (U[:, 1] ** 2 + U[:, 2] ** 2) / U[:, 0])
    np.testing.assert_allclose(values["pressure"], p, rtol=1e-12)
    for w in (writer, wt.mesh._deformation_writer):
        if w is not None and w.writer is not None:
            w.writer.close()


def test_muscl_euler_alpha_derivative(in_tmp_dir):
    """MUSCLEulerModel: the Euler model with the Green-Gauss MUSCL
    reconstruction converges by PTC, takes the condensed path, and its
    adjoint dD/dalpha matches finite differences of forward solves."""
    from dragonfly_sim.core.MUSCL_Euler_model import MUSCLEulerModel
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    wt = DG_windtunnel_model(jittered_square(8), BC, mesh_inner_bdry_function=bottom_wall,
                             poly_order=0, filename_suffix="test_muscl", model_class=MUSCLEulerModel)
    wt.euler_residual_conv_limit = 1e-12
    wt.max_newton_iterations = 400
    wt.log_cm_alpha = False
    wt.release_solver_memory = True
    wt.set_up_sim()
    wt.postprocessor.log_cm_alpha = False
    assert wt.sim_model.has_condensed_jacobian and wt.sim_model.face_traces is not None
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    out.D.set_as_objective()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    sim.run()
    assert wt.sim_model.last_solve_converged
    D0 = float(sim[out.D][0])
    dD = float(np.asarray(sim.compute_totals()[out.D, alpha]).ravel()[0])
    h = 1e-6
    a0 = BC['inlet']['alpha']
    fd = (wt.solve_forward(a0 + h)[2]["D"] - wt.solve_forward(a0 - h)[2]["D"]) / (2 * h)
    assert abs(fd - dD) < 1e-5 * max(abs(fd), 1e-12), (fd, dD)
    assert abs(wt.solve_forward(a0)[2]["D"] - D0) < 1e-10
    w = wt.sim_model.fom_solution_writer
    if w.writer is not None:
        w.writer.close()
