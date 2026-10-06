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
