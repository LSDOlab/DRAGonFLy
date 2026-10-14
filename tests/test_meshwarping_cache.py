"""
IDWarp_jax.compute returns the last forward warp when the wall displacement is
within 1e-12 of the one it was computed for (CSDL re-runs the model for each derivative evaluation of a design).
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

import csdl_alpha as csdl

from dragonfly_sim.core.shape_design import FFDShapeParameterization, ControlPointMotions
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model

from cases import jittered_square
from test_windtunnel_forward import BC, bottom_wall, in_tmp_dir  # noqa: F401


@pytest.fixture
def warper(in_tmp_dir):  # noqa: F811
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    shape = FFDShapeParameterization([5, 3], [2, 2], layers=[ControlPointMotions({1: 0.01})])
    wt = DG_windtunnel_model(jittered_square(8), BC, shape,
                             mesh_inner_bdry_function=bottom_wall,
                             poly_order=0, filename_suffix="test_warp_cache")
    wt.write_mesh_deformation = False
    wt.set_up_sim()
    mw = wt.mesh_warper
    mw.n_warps = 0
    warp = mw._warp

    def counted(*args, **kwargs):
        mw.n_warps += 1
        return warp(*args, **kwargs)

    mw._warp = counted
    mw.M_wall = wt.mesh.boundary_node_set((wt.WALL_TAG,)).coords.shape[0]
    yield mw
    recorder.stop()
    # close the output files before the temporary directory goes
    from dragonfly_sim.utils.filewriter import FileWriter
    for w in vars(getattr(wt.sim_model, 'flow', wt.sim_model)).values():
        if isinstance(w, FileWriter) and w.writer is not None:
            w.writer.close()


def run(mw, bdry):
    out = {}
    mw.compute({"bdry_motions": bdry, "shape_parameters": np.zeros(1)}, out)
    return out["mesh_node_motions"]


def test_same_wall_displacement_reuses_the_warp(warper):
    mw = warper
    bdry = np.zeros((mw.M_wall, mw.dimensions))
    bdry[:, 1] = 0.01 * np.sin(3.0 * np.arange(mw.M_wall))
    first = run(mw, bdry)
    assert mw.n_warps == 1
    second = run(mw, bdry.copy())
    assert mw.n_warps == 1
    assert np.array_equal(first, second)
    # the cache is its own copy: changing a returned array does not change the next one
    second[:] = 99.0
    assert np.array_equal(run(mw, bdry), first)
    assert mw.n_warps == 1


def test_reuse_is_within_the_tolerance_of_the_warped_displacement(warper):
    mw = warper
    assert mw.reuse_tol == 1e-12
    bdry = np.zeros((mw.M_wall, mw.dimensions))
    bdry[:, 1] = 0.01 * np.cos(2.0 * np.arange(mw.M_wall))
    first = run(mw, bdry)
    # inside the tolerance: reused, whatever the sign of the change
    for delta in (0.9e-12, -0.9e-12):
        shifted = bdry.copy()
        shifted[3, 1] += delta
        assert np.array_equal(run(mw, shifted), first)
    assert mw.n_warps == 1
    # outside the tolerance in a single entry: warped again
    shifted = bdry.copy()
    shifted[3, 1] += 2e-12
    run(mw, shifted)
    assert mw.n_warps == 2
    # the reference is the displacement actually warped: small steps do not add up
    ref = shifted.copy()
    run(mw, ref)
    assert mw.n_warps == 2
    for k in range(1, 4):
        step = ref.copy()
        step[0, 0] += k * 0.6e-12
        run(mw, step)
    assert mw.n_warps == 3          # k = 1 reused; k = 2 (1.2e-12 from ref) warped; k = 3 vs that


def test_changed_wall_displacement_warps_again(warper):
    mw = warper
    bdry = np.zeros((mw.M_wall, mw.dimensions))
    bdry[:, 1] = 0.01
    first = run(mw, bdry)
    other = run(mw, 2.0 * bdry)
    assert mw.n_warps == 2
    assert not np.array_equal(first, other)
    # going back to the first displacement is a new warp, equal to the first
    back = run(mw, bdry)
    assert mw.n_warps == 3
    assert np.array_equal(back, first)
