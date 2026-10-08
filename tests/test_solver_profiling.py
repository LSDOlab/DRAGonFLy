"""
utils/solver_profiling.py: the phase profiler, and the phases the package's
solves record (Euler Newton solve, LSPG ROM solve, mesh deformation).

The profiler is a process-wide singleton; the `profiler` fixture disables and
clears it after every test. Under mpirun every phase is entered by every rank
(barriers on), so a missing collective would hang here.
"""
import time

import numpy as np
import pytest
from mpi4py import MPI

from dragonfly_sim.utils.solver_profiling import PROFILER, PhaseProfiler, default_group
from dragonfly_sim.core.reduced_order_model import ReducedOrderModel

from cases import euler_channel_model


@pytest.fixture
def profiler():
    yield PROFILER
    PROFILER.configure(comm=None, enabled=False)
    PROFILER.reset()


def groups(profiler):
    return {default_group(name) for name in profiler.snapshot()}


# ----------------------------------------------------------------------
# The profiler itself
# ----------------------------------------------------------------------
def test_nested_phases_partition_the_time():
    prof = PhaseProfiler().configure(comm=None, enabled=True)
    with prof.phase("outer:a"):
        time.sleep(0.02)
        with prof.phase("inner:b"):
            time.sleep(0.03)
        with prof.phase("inner:b"):
            time.sleep(0.01)
    snap = prof.snapshot()
    a_self, a_total, a_count = snap["outer:a"]
    b_self, b_total, b_count = snap["inner:b"]
    assert (a_count, b_count) == (1, 2)
    assert b_self == pytest.approx(b_total)                 # no children
    assert a_self == pytest.approx(a_total - b_total, abs=1e-9)
    assert a_self == pytest.approx(0.02, abs=0.015)
    assert b_total == pytest.approx(0.04, abs=0.015)


def test_disabled_profiler_records_nothing():
    prof = PhaseProfiler()
    with prof.phase("assemble:residual"):
        pass
    assert prof.snapshot() == {}
    assert prof.report(print_fn=lambda *a: None) == {}


def test_report_unions_phases_across_ranks():
    """Ranks recording different phases -- or none at all -- reduce over the
    union of the names, without hanging (each rank calls report once)."""
    comm = MPI.COMM_WORLD
    prof = PhaseProfiler().configure(comm=comm, enabled=True, barriers=False)
    if comm.rank == 0:
        with prof.phase("assemble:residual"):
            time.sleep(0.01)
    if comm.rank == comm.size - 1:
        with prof.phase("solve:linear"):
            time.sleep(0.01)
    out = prof.report(print_fn=lambda *a: None, group_map=default_group)
    expected = {"assemble:residual", "solve:linear"}
    assert set(out["names"]) == expected
    assert set(out["groups"]) == {"assemble", "solve"}
    assert np.all(out["self_max"] > 0)


def test_default_group():
    assert default_group("assemble:jacobian") == "assemble"
    assert default_group("rom:project:extra") == "rom"
    assert default_group("other") == "other"


# ----------------------------------------------------------------------
# Instrumented solves
# ----------------------------------------------------------------------
def test_profiled_euler_solve_records_its_phases_and_changes_nothing(profiler):
    plain = euler_channel_model(n=6)
    plain.set_up_solver()
    plain.solve_system()
    reference = plain.u_vec.x.array.copy()

    model = euler_channel_model(n=6)
    model.enable_profiling(enabled=True, reset=True)
    model.set_up_solver()
    model.solve_system()
    assert model.solver_converged
    assert {"assemble", "solve", "limiter", "setup"} <= groups(profiler)
    snap = profiler.snapshot()
    for name in ("assemble:residual", "assemble:jacobian", "assemble:physical_residual",
                 "solve:linear", "limiter:positivity", "setup:solver"):
        assert snap[name][2] > 0, name
    out = model.profile_report(reset=True)
    assert out["grand"] > 0
    assert profiler.snapshot() == {}
    # bit-identical serially; in parallel two identical solves already differ
    # by ~1e-15 run to run
    if MPI.COMM_WORLD.size == 1:
        np.testing.assert_array_equal(model.u_vec.x.array, reference)
    else:
        np.testing.assert_allclose(model.u_vec.x.array, reference, rtol=1e-12, atol=1e-12)


def test_profiled_rom_solve_records_rom_phases(profiler):
    model = euler_channel_model(n=6)
    model.set_up_solver()
    model.solve_system()
    rom = ReducedOrderModel(rb_size=2, min_snapshots=1)
    rom.set_up(model)
    rom.add_snapshot(model, np.array([0.0]))
    model.u_vec.x.array[:] *= 1.001
    model.u_vec.x.scatter_forward()
    rom.add_snapshot(model, np.array([1.0]))

    model.enable_profiling(enabled=True, reset=True)
    model.interpolate_solution_vector(None)
    result = rom.solve(model, np.array([0.5]))
    assert result is not None
    snap = profiler.snapshot()
    for name in ("rom:basis", "rom:project", "rom:expand", "solve:reduced", "assemble:residual",
                 "assemble:jacobian", "limiter:positivity", "setup:interpolate_ic"):
        assert name in snap, name
    assert "solve:linear" not in snap          # no full-order linear solve in a ROM solve


def test_mesh_phases(profiler):
    model = euler_channel_model(n=4)
    mesh = model.mesh_obj
    PROFILER.configure(comm=mesh.mesh.comm, enabled=True)
    PROFILER.reset()
    mesh.apply_node_motions(np.zeros((mesh.n_owned_nodes, mesh.mesh.geometry.dim)))
    mesh.reset_nodes()
    snap = PROFILER.snapshot()
    assert snap["mesh:deform"][2] == 1
    assert snap["mesh:reset"][2] == 2            # once inside apply_node_motions, once directly
