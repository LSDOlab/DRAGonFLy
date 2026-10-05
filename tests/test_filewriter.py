"""
pytest suite for filewriter.FileWriter.

What these tests guard against is output that looks right in serial and is
wrong in parallel. ParaView's VTX reader concatenates the per-rank blocks of
a parallel write without merging the points the blocks share, so DG0 and
continuous fields written per rank show the MPI partition boundaries as
cracks. FileWriter gathers those spaces to rank 0 and writes DG p >= 1 per
rank (VTX writes those with separate points for every cell, so partition
boundaries add nothing).

Two layers:

- In-process tests of the output meshes and functions FileWriter hands to
  dolfinx's VTXWriter, including the mesh_deformation and time_dependent
  keywords. They are communicator-agnostic; a serial run re-runs them under
  mpirun -n 3 (test_runs_under_multiple_mpi_ranks).
- test_vtx_output_is_independent_of_the_number_of_ranks writes the cases of
  vtx_replay/write_cases.py on 1 and 3 ranks, replays each file the way
  ParaView's VTX reader assembles it (vtx_replay/vtxdump.cpp, compiled
  against this environment's ADIOS2) and requires identical structure and
  values. It skips without mpirun, a C++ compiler, the ADIOS2 C++ headers or
  the vtk package.

    OMP_NUM_THREADS=1 python -m pytest tests/test_filewriter.py -v
    OMP_NUM_THREADS=1 mpirun -n 3 python -m pytest tests/test_filewriter.py -q \\
        -p no:cacheprovider -k "not runs_under_multiple_mpi_ranks and not number_of_ranks"
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from dragonfly_sim.utils.filewriter import FileWriter

_MPI_CHILD_ENV = "FILEWRITER_TEST_MPI_CHILD"
_REPLAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vtx_replay")


def _load_helper(name):
    spec = importlib.util.spec_from_file_location(
        "vtx_replay_" + name, os.path.join(_REPLAY_DIR, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def outdir():
    """A temporary directory with the same path on every rank."""
    comm = MPI.COMM_WORLD
    path = tempfile.mkdtemp(prefix="filewriter_test_") if comm.rank == 0 else None
    path = comm.bcast(path, root=0)
    yield path
    comm.Barrier()
    if comm.rank == 0:
        shutil.rmtree(path, ignore_errors=True)


def quad_mesh(comm=MPI.COMM_WORLD):
    """A jittered quad mesh, so the cells are not all congruent."""
    msh = dolfinx.mesh.create_rectangle(
        comm, [[0, 0], [2, 1]], [12, 6], dolfinx.mesh.CellType.quadrilateral)
    rng = np.random.default_rng(5)
    jitter = 0.02 * rng.standard_normal((msh.geometry.index_map().size_global, 2))
    global_nodes = msh.geometry.index_map().local_to_global(
        np.arange(msh.geometry.x.shape[0], dtype=np.int32))
    # Indexed by global node number, so owners and ghosts move together
    msh.geometry.x[:, :2] += jitter[global_nodes]
    return msh


def deformation(x, amp):
    d = np.zeros_like(x)
    d[:, 0] = amp * np.sin(np.pi * x[:, 1])
    d[:, 1] = 0.5 * amp * np.sin(np.pi * x[:, 0])
    return d


def field(x, bs):
    """(ufl expression, numpy evaluator) pair with `bs` components."""
    comps_ufl = [ufl.sin(3 * x[0] + i) * ufl.cos(2 * x[1]) + i * x[0] for i in range(bs)]
    expr = comps_ufl[0] if bs == 1 else ufl.as_vector(comps_ufl)

    def evaluate(p):
        return np.stack([np.sin(3 * p[:, 0] + i) * np.cos(2 * p[:, 1]) + i * p[:, 0]
                         for i in range(bs)], axis=1)
    return expr, evaluate


SPACES = {
    "dg0vec": (("DG", 0, (4,)), True),
    "dg0": (("DG", 0), True),
    "dg1vec": (("DG", 1, (4,)), False),
    "dg3": (("DG", 3), False),
    "cg1vec": (("Lagrange", 1, (2,)), True),
    "cg3": (("Lagrange", 3), True),
}


@pytest.mark.parametrize("space", SPACES)
def test_dg0_and_continuous_spaces_are_gathered_and_dg_p_ge_1_is_written_per_rank(space, outdir):
    element, gathered = SPACES[space]
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), element)
    w = FileWriter(os.path.join(outdir, space), comm, V)
    assert w.gather == gathered
    has_writer = comm.allgather(w.writer is not None)
    assert has_writer == ([True] + [False] * (comm.size - 1) if gathered else [True] * comm.size)


def test_gathered_output_mesh_holds_every_cell_and_node_exactly_once(outdir):
    """A single block without duplicated points is what removes the seams."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    V = dolfinx.fem.functionspace(msh, ("DG", 0, (4,)))
    w = FileWriter(os.path.join(outdir, "dg0"), comm, V)
    if comm.rank == 0:
        tdim = msh.topology.dim
        assert w.target_mesh.comm.size == 1
        assert w.target_mesh.topology.index_map(tdim).size_local == \
            msh.topology.index_map(tdim).size_global
        assert w.target_mesh.geometry.x.shape[0] == msh.geometry.index_map().size_global


def test_per_rank_output_mesh_holds_the_owned_cells_and_no_ghosts(outdir):
    """Ghost cells would be written by two ranks and overlap."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    tdim = msh.topology.dim
    V = dolfinx.fem.functionspace(msh, ("DG", 1, (4,)))
    w = FileWriter(os.path.join(outdir, "dg1"), comm, V)
    cells = w.target_mesh.topology.index_map(tdim)
    assert cells.num_ghosts == 0
    assert cells.size_local == msh.topology.index_map(tdim).size_local
    assert np.all(w.target_rows < msh.topology.index_map(tdim).size_local)


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("path", ["function", "expression"])
def test_written_values_are_the_nodal_values_at_the_output_dof_coordinates(space, path, outdir):
    """Lagrange interpolation is nodal, so every output dof must hold the
    field at its own coordinates -- whichever rank the parent dof lived on."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    V = dolfinx.fem.functionspace(msh, SPACES[space][0])
    bs = V.dofmap.bs
    expr, evaluate = field(ufl.SpatialCoordinate(msh), bs)
    w = FileWriter(os.path.join(outdir, space), comm, V)
    if path == "function":
        u = dolfinx.fem.Function(V)
        u.interpolate(dolfinx.fem.Expression(expr, V.element.interpolation_points))
        w.interpolate_and_write(u, write_counter=0)
    else:
        w.interpolate_and_write(expr, write_counter=0)
    if w.writer is not None:
        coords = w.target_funcspace.tabulate_dof_coordinates()
        values = w.function.x.array.reshape(-1, bs)
        assert np.allclose(values, evaluate(coords), rtol=0, atol=1e-12)


def test_dof_mapping_check_rejects_a_scrambled_mapping(outdir):
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), ("DG", 1, (4,)))
    w = FileWriter(os.path.join(outdir, "dg1"), comm, V)
    w.target_rows = w.target_rows[::-1].copy()
    with pytest.raises(RuntimeError, match="dof mapping"):
        w._check_dof_mapping()


@pytest.mark.parametrize("element", [("DG", 0, (4,)), ("DG", 1, (4,)), ("Lagrange", 1, (2,))])
def test_mesh_deformation_true_follows_the_parent_mesh_and_false_keeps_the_baseline(element, outdir):
    """Both writers live on the same space, so a deforming writer must not
    move the geometry of the baseline writer's (shared, cached) output mesh."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    V = dolfinx.fem.functionspace(msh, element)
    expr, _ = field(ufl.SpatialCoordinate(msh), V.dofmap.bs)
    still = FileWriter(os.path.join(outdir, "still"), comm, V, mesh_deformation=False)
    moving = FileWriter(os.path.join(outdir, "moving"), comm, V, mesh_deformation=True)
    baseline = {w: w.target_mesh.geometry.x.copy() for w in (still, moving) if w.writer is not None}

    x0 = msh.geometry.x.copy()
    for k, amp in enumerate((0.1, -0.05)):
        msh.geometry.x[:] = x0 + deformation(x0, amp)
        still.interpolate_and_write(expr, write_counter=k)
        moving.interpolate_and_write(expr, write_counter=k)
        if still.writer is not None:
            assert np.array_equal(still.target_mesh.geometry.x, baseline[still])
        if moving.writer is not None:
            expected = baseline[moving] + deformation(baseline[moving], amp)
            assert np.allclose(moving.target_mesh.geometry.x, expected, rtol=0, atol=1e-14)


def test_time_dependent_frames_are_labelled_by_increasing_time(outdir):
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), ("DG", 1, (4,)))
    expr, _ = field(ufl.SpatialCoordinate(V.mesh), 4)
    w = FileWriter(os.path.join(outdir, "timed"), comm, V, time_dependent=True)
    for t in (0.0, 0.05, 0.1):
        w.interpolate_and_write(expr, time=t)
        assert w.last_time == t
    with pytest.raises(ValueError, match="time="):
        w.interpolate_and_write(expr)
    with pytest.raises(ValueError, match="time="):
        w.interpolate_and_write(expr, write_counter=3, time=0.2)
    with pytest.raises(ValueError, match="does not increase"):
        w.interpolate_and_write(expr, time=0.1)


def test_counter_labelled_writer_rejects_time_and_keeps_a_pinned_counter(outdir):
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), ("DG", 0, (4,)))
    expr, _ = field(ufl.SpatialCoordinate(V.mesh), 4)
    w = FileWriter(os.path.join(outdir, "counted"), comm, V)
    w.interpolate_and_write(expr, write_counter=5)
    assert w.write_counter == 0
    w.interpolate_and_write(expr)
    assert w.write_counter == 1
    with pytest.raises(ValueError, match="time_dependent"):
        w.interpolate_and_write(expr, time=1.0)


def _run_mpi(n, args, timeout=900):
    env = dict(os.environ)
    env[_MPI_CHILD_ENV] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    try:
        return subprocess.run(["mpirun", "-n", str(n)] + args, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the {}-rank run did not finish within {}s, which usually means one rank failed "
            "and the rest are blocked in a collective".format(n, timeout))


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun is not available")
@pytest.mark.skipif(os.environ.get(_MPI_CHILD_ENV) == "1",
                    reason="already running inside the spawned MPI child")
def test_runs_under_multiple_mpi_ranks():
    """Re-run this file under mpirun, so a serial pytest invocation also
    covers the gather and the per-rank paths with more than one rank."""
    proc = _run_mpi(3, [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                        __file__, "-k", "not runs_under_multiple_mpi_ranks and not number_of_ranks"])
    assert proc.returncode == 0, "3-rank run failed:\n{}\n{}".format(proc.stdout, proc.stderr)


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun is not available")
@pytest.mark.skipif(os.environ.get(_MPI_CHILD_ENV) == "1" or MPI.COMM_WORLD.size > 1,
                    reason="spawns its own MPI runs")
def test_vtx_output_is_independent_of_the_number_of_ranks(tmp_path):
    """The files ParaView assembles from a 3-rank write must match a serial
    write: same points, no duplicated points in gathered output, the same
    exterior edges/faces, the same values. A partition seam shows up here as
    extra points and extra exterior edges/faces."""
    pytest.importorskip("vtk")
    replay = _load_helper("replay")
    vtxdump = replay.build_vtxdump(str(tmp_path))
    if vtxdump is None:
        pytest.skip("no C++ compiler or no ADIOS2 C++ headers/libraries in this environment")

    runs = {}
    for n in (1, 3):
        out = tmp_path / "np{}".format(n)
        out.mkdir()
        proc = _run_mpi(n, [sys.executable, os.path.join(_REPLAY_DIR, "write_cases.py"), str(out)])
        assert proc.returncode == 0, "{}-rank write failed:\n{}\n{}".format(n, proc.stdout, proc.stderr)
        runs[n] = out
    manifest = json.loads((runs[3] / "manifest.json").read_text())
    amplitudes, times = manifest["amplitudes"], manifest["times"]

    failures = []
    for entry in manifest["files"]:
        name = entry["name"]
        dumps = {n: [replay.dump(vtxdump, str(runs[n] / (name + ".bp")),
                                 str(runs[n] / "dump" / name / str(k)), k)
                     for k in range(len(amplitudes))] for n in runs}
        first = {n: replay.load(dumps[n][0]) for n in runs}
        for k in range(len(amplitudes)):
            serial, parallel = replay.summary(dumps[1][k]), replay.summary(dumps[3][k])
            expected_blocks = 1 if entry["gathered"] else 3
            if parallel["blocks"] != expected_blocks:
                failures.append("{} frame {}: {} blocks, expected {}".format(
                    name, k, parallel["blocks"], expected_blocks))
            for key in ("points", "unique_points", "cells", "exterior"):
                if serial[key] != parallel[key]:
                    failures.append("{} frame {}: {} = {} on 3 ranks vs {} on 1 rank".format(
                        name, k, key, parallel[key], serial[key]))
            if entry["gathered"] and parallel["points"] != parallel["unique_points"]:
                failures.append("{} frame {}: duplicated points in gathered output".format(name, k))
            diff = replay.max_value_difference(dumps[1][k], dumps[3][k])
            if diff > 1e-12:
                failures.append("{} frame {}: values differ by {:.2e}".format(name, k, diff))

            frame = replay.load(dumps[3][k])
            if entry["points_are_nodes"]:
                x0 = first[3]["geometry"]
                amp = amplitudes[k] if entry["deform"] else 0.0
                err = np.abs(frame["geometry"] - (x0 + deformation(x0, amp))).max()
                if err > 1e-12:
                    failures.append("{} frame {}: geometry off by {:.2e}".format(name, k, err))
            if entry["time"] and frame["time"] != times[k]:
                failures.append("{} frame {}: time label {} != {}".format(
                    name, k, frame["time"], times[k]))
            elif not entry["time"] and frame["time"] != k:
                failures.append("{} frame {}: counter label {}".format(name, k, frame["time"]))
    assert not failures, "\n".join(failures)
