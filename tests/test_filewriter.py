"""
pytest suite for filewriter.FileWriter.

What these tests guard against is output that looks right in serial and is
wrong in parallel. ParaView's VTX reader concatenates the per-rank blocks of
a parallel write without merging the points the blocks share, so DG0 and
continuous fields written per rank show the MPI partition boundaries as
cracks. FileWriter gathers those spaces to rank 0 and writes DG p >= 1 per
rank (VTX writes those with separate points for every cell, so partition
boundaries add nothing).

The tests check the output meshes and functions FileWriter hands to dolfinx's
VTXWriter, including the mesh_deformation and time_dependent keywords, on a
small mesh in a temporary directory. They are communicator-agnostic; a
serial run re-runs them under mpirun -n 3 (test_runs_under_multiple_mpi_ranks),
which is where the gather and per-rank paths actually differ.

    OMP_NUM_THREADS=1 python -m pytest tests/test_filewriter.py -v
    OMP_NUM_THREADS=1 mpirun -n 3 python -m pytest tests/test_filewriter.py -q \\
        -p no:cacheprovider -k "not runs_under_multiple_mpi_ranks"
"""

import os
import shutil
import subprocess
import sys
import tempfile

import dolfinx
import dolfinx.plot
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from dragonfly_sim.utils.filewriter import FileWriter

_MPI_CHILD_ENV = "FILEWRITER_TEST_MPI_CHILD"


@pytest.fixture
def make_writer():
    """FileWriter factory writing into a temporary directory shared by all ranks.

    The writers' files are closed before the directory is removed: a failed
    test keeps its writers alive, and ADIOS2 aborts the process when a writer
    destroyed later finds its directory gone.
    """
    comm = MPI.COMM_WORLD
    path = tempfile.mkdtemp(prefix="filewriter_test_") if comm.rank == 0 else None
    path = comm.bcast(path, root=0)
    writers = []

    def make(name, V, **keywords):
        writers.append(FileWriter(os.path.join(path, name), comm, V, **keywords))
        return writers[-1]

    yield make
    for w in writers:
        if w.writer is not None:
            w.writer.close()
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
def test_dg0_and_continuous_spaces_are_gathered_and_dg_p_ge_1_is_written_per_rank(space, make_writer):
    element, gathered = SPACES[space]
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), element)
    w = make_writer(space, V)
    assert w.gather == gathered
    has_writer = comm.allgather(w.writer is not None)
    assert has_writer == ([True] + [False] * (comm.size - 1) if gathered else [True] * comm.size)


def test_gathered_output_mesh_holds_every_cell_and_node_exactly_once(make_writer):
    """A single block without duplicated points is what removes the seams."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    V = dolfinx.fem.functionspace(msh, ("DG", 0, (4,)))
    w = make_writer("dg0", V)
    if comm.rank == 0:
        tdim = msh.topology.dim
        assert w.target_mesh.comm.size == 1
        assert w.target_mesh.topology.index_map(tdim).size_local == \
            msh.topology.index_map(tdim).size_global
        assert w.target_mesh.geometry.x.shape[0] == msh.geometry.index_map().size_global


def test_per_rank_output_mesh_holds_the_owned_cells_and_no_ghosts(make_writer):
    """Ghost cells would be written by two ranks and overlap."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    tdim = msh.topology.dim
    V = dolfinx.fem.functionspace(msh, ("DG", 1, (4,)))
    w = make_writer("dg1", V)
    cells = w.target_mesh.topology.index_map(tdim)
    assert cells.num_ghosts == 0
    assert cells.size_local == msh.topology.index_map(tdim).size_local
    assert np.all(w.target_rows < msh.topology.index_map(tdim).size_local)


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("path", ["function", "expression"])
def test_written_values_are_the_nodal_values_at_the_output_dof_coordinates(space, path, make_writer):
    """Lagrange interpolation is nodal, so every output dof must hold the
    field at its own coordinates -- whichever rank the parent dof lived on."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    V = dolfinx.fem.functionspace(msh, SPACES[space][0])
    bs = V.dofmap.bs
    expr, evaluate = field(ufl.SpatialCoordinate(msh), bs)
    w = make_writer(space, V)
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


@pytest.mark.parametrize("space", SPACES)
def test_vtx_blocks_add_up_to_a_serial_write(space, make_writer):
    """ParaView concatenates the per-rank VTX blocks without merging shared
    points, so the blocks must add up to the points and cells of a serial
    write; a partition seam shows up as extra (duplicated) points or cells.
    dolfinx.plot.vtk_mesh builds the same arrays VTXWriter writes: the mesh
    geometry when only DG0 fields are written, the function-space nodes
    otherwise."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    tdim = msh.topology.dim
    V = dolfinx.fem.functionspace(msh, SPACES[space][0])
    w = make_writer(space, V)

    points = cells = 0
    if w.writer is not None:
        cellwise = V.dofmap.dof_layout.num_dofs == 1 and V.ufl_element().discontinuous
        topology, _, x = dolfinx.plot.vtk_mesh(w.target_mesh if cellwise else w.target_funcspace)
        points, cells = x.shape[0], len(topology) // (topology[0] + 1)
    points, cells = comm.allreduce(points), comm.allreduce(cells)

    num_cells = msh.topology.index_map(tdim).size_global
    if V.dofmap.dof_layout.num_dofs == 1:
        serial_points = msh.geometry.index_map().size_global
    elif V.ufl_element().discontinuous:
        serial_points = num_cells * V.dofmap.dof_layout.num_dofs
    else:
        serial_points = V.dofmap.index_map.size_global
    assert cells == num_cells
    assert points == serial_points


def test_dof_mapping_check_rejects_a_scrambled_mapping(make_writer):
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), ("DG", 1, (4,)))
    w = make_writer("dg1", V)
    w.target_rows = w.target_rows[::-1].copy()
    with pytest.raises(RuntimeError, match="dof mapping"):
        w._check_dof_mapping()


@pytest.mark.parametrize("element", [("DG", 0, (4,)), ("DG", 1, (4,)), ("Lagrange", 1, (2,))])
def test_mesh_deformation_true_follows_the_parent_mesh_and_false_keeps_the_baseline(element, make_writer):
    """Both writers live on the same space, so a deforming writer must not
    move the geometry of the baseline writer's (shared, cached) output mesh."""
    comm = MPI.COMM_WORLD
    msh = quad_mesh()
    V = dolfinx.fem.functionspace(msh, element)
    expr, _ = field(ufl.SpatialCoordinate(msh), V.dofmap.bs)
    still = make_writer("still", V, mesh_deformation=False)
    moving = make_writer("moving", V, mesh_deformation=True)
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


def test_time_dependent_frames_are_labelled_by_increasing_time(make_writer):
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), ("DG", 1, (4,)))
    expr, _ = field(ufl.SpatialCoordinate(V.mesh), 4)
    w = make_writer("timed", V, time_dependent=True)
    for t in (0.0, 0.05, 0.1):
        w.interpolate_and_write(expr, time=t)
        assert w.last_time == t
    with pytest.raises(ValueError, match="time="):
        w.interpolate_and_write(expr)
    with pytest.raises(ValueError, match="time="):
        w.interpolate_and_write(expr, write_counter=3, time=0.2)
    with pytest.raises(ValueError, match="does not increase"):
        w.interpolate_and_write(expr, time=0.1)


def test_counter_labelled_writer_rejects_time_and_keeps_a_pinned_counter(make_writer):
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(quad_mesh(), ("DG", 0, (4,)))
    expr, _ = field(ufl.SpatialCoordinate(V.mesh), 4)
    w = make_writer("counted", V)
    w.interpolate_and_write(expr, write_counter=5)
    assert w.write_counter == 0
    w.interpolate_and_write(expr)
    assert w.write_counter == 1
    with pytest.raises(ValueError, match="time_dependent"):
        w.interpolate_and_write(expr, time=1.0)


@pytest.mark.parametrize("degree", [0, 1])
def test_several_named_fields_share_one_file(degree, make_writer):
    """Scalar and vector fields in one writer: each output function holds its
    own field at the output dof coordinates, under its own name."""
    msh = quad_mesh()
    x = ufl.SpatialCoordinate(msh)
    Vs = dolfinx.fem.functionspace(msh, ("DG", degree))
    Vv = dolfinx.fem.functionspace(msh, ("DG", degree, (2,)))
    (e_s, f_s), (e_v, f_v) = field(x, 1), field(x, 2)
    w = make_writer("multi", {"density": Vs, "momentum": Vv})
    assert w.field_names == ["density", "momentum"]
    u = dolfinx.fem.Function(Vv)
    u.interpolate(dolfinx.fem.Expression(e_v, Vv.element.interpolation_points))
    w.interpolate_and_write({"density": e_s, "momentum": u}, write_counter=0)
    if w.writer is not None:
        for f, evaluate, bs in zip(w.fields, (f_s, f_v), (1, 2)):
            assert f.function.name == f.name
            coords = f.target_funcspace.tabulate_dof_coordinates()
            assert np.allclose(f.function.x.array.reshape(-1, bs), evaluate(coords), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="expected the fields"):
        w.interpolate_and_write({"density": e_s}, write_counter=1)


def test_fields_of_one_file_must_share_the_write_mode(make_writer):
    msh = quad_mesh()
    with pytest.raises(ValueError, match="separate files"):
        make_writer("mixed", {"a": dolfinx.fem.functionspace(msh, ("DG", 0)),
                              "b": dolfinx.fem.functionspace(msh, ("DG", 1))})


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun is not available")
@pytest.mark.skipif(os.environ.get(_MPI_CHILD_ENV) == "1",
                    reason="already running inside the spawned MPI child")
def test_runs_under_multiple_mpi_ranks():
    """Re-run this file under mpirun, so a serial pytest invocation also
    covers the gather and the per-rank paths with more than one rank."""
    cmd = [
        "mpirun", "-n", "3",
        sys.executable, "-m", "pytest", "-x", "-q",
        "-p", "no:cacheprovider",
        __file__,
        "-k", "not runs_under_multiple_mpi_ranks",
    ]
    env = dict(os.environ)
    env[_MPI_CHILD_ENV] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the 3-rank run did not finish within 300s, which usually means "
            "one rank failed an assertion and the rest are blocked in a "
            "collective")

    assert proc.returncode == 0, (
        "3-rank run failed:\n{}\n{}".format(proc.stdout, proc.stderr))
