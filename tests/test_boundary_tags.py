"""
The wall and symmetry boundaries from the mesh's own facet tags: Mesh and
DG_windtunnel_model with wall_tags / symmetry_tags instead of a function of
the facet midpoints, the warning when a function is used on a tagged mesh,
and load_dolfinx_mesh reading the tags of XDMF, gmsh and CGNS-style meshes.
"""
import contextlib

import numpy as np
import pytest
import dolfinx
from mpi4py import MPI

from dragonfly_sim.core.mesh_manager import Mesh
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.utils.mesh_io_utils import (attach_facet_tags, load_dolfinx_mesh,
                                               _cgns_facet_tags)

from cases import FREESTREAM, jittered_square
from test_windtunnel_forward import bottom_wall, in_tmp_dir  # noqa: F401  (fixture)

BOTTOM, SIDES, TOP = 1, 2, 3


def warns_on_root(match=None):
    """The warning is raised on rank 0 only (one message per run, not one per rank)."""
    if MPI.COMM_WORLD.rank == 0:
        return pytest.warns(UserWarning, match=match)
    return contextlib.nullcontext([])


def tagged_square(n=6, top=TOP, comm=None):
    """Unit square with facet tags: bottom 1, left and right 2, top `top`."""
    msh = jittered_square(n, comm)
    msh.topology.create_entities(1)
    selections = [(BOTTOM, lambda x: x[1] < 1e-8), (SIDES, lambda x: (x[0] < 1e-8) | (x[0] > 1 - 1e-8)),
                  (top, lambda x: x[1] > 1 - 1e-8)]
    facets, values = [], []
    for value, marker in selections:
        found = dolfinx.mesh.locate_entities_boundary(msh, 1, marker)
        facets.append(found)
        values.append(np.full(found.size, value, dtype=np.int32))
    facets, values = np.concatenate(facets), np.concatenate(values)
    order = np.argsort(facets)
    tags = dolfinx.mesh.meshtags(msh, 1, facets[order], values[order])
    return attach_facet_tags(msh, tags, {"wall": BOTTOM, "farfield": SIDES, "sym": TOP})


def test_wall_tags_select_the_same_facets_as_the_function():
    msh = tagged_square()
    by_function = Mesh(msh, lambda pts: pts[1] < 1e-8)
    by_tags = Mesh(msh, wall_tags=BOTTOM)
    np.testing.assert_array_equal(by_tags.inner_bdry_facet_mask, by_function.inner_bdry_facet_mask)
    assert by_tags.symmetry_facet_mask is None
    assert by_tags.inner_bdry_facet_mask.any() or MPI.COMM_WORLD.size > 1


def test_wall_and_symmetry_tags_by_value_name_and_sequence():
    msh = tagged_square()
    m = Mesh(msh, wall_tags=[BOTTOM], symmetry_tags=TOP)
    n = Mesh(msh, wall_tags="wall", symmetry_tags=["sym"])
    for mask in ("inner_bdry_facet_mask", "symmetry_facet_mask"):
        np.testing.assert_array_equal(getattr(m, mask), getattr(n, mask))
    mid = m.bdry_midpoints
    np.testing.assert_array_equal(m.symmetry_facet_mask, mid[:, 1] > 1 - 1e-8)
    np.testing.assert_array_equal(m.inner_bdry_facet_mask, mid[:, 1] < 1e-8)
    several = Mesh(msh, wall_tags=[BOTTOM, TOP])
    np.testing.assert_array_equal(several.inner_bdry_facet_mask, (mid[:, 1] < 1e-8) | (mid[:, 1] > 1 - 1e-8))


def test_function_on_a_tagged_mesh_warns_and_lists_the_tags():
    msh = tagged_square()
    with warns_on_root() as record:
        Mesh(msh, lambda pts: pts[1] < 1e-8)
    if MPI.COMM_WORLD.rank == 0:
        text = str(record[0].message)
        for expected in ("1 'wall' (6 facets)", "2 'farfield' (12 facets)", "3 'sym' (6 facets)",
                         "wall_tags", "symmetry_tags"):
            assert expected in text, (expected, text)


def test_tags_given_with_a_function_are_reported_as_ignored():
    with warns_on_root("wall_tags and symmetry_tags are ignored"):
        Mesh(tagged_square(), lambda pts: pts[1] < 1e-8, wall_tags=BOTTOM)


def test_no_warning_without_tags_or_when_the_tags_are_used(recwarn):
    Mesh(jittered_square(4), lambda pts: pts[1] < 1e-8)
    Mesh(tagged_square(), wall_tags=BOTTOM)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)
                and "boundary facet tags" in str(w.message)]


def test_tags_without_wall_tags_or_function_name_the_tags_in_the_error():
    with pytest.raises(ValueError, match=r"wall_tags.*1 'wall'"):
        Mesh(tagged_square())
    with pytest.raises(ValueError, match="boundary function"):
        Mesh(jittered_square(4))


@pytest.mark.parametrize("kwargs, message", [
    (dict(wall_tags=9), "tag 9 is not on the mesh's boundary facets"),
    (dict(wall_tags="nope"), "no boundary tag named 'nope'"),
    (dict(wall_tags=[BOTTOM], symmetry_tags=[BOTTOM]), "both wall and symmetry"),
    (dict(wall_tags=[]), "empty"),
])
def test_wrong_tags_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Mesh(tagged_square(), **kwargs)


def test_wall_tags_need_a_tagged_mesh():
    with pytest.raises(ValueError, match="no boundary facet tags"):
        Mesh(jittered_square(4), wall_tags=1)


def make_windtunnel(msh, **kwargs):
    wt = DG_windtunnel_model(msh, FREESTREAM, poly_order=0, filename_suffix="test_tags", **kwargs)
    wt.euler_residual_conv_limit = 1e-11
    wt.set_up_sim()
    return wt


def test_windtunnel_with_wall_tags_matches_the_function(in_tmp_dir):  # noqa: F811
    by_function = make_windtunnel(tagged_square(8), mesh_inner_bdry_function=bottom_wall)
    by_tags = make_windtunnel(tagged_square(8), wall_tags=BOTTOM)
    _, ok_f, forces_f = by_function.solve_forward()
    _, ok_t, forces_t = by_tags.solve_forward()
    assert ok_f and ok_t
    for key in ("L", "D", "c_l", "c_d"):
        assert forces_t[key] == pytest.approx(forces_f[key], rel=1e-12, abs=1e-14)


def test_windtunnel_symmetry_tags_become_slip_walls(in_tmp_dir):  # noqa: F811
    wt = make_windtunnel(tagged_square(8), wall_tags=BOTTOM, symmetry_tags="sym")
    comm = wt.mesh.mesh.comm
    present = set(sum(comm.allgather([int(t) for t in np.unique(wt.mesh.meshtags.values)]), []))
    assert present == {wt.WALL_TAG, wt.SYMMETRY_TAG, wt.FARFIELD_TAG}
    n_sym = comm.allreduce(int(np.sum(wt.mesh.symmetry_facet_mask)), MPI.SUM)
    assert n_sym >= 8
    _, ok, forces = wt.solve_forward()
    assert ok and np.isfinite(forces["c_l"])


def test_windtunnel_wall_tags_win_over_the_ffd_fallback(in_tmp_dir):  # noqa: F811
    class FFDOnly:
        def setup(self, *args):
            pass

        def inner_boundary_function(self):
            raise AssertionError("the FFD fallback must not be used when wall_tags is given")
    with pytest.raises(AssertionError):
        DG_windtunnel_model(tagged_square(4), FREESTREAM, FFDOnly(), poly_order=0)  # no tags: falls back
    DG_windtunnel_model(tagged_square(4), FREESTREAM, FFDOnly(), wall_tags=BOTTOM, poly_order=0)


def test_windtunnel_without_any_wall_definition_names_the_options():
    with pytest.raises(ValueError, match="wall_tags"):
        DG_windtunnel_model(tagged_square(4), FREESTREAM, poly_order=0)


# ----------------------------------------------------------------------
# Reading the tags from files
# ----------------------------------------------------------------------
def shared_path(tmp_path):
    comm = MPI.COMM_WORLD
    return comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)


def test_xdmf_facet_tags_are_read(tmp_path):
    comm = MPI.COMM_WORLD
    path = shared_path(tmp_path) + "/square.xdmf"
    msh = tagged_square(6)
    with dolfinx.io.XDMFFile(comm, path, "w") as xdmf:
        xdmf.write_mesh(msh)
        msh.facet_tags.name = "Facet tags"
        xdmf.write_meshtags(msh.facet_tags, msh.geometry)
    loaded, _ = load_dolfinx_mesh(path, comm)
    assert loaded.facet_tags is not None and loaded.facet_tags.dim == 1
    assert loaded.facet_tag_names == {}
    mesh = Mesh(loaded, wall_tags=BOTTOM, symmetry_tags=TOP)
    assert comm.allreduce(int(np.sum(mesh.inner_bdry_facet_mask)), MPI.SUM) >= 6
    with pytest.raises(ValueError, match="no grid 'other'"):
        load_dolfinx_mesh(path, comm, facet_tags_name="other")


def test_xdmf_without_tags_has_none(tmp_path):
    comm = MPI.COMM_WORLD
    path = shared_path(tmp_path) + "/plain.xdmf"
    with dolfinx.io.XDMFFile(comm, path, "w") as xdmf:
        xdmf.write_mesh(jittered_square(4))
    loaded, _ = load_dolfinx_mesh(path, comm)
    assert loaded.facet_tags is None and loaded.facet_tag_names == {}


def test_gmsh_physical_groups_are_the_tags(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    comm = MPI.COMM_WORLD
    path = shared_path(tmp_path) + "/square.msh"
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("square")
        gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        lines = gmsh.model.getEntities(1)
        bottom = [t for d, t in lines if abs(gmsh.model.getBoundingBox(d, t)[4]) < 1e-6]  # ymax == 0 (gmsh pads boxes by 1e-7)
        others = [t for d, t in lines if t not in bottom]
        gmsh.model.addPhysicalGroup(1, bottom, 7, "wall")
        gmsh.model.addPhysicalGroup(1, others, 8, "farfield")
        gmsh.model.addPhysicalGroup(2, [1], 1, "fluid")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
        gmsh.finalize()
    comm.Barrier()
    loaded, _ = load_dolfinx_mesh(path, comm, tdim=2)
    assert loaded.facet_tag_names == {"wall": 7, "farfield": 8}
    mesh = Mesh(loaded, wall_tags="wall")
    n_wall = comm.allreduce(int(np.sum(mesh.inner_bdry_facet_mask)), MPI.SUM)
    assert n_wall >= 5
    np.testing.assert_allclose(mesh.bdry_midpoints[mesh.inner_bdry_facet_mask][:, 1], 0.0, atol=1e-12)


def test_cgns_style_edge_tags_land_on_the_right_facets():
    """_cgns_facet_tags: node index pairs on rank 0 become facet tags on the distributed mesh."""
    import basix.ufl
    comm = MPI.COMM_WORLD
    n = 5
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    x = np.column_stack([ii.ravel(), jj.ravel()]).astype(np.float64) / n
    node = lambda i, j: j * (n + 1) + i
    cells = np.array([[node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)]
                      for j in range(n) for i in range(n)], dtype=np.int64)
    wall = np.array([[node(i, 0), node(i + 1, 0)] for i in range(n)], dtype=np.int64)
    far = np.array([[node(i, n), node(i + 1, n)] for i in range(n)]
                   + [[node(0, j), node(0, j + 1)] for j in range(n)]
                   + [[node(n, j), node(n, j + 1)] for j in range(n)], dtype=np.int64)
    empty = np.empty((0, 2), dtype=np.int64)
    root = comm.rank == 0
    element = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet, 2)
    msh = dolfinx.mesh.create_mesh(comm, cells if root else np.empty((0, 4), dtype=np.int64), element,
                                   x if root else np.empty((0, 2)), partitioner=partitioner)
    msh.topology.create_entities(1)
    tags = _cgns_facet_tags(msh, {1: wall if root else empty, 2: far if root else empty}, comm)
    attach_facet_tags(msh, tags, {"Wall": 1, "Farfield": 2})
    mesh = Mesh(msh, wall_tags="Wall")
    assert comm.allreduce(int(np.sum(mesh.inner_bdry_facet_mask)), MPI.SUM) >= n
    np.testing.assert_allclose(mesh.bdry_midpoints[mesh.inner_bdry_facet_mask][:, 1], 0.0, atol=1e-12)
    total = comm.allreduce(int(np.sum(~mesh.inner_bdry_facet_mask)), MPI.SUM)
    assert total >= 3 * n
