"""
utils/mesh_io_utils: merging structured blocks into 2D quad and 3D hex meshes,
unstructured zones into tet and hex meshes, boundary patches of an unstructured
CGNS grid, and the CGNS -> dolfinx path on ONERA's OAT15A grid 1, the Simple
Transonic Wing's L3 volume grid and the DLR-F25 tetrahedral grid (each skipped
when the git-ignored meshes/ folder does not hold it).
"""
import os
import shutil
import subprocess

import numpy as np
import pytest
import ufl
import basix.ufl
import dolfinx
from mpi4py import MPI

from dragonfly_sim.utils.mesh_io_utils import (merge_structured_blocks, read_cgns_planar,
                                               merge_structured_volume_blocks, read_cgns_volume,
                                               merge_unstructured_zones, read_cgns_patches,
                                               load_dolfinx_mesh, _signed_areas, _quad_edges,
                                               _center_jacobians, _hex_faces, _cell_faces, _unique_rows,
                                               _tet_volumes, _unstructured_cells)

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")
OAT15A_GRID1 = os.path.join(MESH_DIR, "ONERA-ONERA-OAT15A-Rizzi", "grid1", "OAT15A_Rizzi_1.cgns")
needs_grid1 = pytest.mark.skipif(not os.path.isfile(OAT15A_GRID1), reason="OAT15A grid 1 not in meshes/")
WING_L3 = os.path.join(MESH_DIR, "wing_vol_L3.cgns")
needs_wing_l3 = pytest.mark.skipif(not os.path.isfile(WING_L3), reason="wing_vol_L3.cgns not in meshes/")
DLR_F25 = os.path.join(MESH_DIR, "DLR-F25", "DLR-F25_patched_2.5Mc.cgns")
needs_f25 = pytest.mark.skipif(not os.path.isfile(DLR_F25) or shutil.which("cgnsconvert") is None,
                               reason="DLR-F25_patched_2.5Mc.cgns not in meshes/DLR-F25/, or no cgnsconvert")


pytestmark = [pytest.mark.unit_test]


def two_blocks():
    """Node blocks on [0, 1] x [0, 1] (i along x) and [1, 2] x [0, 1] (i along
    y: left-handed), sharing their 3 nodes on x = 1."""
    ya = np.linspace(0.0, 1.0, 3)
    xa = np.linspace(0.0, 1.0, 4)
    A = np.stack(np.meshgrid(xa, ya), axis=-1)            # (nj=3, ni=4, 2), i along x
    yb = np.linspace(0.0, 1.0, 3)
    xb = np.linspace(1.0, 2.0, 4)
    B = np.stack(np.meshgrid(yb, xb)[::-1], axis=-1)      # (nj=4, ni=3, 2), i along y
    return A, B


def test_merge_structured_blocks():
    A, B = two_blocks()
    points, quads = merge_structured_blocks([A, B])
    # 12 + 12 nodes, the 3 on x = 1 shared
    assert len(points) == 21 and len(quads) == 6 + 6
    # the left-handed block was flipped: every cell counterclockwise, total area 2
    areas = _signed_areas(points, quads)
    assert np.all(areas > 0) and np.isclose(areas.sum(), 2.0)
    # conforming: the only one-cell edges are the perimeter of [0, 2] x [0, 1]
    edges, counts = np.unique(_quad_edges(quads), axis=0, return_counts=True)
    assert counts.max() == 2 and np.sum(counts == 1) == 2 * (3 + 3) + 2 * 2
    # dolfinx takes the ordering as is (a twisted cell would not give area 2)
    element = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, quads.astype(np.int64), element, points)
    area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh)))
    assert np.isclose(area, 2.0)


@needs_grid1
def test_read_oat15a_grid1():
    data = read_cgns_planar(OAT15A_GRID1)
    assert len(data["quads"]) == 15872                    # the grids' ReadMe
    assert data["n_raw_points"] - len(data["points"]) == 2 * 97 + 2 * 33   # 1-to-1 interfaces
    assert len(data["wall_edges"]) == 96 + 16             # airfoil + blunt trailing edge base
    assert data["chord"] == 230.0
    wall = data["points"][np.unique(data["wall_edges"])]
    assert wall[:, 0].min() == 0.0 and wall[:, 0].max() == 1.0
    assert np.abs(wall[:, 1]).max() < 0.1


@needs_grid1
def test_load_oat15a_grid1_distributed():
    comm = MPI.COMM_WORLD
    msh, info = load_dolfinx_mesh(OAT15A_GRID1, comm)
    assert msh.topology.index_map(2).size_global == info["n_cells"] == 15872
    msh.topology.create_connectivity(1, 2)
    n_ext = comm.allreduce(len(dolfinx.mesh.exterior_facet_indices(msh.topology)), MPI.SUM)
    assert n_ext == info["n_wall_edges"] + info["n_farfield_edges"]
    # the same mesh on any number of ranks: total area and wall length
    area = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh))), MPI.SUM)
    serial = read_cgns_planar(OAT15A_GRID1)
    assert np.isclose(area, _signed_areas(serial["points"], serial["quads"]).sum(), rtol=1e-12)


def test_merge_structured_volume_blocks():
    """[0, 1]^3 (i along x) and [1, 2] x [0, 1]^2 with i and j swapped
    (left-handed), sharing their 3 x 3 nodes on x = 1."""
    t = np.linspace(0.0, 1.0, 3)
    z, y, x = np.meshgrid(t, t, np.linspace(0.0, 1.0, 4), indexing="ij")                # (k, j, i)
    A = np.stack([x, y, z], axis=-1)
    z, x, y = np.meshgrid(t, np.linspace(1.0, 2.0, 4), t, indexing="ij")
    B = np.stack([x, y, z], axis=-1)                                                     # i -> y, j -> x
    points, hexes = merge_structured_volume_blocks([A, B])
    assert len(points) == 36 + 36 - 9 and len(hexes) == 12 + 12
    jac = _center_jacobians(points, hexes)
    assert np.all(jac > 0)
    faces, counts = np.unique(_hex_faces(hexes), axis=0, return_counts=True)
    # [0, 2] x [0, 1]^2 in 6 x 2 x 2 cells: 2 x 20 interior faces per block plus
    # the 2 x 2 interface; the boundary is conforming too
    assert counts.max() == 2 and np.sum(counts == 2) == 44 and np.sum(counts == 1) == 56
    element = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, hexes.astype(np.int64), element, points)
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh)))
    assert np.isclose(volume, 2.0)


@needs_wing_l3
def test_read_wing_l3():
    data = read_cgns_volume(WING_L3)
    assert data["n_blocks"] == 12
    assert len(data["hexes"]) == 181152 and len(data["points"]) == 187425
    xdmf_h5 = os.path.join(MESH_DIR, "wing_vol_L3_xdmf.h5")
    if os.path.isfile(xdmf_h5):
        # the same mesh as the XDMF conversion: nodes, and cells as node sets
        import h5py
        with h5py.File(xdmf_h5, "r") as f:
            points, hexes = f["data0"][:], f["data1"][:]
        order, order_x = (np.lexsort(p.T[::-1]) for p in (data["points"], points))
        np.testing.assert_array_equal(data["points"][order], points[order_x])
        to_x = np.empty(len(order), dtype=np.int64)
        to_x[order] = order_x
        mine, theirs = np.sort(to_x[data["hexes"]], axis=1), np.sort(hexes, axis=1)
        np.testing.assert_array_equal(mine[np.lexsort(mine.T[::-1])], theirs[np.lexsort(theirs.T[::-1])])


@needs_wing_l3
def test_load_wing_l3_distributed():
    comm = MPI.COMM_WORLD
    msh, info = load_dolfinx_mesh(WING_L3, comm, tdim=3)
    assert msh.topology.index_map(3).size_global == info["n_cells"] == 181152
    msh.topology.create_connectivity(2, 3)
    n_ext = comm.allreduce(len(dolfinx.mesh.exterior_facet_indices(msh.topology)), MPI.SUM)
    assert n_ext == info["n_boundary_faces"]
    volume = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh))), MPI.SUM)
    assert volume > 0


def test_unsupported_extension():
    with pytest.raises(ValueError):
        load_dolfinx_mesh("mesh.msh")


# --- unstructured grids: tetrahedra and hexahedra ---------------------------------------------------------------

def cube_tets(n, x0=0.0, shape=(1.0, 1.0, 1.0)):
    """Kuhn tetrahedralisation of a box with n x n x n cells starting at (x0, 0, 0): nodes (N, 3) and tets (M, 4),
    all right-handed."""
    g = np.linspace(0.0, 1.0, n + 1)
    z, y, x = np.meshgrid(g * shape[2], g * shape[1], x0 + g * shape[0], indexing="ij")
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    idx = np.arange(len(points)).reshape(n + 1, n + 1, n + 1)               # (k, j, i)
    tets = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                corner = np.array([k, j, i])
                for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                    path, v = [corner.copy()], corner.copy()
                    for a in perm:
                        v = v.copy()
                        v[a] += 1
                        path.append(v)
                    tets.append([idx[tuple(q)] for q in path])
    tets = np.array(tets)
    neg = _tet_volumes(points, tets) < 0
    tets[neg] = tets[neg][:, [0, 1, 3, 2]]
    return points, tets


def test_unique_rows_matches_numpy():
    rng = np.random.default_rng(0)
    rows = np.sort(rng.integers(0, 6, size=(500, 3)), axis=1)
    mine, counts = _unique_rows(rows)
    ref, ref_counts = np.unique(rows, axis=0, return_counts=True)
    np.testing.assert_array_equal(mine, ref)
    np.testing.assert_array_equal(counts, ref_counts)


def test_merge_unstructured_zones_tets():
    """Two tet boxes, [0, 1]^3 and [1, 2] x [0, 1]^2 (the second left-handed), sharing their nodes on x = 1."""
    pa, ta = cube_tets(2)
    pb, tb = cube_tets(2, x0=1.0)
    tb = tb[:, [0, 1, 3, 2]]
    assert _tet_volumes(pb, tb).sum() < 0
    points, tets, node_map = merge_unstructured_zones([(pa, ta), (pb, tb)], "tetrahedron")
    assert len(points) == 27 + 27 - 9 and len(tets) == 2 * 48 and len(node_map) == 54
    volumes = _tet_volumes(points, tets)
    assert np.all(volumes > 0) and np.isclose(volumes.sum(), 2.0)
    faces, counts = _unique_rows(_cell_faces(tets, "tetrahedron"))
    assert counts.max() == 2
    # the boundary of [0, 2] x [0, 1]^2 in 4 x 2 x 2 cells: (2 x 4 + 2 x 8 + 2 x 8) squares, 2 triangles each
    assert np.sum(counts == 1) == 80
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, tets.astype(np.int64),
                                   basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)), points)
    assert np.isclose(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh))), 2.0)
    # a single zone is not merged
    assert merge_unstructured_zones([(pa, ta)], "tetrahedron")[2] is None


def test_unstructured_cells_vtk_ordering():
    """VTK hexahedra (bottom face counter-clockwise, then the top) become dolfinx hexahedra; tets are unchanged."""
    import pyvista as pv
    c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    grid = pv.UnstructuredGrid(np.array([8, *range(8)]), np.array([12]), c)
    points, cells, cell_type = _unstructured_cells(grid)
    assert cell_type == "hexahedron" and np.all(_center_jacobians(points, cells) > 0)
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells.astype(np.int64),
                                   basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)), points)
    assert np.isclose(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh))), 1.0)
    pt, tt = cube_tets(1)
    grid = pv.UnstructuredGrid(np.hstack([np.full((len(tt), 1), 4), tt]).ravel(), np.full(len(tt), 10), pt)
    points, cells, cell_type = _unstructured_cells(grid)
    assert cell_type == "tetrahedron" and np.array_equal(cells, tt)
    wedge = pv.UnstructuredGrid(np.array([6, *range(6)]), np.array([13]), np.vstack([c[:3], c[4:7]]))
    with pytest.raises(ValueError, match="only tetrahedra"):
        _unstructured_cells(wedge)


def write_tet_cgns(path, points, tets, patches):
    """A one-zone unstructured CGNS file (HDF5 container) in the layout of the DPW grids: an Elem_TETRA_4 section,
    a TRI_3 section per patch, and a ZoneBC with the BC type and ElementRange of every patch.
    patches: list of (name, BC type, triangles (k, 3), 0-based nodes)."""
    import h5py
    h5py.get_config().track_order = True      # libcgns reads the children in creation order

    def node(parent, name, label, typ, data=None):
        g = parent.create_group(name)
        g.attrs["flags"] = np.array([1], np.int32)
        for k, v, n in (("label", label, 33), ("name", name, 33), ("type", typ, 3)):
            g.attrs[k] = np.array(v.encode(), dtype="S{}".format(n))
        if data is not None:
            g.create_dataset(" data", data=data)
        return g

    text = lambda s: np.frombuffer(s.encode(), np.int8).copy()
    with h5py.File(path, "w") as f:
        f.attrs["label"] = np.array(b"Root Node of HDF5 File", dtype="S33")
        f.attrs["name"] = np.array(b"HDF5 MotherNode", dtype="S33")
        f.attrs["type"] = np.array(b"MT", dtype="S3")
        f.create_dataset(" format", data=text("IEEE_LITTLE_32\0"))
        f.create_dataset(" hdf5version", data=np.pad(text("HDF5 Version 1.14.4"), (0, 14)))
        node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4", np.array([2.52], np.float32))
        zone = node(node(f, "Base", "CGNSBase_t", "I4", np.array([3, 3], np.int32)), "Zone", "Zone_t", "I4",
                    np.array([[len(points)], [len(tets)], [0]], np.int32))
        node(zone, "ZoneType", "ZoneType_t", "C1", text("Unstructured"))
        coords = node(zone, "GridCoordinates", "GridCoordinates_t", "MT")
        for ax, name in enumerate("XYZ"):
            node(coords, "Coordinate" + name, "DataArray_t", "R8", np.ascontiguousarray(points[:, ax]))
        sec = node(zone, "Elem_TETRA_4", "Elements_t", "I4", np.array([10, 0], np.int32))
        node(sec, "ElementRange", "IndexRange_t", "I4", np.array([1, len(tets)], np.int32))
        node(sec, "ElementConnectivity", "DataArray_t", "I4", (tets + 1).astype(np.int32).ravel())
        zbc, first = node(zone, "ZoneBC", "ZoneBC_t", "MT"), len(tets) + 1
        for name, bc, tris in patches:
            last = first + len(tris) - 1
            sec = node(zone, name, "Elements_t", "I4", np.array([5, 0], np.int32))
            node(sec, "ElementRange", "IndexRange_t", "I4", np.array([first, last], np.int32))
            node(sec, "ElementConnectivity", "DataArray_t", "I4", (tris + 1).astype(np.int32).ravel())
            node(node(zbc, name, "BC_t", "C1", text(bc)), "ElementRange", "IndexRange_t", "I4",
                 np.array([[first], [last]], np.int32))
            first = last + 1


@pytest.fixture(scope="module")
def tet_cube_cgns(tmp_path_factory):
    """The unit cube in 3 x 3 x 3 x 6 tets, as a one-zone CGNS file with Inlet (x = 0), Outlet (x = 1) and Wall (the
    other four sides) patches. Returns path, points, tets and the patches."""
    points, tets = cube_tets(3)
    faces, counts = _unique_rows(_cell_faces(tets, "tetrahedron"))
    bnd = faces[counts == 1]
    x = points[bnd][:, :, 0]
    inlet, outlet = np.all(x == 0.0, axis=1), np.all(x == 1.0, axis=1)
    patches = [("Inlet", "BCInflow", bnd[inlet]), ("Outlet", "BCOutflow", bnd[outlet]),
               ("Wall", "BCWallViscous", bnd[~inlet & ~outlet])]
    path = str(tmp_path_factory.mktemp("cgns") / "cube_tets.cgns")
    write_tet_cgns(path, points, tets, patches)
    return path, points, tets, patches


def test_read_unstructured_tet_cgns(tet_cube_cgns):
    path, points, tets, patches = tet_cube_cgns
    data = read_cgns_volume(path, patches=True)
    assert data["cell_type"] == "tetrahedron" and data["n_blocks"] == 1 and data["tets"] is data["cells"]
    np.testing.assert_array_equal(data["points"], points)
    assert np.array_equal(np.sort(data["tets"], axis=1), np.sort(tets, axis=1))
    assert np.all(_tet_volumes(data["points"], data["tets"]) > 0)
    assert list(data["patches"]) == ["Inlet", "Outlet", "Wall"]
    assert [p["bc"] for p in data["patches"].values()] == ["BCInflow", "BCOutflow", "BCWallViscous"]
    assert sum(len(p["faces"]) for p in data["patches"].values()) == len(data["boundary_faces"]) == 6 * 18
    for (name, _, tris), p in zip(patches, data["patches"].values()):
        np.testing.assert_array_equal(p["faces"], tris)
    # without patches, nothing but the mesh
    assert "patches" not in read_cgns_volume(path)
    # the patch reader alone
    assert list(read_cgns_patches(path)) == ["Inlet", "Outlet", "Wall"]


def test_read_unstructured_adf_cgns(tet_cube_cgns, tmp_path):
    """The DPW grids are ADF files: the patches come from a temporary HDF5 copy (cgnsconvert)."""
    if shutil.which("cgnsconvert") is None:
        pytest.skip("no cgnsconvert")
    adf = str(tmp_path / "cube_tets_adf.cgns")
    subprocess.run(["cgnsconvert", "-a", tet_cube_cgns[0], adf], check=True, capture_output=True)
    with open(adf, "rb") as fh:
        assert fh.read(8)[4:7] == b"ADF"
    data = read_cgns_volume(adf, patches=True)
    assert len(data["tets"]) == len(tet_cube_cgns[2]) and list(data["patches"]) == ["Inlet", "Outlet", "Wall"]


def test_patch_not_on_the_boundary_is_rejected(tet_cube_cgns, tmp_path):
    path, points, tets, patches = tet_cube_cgns
    bad = [("Inlet", "BCInflow", tets[:5, :3])] + patches[1:]                  # interior faces
    write_tet_cgns(str(tmp_path / "bad.cgns"), points, tets, bad)
    with pytest.raises(ValueError, match="not boundary faces"):
        read_cgns_volume(str(tmp_path / "bad.cgns"), patches=True)


def test_load_tet_cgns_distributed(tet_cube_cgns):
    path, points, tets, patches = tet_cube_cgns
    comm = MPI.COMM_WORLD
    msh, info = load_dolfinx_mesh(path, comm, tdim=3, patches=True)
    assert info["cell_type"] == "tetrahedron" and msh.topology.index_map(3).size_global == len(tets)
    assert np.isclose(comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh))), MPI.SUM), 1.0)
    tags = info["facet_tags"]
    owned = tags.indices < msh.topology.index_map(2).size_local
    for name, p in info["patches"].items():
        n = comm.allreduce(int(np.sum((tags.values == p["value"]) & owned)), MPI.SUM)
        assert n == p["n_faces"] == len(dict((a, b) for a, _, b in patches)[name])
    # every tagged facet lies on its patch's side of the cube
    msh.topology.create_connectivity(2, 3)
    mids = dolfinx.mesh.compute_midpoints(msh, 2, tags.indices)
    inlet = tags.values == info["patches"]["Inlet"]["value"]
    assert np.allclose(mids[inlet, 0], 0.0)
    assert np.allclose(mids[tags.values == info["patches"]["Outlet"]["value"], 0], 1.0)
    # without patches no tags, and the same mesh
    msh2, info2 = load_dolfinx_mesh(path, comm, tdim=3)
    assert "facet_tags" not in info2 and msh2.topology.index_map(3).size_global == len(tets)


@needs_f25
def test_read_dlr_f25():
    data = read_cgns_volume(DLR_F25, patches=True)
    assert data["cell_type"] == "tetrahedron" and len(data["tets"]) == 2519327 and len(data["points"]) == 434918
    assert len(data["boundary_faces"]) == 64238 and len(data["patches"]) == 20
    assert sum(len(p["faces"]) for p in data["patches"].values()) == 64238          # every boundary face is in a patch
    bcs = {p["bc"] for p in data["patches"].values()}
    assert bcs == {"BCWallViscous", "BCSymmetryPlane", "BCFarfield"}
    assert np.all(_tet_volumes(data["points"], data["tets"]) > 0)


@needs_f25
def test_load_dlr_f25_distributed():
    comm = MPI.COMM_WORLD
    msh, info = load_dolfinx_mesh(DLR_F25, comm, tdim=3, patches=True)
    assert msh.topology.index_map(3).size_global == info["n_cells"] == 2519327
    tags = info["facet_tags"]
    owned = tags.indices < msh.topology.index_map(2).size_local
    assert comm.allreduce(int(owned.sum()), MPI.SUM) == info["n_boundary_faces"] == 64238
    msh.topology.create_connectivity(2, 3)
    exterior = set(dolfinx.mesh.exterior_facet_indices(msh.topology).tolist())
    assert all(f in exterior for f in tags.indices[owned])
    volume = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(msh))), MPI.SUM)
    assert np.isclose(volume, 1.45798e17, rtol=1e-5)                                 # mm^3
