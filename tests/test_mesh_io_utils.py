"""
utils/mesh_io_utils: merging structured blocks into 2D quad and 3D hex meshes,
and the CGNS -> dolfinx path on ONERA's OAT15A grid 1 and the Simple Transonic
Wing's L3 volume grid (each skipped when the git-ignored meshes/ folder does
not hold it).
"""
import os

import numpy as np
import pytest
import ufl
import basix.ufl
import dolfinx
from mpi4py import MPI

from dragonfly_sim.utils.mesh_io_utils import (merge_structured_blocks, read_cgns_planar,
                                               merge_structured_volume_blocks, read_cgns_volume,
                                               load_dolfinx_mesh, _signed_areas, _quad_edges,
                                               _center_jacobians, _hex_faces)

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")
OAT15A_GRID1 = os.path.join(MESH_DIR, "ONERA-ONERA-OAT15A-Rizzi", "grid1", "OAT15A_Rizzi_1.cgns")
needs_grid1 = pytest.mark.skipif(not os.path.isfile(OAT15A_GRID1), reason="OAT15A grid 1 not in meshes/")
WING_L3 = os.path.join(MESH_DIR, "wing_vol_L3.cgns")
needs_wing_l3 = pytest.mark.skipif(not os.path.isfile(WING_L3), reason="wing_vol_L3.cgns not in meshes/")


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
