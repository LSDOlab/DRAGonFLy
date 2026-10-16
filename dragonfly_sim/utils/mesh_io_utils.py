"""
Mesh input: dolfinx meshes from XDMF files or from CGNS grids.

A structured multi-block CGNS grid of a 2D case is often stored as a 3D grid
one cell thick in span (e.g. ONERA's OAT15A grids). read_cgns_planar reads it
with PyVista (VTK's CGNS reader), keeps one span plane of every block as
quadrilaterals, merges the nodes the blocks share, and finds the wall and far
field edges from the CGNS boundary condition families. read_cgns_volume reads
a 3D grid into one mesh the same way: structured multi-block grids (e.g. the
Simple Transonic Wing's wing_vol_L*.cgns) as hexahedra, unstructured zones
(e.g. the DPW wing-body-tail grids, or meshes/DLR-F25/DLR-F25_patched_*.cgns)
as tetrahedra or hexahedra, one cell type per mesh. read_cgns_patches reads
the named boundary patches of an unstructured grid, which VTK does not expose.
cgns_to_dolfinx_mesh turns either into a distributed dolfinx mesh in memory:
no intermediate file.

Structured block interface nodes (and the nodes shared by several
unstructured zones) are merged by EXACT coordinate match, which holds for
grids whose 1-to-1 interfaces were written point-matched (Pointwise exports
are); a grid that needs a tolerance merge fails the boundary check below.
"""
import os

import numpy as np
from mpi4py import MPI
import basix.ufl
import dolfinx


def _plane_points(grid, plane_index):
    """Points of k-plane `plane_index` of a StructuredGrid, shape (nj, ni, 3)."""
    ni, nj, nk = grid.dimensions
    if nk > 2:
        raise ValueError("block has {} nodes in its third index direction; only grids one cell "
                         "thick (or planar) can be reduced to 2D".format(nk))
    k = min(plane_index, nk - 1)
    # VTK structured points run i fastest, then j, then k
    return np.asarray(grid.points).reshape(nk, nj, ni, 3)[k]


def _block_quads(nj, ni, offset):
    """Quads of an nj x ni node block in dolfinx ordering [(i,j), (i+1,j), (i,j+1), (i+1,j+1)]."""
    idx = offset + np.arange(nj * ni).reshape(nj, ni)
    return np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], axis=-1).reshape(-1, 4)


def _signed_areas(points, quads):
    """Signed areas of quads in dolfinx ordering (positive: counterclockwise)."""
    p0, p1, p2, p3 = (points[quads[:, i]] for i in range(4))
    d1, d2 = p3 - p0, p2 - p1
    return 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])


def _quad_edges(quads):
    """The four edges of every quad (dolfinx ordering), as sorted node pairs."""
    e = np.concatenate([quads[:, [0, 1]], quads[:, [1, 3]], quads[:, [3, 2]], quads[:, [2, 0]]])
    return np.sort(e, axis=1)


def _family_patch_lines(reader, family, plane_index):
    """Points of every boundary patch of `family`, cut at the span plane: a list
    of (n, 3) arrays, one node line per patch, in order along the line."""
    reader.disable_all_families()
    reader.reader.SetFamilyArrayStatus(family, 1)
    lines = []
    for zone in reader.read()["Base"]:
        if zone is None or "Patches" not in zone.keys():
            continue
        for patch in zone["Patches"]:
            pts = _plane_points(patch, plane_index)
            if min(pts.shape[:2]) != 1:
                raise ValueError("a {} patch is not a line at the span plane (node grid {} x {}): "
                                 "it lies in the plane itself".format(family, *pts.shape[:2]))
            lines.append(pts.reshape(-1, 3))
    return lines


def merge_structured_blocks(blocks):
    """
    Merge 2D structured node blocks into one quad mesh.

    blocks: list of (nj, ni, 2) node arrays. Returns (points (N, 2), quads
    (M, 4) in dolfinx ordering, counterclockwise). Nodes shared by blocks are
    merged by exact coordinate match; a block whose (i, j) axes are left-handed
    has its quads flipped.
    """
    quads, offset = [], 0
    raw = np.concatenate([b.reshape(-1, 2) for b in blocks])
    for b in blocks:
        nj, ni = b.shape[:2]
        q = _block_quads(nj, ni, offset)
        if _signed_areas(raw, q).sum() < 0:
            q = q[:, [1, 0, 3, 2]]
        quads.append(q)
        offset += nj * ni
    points, inverse = np.unique(raw, axis=0, return_inverse=True)
    quads = inverse.reshape(-1)[np.concatenate(quads)]
    return points, quads


def read_cgns_planar(cgns_path, plane_index=0, wall_family="Wall", farfield_family="Farfield",
                     scale_to_chord=True):
    """
    Read a structured multi-block CGNS grid that is one cell thick (or planar)
    and reduce it to a 2D quadrilateral mesh. Serial.

    plane_index: which span plane of every block to keep (0 or 1). The span
    direction is the coordinate axis that is constant on that plane; the other
    two, in ascending order, become (x, y) -- an X-Z airfoil grid maps
    (X, Z) -> (x, y).

    scale_to_chord: shift the points so the wall's leading edge (its smallest
    x) is at x = 0 and scale them by 1 / chord, the wall's x extent.

    Returns a dict: points (N, 2), quads (M, 4) (dolfinx ordering,
    counterclockwise), wall_edges and farfield_edges (node index pairs),
    chord (in the file's units), n_raw_points (before merging).

    Raises ValueError if a cell has non-positive area, an edge has more than
    two cells, or the boundary edges are not exactly the wall and far field
    edges.
    """
    import pyvista as pv

    if not os.path.isfile(cgns_path):
        raise FileNotFoundError(cgns_path)
    reader = pv.get_reader(cgns_path)
    planes = [_plane_points(zone["Internal"], plane_index) for zone in reader.read()["Base"]]

    span_axes = [ax for ax in range(3) if all(np.ptp(p[..., ax]) == 0 for p in planes)]
    if len(span_axes) != 1 or len({p[0, 0, span_axes[0]] for p in planes}) != 1:
        raise ValueError("the span plane is not one coordinate plane shared by all blocks")
    in_plane = [ax for ax in range(3) if ax != span_axes[0]]
    points, quads = merge_structured_blocks([p[..., in_plane] for p in planes])
    n_raw = sum(p.shape[0] * p.shape[1] for p in planes)

    # boundary edges from the CGNS families, as merged node index pairs
    lookup = {tuple(x): i for i, x in enumerate(points)}
    reader.load_boundary_patch = True
    edges = {}
    for name, family in (("wall", wall_family), ("farfield", farfield_family)):
        pairs = []
        for line in _family_patch_lines(reader, family, plane_index):
            try:
                ids = np.array([lookup[tuple(x)] for x in line[:, in_plane]])
            except KeyError:
                raise ValueError("a {} patch node is not a grid node".format(family)) from None
            pairs.append(np.column_stack([ids[:-1], ids[1:]]))
        if not pairs:
            raise ValueError("no boundary patches of family '{}' in {}".format(family, cgns_path))
        edges[name] = np.concatenate(pairs)

    # checks
    areas = _signed_areas(points, quads)
    if not np.all(areas > 0):
        raise ValueError("{} cells have non-positive area".format(int(np.sum(areas <= 0))))
    all_edges, counts = np.unique(_quad_edges(quads), axis=0, return_counts=True)
    if counts.max() > 2:
        raise ValueError("{} edges have more than two cells".format(int(np.sum(counts > 2))))
    bdry = {tuple(e) for e in all_edges[counts == 1]}
    tagged = [{tuple(e) for e in np.sort(edges[k], axis=1)} for k in ("wall", "farfield")]
    if tagged[0] & tagged[1] or tagged[0] | tagged[1] != bdry:
        raise ValueError("the boundary edges are not exactly the {} and {} edges ({} boundary, {} "
                         "{}, {} {})".format(wall_family, farfield_family, len(bdry), len(tagged[0]),
                                             wall_family, len(tagged[1]), farfield_family))

    wall_x = points[np.unique(edges["wall"]), 0]
    chord = float(wall_x.max() - wall_x.min())
    if scale_to_chord:
        points = (points - np.array([wall_x.min(), 0.0])) / chord
    return dict(points=points, quads=quads, wall_edges=edges["wall"],
                farfield_edges=edges["farfield"], chord=chord, n_raw_points=n_raw)


# The six faces of a hexahedron and the four of a tetrahedron in dolfinx ordering, as local node indices
_HEX_FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5],
                       [2, 3, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]])
_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

# What differs between the 3D cell types a CGNS volume mesh can have:
#   n_nodes   nodes per cell
#   faces     local node indices of the faces (dolfinx ordering)
#   vtk_id    VTK cell type of the cells (CGNS HEXA_8 / TETRA_4 use VTK's node order)
#   from_vtk  columns that turn VTK node order into dolfinx order (a hexahedron's nodes 2, 3 and 6, 7 swap)
#   mirror    columns that turn a left-handed cell into a right-handed one
#   key       name of the cell array in read_cgns_volume's result (besides "cells")
_CELL_TYPES = {
    "hexahedron": dict(n_nodes=8, faces=_HEX_FACES, vtk_id=12, from_vtk=[0, 1, 3, 2, 4, 5, 7, 6],
                       mirror=[1, 0, 3, 2, 5, 4, 7, 6], key="hexes"),
    "tetrahedron": dict(n_nodes=4, faces=_TET_FACES, vtk_id=10, from_vtk=[0, 1, 2, 3],
                        mirror=[0, 1, 3, 2], key="tets"),
}
# CGNS element types of boundary patches (nodes per face)
_CGNS_FACE_TYPES = {5: 3, 7: 4}               # TRI_3, QUAD_4


def _block_hexes(nk, nj, ni, offset):
    """Hexes of an nk x nj x ni node block in dolfinx ordering (i fastest, then
    j, then k: [(i,j,k), (i+1,j,k), (i,j+1,k), (i+1,j+1,k), the same at k+1])."""
    idx = offset + np.arange(nk * nj * ni).reshape(nk, nj, ni)
    bottom = [idx[:-1, :-1, :-1], idx[:-1, :-1, 1:], idx[:-1, 1:, :-1], idx[:-1, 1:, 1:]]
    top = [idx[1:, :-1, :-1], idx[1:, :-1, 1:], idx[1:, 1:, :-1], idx[1:, 1:, 1:]]
    return np.stack(bottom + top, axis=-1).reshape(-1, 8)


def _center_jacobians(points, hexes):
    """Jacobian determinants of the trilinear map at the cell centers, hexes in
    dolfinx ordering (positive: right-handed; 8x the volume for a parallelepiped)."""
    v = [points[hexes[:, i]] for i in range(8)]
    d_i = (v[1] - v[0]) + (v[3] - v[2]) + (v[5] - v[4]) + (v[7] - v[6])
    d_j = (v[2] - v[0]) + (v[3] - v[1]) + (v[6] - v[4]) + (v[7] - v[5])
    d_k = (v[4] - v[0]) + (v[5] - v[1]) + (v[6] - v[2]) + (v[7] - v[3])
    return np.einsum("ij,ij->i", d_i, np.cross(d_j, d_k)) / 64.0


def _cell_faces(cells, cell_type="hexahedron"):
    """The faces of every cell (6 per hexahedron, 4 per tetrahedron), as sorted node tuples."""
    f = _CELL_TYPES[cell_type]["faces"]
    return np.sort(cells[:, f].reshape(-1, f.shape[1]), axis=1)


def _hex_faces(hexes):
    """The six faces of every hex, as sorted node quadruples."""
    return _cell_faces(hexes, "hexahedron")


def _unique_rows(rows):
    """Unique rows of an integer array, sorted lexicographically, with their counts. The same result as
    np.unique(rows, axis=0, return_counts=True), several times faster on millions of rows."""
    order = np.lexsort(rows.T[::-1])
    srt = rows[order]
    new = np.ones(len(srt), dtype=bool)
    new[1:] = np.any(srt[1:] != srt[:-1], axis=1)
    starts = np.flatnonzero(new)
    return srt[starts], np.diff(np.append(starts, len(srt)))


def _tet_volumes(points, tets):
    """Signed volumes of tetrahedra (positive: right-handed, (v1-v0) x (v2-v0) . (v3-v0) > 0)."""
    p = points[tets]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def _cell_orientation(points, cells, cell_type):
    """Signed measure of every cell, positive when right-handed: the Jacobian determinant at the center of a
    hexahedron, the volume of a tetrahedron."""
    return _center_jacobians(points, cells) if cell_type == "hexahedron" else _tet_volumes(points, cells)


def merge_structured_volume_blocks(blocks):
    """
    Merge 3D structured node blocks into one hexahedral mesh.

    blocks: list of (nk, nj, ni, 3) node arrays. Returns (points (N, 3), hexes
    (M, 8) in dolfinx ordering, right-handed). Nodes shared by blocks are
    merged by exact coordinate match; a block whose (i, j, k) axes are
    left-handed has its hexes mirrored in i.
    """
    hexes, offset = [], 0
    raw = np.concatenate([b.reshape(-1, 3) for b in blocks])
    for b in blocks:
        nk, nj, ni = b.shape[:3]
        h = _block_hexes(nk, nj, ni, offset)
        if _center_jacobians(raw, h).sum() < 0:
            h = h[:, [1, 0, 3, 2, 5, 4, 7, 6]]
        hexes.append(h)
        offset += nk * nj * ni
    points, inverse = np.unique(raw, axis=0, return_inverse=True)
    hexes = inverse.reshape(-1)[np.concatenate(hexes)]
    return points, hexes


def merge_unstructured_zones(zones, cell_type):
    """
    Merge unstructured node/cell zones of one cell type into one mesh.

    zones: list of (points (n, 3), cells (m, k)) with the cells in dolfinx ordering. Returns (points (N, 3), cells
    (M, k), node_map), node_map taking a node of the concatenated zones to its merged index (None for one zone,
    which needs no merging). Nodes shared by zones are merged by exact coordinate match; a zone whose cells are
    left-handed has its cells mirrored.
    """
    mirror = _CELL_TYPES[cell_type]["mirror"]
    raw = np.concatenate([p for p, _ in zones])
    cells, offset = [], 0
    for p, c in zones:
        c = c + offset
        if _cell_orientation(raw, c, cell_type).sum() < 0:
            c = c[:, mirror]
        cells.append(c)
        offset += len(p)
    cells = np.concatenate(cells)
    if len(zones) == 1:
        return raw, cells, None
    points, inverse = np.unique(raw, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return points, inverse[cells], inverse


def _unstructured_cells(grid):
    """(points, cells in dolfinx ordering, cell type name) of a VTK UnstructuredGrid of tetrahedra or hexahedra."""
    types = np.unique(grid.celltypes)
    by_id = {v["vtk_id"]: k for k, v in _CELL_TYPES.items()}
    if len(types) != 1 or int(types[0]) not in by_id:
        raise ValueError("an unstructured zone has VTK cell types {}: only tetrahedra (10) or hexahedra (12), "
                         "one type per zone, are supported".format(types.tolist()))
    cell_type = by_id[int(types[0])]
    cells = np.asarray(grid.cells_dict[int(types[0])], dtype=np.int64)[:, _CELL_TYPES[cell_type]["from_vtk"]]
    return np.asarray(grid.points, dtype=np.float64), cells, cell_type


def _h5_label(node):
    return node.attrs.get("label", b"").tobytes().decode().strip("\x00") if "label" in node.attrs else ""


def _h5_value(node):
    return node[" data"][()] if " data" in node else None


def read_cgns_patches(cgns_path):
    """
    The boundary patches of a one-zone unstructured CGNS grid, which VTK's reader skips: every BC_t node of the
    zone's ZoneBC (a patch of faces, found from the BC's ElementRange, PointRange or PointList and the TRI_3 / QUAD_4
    element sections), or, if the zone has no BCs, every such element section.

    Needs the HDF5 container of CGNS (h5py); an ADF file, like the DPW grids, is converted to a temporary HDF5 copy
    with the `cgnsconvert -h` tool of the CGNS library.

    Returns a dict in file order: name -> dict(bc=BC type name or None, faces=(k, 3 or 4) int64 array of node
    indices, 0-based in the zone's node numbering, i.e. the order of its coordinates).
    """
    import subprocess
    import tempfile
    import h5py

    if not os.path.isfile(cgns_path):
        raise FileNotFoundError(cgns_path)
    with tempfile.TemporaryDirectory() as tmp:
        path = cgns_path
        if not h5py.is_hdf5(cgns_path):
            path = os.path.join(tmp, "grid_hdf5.cgns")
            try:
                subprocess.run(["cgnsconvert", "-h", cgns_path, path], check=True, capture_output=True)
            except FileNotFoundError:
                raise RuntimeError("{} is an ADF file and the patch names need the HDF5 container: convert it with "
                                   "`cgnsconvert -h` (CGNS library tools)".format(cgns_path)) from None
        with h5py.File(path, "r") as f:
            zones = [z for b in f.values() if isinstance(b, h5py.Group) and _h5_label(b) == "CGNSBase_t"
                     for z in b.values() if isinstance(z, h5py.Group) and _h5_label(z) == "Zone_t"]
            if len(zones) != 1:
                raise ValueError("boundary patches are read for one-zone grids; {} has {} zones".format(cgns_path, len(zones)))
            zone = zones[0]
            sections = []                          # (first element id, last, faces) of the face sections
            for node in zone.values():
                if isinstance(node, h5py.Group) and _h5_label(node) == "Elements_t":
                    etype = int(_h5_value(node)[0])
                    if etype in _CGNS_FACE_TYPES:
                        first, last = (int(v) for v in np.ravel(_h5_value(node["ElementRange"])))
                        conn = np.asarray(_h5_value(node["ElementConnectivity"]), dtype=np.int64) - 1
                        sections.append((node.name.split("/")[-1], first, last, conn.reshape(-1, _CGNS_FACE_TYPES[etype])))
            bcs = []
            if "ZoneBC" in zone:
                for node in zone["ZoneBC"].values():
                    if isinstance(node, h5py.Group) and _h5_label(node) == "BC_t":
                        bc = _h5_value(node).tobytes().decode().strip("\x00")
                        if "ElementRange" in node or "PointRange" in node:
                            lo, hi = (int(v) for v in np.ravel(_h5_value(node["ElementRange" if "ElementRange" in node else "PointRange"])))
                            ids = np.arange(lo, hi + 1)
                        elif "PointList" in node or "ElementList" in node:
                            ids = np.ravel(_h5_value(node["PointList" if "PointList" in node else "ElementList"])).astype(np.int64)
                        else:
                            raise ValueError("BC '{}' has no ElementRange, PointRange or PointList".format(node.name))
                        bcs.append((node.name.split("/")[-1], bc, ids))
            if not bcs:
                bcs = [(name, None, np.arange(first, last + 1)) for name, first, last, _ in sections]
            if not bcs:
                raise ValueError("{} has no boundary patches (no BC_t nodes and no TRI_3 / QUAD_4 sections)".format(cgns_path))
            patches = {}
            for name, bc, ids in bcs:
                parts = []
                for _, first, last, faces in sections:
                    sel = ids[(ids >= first) & (ids <= last)]
                    if len(sel):
                        parts.append(faces[sel - first])
                if not parts:
                    raise ValueError("BC '{}' refers to no TRI_3 / QUAD_4 elements".format(name))
                if len({p.shape[1] for p in parts}) != 1:
                    raise ValueError("BC '{}' mixes triangles and quadrilaterals".format(name))
                patches[name] = dict(bc=bc, faces=np.concatenate(parts))
    return patches


def read_cgns_volume(cgns_path, interface_rel_tol=1e-10, patches=False):
    """
    Read a 3D CGNS grid into one mesh. Serial.

    Structured multi-block grids become one hexahedral mesh; unstructured zones become one mesh of tetrahedra or
    hexahedra (a single cell type; the zones of a multi-zone grid are merged like structured blocks). Nodes shared
    by blocks or zones are merged by exact coordinate match (see merge_structured_volume_blocks and
    merge_unstructured_zones), in the file's units. The CGNS boundary conditions are not read by VTK -- it skips BC_t
    types such as BCWallViscous and BCFarfield -- so the solver finds its boundaries geometrically, as for an untagged
    XDMF mesh, unless `patches` is set.

    interface_rel_tol: two boundary faces whose centroids are closer than this
    times the grid's bounding-box diagonal are an unmerged block interface.

    patches: also read the named boundary patches (read_cgns_patches; unstructured one-zone grids only). Every patch
    face must be a boundary face of the mesh; boundary faces in no patch are allowed.

    Returns a dict: points (N, 3), cells (M, 8 or 4), cell_type ("hexahedron" or "tetrahedron"), "hexes" or "tets"
    (the same array as cells), boundary_faces (node index quadruples or triples), n_raw_points (before merging),
    n_blocks (blocks or zones) and, with `patches`, patches: name -> dict(bc, faces) as read_cgns_patches, faces in
    the mesh's node numbering.
    Cells are in dolfinx ordering, right-handed.

    Raises ValueError if a cell is inverted or degenerate (at its center, for a hexahedron), a face has more than two
    cells, block interfaces did not merge, or a patch face is not a boundary face.
    """
    import pyvista as pv
    import vtk
    from scipy.spatial import cKDTree

    if not os.path.isfile(cgns_path):
        raise FileNotFoundError(cgns_path)
    reader = pv.get_reader(cgns_path)
    # VTK warns once per block and BC that it skips the BC_t node
    warnings_on = vtk.vtkObject.GetGlobalWarningDisplay()
    vtk.vtkObject.GlobalWarningDisplayOff()
    try:
        bases = reader.read()
    finally:
        vtk.vtkObject.SetGlobalWarningDisplay(warnings_on)
    grids = [zone["Internal"] for base in bases for zone in base if zone is not None]
    if not grids:
        raise ValueError("no zones in {}".format(cgns_path))
    structured = [isinstance(g, pv.StructuredGrid) for g in grids]
    if any(structured) and not all(structured):
        raise ValueError("{} mixes structured and unstructured zones".format(cgns_path))

    node_map = None
    if all(structured):
        blocks = []
        for grid in grids:
            ni, nj, nk = grid.dimensions
            if min(ni, nj, nk) < 2:
                raise ValueError("a block has {} x {} x {} nodes: not a volume grid".format(ni, nj, nk))
            # VTK structured points run i fastest, then j, then k
            blocks.append(np.asarray(grid.points, dtype=np.float64).reshape(nk, nj, ni, 3))
        points, cells = merge_structured_volume_blocks(blocks)
        cell_type = "hexahedron"
        n_raw = sum(b.shape[0] * b.shape[1] * b.shape[2] for b in blocks)
        if patches:
            raise ValueError("boundary patches are read for unstructured grids only")
    else:
        zones = [_unstructured_cells(g) for g in grids]
        types = {z[2] for z in zones}
        if len(types) != 1:
            raise ValueError("the zones have different cell types: {}".format(sorted(types)))
        cell_type = types.pop()
        points, cells, node_map = merge_unstructured_zones([z[:2] for z in zones], cell_type)
        n_raw = sum(len(z[0]) for z in zones)

    orientation = _cell_orientation(points, cells, cell_type)
    if not np.all(orientation > 0):
        raise ValueError("{} cells are inverted or degenerate".format(int(np.sum(orientation <= 0))))
    faces, counts = _unique_rows(_cell_faces(cells, cell_type))
    if counts.max() > 2:
        raise ValueError("{} faces have more than two cells".format(int(np.sum(counts > 2))))
    boundary_faces = faces[counts == 1]
    diagonal = np.linalg.norm(np.ptp(points, axis=0))
    pairs = cKDTree(points[boundary_faces].mean(axis=1)).query_pairs(interface_rel_tol * diagonal)
    if pairs:
        raise ValueError("{} pairs of boundary faces coincide: block interfaces whose nodes do not "
                         "match exactly".format(len(pairs)))
    out = dict(points=points, cells=cells, cell_type=cell_type, boundary_faces=boundary_faces,
               n_raw_points=n_raw, n_blocks=len(grids))
    out[_CELL_TYPES[cell_type]["key"]] = cells
    if patches:
        out["patches"] = _checked_patches(read_cgns_patches(cgns_path), node_map, len(points), n_raw, boundary_faces)
    return out


def _checked_patches(patches, node_map, n_points, n_raw, boundary_faces):
    """patches in the merged node numbering, after checking that they lie on the boundary and do not overlap."""
    boundary = {tuple(f) for f in boundary_faces}
    seen = {}
    for name, p in patches.items():
        faces = p["faces"]
        if faces.max() >= n_raw:
            raise ValueError("patch '{}' refers to node {} of a grid with {} nodes".format(name, faces.max() + 1, n_raw))
        if node_map is not None:
            faces = node_map[faces]
        p["faces"] = faces
        keys = {tuple(f) for f in np.sort(faces, axis=1)}
        if len(keys) != len(faces):
            raise ValueError("patch '{}' lists a face twice".format(name))
        if keys - boundary:
            raise ValueError("{} faces of patch '{}' are not boundary faces of the mesh".format(len(keys - boundary), name))
        for other, k in seen.items():
            if keys & k:
                raise ValueError("patches '{}' and '{}' share {} faces".format(name, other, len(keys & k)))
        seen[name] = keys
    return patches


def cgns_to_dolfinx_mesh(cgns_path, comm=MPI.COMM_WORLD, tdim=2, **kwargs):
    """
    A distributed dolfinx mesh from a CGNS grid: rank 0 reads it and dolfinx
    partitions it, ghosting cells across shared facets as XDMFFile.read_mesh
    does (the interior-facet forms need it). Collective.

    tdim=2: a structured grid one cell thick, reduced to quadrilaterals
    (read_cgns_planar, kwargs passed on). info holds chord, n_points, n_cells,
    n_raw_points, n_wall_edges and n_farfield_edges.
    tdim=3: a volume grid, as hexahedra or tetrahedra, whichever the grid holds
    (read_cgns_volume, kwargs passed on). info holds n_points, n_cells,
    n_raw_points, n_blocks, n_boundary_faces and cell_type.

    With `patches=True` (tdim=3, unstructured grids) the boundary patches are
    read too: info["patches"] maps every patch name to dict(value, bc,
    n_faces), `value` being its tag, and info["facet_tags"] is the dolfinx
    MeshTags of the facets (dimension 2) with those values, untagged facets
    absent. `patches` and the other entries of info are the same on every rank;
    `facet_tags` holds each rank's own facets, including the ghost copies of
    facets on process boundaries: count a patch's faces over the facets a rank
    owns (index < mesh.topology.index_map(2).size_local), or they are counted
    twice.

    Returns (mesh, info), info being the same on every rank (except facet_tags).
    """
    if tdim not in (2, 3):
        raise ValueError("tdim must be 2 (planar grid) or 3 (volume grid), got {}".format(tdim))
    if comm.rank == 0:
        try:
            data = (read_cgns_planar if tdim == 2 else read_cgns_volume)(cgns_path, **kwargs)
            error = None
        except Exception as e:  # raise on every rank, not only rank 0
            data, error = None, e
    else:
        data, error = None, None
    error = comm.bcast(error, root=0)
    if error is not None:
        raise error
    if comm.rank == 0:
        if tdim == 2:
            cells, x = data["quads"].astype(np.int64), data["points"]
            info = dict(chord=data["chord"], n_points=len(x), n_cells=len(cells),
                        n_raw_points=data["n_raw_points"], n_wall_edges=len(data["wall_edges"]),
                        n_farfield_edges=len(data["farfield_edges"]))
        else:
            cells, x = data["cells"].astype(np.int64), data["points"]
            info = dict(n_points=len(x), n_cells=len(cells), n_raw_points=data["n_raw_points"],
                        n_blocks=data["n_blocks"], n_boundary_faces=len(data["boundary_faces"]),
                        cell_type=data["cell_type"])
            if "patches" in data:
                info["patches"] = {name: dict(value=i + 1, bc=p["bc"], n_faces=len(p["faces"]))
                                   for i, (name, p) in enumerate(data["patches"].items())}
    else:
        info = None
    info = comm.bcast(info, root=0)
    cell_type = "quadrilateral" if tdim == 2 else info["cell_type"]
    n_nodes = 4 if tdim == 2 else _CELL_TYPES[cell_type]["n_nodes"]
    if comm.rank != 0:
        cells, x = np.empty((0, n_nodes), dtype=np.int64), np.empty((0, tdim))
    element = basix.ufl.element("Lagrange", cell_type, 1, shape=(tdim,))
    partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet, 2)
    mesh = dolfinx.mesh.create_mesh(comm, cells, element, x, partitioner=partitioner)
    if "patches" in info:
        info["facet_tags"] = _facet_tags(mesh, data["patches"] if comm.rank == 0 else None, info["patches"], tdim)
    return mesh, info


def _facet_tags(mesh, patches, values, tdim):
    """MeshTags of the facets of the patches, valued as `values` says. Collective: rank 0 holds `patches` (name ->
    dict(faces=node index array)), the others pass None."""
    comm = mesh.comm
    if comm.rank == 0:
        entities = np.concatenate([p["faces"] for p in patches.values()]).astype(np.int64)
        vals = np.concatenate([np.full(len(p["faces"]), values[name]["value"], dtype=np.int32)
                               for name, p in patches.items()])
    width = comm.bcast(entities.shape[1] if comm.rank == 0 else None, root=0)
    if comm.rank != 0:
        entities, vals = np.empty((0, width), dtype=np.int64), np.empty(0, dtype=np.int32)
    local_entities, local_values = dolfinx.io.distribute_entity_data(mesh, tdim - 1, entities, vals)
    mesh.topology.create_connectivity(tdim - 1, 0)
    return dolfinx.mesh.meshtags_from_entities(mesh, tdim - 1, dolfinx.graph.adjacencylist(local_entities),
                                               local_values.astype(np.int32))


def load_dolfinx_mesh(path, comm=MPI.COMM_WORLD, mesh_name="mesh", tdim=2, **cgns_kwargs):
    """
    A dolfinx mesh from an XDMF file (grid `mesh_name`) or a CGNS grid
    (cgns_to_dolfinx_mesh with `tdim`, cgns_kwargs passed on: e.g.
    patches=True for the boundary patches of an unstructured grid), by file
    extension. Returns (mesh, info); info is cgns_to_dolfinx_mesh's dict, or {}
    for XDMF. Collective.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cgns":
        return cgns_to_dolfinx_mesh(path, comm, tdim=tdim, **cgns_kwargs)
    if ext == ".xdmf":
        with dolfinx.io.XDMFFile(comm, path, "r") as xdmf:
            return xdmf.read_mesh(name=mesh_name), {}
    raise ValueError("unsupported mesh file '{}': expected .xdmf or .cgns".format(path))
