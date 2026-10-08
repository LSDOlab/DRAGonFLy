"""
Mesh input: dolfinx meshes from XDMF files or from structured CGNS grids.

A structured multi-block CGNS grid of a 2D case is often stored as a 3D grid
one cell thick in span (e.g. ONERA's OAT15A grids). read_cgns_planar reads it
with PyVista (VTK's CGNS reader), keeps one span plane of every block as
quadrilaterals, merges the nodes the blocks share, and finds the wall and far
field edges from the CGNS boundary condition families. read_cgns_volume reads
a 3D grid (e.g. the Simple Transonic Wing's wing_vol_L*.cgns) into one
hexahedral mesh the same way. cgns_to_dolfinx_mesh turns either into a
distributed dolfinx mesh in memory: no intermediate file.

Block interface nodes are merged by EXACT coordinate match, which holds for
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


# The six faces of a hexahedron in dolfinx ordering, as local node indices
_HEX_FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5],
                       [2, 3, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]])


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


def _hex_faces(hexes):
    """The six faces of every hex, as sorted node quadruples."""
    return np.sort(hexes[:, _HEX_FACES].reshape(-1, 4), axis=1)


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


def read_cgns_volume(cgns_path, interface_rel_tol=1e-10):
    """
    Read a structured multi-block 3D CGNS grid into one hexahedral mesh. Serial.

    The blocks' nodes are merged by exact coordinate match (see
    merge_structured_volume_blocks), in the file's units. The CGNS boundary
    conditions are not read -- VTK skips BC_t types such as BCWallViscous and
    BCFarfield -- so the solver finds its boundaries geometrically, as for an
    untagged XDMF mesh.

    interface_rel_tol: two boundary faces whose centroids are closer than this
    times the grid's bounding-box diagonal are an unmerged block interface.

    Returns a dict: points (N, 3), hexes (M, 8) (dolfinx ordering,
    right-handed), boundary_faces (node index quadruples), n_raw_points
    (before merging), n_blocks.

    Raises ValueError if a cell is inverted or degenerate at its center, a
    face has more than two cells, or block interfaces did not merge.
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
    blocks = []
    for base in bases:
        for zone in base:
            if zone is None:
                continue
            grid = zone["Internal"]
            ni, nj, nk = grid.dimensions
            if min(ni, nj, nk) < 2:
                raise ValueError("a block has {} x {} x {} nodes: not a volume grid".format(ni, nj, nk))
            # VTK structured points run i fastest, then j, then k
            blocks.append(np.asarray(grid.points, dtype=np.float64).reshape(nk, nj, ni, 3))
    if not blocks:
        raise ValueError("no structured blocks in {}".format(cgns_path))
    points, hexes = merge_structured_volume_blocks(blocks)
    n_raw = sum(b.shape[0] * b.shape[1] * b.shape[2] for b in blocks)

    jac = _center_jacobians(points, hexes)
    if not np.all(jac > 0):
        raise ValueError("{} cells are inverted or degenerate".format(int(np.sum(jac <= 0))))
    faces, counts = np.unique(_hex_faces(hexes), axis=0, return_counts=True)
    if counts.max() > 2:
        raise ValueError("{} faces have more than two cells".format(int(np.sum(counts > 2))))
    boundary_faces = faces[counts == 1]
    diagonal = np.linalg.norm(np.ptp(points, axis=0))
    pairs = cKDTree(points[boundary_faces].mean(axis=1)).query_pairs(interface_rel_tol * diagonal)
    if pairs:
        raise ValueError("{} pairs of boundary faces coincide: block interfaces whose nodes do not "
                         "match exactly".format(len(pairs)))
    return dict(points=points, hexes=hexes, boundary_faces=boundary_faces, n_raw_points=n_raw,
                n_blocks=len(blocks))


def cgns_to_dolfinx_mesh(cgns_path, comm=MPI.COMM_WORLD, tdim=2, **kwargs):
    """
    A distributed dolfinx mesh from a structured CGNS grid: rank 0 reads it
    and dolfinx partitions it, ghosting cells across shared facets as
    XDMFFile.read_mesh does (the interior-facet forms need it). Collective.

    tdim=2: a grid one cell thick, reduced to quadrilaterals (read_cgns_planar,
    kwargs passed on). info holds chord, n_points, n_cells, n_raw_points,
    n_wall_edges and n_farfield_edges.
    tdim=3: a volume grid, as hexahedra (read_cgns_volume, kwargs passed on).
    info holds n_points, n_cells, n_raw_points, n_blocks and n_boundary_faces.

    Returns (mesh, info), info being the same on every rank.
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
    cells_key, nodes_per_cell, cell_name = (("quads", 4, "quadrilateral") if tdim == 2
                                            else ("hexes", 8, "hexahedron"))
    if comm.rank == 0:
        cells, x = data[cells_key].astype(np.int64), data["points"]
        if tdim == 2:
            info = dict(chord=data["chord"], n_points=len(x), n_cells=len(cells),
                        n_raw_points=data["n_raw_points"], n_wall_edges=len(data["wall_edges"]),
                        n_farfield_edges=len(data["farfield_edges"]))
        else:
            info = dict(n_points=len(x), n_cells=len(cells), n_raw_points=data["n_raw_points"],
                        n_blocks=data["n_blocks"], n_boundary_faces=len(data["boundary_faces"]))
    else:
        cells, x, info = (np.empty((0, nodes_per_cell), dtype=np.int64), np.empty((0, tdim)),
                          None)
    info = comm.bcast(info, root=0)
    element = basix.ufl.element("Lagrange", cell_name, 1, shape=(tdim,))
    partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet, 2)
    mesh = dolfinx.mesh.create_mesh(comm, cells, element, x, partitioner=partitioner)
    return mesh, info


def load_dolfinx_mesh(path, comm=MPI.COMM_WORLD, mesh_name="mesh", tdim=2, **cgns_kwargs):
    """
    A dolfinx mesh from an XDMF file (grid `mesh_name`) or a structured CGNS
    grid (cgns_to_dolfinx_mesh with `tdim`, cgns_kwargs passed on), by file
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
