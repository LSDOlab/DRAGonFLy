"""
Mesh input: dolfinx meshes from XDMF files or from structured CGNS grids.

A structured multi-block CGNS grid of a 2D case is often stored as a 3D grid
one cell thick in span (e.g. ONERA's OAT15A grids). read_cgns_planar reads it
with PyVista (VTK's CGNS reader), keeps one span plane of every block as
quadrilaterals, merges the nodes the blocks share, and finds the wall and far
field edges from the CGNS boundary condition families. cgns_to_dolfinx_mesh
turns that into a distributed dolfinx mesh in memory: no intermediate file.

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


def cgns_to_dolfinx_mesh(cgns_path, comm=MPI.COMM_WORLD, **kwargs):
    """
    A distributed dolfinx quadrilateral mesh from a structured CGNS grid one
    cell thick: rank 0 reads it (read_cgns_planar, kwargs passed on) and
    dolfinx partitions it, ghosting cells across shared facets as
    XDMFFile.read_mesh does (the interior-facet forms need it). Collective.

    Returns (mesh, info), info being a dict with chord, n_points, n_cells,
    n_raw_points, n_wall_edges and n_farfield_edges (the same on every rank).
    """
    if comm.rank == 0:
        try:
            data = read_cgns_planar(cgns_path, **kwargs)
            error = None
        except Exception as e:  # raise on every rank, not only rank 0
            data, error = None, e
    else:
        data, error = None, None
    error = comm.bcast(error, root=0)
    if error is not None:
        raise error
    if comm.rank == 0:
        cells, x = data["quads"].astype(np.int64), data["points"]
        info = dict(chord=data["chord"], n_points=len(x), n_cells=len(cells),
                    n_raw_points=data["n_raw_points"], n_wall_edges=len(data["wall_edges"]),
                    n_farfield_edges=len(data["farfield_edges"]))
    else:
        cells, x, info = np.empty((0, 4), dtype=np.int64), np.empty((0, 2)), None
    info = comm.bcast(info, root=0)
    element = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet, 2)
    mesh = dolfinx.mesh.create_mesh(comm, cells, element, x, partitioner=partitioner)
    return mesh, info


def load_dolfinx_mesh(path, comm=MPI.COMM_WORLD, mesh_name="mesh", **cgns_kwargs):
    """
    A dolfinx mesh from an XDMF file (grid `mesh_name`) or a structured CGNS
    grid (cgns_to_dolfinx_mesh, cgns_kwargs passed on), by file extension.
    Returns (mesh, info); info is cgns_to_dolfinx_mesh's dict, or {} for XDMF.
    Collective.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cgns":
        return cgns_to_dolfinx_mesh(path, comm, **cgns_kwargs)
    if ext == ".xdmf":
        with dolfinx.io.XDMFFile(comm, path, "r") as xdmf:
            return xdmf.read_mesh(name=mesh_name), {}
    raise ValueError("unsupported mesh file '{}': expected .xdmf or .cgns".format(path))
