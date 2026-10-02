import numpy as np

def inner_bdry_function_from_corner_list(ffd_block_corner_list, overset_margin=1e-4, rel_tol=1e-6):
    """
    Build an inner-boundary indicator function out of an FFD block corner list.

    ffd_block_corner_list carries the same information an inner_bdry_function
    encodes: the inner boundary facets are the ones contained in the FFD block,
    so the block itself can serve as the containment test. The returned closure
    has the signature Mesh expects: it takes x of shape (n_dimensions, n_points)
    and returns a boolean mask of the points inside the block.

    The block is reconstructed exactly as in
    shape_parameterization.ffd_corners_from_corner_list -- two surfaces offset
    along a single connecting direction, linearly interpolated in between -- so a swept
    or tapered block is tested against its actual tapered extent rather than its
    bounding box.

    overset_margin matches the one the control point construction applies, so the
    test region is the FFD block itself rather than the bare corner list; rel_tol
    widens it further by that fraction of the block extent in each direction, so
    that geometry sitting exactly on the block boundary still counts as inside.

    NOTE the test is purely geometric: any OTHER boundary passing through the FFD
    block (e.g. the symmetry plane at the root of a wing) is flagged as well,
    whereas a hand-written inner_bdry_function can exclude it by construction.
    """
    num_surfaces = len(ffd_block_corner_list)
    assert num_surfaces == 2, "Expected exactly 2 surface corner pairs, got {}".format(num_surfaces)
    ndim = len(ffd_block_corner_list[0][0])

    # Per-surface min/max in each coordinate, as in the control point construction
    surface_mins = np.zeros((num_surfaces, ndim))
    surface_maxs = np.zeros((num_surfaces, ndim))
    for s, (corner_a, corner_b) in enumerate(ffd_block_corner_list):
        corner_a, corner_b = np.array(corner_a), np.array(corner_b)
        surface_mins[s] = np.minimum(corner_a, corner_b)
        surface_maxs[s] = np.maximum(corner_a, corner_b)

    surface_midpoints = 0.5 * (surface_mins + surface_maxs)
    coord_diffs = np.abs(surface_midpoints[1] - surface_midpoints[0])
    connecting_dir = int(np.argmax(coord_diffs))
    span_dirs = [d for d in range(ndim) if d != connecting_dir]

    # Same overset margin the control points are laid out with, so that the
    # containment test covers the whole FFD block
    for s in range(num_surfaces):
        for d in range(ndim):
            delta = surface_maxs[s, d] - surface_mins[s, d]
            if delta > 0:
                surface_mins[s, d] -= overset_margin * delta
                surface_maxs[s, d] += overset_margin * delta

    # Tolerances, scaled per direction by the block's bounding box extent. A
    # direction the block is collapsed in falls back to the largest extent, so
    # its tolerance stays a real (finite) length.
    block_mins = np.minimum(surface_mins[0], surface_mins[1])
    block_maxs = np.maximum(surface_maxs[0], surface_maxs[1])
    extents = block_maxs - block_mins
    tol = rel_tol * np.where(extents > 0., extents, extents.max())

    connect_lo = surface_midpoints[0][connecting_dir]
    connect_hi = surface_midpoints[1][connecting_dir]
    connect_span = connect_hi - connect_lo
    t_tol = tol[connecting_dir] / abs(connect_span)

    print("inner_bdry_function_from_corner_list: FFD block used as inner boundary test")
    print("  connecting direction: {}".format(connecting_dir))
    print("  block min corner: {}".format(list(block_mins)))
    print("  block max corner: {}".format(list(block_maxs)))

    def inner_bdry_function(x):
        """
        Input: x is an array of size (n_dimensions, n_points)
        """
        coords = np.asarray(x)[:ndim, :]

        # Parametric position along the connecting direction; points outside
        # [0, 1] are beyond the two surfaces bounding the block
        t = (coords[connecting_dir, :] - connect_lo) / connect_span
        inside = (t >= -t_tol) & (t <= 1. + t_tol)

        # Within the block, the span-direction bounds are the linear
        # interpolation of the two surfaces' bounds
        t_clipped = np.clip(t, 0., 1.)
        for d in span_dirs:
            lower = (1. - t_clipped) * surface_mins[0, d] + t_clipped * surface_mins[1, d]
            upper = (1. - t_clipped) * surface_maxs[0, d] + t_clipped * surface_maxs[1, d]
            inside &= (coords[d, :] >= lower - tol[d]) & (coords[d, :] <= upper + tol[d])

        return inside

    return inner_bdry_function

def find_local_surface_nodes(msh, facet_tags, surface_tag: int) -> np.ndarray:
    """Return *local* geometry indices of owned surface nodes (ghosts excluded).

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
    facet_tags : dolfinx.mesh.MeshTags | None
    surface_tag : int

    Returns
    -------
    np.ndarray, int64, shape (M_local,)
    """

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, 0)
    f_to_v = msh.topology.connectivity(fdim, 0)

    node_set: set[int] = set()
    if facet_tags is not None:
        for f in facet_tags.find(surface_tag):
            node_set.update(f_to_v.links(int(f)).tolist())
    else:
        msh.topology.create_entities(fdim)
        msh.topology.create_connectivity(fdim, tdim)
        f_to_c = msh.topology.connectivity(fdim, tdim)
        for f in range(msh.topology.index_map(fdim).size_local):
            if len(f_to_c.links(f)) == 1:
                node_set.update(f_to_v.links(f).tolist())

    n_local_geom = msh.geometry.index_map().size_local  # excludes ghosts
    return np.array(
        sorted(idx for idx in node_set if idx < n_local_geom),
        dtype=np.int64,
    )
