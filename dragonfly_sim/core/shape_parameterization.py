"""Wall-shape parameterization built on lsdo_geo.

Maps shape design variables to the motion of the mesh wall nodes, and measures
the deformed wall for geometric constraints. Three pieces:

* ``WallFFD`` -- an ``lsdo_geo.FFDBlock`` around the wall nodes. Its output,
  ``wall_displacement(C)``, is the replicated ``(M_global, dim)`` array that
  ``IDWarp_jax.evaluate`` consumes.
* Shape layers that produce the block's coefficients ``C``:
  ``SectionalShape`` (lsdo_geo ``SectionalParameterization``: camber/thickness
  per chordwise station in 2D; twist/chord/sweep/dihedral/span per spanwise
  section in 3D) and ``WallFFD.apply_cp_motions`` (raw control-point motions).
  Compose them as ``C = apply_cp_motions(SectionalShape.apply(C0), cp_motions)``
  -- see ``SectionalShape`` for why the sectional layer must act on the
  constant baseline.
* ``WallGeometry`` -- geometric constraint quantities of the deformed wall:
  enclosed area (2D) / volume (3D), thickness at stations, and 3D planform
  (section chords, span, planform area).

Everything here is csdl_alpha graph code on replicated arrays, so derivatives
come from csdl's autodiff and every rank computes identical values.
"""
import itertools

import numpy as np
import scipy.sparse as sps
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo as lg

from dragonfly_sim.utils.mpi_utils import allgather_numpy


def validate_cp_coord_opt_idxs(cp_coord_opt_idxs, dim):
    """Check and return the optimized control-point directions as an int array.

    Strictly increasing is required, not merely unique: the cp_motions design
    variable's trailing axis is a compressed index into this array (see
    ``WallFFD.apply_cp_motions``), so a fixed order is what lets the bounds
    built by ``ffd_dv_utils.build_cp_motion_dv`` line up with the DV columns
    without either side having to know how the other was written.
    """
    if cp_coord_opt_idxs is None:
        return np.arange(dim)
    idxs = np.asarray(cp_coord_opt_idxs)
    if not np.issubdtype(idxs.dtype, np.integer):
        raise TypeError("cp_coord_opt_idxs must be integer spatial direction "
                        "indices, got dtype {}".format(idxs.dtype))
    if idxs.ndim != 1:
        raise ValueError("cp_coord_opt_idxs must be 1-D, got shape {}".format(idxs.shape))
    if idxs.size == 0:
        raise ValueError("cp_coord_opt_idxs is empty: at least one coordinate "
                         "direction must be a design variable")
    if np.any(np.diff(idxs) <= 0):
        raise ValueError("cp_coord_opt_idxs must be strictly increasing, got {}"
                         .format(idxs.tolist()))
    if idxs[0] < 0 or idxs[-1] >= dim:
        raise ValueError("cp_coord_opt_idxs must lie in [0, {}) for a {}-D mesh, got {}"
                         .format(dim, dim, idxs.tolist()))
    return idxs


def ffd_corners_from_bounding_box(coords, margin=1e-4):
    """Axis-aligned FFD block corners around `coords`, widened by margin*extent.

    Returns the ``(2,)*dim + (dim,)`` corner array that
    ``lfs.create_b_spline_from_corners`` takes: ``corners[i0, i1, ..., d]`` is
    the min (i_d = 0) or max (i_d = 1) of coordinate d.
    """
    coords = np.asarray(coords, dtype=np.float64)
    dim = coords.shape[1]
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    delta = hi - lo
    bounds = np.stack([lo - margin * delta, hi + margin * delta])  # (2, dim)
    corners = np.zeros((2,) * dim + (dim,))
    for idx in itertools.product((0, 1), repeat=dim):
        corners[idx] = [bounds[idx[d], d] for d in range(dim)]
    return corners


def ffd_corners_from_corner_list(ffd_block_corner_list, margin=1e-4):
    """FFD block corners lofted between two rectangular end surfaces.

    ``ffd_block_corner_list`` holds exactly two surfaces, each given by two
    opposite corners, e.g. the root and tip rectangles of a wing block. The
    "connecting" direction is the axis along which the surface midpoints differ
    most (spanwise for a wing); each surface spans the remaining axes. Each
    surface is widened by margin*extent along the axes it spans. The resulting
    block is tapered/swept, and ``lfs.create_b_spline_from_corners`` fills it
    with the same trilinear lattice the pre-lsdo_geo construction produced.
    """
    if len(ffd_block_corner_list) != 2:
        raise ValueError("Expected exactly 2 surface corner pairs, got {}"
                         .format(len(ffd_block_corner_list)))
    dim = len(ffd_block_corner_list[0][0])
    surface_mins = np.zeros((2, dim))
    surface_maxs = np.zeros((2, dim))
    for s, (corner_a, corner_b) in enumerate(ffd_block_corner_list):
        corner_a, corner_b = np.asarray(corner_a, float), np.asarray(corner_b, float)
        surface_mins[s] = np.minimum(corner_a, corner_b)
        surface_maxs[s] = np.maximum(corner_a, corner_b)

    midpoints = 0.5 * (surface_mins + surface_maxs)
    connecting_dir = int(np.argmax(np.abs(midpoints[1] - midpoints[0])))

    delta = surface_maxs - surface_mins
    widen = np.where(delta > 0, margin * delta, 0.)
    surface_mins = surface_mins - widen
    surface_maxs = surface_maxs + widen

    corners = np.zeros((2,) * dim + (dim,))
    for idx in itertools.product((0, 1), repeat=dim):
        s = idx[connecting_dir]
        corners[idx] = [surface_mins[s, d] if (d == connecting_dir or idx[d] == 0)
                        else surface_maxs[s, d] for d in range(dim)]
    return corners


class WallFFD:
    """lsdo_geo FFD block embedding the mesh wall nodes.

    Replicated across ranks: every rank evaluates the FFD at all ``M_global``
    wall nodes and returns the full global array, rather than evaluating only
    the nodes it owns and having the pieces assembled afterwards.

    That is deliberate. A rank owning no wall nodes would otherwise return a
    ``(0, dim)`` array, and the reverse chain back to the design variables would
    reduce an empty array -- which XLA constant-folds at compile time. The
    downstream gradient reduction would then have no data dependence on
    anything on that rank, and since ``jax.pure_callback`` is declared
    effect-free, its collective could be scheduled arbitrarily early relative
    to the other ranks'. That is an ordering deadlock the ranks cannot recover
    from. Evaluating the full wall everywhere keeps every rank's chain
    non-degenerate and lets the gradient reduction happen inside
    ``IDWarp_jax.compute_jacvec_product``, on an array every rank genuinely
    computes.

    Only the projection is rank-local: each rank projects the wall nodes it
    owns (``local_rows``) and the resulting parametric coordinates are gathered
    in rank order.

    Parameters
    ----------
    comm : mpi4py communicator
    global_wall_coords : (M_global, dim) array
        All wall node coordinates, in the rank-block order of
        ``Mesh.boundary_node_set((wall_tag,)).coords``.
    local_rows : int array
        The contiguous rows of ``global_wall_coords`` this rank owns
        (that node set's ``global_idx``).
    shape : sequence of int
        Control points per parametric direction.
    degree : int or sequence of int
    ffd_block_corner_list : optional
        Two-surface block definition, see ``ffd_corners_from_corner_list``.
        None wraps the wall's bounding box.
    margin : float
        Relative widening of the block beyond the wall / the given surfaces.
    projection_newton_tol : float
        Newton tolerance of the wall node projection into the block.
    """

    def __init__(self, comm, global_wall_coords, local_rows, shape, degree=2,
                 ffd_block_corner_list=None, margin=1e-4, projection_newton_tol=1e-12,
                 name='wall_ffd'):
        self.comm = comm
        self.baseline_wall_coords = np.ascontiguousarray(global_wall_coords, dtype=np.float64)
        self.M_global, self.dim = self.baseline_wall_coords.shape
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != self.dim:
            raise ValueError("FFD shape {} does not match the {}-D wall"
                             .format(self.shape, self.dim))
        degree = (degree,) * self.dim if np.isscalar(degree) else tuple(int(p) for p in degree)
        self.degree = degree

        if ffd_block_corner_list is None:
            corners = ffd_corners_from_bounding_box(self.baseline_wall_coords, margin)
            self.ffd_block_mode = 'automatic'
        else:
            corners = ffd_corners_from_corner_list(ffd_block_corner_list, margin)
            self.ffd_block_mode = 'manual'
        self.corners = corners

        b_spline = lfs.create_b_spline_from_corners(
            corners, degree=self.degree, num_coefficients=self.shape, name=name)
        # No embedded_entities here: FFDBlock.embed_entities would project with
        # refine_projection, which assumes 3-D points, and would pickle the
        # result into ./stored_files/projections. The wall and the probe sets
        # are projected below instead, and appended as embedded entities.
        self.block = lg.FFDBlock(space=b_spline.space, coefficients=b_spline.coefficients,
                                 name=name)
        self.baseline_coefficients = np.array(self.block.coefficients.value, dtype=np.float64)

        local_rows = np.asarray(local_rows, dtype=np.int64)
        if local_rows.size and np.any(np.diff(local_rows) != 1):
            raise ValueError("local_rows must be a contiguous block of wall rows")
        self.projection_newton_tol = projection_newton_tol
        local_u = self._project(self.baseline_wall_coords[local_rows])
        self.wall_parametric_coordinates = allgather_numpy(
            comm, local_u.ravel(), self.M_global * self.dim,
        ).reshape(self.M_global, self.dim)
        self.block.embedded_entity_parametric_coordinates.append(self.wall_parametric_coordinates)

        self._probe_points = []

    def _project(self, points):
        if points.shape[0] == 0:
            return np.zeros((0, self.dim))
        return np.asarray(self.block.project(
            points=points, plot=False, force_reproject=True,
            grid_search_density_parameter=10, grid_search_subtraction_cutoff=1000,
            newton_tolerance=self.projection_newton_tol, projection_tolerance=None,
            do_pickles=False), dtype=np.float64).reshape(-1, self.dim)

    def add_probe_points(self, points):
        """Embed extra points (replicated input); returns their probe index.

        Projected on rank 0 and broadcast, so every rank holds identical
        parametric coordinates.
        """
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, self.dim)
        u = self._project(points) if self.comm.rank == 0 else None
        u = self.comm.bcast(u, root=0)
        self.block.embedded_entity_parametric_coordinates.append(u)
        self._probe_points.append((points, u))
        return len(self._probe_points) - 1

    def baseline_variable(self):
        """The baseline coefficients as a constant csdl Variable."""
        return csdl.Variable(value=self.baseline_coefficients.copy())

    def apply_cp_motions(self, coefficients, cp_motions, cp_coord_opt_idxs):
        """Add raw control-point motions to `coefficients`.

        ``cp_motions`` has shape ``ffd_shape + (len(cp_coord_opt_idxs),)``;
        column i moves spatial component ``cp_coord_opt_idxs[i]`` and the other
        directions are left unchanged. The index tuples are built rather than
        written as literals so this works for any dimension.
        """
        idxs = validate_cp_coord_opt_idxs(cp_coord_opt_idxs, self.dim)
        expected_shape = self.shape + (idxs.size,)
        if tuple(cp_motions.shape) != expected_shape:
            raise ValueError(
                "cp_motions has shape {}, expected {} (ffd_shape + (n_opt,), where "
                "n_opt = len(cp_coord_opt_idxs) = {}). See ffd_dv_utils.build_cp_motion_dv "
                "for constructing a design variable of the right shape."
                .format(tuple(cp_motions.shape), expected_shape, idxs.size))
        updated = coefficients
        for i, coord in enumerate(idxs):
            cp_index = tuple([slice(None)] * self.dim + [int(coord)])
            motion_index = tuple([slice(None)] * self.dim + [i])
            updated = updated.set(csdl.slice[cp_index],
                                  coefficients[cp_index] + cp_motions[motion_index])
        return updated

    def _displacement_at(self, parametric_coordinates, coefficients):
        # The FFD map is linear in the coefficients, so the displacement is the
        # map applied to C - C0: exact zero at the baseline, whatever the
        # projection residual of the baseline points.
        delta = coefficients - self.baseline_variable()
        return self.block.evaluate(parametric_coordinates, coefficients=delta)

    def wall_displacement(self, coefficients):
        """Replicated ``(M_global, dim)`` wall node displacement for `coefficients`."""
        return self._displacement_at(self.wall_parametric_coordinates, coefficients)

    def probe_positions(self, coefficients, probe):
        """Deformed positions of probe set `probe` (index from add_probe_points)."""
        points, u = self._probe_points[probe]
        return self._displacement_at(u, coefficients) + points


class SectionalShape:
    """Sectional shape variables on a WallFFD, via lsdo_geo's SectionalParameterization.

    The block is seen as a stack of sections normal to ``principal_dim``: in 2D
    use principal_dim = 0 (chordwise stations, each a vertical column of control
    points); for a wing, the spanwise parametric direction. Each variable holds
    either one value per section, or -- when it has fewer entries -- the
    coefficients of a B-spline profile over the sections.

    The sectional layer must act on the *constant* baseline coefficients.
    lsdo_geo takes stretch and translation axes, and stretch origins, from the
    points' values when the graph is built, so feeding it DV-dependent points
    would bake the initial DV values into the map. Add raw control-point
    motions afterwards (``WallFFD.apply_cp_motions``).

    Named presets (``add``), with lsdo_geo's conventions: a stretch is the
    additive change in section extent along the axis (linear across the
    section, zero at the pivot); a rotation is in radians about the axis
    through the pivot; a translation is a length.

    ========== ======================================== =======
    kind       operation                                 dims
    ========== ======================================== =======
    camber     translation along the vertical axis      2D
    thickness  stretch along the vertical param. axis   2D, 3D
    twist      rotation about the principal axis        3D
    chord      stretch along parametric axis 0          3D
    sweep      translation along x                      3D
    dihedral   translation along z                      3D
    span       translation along y                      3D
    ========== ======================================== =======
    """

    def __init__(self, ffd, principal_dim=0):
        self.ffd = ffd
        self.dim = ffd.dim
        if not 0 <= principal_dim < self.dim:
            raise ValueError("principal_dim must lie in [0, {})".format(self.dim))
        self.principal_dim = principal_dim
        self.num_sections = ffd.shape[principal_dim]
        # (kind, values, axis, pivot, profile_degree); turned into lsdo_geo
        # SectionalParameters inside apply(), see there for why
        self._specs = []

    def _per_section(self, values, profile_degree=2):
        n = int(np.prod(values.shape))
        if n == self.num_sections:
            return values.reshape((self.num_sections,))
        if not 1 <= n < self.num_sections:
            raise ValueError("a sectional variable needs {} entries (one per section) "
                             "or fewer (profile coefficients), got {}"
                             .format(self.num_sections, n))
        space = lfs.BSplineSpace(num_parametric_dimensions=1,
                                 degree=min(profile_degree, n - 1), coefficients_shape=(n,))
        basis = space.compute_basis_matrix(
            np.linspace(0., 1., self.num_sections).reshape(-1, 1))
        basis = basis.toarray() if sps.issparse(basis) else np.asarray(basis)
        return csdl.matvec(basis, values.reshape((n,)))

    def _section_pivot(self, pivot):
        n_section_dims = self.dim - 1
        if pivot is None:
            return None
        pivot = np.asarray(pivot, dtype=np.float64).reshape(-1)
        if pivot.size != n_section_dims:
            raise ValueError("pivot is a parametric coordinate within the section "
                             "({} entries), got {}".format(n_section_dims, pivot.size))
        return pivot

    def _check_size(self, values):
        n = int(np.prod(values.shape))
        if not 1 <= n <= self.num_sections:
            raise ValueError("a sectional variable needs {} entries (one per section) "
                             "or fewer (profile coefficients), got {}"
                             .format(self.num_sections, n))

    def add_translation(self, values, direction, profile_degree=2):
        self._check_size(values)
        self._specs.append(('translation', values, np.asarray(direction, dtype=np.float64),
                            None, profile_degree))

    def add_stretch(self, values, parametric_axis, pivot=None, profile_degree=2):
        self._check_size(values)
        self._specs.append(('stretch', values, int(parametric_axis),
                            self._section_pivot(pivot), profile_degree))

    def add_rotation(self, values, parametric_axis=None, pivot=None, profile_degree=2):
        if self.dim != 3:
            raise ValueError("sectional rotations need a 3-D block (lsdo_geo "
                             "rotates with quaternions)")
        self._check_size(values)
        axis = self.principal_dim if parametric_axis is None else int(parametric_axis)
        self._specs.append(('rotation', values, axis, self._section_pivot(pivot), profile_degree))

    def add(self, kind, values, pivot=None, profile_degree=2):
        """Add a named sectional variable; see the class docstring."""
        vertical = self.dim - 1
        unit = np.eye(self.dim)
        if kind == 'camber' and self.dim == 2:
            self.add_translation(values, unit[vertical], profile_degree)
        elif kind == 'thickness':
            self.add_stretch(values, vertical, pivot, profile_degree)
        elif kind == 'twist':
            self.add_rotation(values, None, pivot, profile_degree)
        elif kind == 'chord' and self.dim == 3:
            self.add_stretch(values, 0, pivot, profile_degree)
        elif kind == 'sweep' and self.dim == 3:
            self.add_translation(values, unit[0], profile_degree)
        elif kind == 'dihedral' and self.dim == 3:
            self.add_translation(values, unit[2], profile_degree)
        elif kind == 'span' and self.dim == 3:
            self.add_translation(values, unit[1], profile_degree)
        else:
            raise ValueError("unknown sectional variable {!r} for a {}-D block"
                             .format(kind, self.dim))

    def apply(self, coefficients):
        """Sectionally deformed coefficients. Pass ``ffd.baseline_variable()``."""
        if not self._specs:
            return coefficients
        # lsdo_geo reads intermediate .value's while building this subgraph
        # (stretch axes and origins, integer-axis directions), which a
        # non-inline recorder -- the drivers' -- leaves as None. Every input
        # here has a value (the baseline and the design variables), so record
        # just this small subgraph, profiles included, inline.
        recorder = csdl.get_current_recorder()
        was_inline = recorder.inline
        recorder.inline = True
        try:
            parameters = lg.SectionalParameters()
            for kind, values, axis, pivot, profile_degree in self._specs:
                per_section = self._per_section(values, profile_degree)
                if kind == 'translation':
                    parameters.add_translation(axis, per_section)
                elif kind == 'stretch':
                    parameters.add_stretch(axis, per_section, pivot)
                else:
                    parameters.add_rotation(axis, per_section, pivot)
            parameterization = lg.SectionalParameterization(
                parameterized_points=coefficients,
                principal_parametric_dimension=self.principal_dim)
            return parameterization.evaluate(parameters)
        finally:
            recorder.inline = was_inline


def _selection_matrix(ids, n_cols):
    ids = np.asarray(ids, dtype=np.int64)
    return sps.csr_matrix((np.ones(ids.size), (np.arange(ids.size), ids)),
                          shape=(ids.size, n_cols))


class WallGeometry:
    """Geometric constraint quantities of the FFD-deformed wall.

    Parameters
    ----------
    ffd : WallFFD
    wall_facets : (n_facets, k) int array
        Replicated wall facets as rows of wall node ids in ``[0, M_global)``
        (``Mesh.wall_facets(wall_tag)``): segments in 2D, triangles or quads
        (cyclic vertex order) in 3D. Their winding only needs to be consistent;
        the sign of the enclosed measure is fixed from the baseline.

    In 3D the wall may be open on a symmetry plane through the origin (y = 0):
    that plane contributes nothing to the divergence-theorem volume, as long as
    the root section keeps y = 0.
    """

    def __init__(self, ffd, wall_facets):
        self.ffd = ffd
        self.dim = ffd.dim
        facets = np.asarray(wall_facets, dtype=np.int64)
        if facets.ndim != 2 or facets.size == 0:
            raise ValueError("wall_facets must be a non-empty (n_facets, k) array")
        if facets.min() < 0 or facets.max() >= ffd.M_global:
            raise ValueError("wall_facets must index wall nodes in [0, {})".format(ffd.M_global))
        if self.dim == 2:
            if facets.shape[1] != 2:
                raise ValueError("2D wall facets must be segments")
            self.simplices = facets
        else:
            if facets.shape[1] == 3:
                self.simplices = facets
            elif facets.shape[1] == 4:
                self.simplices = np.concatenate([facets[:, [0, 1, 2]], facets[:, [0, 2, 3]]])
            else:
                raise ValueError("3D wall facets must be triangles or quads")
        self._selectors = [_selection_matrix(self.simplices[:, j], ffd.M_global)
                           for j in range(self.simplices.shape[1])]
        raw = self._raw_measure_numpy(ffd.baseline_wall_coords)
        if not abs(raw) > 0.:
            raise ValueError("the baseline wall encloses no area/volume; "
                             "are the wall facets complete?")
        self._sign = float(np.sign(raw))
        self.baseline_measure = abs(raw)

    # ---- enclosed area / volume -------------------------------------------
    def _raw_measure_numpy(self, X):
        v = [X[self.simplices[:, j]] for j in range(self.simplices.shape[1])]
        if self.dim == 2:
            return 0.5 * np.sum(v[0][:, 0] * v[1][:, 1] - v[1][:, 0] * v[0][:, 1])
        return np.sum(np.einsum('ij,ij->i', v[0], np.cross(v[1], v[2]))) / 6.

    def _gather(self, X, j):
        cols = [csdl.sparse.matvec(self._selectors[j], X[:, d].reshape((self.ffd.M_global, 1)))
                for d in range(self.dim)]
        return csdl.concatenate(cols, axis=1)

    def enclosed_measure(self, wall_displacement):
        """Enclosed area (2D) or volume (3D) of the deformed wall, positive."""
        X = wall_displacement + self.ffd.baseline_wall_coords
        v = [self._gather(X, j) for j in range(self.simplices.shape[1])]
        if self.dim == 2:
            raw = 0.5 * csdl.sum(v[0][:, 0] * v[1][:, 1] - v[1][:, 0] * v[0][:, 1])
        else:
            raw = csdl.sum(v[0] * csdl.cross(v[1], v[2], axis=1)) / 6.
        return self._sign * raw

    # ---- baseline-geometry queries (numpy, replicated) ---------------------
    def _vertical_line_hits(self, x_in_plane):
        """Vertical-coordinate values where the vertical line through
        `x_in_plane` (x in 2D, (x, y) in 3D) crosses the baseline wall."""
        X = self.ffd.baseline_wall_coords
        if self.dim == 2:
            a, b = X[self.simplices[:, 0]], X[self.simplices[:, 1]]
            x0 = float(x_in_plane)
            crosses = (a[:, 0] - x0) * (b[:, 0] - x0) <= 0.
            crosses &= a[:, 0] != b[:, 0]
            t = (x0 - a[crosses, 0]) / (b[crosses, 0] - a[crosses, 0])
            return a[crosses, 1] + t * (b[crosses, 1] - a[crosses, 1])
        p = np.asarray(x_in_plane, dtype=np.float64)
        a, b, c = (X[self.simplices[:, j]] for j in range(3))
        # barycentric coordinates in the xy projection
        v0, v1, v2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], p - a[:, :2]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-14 * (np.abs(v0).max() * np.abs(v1).max() + 1e-300)
        den = np.where(ok, den, 1.)
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        tol = 1e-12
        inside = ok & (l1 >= -tol) & (l2 >= -tol) & (l1 + l2 <= 1. + tol)
        return (a[inside, 2] + l1[inside] * (b[inside, 2] - a[inside, 2])
                + l2[inside] * (c[inside, 2] - a[inside, 2]))

    def _section_leading_trailing_edges(self, y0):
        """Baseline (LE, TE) points of the 3D wall's section at span y0: the
        min-x and max-x points where the plane y = y0 cuts the wall."""
        X = self.ffd.baseline_wall_coords
        pts = []
        for j, k in ((0, 1), (1, 2), (2, 0)):
            a, b = X[self.simplices[:, j]], X[self.simplices[:, k]]
            crosses = ((a[:, 1] - y0) * (b[:, 1] - y0) <= 0.) & (a[:, 1] != b[:, 1])
            t = (y0 - a[crosses, 1]) / (b[crosses, 1] - a[crosses, 1])
            pts.append(a[crosses] + t[:, None] * (b[crosses] - a[crosses]))
        pts = np.concatenate(pts)
        if pts.shape[0] == 0:
            raise ValueError("the plane y = {} does not cut the wall".format(y0))
        return pts[np.argmin(pts[:, 0])], pts[np.argmax(pts[:, 0])]

    def _chord_range(self, y0=None):
        if self.dim == 2:
            x = self.ffd.baseline_wall_coords[:, 0]
            return x.min(), x.max()
        le, te = self._section_leading_trailing_edges(y0)
        return le[0], te[0]

    # ---- thickness at stations --------------------------------------------
    def add_thickness_stations(self, chord_fractions, span_stations=None):
        """Embed upper/lower wall points at chordwise stations; returns a handle.

        2D: one station per chord fraction. 3D: the tensor product of chord
        fractions (of the local section chord) and span stations (y values).
        """
        fractions = np.atleast_1d(np.asarray(chord_fractions, dtype=np.float64))
        upper, lower = [], []
        spans = [None] if self.dim == 2 else np.atleast_1d(span_stations)
        for y0 in spans:
            x_le, x_te = self._chord_range(y0)
            for f in fractions:
                x0 = x_le + f * (x_te - x_le)
                where = x0 if self.dim == 2 else np.array([x0, y0])
                hits = self._vertical_line_hits(where)
                if hits.size < 2:
                    raise ValueError("thickness station at {} does not cross the wall "
                                     "twice".format(where))
                base = [x0] if self.dim == 2 else [x0, y0]
                upper.append(base + [hits.max()])
                lower.append(base + [hits.min()])
        upper, lower = np.array(upper), np.array(lower)
        handle = {'upper': self.ffd.add_probe_points(upper),
                  'lower': self.ffd.add_probe_points(lower),
                  'baseline': np.linalg.norm(upper - lower, axis=1)}
        return handle

    def thickness(self, coefficients, handle):
        """Deformed thickness at the stations of `handle`."""
        up = self.ffd.probe_positions(coefficients, handle['upper'])
        lo = self.ffd.probe_positions(coefficients, handle['lower'])
        return csdl.norm(up - lo, axes=(1,))

    def thickness_ratio(self, coefficients, handle):
        """Deformed over baseline thickness at the stations of `handle`."""
        return self.thickness(coefficients, handle) / handle['baseline']

    # ---- 3D planform ------------------------------------------------------
    def add_planform_stations(self, span_stations):
        """Embed the LE and TE points of the wall sections at the given span
        stations (increasing y, root first); returns a handle.

        ``handle['baseline']`` holds the undeformed chord, span and area as
        numbers, for constraint bounds (the csdl values are not available while
        the graph is recorded).
        """
        if self.dim != 3:
            raise ValueError("planform quantities are for 3D walls")
        span_stations = np.asarray(span_stations, dtype=np.float64)
        if span_stations.size < 2 or np.any(np.diff(span_stations) <= 0):
            raise ValueError("need at least two increasing span stations")
        edges = [self._section_leading_trailing_edges(y0) for y0 in span_stations]
        le = np.array([e[0] for e in edges])
        te = np.array([e[1] for e in edges])
        chord = np.linalg.norm((te - le)[:, :2], axis=1)
        y = le[:, 1]
        baseline = {'chord': chord, 'span': float(y[-1] - y[0]),
                    'area': float(np.sum(0.5 * (chord[1:] + chord[:-1]) * np.diff(y)))}
        return {'le': self.ffd.add_probe_points(le), 'te': self.ffd.add_probe_points(te),
                'n': span_stations.size, 'baseline': baseline}

    def planform(self, coefficients, handle):
        """Deformed section chords, span and trapezoidal planform area.

        Chords are LE-TE distances projected onto the xy plane; span and area
        use the y coordinates of the leading edges.
        """
        le = self.ffd.probe_positions(coefficients, handle['le'])
        te = self.ffd.probe_positions(coefficients, handle['te'])
        chord = csdl.norm((te - le)[:, :2], axes=(1,))
        y = le[:, 1]
        n = handle['n']
        dy = y[1:] - y[:n - 1]
        area = csdl.sum(0.5 * (chord[1:] + chord[:n - 1]) * dy)
        span = y[n - 1] - y[0]
        return {'chord': chord, 'span': span, 'area': area}
