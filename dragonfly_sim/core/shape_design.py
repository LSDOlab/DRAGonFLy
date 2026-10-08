"""Shape design variables as modular layers on an lsdo_geo FFD block.

``DG_windtunnel_model`` only sees an ``FFDShapeParameterization``: it hands it
the wall nodes in ``set_up_sim`` (``setup``) and takes the resulting wall
displacement into the mesh warper (``deform_mesh``). Which design variables
exist, their bounds, and how they move the control points is decided entirely
by the layers given to the parameterization:

* ``ControlPointMotions`` -- raw control-point motions (``cp_motions``).
* ``SectionalVariables`` -- the ``SectionalShape`` presets (2D camber and
  thickness per chordwise station; 3D twist, chord, ... per spanwise section).
* ``WingShape`` -- a trapezoidal wing: quarter-chord (or any chord-line) sweep,
  aspect ratio, root chord, span and taper ratio, plus chordwise thickness and
  camber modes per spanwise station.

A driver then reads::

    shape = FFDShapeParameterization(ffd_shape, degree, corner_list,
                                     layers=[WingShape(planform={...}, sections={...})])
    model = DG_windtunnel_model(mesh, boundary_dict, shape, ...)
    model.set_up_sim()
    shape_parameters = shape.declare_design_variables()
    ...
    u_vec = model.evaluate(model.deform_mesh(), shape_parameters, alpha)

and constrains geometry through ``shape.outputs()`` (layer quantities such as
the wing's planform area) and ``shape.geometry`` (``WallGeometry``: enclosed
measure, thickness at stations, planform stations).

The coefficient pipeline is ``C0 -> one lsdo_geo SectionalParameterization ->
additive layers``. lsdo_geo freezes axes and stretch origins from the points'
values when the graph is built, so every layer's sectional operations are
collected into a single evaluation on the constant baseline (and must share
one principal direction); additive motions (cp motions, sectional modes) are
linear and go on top.

A new layer subclasses ``ShapeLayer`` and overrides what it needs.
"""
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import scipy.sparse as sps
from scipy.optimize import brentq

from dragonfly_sim.core.shape_parameterization import (WallFFD, SectionalShape, WallGeometry,
                                                       _inline_recording)
from dragonfly_sim.utils.ffd_dv_utils import (ShapeDVSet, cp_dv_directions, build_cp_motion_dv,
                                              build_sectional_dv, build_planform_dv)
from dragonfly_sim.utils.meshwarping_utils import inner_bdry_function_from_corner_list


class FFDShapeParameterization:
    """An FFD block around the wall plus the design-variable layers that move it.

    Parameters
    ----------
    shape : sequence of int
        Control points per parametric direction.
    degree : int or sequence of int
    block_corner_list : optional
        Two-surface block definition (``shape_parameterization.
        ffd_corners_from_corner_list``); None wraps the wall's bounding box.
        Also locates the wall when the model gets no explicit
        ``mesh_inner_bdry_function`` (``inner_boundary_function``).
    layers : sequence of ShapeLayer
        Applied in order; their design variables are declared in this order.
    margin, projection_newton_tol : see ``WallFFD``.
    inner_bdry_rel_tol, inner_bdry_overset_margin : containment test of
        ``inner_boundary_function``.
    """

    def __init__(self, shape, degree=2, block_corner_list=None, layers=(), margin=1e-4,
                 projection_newton_tol=1e-12, inner_bdry_rel_tol=1e-6,
                 inner_bdry_overset_margin=1e-4):
        self.shape = tuple(int(n) for n in shape)
        self.degree = degree
        self.block_corner_list = block_corner_list
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("FFDShapeParameterization needs at least one layer")
        names = [layer.name for layer in self.layers]
        if len(set(names)) != len(names):
            raise ValueError("layer names must be unique, got {}".format(names))
        self.margin = margin
        self.projection_newton_tol = projection_newton_tol
        self.inner_bdry_rel_tol = inner_bdry_rel_tol
        self.inner_bdry_overset_margin = inner_bdry_overset_margin

        self.ffd = None
        self._wall_facets = None
        self._geometry = None
        self.dvs = None
        self._coefficients = None
        self._wall_displacement = None

    # ---- set-up (called by the model) -------------------------------------
    def inner_boundary_function(self):
        """Wall facet marker from the block corners, or None without them."""
        if self.block_corner_list is None:
            return None
        return inner_bdry_function_from_corner_list(
            self.block_corner_list, overset_margin=self.inner_bdry_overset_margin,
            rel_tol=self.inner_bdry_rel_tol)

    def setup(self, comm, wall_coords, wall_rows, wall_facets=None):
        """Build the FFD block around the wall and set up the layers. Collective.

        wall_coords : (M_global, dim) replicated wall node coordinates, in the
            rank-block order of the mesh's wall node set; wall_rows : the rows
            this rank owns; wall_facets : replicated wall facets (rows of wall
            node ids), needed for ``geometry`` and by layers that measure the
            baseline wall.
        """
        if self.ffd is not None:
            raise RuntimeError("FFDShapeParameterization.setup called twice")
        self.ffd = WallFFD(comm, wall_coords, wall_rows, self.shape, degree=self.degree,
                           ffd_block_corner_list=self.block_corner_list, margin=self.margin,
                           projection_newton_tol=self.projection_newton_tol)
        self._wall_facets = wall_facets
        for layer in self.layers:
            layer.setup(self)

    @property
    def dim(self):
        return self._require_setup().dim

    @property
    def geometry(self):
        """``WallGeometry`` of the baseline wall, built on first use."""
        self._require_setup()
        if self._geometry is None:
            if self._wall_facets is None:
                raise RuntimeError("no wall facets were given to setup()")
            self._geometry = WallGeometry(self.ffd, self._wall_facets)
        return self._geometry

    def _require_setup(self):
        if self.ffd is None:
            raise RuntimeError("the shape parameterization is not set up yet "
                               "(DG_windtunnel_model.set_up_sim calls setup)")
        return self.ffd

    # ---- design variables ---------------------------------------------------
    def declare_design_variables(self):
        """Create every layer's csdl design variables (with bounds), in layer
        order. Returns ``parameter_vector()``. Call inside the recorder, after
        ``set_up_sim``."""
        self._require_setup()
        if self.dvs is not None:
            raise RuntimeError("design variables already declared")
        self.dvs = ShapeDVSet()
        for layer in self.layers:
            layer.declare(self.dvs)
        if not self.dvs.names:
            raise ValueError("the layers declared no design variables")
        return self.parameter_vector()

    def parameter_vector(self):
        """All shape design variables as one flat vector (declaration order)."""
        if self.dvs is None:
            raise RuntimeError("call declare_design_variables() first")
        return self.dvs.flat_variable()

    def __getitem__(self, name):
        """A declared design variable by name."""
        return self.dvs[name]

    def print_design_variables(self, printer=print):
        printer("Shape design variables ({} entries):".format(self.dvs.size))
        for name, dv in zip(self.dvs.names, self.dvs.dvs):
            line = "  {}: shape {}, lower [{:.4g}, {:.4g}], upper [{:.4g}, {:.4g}]".format(
                name, dv.shape, dv.lower.min(), dv.lower.max(), dv.upper.min(), dv.upper.max())
            if dv.value.size == 1:
                line += ", initial {:.6g}".format(float(dv.value.ravel()[0]))
            printer(line)
        for layer in self.layers:
            for line in layer.describe():
                printer(line)

    # ---- the coefficient pipeline -------------------------------------------
    def coefficients(self):
        """Deformed FFD coefficients (built once, then cached)."""
        if self._coefficients is not None:
            return self._coefficients
        if self.dvs is None:
            raise RuntimeError("call declare_design_variables() first")
        coefficients = self.ffd.baseline_variable()

        sectional_layers = [layer for layer in self.layers if layer.principal_dim is not None]
        principal_dims = {layer.principal_dim for layer in sectional_layers}
        if len(principal_dims) > 1:
            raise ValueError("layers with sectional operations must share one principal "
                             "direction (they form a single lsdo_geo evaluation on the "
                             "baseline), got {}".format(sorted(principal_dims)))
        if sectional_layers:
            sectional = SectionalShape(self.ffd, principal_dim=principal_dims.pop())
            # The layers may build value-dependent expressions of their design
            # variables for lsdo_geo; record them inline with it.
            with _inline_recording():
                for layer in sectional_layers:
                    layer.sectional(sectional)
                coefficients = sectional.apply(coefficients)

        for layer in self.layers:
            coefficients = layer.additive(coefficients)
        self._coefficients = coefficients
        return coefficients

    def wall_displacement(self):
        """Replicated ``(M_global, dim)`` wall node displacement (cached)."""
        if self._wall_displacement is None:
            self._wall_displacement = self.ffd.wall_displacement(self.coefficients())
        return self._wall_displacement

    def outputs(self):
        """Constraint quantities of the layers: ``{layer.name: {...}}`` for the
        layers that have any (e.g. ``outputs()['wing']['area']``)."""
        self.coefficients()
        return {layer.name: layer.outputs() for layer in self.layers if layer.outputs()}


class ShapeLayer:
    """Base of the design-variable layers of an FFDShapeParameterization.

    A layer with sectional (lsdo_geo) operations sets ``principal_dim`` and
    registers them in ``sectional``; that call happens inside an inline
    recording, so expressions of the design variables built there have values,
    which lsdo_geo needs. ``additive`` adds motions that are linear in the
    variables on top of the sectional result.
    """
    name = 'layer'
    principal_dim = None

    def setup(self, parameterization):
        """Numpy precomputation once the FFD block exists (``parameterization.ffd``,
        ``parameterization.geometry``)."""
        self.parameterization = parameterization

    def declare(self, dvs):
        """Create this layer's design variables in ``dvs`` (a ShapeDVSet)."""

    def sectional(self, sectional):
        """Register sectional operations on ``sectional`` (a SectionalShape)."""

    def additive(self, coefficients):
        return coefficients

    def outputs(self):
        return {}

    def describe(self):
        """Lines for ``FFDShapeParameterization.print_design_variables``."""
        return []


class ControlPointMotions(ShapeLayer):
    """Raw control-point motions: one design variable ``cp_motions`` of shape
    ``ffd_shape + (n_opt,)``, with bounds from ``dv_spec`` (see ffd_dv_utils)."""
    name = 'cp_motions'

    def __init__(self, dv_spec, scaler_mode='bounds', name='cp_motions'):
        self.dv_spec = dv_spec
        self.coord_idxs = cp_dv_directions(dv_spec)
        self.scaler_mode = scaler_mode
        self.name = name

    def declare(self, dvs):
        ffd = self.parameterization.ffd
        self.dv = build_cp_motion_dv(ffd.baseline_coefficients, self.dv_spec,
                                     scaler_mode=self.scaler_mode)
        self.variable = dvs.add(self.name, self.dv)

    def additive(self, coefficients):
        return self.parameterization.ffd.apply_cp_motions(coefficients, self.variable,
                                                          self.coord_idxs)

    def describe(self):
        axis_names = ('x', 'y', 'z')
        return ["  {}: direction {} ({})".format(self.name, int(d), axis_names[int(d)])
                for d in self.coord_idxs]


class SectionalVariables(ShapeLayer):
    """``SectionalShape`` presets as design variables.

    variables : {kind: spec}, kind one of ``SectionalShape.add``'s presets;
        spec is the bounds (anything ``build_sectional_dv`` takes) or a dict
        with keys ``bounds`` and optionally ``num_values`` (fewer than the
        number of sections: a B-spline profile), ``pivot`` and
        ``profile_degree``. Each variable is a delta from the baseline.
    """
    name = 'sectional'

    def __init__(self, principal_dim, variables, scaler_mode='bounds', name='sectional'):
        self.principal_dim = int(principal_dim)
        if not variables:
            raise ValueError("SectionalVariables needs at least one variable")
        self.specs = {}
        for kind, spec in variables.items():
            if not isinstance(spec, dict):
                spec = {'bounds': spec}
            unknown = set(spec) - {'bounds', 'num_values', 'pivot', 'profile_degree'}
            if unknown or 'bounds' not in spec:
                raise ValueError("sectional variable {!r}: spec needs 'bounds' and may have "
                                 "num_values/pivot/profile_degree, got {}".format(kind, sorted(spec)))
            self.specs[kind] = spec
        self.scaler_mode = scaler_mode
        self.name = name

    def declare(self, dvs):
        num_sections = self.parameterization.ffd.shape[self.principal_dim]
        self.variables = {}
        for kind, spec in self.specs.items():
            n = spec.get('num_values', num_sections)
            self.variables[kind] = dvs.add(kind, build_sectional_dv(n, spec['bounds'],
                                                                    scaler_mode=self.scaler_mode))

    def sectional(self, sectional):
        for kind, spec in self.specs.items():
            sectional.add(kind, self.variables[kind], pivot=spec.get('pivot'),
                          profile_degree=spec.get('profile_degree', 2))


def _bspline_basis(degree, n_coefficients, u):
    space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=min(degree, n_coefficients - 1),
                             coefficients_shape=(n_coefficients,))
    basis = space.compute_basis_matrix(np.asarray(u, dtype=np.float64).reshape(-1, 1))
    return basis.toarray() if sps.issparse(basis) else np.asarray(basis)


def _invert_monotone_bspline(degree, values, target):
    """Parametric coordinate u in [0, 1] where the 1D B-spline with control
    values ``values`` equals ``target``."""
    def f(u):
        return float((_bspline_basis(degree, len(values), [u]) @ values)[0]) - target
    lo, hi = f(0.), f(1.)
    if lo * hi > 0.:
        raise ValueError("target {} lies outside the block's range [{}, {}]"
                         .format(target, lo + target, hi + target))
    return brentq(f, 0., 1., xtol=1e-15, rtol=4 * np.finfo(float).eps)


def _bernstein(degree, xi):
    """Bernstein polynomials of `degree` at xi: shape (len(xi), degree + 1)."""
    from scipy.special import comb
    xi = np.asarray(xi, dtype=np.float64)[:, None]
    m = np.arange(degree + 1)[None, :]
    return comb(degree, m) * xi ** m * (1. - xi) ** (degree - m)


def _section_mode_profiles(kind, num_modes, xi):
    """Chordwise profiles of the WingShape section modes at chord fractions xi:
    shape (len(xi), num_modes). Thickness: the Bernstein polynomials of degree
    num_modes - 1 (they sum to one: equal values scale the section uniformly).
    Camber: 4 xi (1 - xi) times those (zero at the leading and trailing edge;
    equal values give the parabolic camber line)."""
    profiles = _bernstein(num_modes - 1, xi)
    if kind == 'camber':
        profiles = profiles * (4. * xi * (1. - xi))[:, None]
    return profiles


class WingShape(ShapeLayer):
    """Planform and sectional design variables of a trapezoidal half wing.

    The wall must be a straight-tapered half wing with its root on the y = 0
    symmetry plane, x chordwise and z up, inside a lofted FFD block whose
    parametric axes are (chord, span, vertical) -- the corner-list blocks of
    the wing drivers.

    planform : {name: (lower, upper)} for any of
        ``sweep``         sweep angle [rad] of the chord line at
                          ``sweep_chord_fraction`` (0.25: quarter chord)
        ``aspect_ratio``  full-wing AR = b^2 / S
        ``root_chord``
        ``span``          full-wing span b (twice the half wing's semi-span)
        ``taper_ratio``   tip chord / root chord
        Absolute values: each variable starts at the baseline wing's value and
        its bounds must bracket it. A name left out stays at its baseline,
        except the ``derived`` one.
    sections : {name: spec} for ``thickness`` and/or ``camber``. spec is the
        bounds, or a dict {'bounds': ..., 'modes': n} for n chordwise modes per
        FFD spanwise section (default 1). The variable then has n_sections x n
        entries, section-major (root first: all of the root's modes, then the
        next section's); bounds as for ``build_sectional_dv`` -- a scalar, a
        pair, or a pair of arrays of shape (n_sections,) (the same for every
        mode of a section), (n_sections, n) or (n_sections * n,). Pin an entry
        with (0., 0.). Deltas from the baseline, starting at zero, with xi the
        local chord fraction and B_m the Bernstein polynomials of degree n - 1:
        ``thickness``  relative change of the section's thickness (t/c),
                       sum_m tau_m B_m(xi) -- 0.1 in every mode is 10% thicker
        ``camber``     change of camber over local chord, sum_m kappa_m
                       4 xi (1 - xi) B_m(xi), so the leading and trailing edge
                       stay put; kappa in every mode is a parabolic camber line
                       of maximum kappa
        With one mode these are a uniform thickness change and a parabolic
        camber line. The modes are least-squares fits on the block's chordwise
        basis; they are exact when it is a Bezier (n_chord = degree + 1) of
        degree >= n - 1 for thickness and >= n + 1 for camber (``describe``
        reports the fit residual).
    derived : which of ``aspect_ratio``, ``root_chord`` and ``span`` follows from
        the other two and the taper ratio, s = AR c_r (1 + taper) / 4. Default:
        ``span``, unless span is a variable -- then whichever of aspect_ratio
        and root_chord is not (with only span a variable, name it).
    sweep_chord_fraction : chord line the sweep is measured on; sections scale
        about it, and its root point stays fixed.
    semi_span : baseline semi-span; default the wall's largest y (which
        includes any tip cap).
    fit_tol : allowed deviation of the baseline leading/trailing edges from
        straight lines, relative to the root chord.

    The geometry: FFD section j, at baseline span fraction eta_j = y_j / s0,
    moves to y_j' = eta_j s', its chord c_j' = c_r' (1 - (1 - taper') eta_j)
    scales it by r_j = c_j'/c_j0 about the reference chord line in x and about
    the chord plane in z (keeping t/c; the thickness modes act on top), and the
    reference line is placed at x_ref,root + y_j' tan(sweep'). These are
    lsdo_geo sectional stretches and translations; thickness and camber are
    additive sectional modes. With the taper ratio fixed the map is affine, so
    the deformed wall is exactly the trapezoid the variables describe; a taper
    change is exact at the FFD sections and B-spline-blended between them.
    """
    name = 'wing'
    principal_dim = 1
    PLANFORM_KEYS = ('sweep', 'aspect_ratio', 'root_chord', 'span', 'taper_ratio')
    SIZE_KEYS = ('aspect_ratio', 'root_chord', 'span')
    SECTION_KEYS = ('thickness', 'camber')

    def __init__(self, planform=None, sections=None, sweep_chord_fraction=0.25,
                 semi_span=None, derived=None, fit_tol=1e-3, scaler_mode='bounds', name='wing'):
        self.planform_spec = dict(planform or {})
        self.section_spec = {}
        unknown = set(self.planform_spec) - set(self.PLANFORM_KEYS)
        if unknown:
            raise ValueError("unknown planform variable(s) {}; expected any of {}"
                             .format(sorted(unknown), list(self.PLANFORM_KEYS)))
        for key, spec in dict(sections or {}).items():
            if key not in self.SECTION_KEYS:
                raise ValueError("unknown section variable {!r}; expected any of {}"
                                 .format(key, list(self.SECTION_KEYS)))
            if not isinstance(spec, dict):
                spec = {'bounds': spec}
            if 'bounds' not in spec or set(spec) - {'bounds', 'modes'}:
                raise ValueError("section variable {!r}: spec needs 'bounds' and may have "
                                 "'modes', got {}".format(key, sorted(spec)))
            modes = int(spec.get('modes', 1))
            if modes < 1:
                raise ValueError("section variable {!r} needs at least one mode".format(key))
            self.section_spec[key] = {'bounds': spec['bounds'], 'modes': modes}
        if not self.planform_spec and not self.section_spec:
            raise ValueError("WingShape needs at least one planform or section variable")
        self.derived = self._derived_size(derived)
        self.f = float(sweep_chord_fraction)
        self.semi_span_arg = semi_span
        self.fit_tol = fit_tol
        self.scaler_mode = scaler_mode
        self.name = name

    def _derived_size(self, derived):
        variables = [k for k in self.SIZE_KEYS if k in self.planform_spec]
        if len(variables) == 3:
            raise ValueError("aspect_ratio, root_chord and span are related through the taper "
                             "ratio; at most two of them can be design variables")
        if derived is None:
            if 'span' not in variables:
                return 'span'
            free = [k for k in ('aspect_ratio', 'root_chord') if k not in variables]
            if len(free) == 2:
                raise ValueError("with span a design variable, say which of aspect_ratio and "
                                 "root_chord follows from it (derived=...); the other stays "
                                 "at its baseline")
            return free[0]
        if derived not in self.SIZE_KEYS:
            raise ValueError("derived must be one of {}, got {!r}".format(list(self.SIZE_KEYS),
                                                                          derived))
        if derived in variables:
            raise ValueError("{!r} cannot be both a design variable and derived".format(derived))
        return derived

    # ---- baseline (numpy) -----------------------------------------------
    def setup(self, parameterization):
        super().setup(parameterization)
        ffd = parameterization.ffd
        if ffd.dim != 3:
            raise ValueError("WingShape needs a 3D wall")
        self.baseline = self._fit_baseline(parameterization.geometry)
        self._setup_sections(ffd)

    def _fit_baseline(self, geometry):
        X = geometry.ffd.baseline_wall_coords
        y_root, y_max = X[:, 1].min(), X[:, 1].max()
        s0 = float(self.semi_span_arg) if self.semi_span_arg is not None else float(y_max)
        if abs(y_root) > 1e-6 * s0:
            raise ValueError("WingShape needs the wing root on the y = 0 symmetry plane; "
                             "the wall starts at y = {}".format(y_root))
        stations = np.linspace(0.05, 0.9, 12) * s0
        edges = [geometry.section_edges(y) for y in stations]
        le = np.array([e[0] for e in edges])
        te = np.array([e[1] for e in edges])
        le_fit = np.polyfit(stations, le[:, 0], 1)
        te_fit = np.polyfit(stations, te[:, 0], 1)
        z_fit = np.polyfit(stations, 0.5 * (le[:, 2] + te[:, 2]), 1)
        c_root = te_fit[1] - le_fit[1]
        residual = max(np.abs(np.polyval(le_fit, stations) - le[:, 0]).max(),
                       np.abs(np.polyval(te_fit, stations) - te[:, 0]).max())
        if residual > self.fit_tol * c_root:
            raise ValueError("the wing's leading/trailing edges deviate {:.3g} from straight "
                             "lines (fit_tol {} x root chord {:.4g}); WingShape needs a "
                             "straight-tapered planform".format(residual, self.fit_tol, c_root))
        chord_slope = te_fit[0] - le_fit[0]
        c_tip = c_root + chord_slope * s0
        x_ref_root = le_fit[1] + self.f * c_root
        ref_slope = le_fit[0] + self.f * chord_slope
        return {'sweep': float(np.arctan(ref_slope)),
                'aspect_ratio': float(4. * s0 / (c_root + c_tip)),
                'root_chord': float(c_root),
                'span': 2. * s0,
                'taper_ratio': float(c_tip / c_root),
                'semi_span': s0,
                'area': float((c_root + c_tip) * s0),
                'x_ref_root': float(x_ref_root),
                'ref_slope': float(ref_slope),
                'x_le': le_fit, 'z_chord_plane': z_fit,
                'fit_residual': float(residual)}

    def _setup_sections(self, ffd):
        C = ffd.baseline_coefficients
        n0, n1, n2 = ffd.shape
        deg0 = ffd.degree[0]
        b = self.baseline
        y = C[0, :, 0, 1]
        tol = 1e-9 * np.abs(C).max()
        if (np.abs(C[..., 1] - y[None, :, None]).max() > tol
                or np.abs(C[..., 0] - C[:, :, :1, 0]).max() > tol
                or np.abs(C[..., 2] - C[:1, :, :, 2]).max() > tol):
            raise ValueError("WingShape needs a lofted block with parametric axes (chord, span, "
                             "vertical): spanwise sections in planes y = const, x varying "
                             "along axis 0 only and z along axis 2 only")

        s0, cr0, lam0 = b['semi_span'], b['root_chord'], b['taper_ratio']
        self.y = y
        self.eta = y / s0
        self.c0 = cr0 * (1. - (1. - lam0) * self.eta)
        self.x_ref0 = b['x_ref_root'] + b['ref_slope'] * y
        x_le0 = np.polyval(b['x_le'], y)
        z_ref0 = np.polyval(b['z_chord_plane'], y)
        # lsdo_geo normalizes a stretch by the section's control-point extent
        self.extent_x = C[:, :, 0, 0].max(axis=0) - C[:, :, 0, 0].min(axis=0)
        self.extent_z = C[0, :, :, 2].max(axis=1) - C[0, :, :, 2].min(axis=1)

        # Pivots as parametric coordinates within each section: (axis 0, axis 2).
        # lsdo_geo evaluates them on a degree-1 B-spline through the section's
        # control points (SectionalParameterization.helpful_section_b_spline_space),
        # not on the FFD's own basis, so invert that one.
        self.pivots = np.zeros((n1, 2))
        for j in range(n1):
            self.pivots[j, 0] = _invert_monotone_bspline(1, C[:, j, 0, 0], self.x_ref0[j])
            self.pivots[j, 1] = _invert_monotone_bspline(1, C[0, j, :, 2], z_ref0[j])

        # Section modes: chordwise control values whose B-spline is the mode's
        # profile in the wing chord fraction xi (least squares; exact when the
        # chordwise basis contains it). Thickness weights also carry the control
        # points' height above the chord plane, the distance they scale.
        u = np.linspace(0., 1., 8 * n0 + 1)
        basis0 = _bspline_basis(deg0, n0, u)
        self.mode_weights = {}
        self.mode_fit_residual = 0.
        for key, spec in self.section_spec.items():
            weights = np.zeros((spec['modes'],) + ffd.shape)
            for j in range(n1):
                xi = (basis0 @ C[:, j, 0, 0] - x_le0[j]) / self.c0[j]
                profiles = _section_mode_profiles(key, spec['modes'], xi)
                fit = np.linalg.lstsq(basis0, profiles, rcond=None)[0]       # (n0, modes)
                self.mode_fit_residual = max(self.mode_fit_residual,
                                             float(np.abs(basis0 @ fit - profiles).max()))
                if key == 'thickness':
                    weights[:, :, j, :] = fit.T[:, :, None] * (C[0, j, :, 2] - z_ref0[j])[None, None, :]
                else:
                    weights[:, :, j, :] = fit.T[:, :, None]
            self.mode_weights[key] = weights

    # ---- design variables ---------------------------------------------------
    def _section_bounds(self, bounds, n1, modes):
        """Broadcast per-section (n1,) or (n1, modes) bound arrays to the
        variable's flat (n1 * modes,) layout; scalars pass through."""
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            return bounds
        resolved = []
        for bound in bounds:
            bound = np.asarray(bound, dtype=np.float64)
            if bound.ndim == 0:
                resolved.append(float(bound))
            elif bound.shape == (n1,):
                resolved.append(np.repeat(bound, modes))
            elif bound.size == n1 * modes:
                resolved.append(bound.reshape(-1))
            else:
                raise ValueError("section bounds must be scalars or arrays of shape ({0},), "
                                 "({0}, {1}) or ({2},), got {3}".format(n1, modes, n1 * modes,
                                                                       bound.shape))
        return tuple(resolved)

    def declare(self, dvs):
        self.variables = {}
        for key in self.PLANFORM_KEYS:
            if key in self.planform_spec:
                dv = build_planform_dv(self.baseline[key], self.planform_spec[key],
                                       scaler_mode=self.scaler_mode, name=key)
                self.variables[key] = dvs.add(key, dv)
        n1 = self.parameterization.ffd.shape[1]
        for key in self.SECTION_KEYS:
            if key in self.section_spec:
                modes = self.section_spec[key]['modes']
                bounds = self._section_bounds(self.section_spec[key]['bounds'], n1, modes)
                dv = build_sectional_dv(n1 * modes, bounds, scaler_mode=self.scaler_mode)
                self.variables[key] = dvs.add(key, dv)

    def _planform_variable(self, key):
        if key in self.variables:
            return self.variables[key]
        return csdl.Variable(value=np.array([self.baseline[key]]))

    def sectional(self, sectional):
        sweep = self._planform_variable('sweep')
        taper_ratio = self._planform_variable('taper_ratio')
        if self.derived == 'span':
            aspect_ratio = self._planform_variable('aspect_ratio')
            root_chord = self._planform_variable('root_chord')
            semi_span = aspect_ratio * root_chord * (1. + taper_ratio) / 4.
        else:
            semi_span = self._planform_variable('span') / 2.
            if self.derived == 'root_chord':
                aspect_ratio = self._planform_variable('aspect_ratio')
                root_chord = 4. * semi_span / (aspect_ratio * (1. + taper_ratio))
            else:
                root_chord = self._planform_variable('root_chord')
                aspect_ratio = 4. * semi_span / (root_chord * (1. + taper_ratio))
        chord = root_chord * (1. - (1. - taper_ratio) * self.eta)        # c_j'
        ratio = chord / self.c0                                          # r_j
        y_new = semi_span * self.eta
        x_ref = self.baseline['x_ref_root'] + y_new * csdl.tan(sweep)

        # chord: scale about the reference chord line, and about the chord
        # plane (t/c kept)
        sectional.add_stretch((ratio - 1.) * self.extent_x, 0, pivot=self.pivots)
        sectional.add_stretch((ratio - 1.) * self.extent_z, 2, pivot=self.pivots)
        sectional.add_translation(x_ref - self.x_ref0, [1., 0., 0.])
        sectional.add_translation(y_new - self.y, [0., 1., 0.])
        # thickness modes scale the (chord-scaled) height above the chord
        # plane; camber modes move by a fraction of the new chord
        n1 = self.parameterization.ffd.shape[1]
        for key, scale in (('thickness', ratio), ('camber', chord)):
            if key not in self.variables:
                continue
            modes = self.section_spec[key]['modes']
            values = self.variables[key].reshape((n1, modes))
            for m in range(modes):
                sectional.add_mode(values[:, m] * scale, [0., 0., 1.],
                                   self.mode_weights[key][m])

        tip_chord = root_chord * taper_ratio
        self._outputs = {'sweep': sweep, 'aspect_ratio': aspect_ratio,
                         'root_chord': root_chord, 'taper_ratio': taper_ratio,
                         'tip_chord': tip_chord, 'semi_span': semi_span,
                         'span': 2. * semi_span,
                         'area': (root_chord + tip_chord) * semi_span,
                         'section_chords': chord}

    def outputs(self):
        """Planform quantities as csdl variables: sweep, aspect_ratio,
        root_chord, taper_ratio, tip_chord, semi_span, span, area (full wing)
        and section_chords (at the FFD sections)."""
        return getattr(self, '_outputs', {})

    def describe(self):
        b = self.baseline
        lines = ["  wing baseline: sweep({:.2f} c) {:.4f} deg, AR {:.5g}, root chord {:.5g}, "
                 "taper {:.5g}, span {:.5g}, area {:.5g} (edge fit residual {:.2e}); "
                 "{} derived".format(self.f, np.degrees(b['sweep']), b['aspect_ratio'],
                                     b['root_chord'], b['taper_ratio'], b['span'], b['area'],
                                     b['fit_residual'], self.derived)]
        if self.section_spec:
            n1 = self.parameterization.ffd.shape[1]
            lines.append("  wing sections: {} spanwise stations at y = {}; {} (max chordwise "
                         "mode fit residual {:.2e})".format(
                             n1, np.array2string(self.y, precision=3),
                             ", ".join("{} {} modes".format(key, spec['modes'])
                                       for key, spec in self.section_spec.items()),
                             self.mode_fit_residual))
        return lines
