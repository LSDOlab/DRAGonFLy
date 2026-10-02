"""Construction of the FFD control-point-motion design variable and its bounds.

The design variable ``cp_motions`` has shape ``ffd_shape + (n_opt,)``, where the
trailing axis is a *compressed* index over the optimized spatial directions --
``cp_motions[..., m]`` moves spatial component ``coord_idxs[m]`` of every control
point, and directions absent from ``coord_idxs`` stay at their baseline value.
See shape_parameterization.WallFFD.apply_cp_motions for the application of
that mapping.

Everything here is expressed through a single ``dv_spec`` dict, keyed by spatial
direction index, so the choice of optimized directions and the bounds on each of
them cannot drift apart:

    dv_spec = {
        0: 0.02,                    # x: symmetric +/- 0.02 everywhere
        2: (-0.05, 0.10),           # z: asymmetric, uniform over control points
    }

A value may also be a callable ``f(coords) -> (lower, upper)`` for bounds that
vary over the block; see spanwise_linear_bounds.

Sectional shape variables (shape_parameterization.SectionalShape) get their
arrays from build_sectional_dv, and ShapeDVSet collects any mix of these into
the ordered set of shape design variables a driver declares.

Numpy-only apart from ShapeDVSet's variable creation, which imports csdl
lazily -- so the bounds logic can be exercised without an MPI environment.
Callers pass their own ``printer`` (typically ``PETSc.Sys.Print``) to
print_cp_motion_bounds.
"""

import numpy as np
from typing import NamedTuple, Optional


class CPMotionDV(NamedTuple):
    """Everything needed to declare ``cp_motions`` as a design variable.

    coord_idxs : (n_opt,) int array of optimized spatial components, ascending
    shape      : ffd_shape + (n_opt,), the shape of the design variable
    value      : initial value (zeros -- the motions are deltas from baseline)
    lower/upper: bounds, in physical units, same shape as value
    scaler     : per-entry scaler for set_as_design_variable, or None
    """
    coord_idxs: np.ndarray
    shape: tuple
    value: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scaler: Optional[np.ndarray]


def cp_dv_directions(dv_spec):
    """The optimized spatial components named by dv_spec, sorted ascending.

    Call this before constructing DG_windtunnel_model, which needs
    cp_coord_opt_idxs up front; build_cp_motion_dv (which needs the baseline
    control points, and so can only run after set_up_sim) derives the
    same array again and returns it for cross-checking.

    Sorting is load-bearing: it makes the design variable's column order
    independent of the order the dict literal happens to be written in, and
    shape_parameterization.validate_cp_coord_opt_idxs enforces the same
    strictly-increasing invariant.
    """
    if not isinstance(dv_spec, dict):
        raise TypeError("dv_spec must be a dict keyed by spatial direction index, "
                        "got {}".format(type(dv_spec).__name__))
    if len(dv_spec) == 0:
        raise ValueError("dv_spec is empty: at least one coordinate direction must "
                         "be a design variable")

    for direction in dv_spec:
        if not isinstance(direction, (int, np.integer)) or isinstance(direction, bool):
            raise TypeError("dv_spec keys must be integer spatial direction indices, "
                            "got key {!r} of type {}".format(direction, type(direction).__name__))
        if direction < 0:
            raise ValueError("dv_spec keys must be non-negative, got {}".format(direction))

    return np.array(sorted(int(d) for d in dv_spec), dtype=int)


def _resolve_direction_bounds(bound_spec, coords, direction):
    """One direction's bounds as a pair of (num_control_points,) arrays."""
    num_cps = coords.shape[0]

    if callable(bound_spec):
        lower, upper = bound_spec(coords)
    elif isinstance(bound_spec, (tuple, list)):
        if len(bound_spec) != 2:
            raise ValueError("bounds for direction {} must be a (lower, upper) pair, "
                             "got a sequence of length {}".format(direction, len(bound_spec)))
        lower, upper = bound_spec
    elif isinstance(bound_spec, (float, int, np.floating, np.integer)):
        if bound_spec < 0.:
            raise ValueError("a scalar bound is the symmetric half-range and must be "
                             "non-negative; direction {} got {}".format(direction, bound_spec))
        lower, upper = -bound_spec, bound_spec
    else:
        raise TypeError("bounds for direction {} must be a scalar, a (lower, upper) pair, "
                        "or a callable(coords) -> (lower, upper), got {}"
                        .format(direction, type(bound_spec).__name__))

    lower = np.broadcast_to(np.asarray(lower, dtype=np.double), (num_cps,))
    upper = np.broadcast_to(np.asarray(upper, dtype=np.double), (num_cps,))

    # cp_motions starts at zero, so the initial point has to lie inside the box.
    # A pinned direction, (0., 0.), satisfies this.
    if not np.all(lower <= 0.):
        raise ValueError("lower bounds for direction {} must be non-positive "
                         "(cp_motions starts at zero), got max {}".format(direction, lower.max()))
    if not np.all(upper >= 0.):
        raise ValueError("upper bounds for direction {} must be non-negative "
                         "(cp_motions starts at zero), got min {}".format(direction, upper.min()))

    return lower, upper


def build_cp_motion_dv(baseline_controlpoints, dv_spec, scaler_mode='bounds'):
    """Build the cp_motions design variable arrays from a dv_spec.

    baseline_controlpoints : (*ffd_shape, ndim) control point coordinates, as
        held by shape_parameterization.WallFFD.baseline_coefficients
    dv_spec : {direction: bound_spec}, see the module docstring
    scaler_mode : 'bounds' derives a scaler of 1/half-range so every direction's
        box becomes +/-1 in the optimizer's space; None emits no scaler.

    Returns a CPMotionDV. CSDL flattens design variables and their bounds in C
    order (csdl_alpha/backends/simulator.py, build_opt_metadata), and this
    function reshapes in C order too, so no manual index bookkeeping is needed:
    entry [i, j, k, m] bounds the motion of
    baseline_controlpoints[i, j, k, coord_idxs[m]].
    """
    baseline_controlpoints = np.asarray(baseline_controlpoints, dtype=np.double)
    if baseline_controlpoints.ndim < 2:
        raise ValueError("baseline_controlpoints must be (*ffd_shape, ndim), got shape {}"
                         .format(baseline_controlpoints.shape))

    ffd_shape = baseline_controlpoints.shape[:-1]
    ndim = baseline_controlpoints.shape[-1]

    coord_idxs = cp_dv_directions(dv_spec)
    if np.any(coord_idxs >= ndim):
        raise ValueError("dv_spec names direction(s) {} but the control points only have "
                         "{} spatial components".format(coord_idxs[coord_idxs >= ndim].tolist(), ndim))

    coords = baseline_controlpoints.reshape(-1, ndim)
    num_cps = coords.shape[0]
    n_opt = coord_idxs.shape[0]

    lower = np.zeros((num_cps, n_opt), dtype=np.double)
    upper = np.zeros((num_cps, n_opt), dtype=np.double)
    for m, direction in enumerate(coord_idxs):
        lower[:, m], upper[:, m] = _resolve_direction_bounds(dv_spec[int(direction)],
                                                             coords, int(direction))

    dv_shape = tuple(ffd_shape) + (n_opt,)
    lower = lower.reshape(dv_shape)
    upper = upper.reshape(dv_shape)

    if scaler_mode is None:
        scaler = None
    elif scaler_mode == 'bounds':
        # +/-1 per direction in the optimizer's space. Without this, directions
        # whose bounds differ by orders of magnitude give SLSQP's QP badly scaled
        # columns and the tightly bounded ones barely move -- the same failure the
        # alpha scaler in the drivers exists to prevent.
        half_range = np.maximum(np.abs(lower), np.abs(upper))
        # a pinned direction has zero half-range; leave its scaler at 1
        scaler = np.divide(1.0, half_range, out=np.ones_like(half_range), where=half_range > 0.)
    else:
        raise ValueError("scaler_mode must be 'bounds' or None, got {!r}".format(scaler_mode))

    return CPMotionDV(coord_idxs=coord_idxs,
                      shape=dv_shape,
                      value=np.zeros(dv_shape, dtype=np.double),
                      lower=lower,
                      upper=upper,
                      scaler=scaler)


def spanwise_linear_bounds(base_bound, scale_at_root, scale_at_ref, y_ref, y_axis=1):
    """A bound callable whose magnitude varies linearly with the spanwise coordinate.

    The bound is base_bound*scale_at_root at y = 0 and base_bound*scale_at_ref at
    y = y_ref, linearly interpolated in between and clipped to that range outside
    [0, y_ref] so the magnitude never changes sign. Bounds are symmetric:
    lower = -bound, upper = +bound.

    Motivating case: a wing FFD block that tapers from root chord to tip chord, so
    a single scalar bound is a much larger fraction of the local chord at the tip
    than at the root. Shrinking the bound along the span keeps the allowed
    deformation a roughly constant fraction of the local chord.
    """
    def bound_function(coords):
        t = np.clip(coords[:, y_axis]/y_ref, 0., 1.)
        scale = (1. - t)*scale_at_root + t*scale_at_ref
        bound = base_bound*scale
        return -bound, bound

    return bound_function


def print_cp_motion_bounds(cp_dv, printer=print, axis_names=('x', 'y', 'z')):
    """Summarise a CPMotionDV, one line per optimized direction.

    printer is called once per line; pass PETSc.Sys.Print from an MPI driver to
    keep the output on rank 0.
    """
    printer("cp_motions design variable: shape {}, {} optimized direction(s)"
            .format(cp_dv.shape, cp_dv.coord_idxs.shape[0]))
    for m, direction in enumerate(cp_dv.coord_idxs):
        direction = int(direction)
        name = axis_names[direction] if direction < len(axis_names) else str(direction)
        lower_m, upper_m = cp_dv.lower[..., m], cp_dv.upper[..., m]
        line = ("  column {} -> direction {} ({}): lower [{:.4g}, {:.4g}], upper [{:.4g}, {:.4g}]"
                .format(m, direction, name,
                        lower_m.min(), lower_m.max(), upper_m.min(), upper_m.max()))
        if cp_dv.scaler is not None:
            scaler_m = cp_dv.scaler[..., m]
            line += ", scaler [{:.4g}, {:.4g}]".format(scaler_m.min(), scaler_m.max())
        printer(line)


def build_sectional_dv(num_values, bounds, scaler_mode='bounds'):
    """Arrays for one sectional shape variable (twist, camber, ...).

    num_values : entries of the variable -- one per section, or fewer for a
        B-spline profile over the sections (see SectionalShape)
    bounds : a scalar symmetric half-range, a (lower, upper) pair, or a pair of
        (num_values,) arrays. As for cp_motions the variable starts at zero (a
        delta from the baseline), so zero must lie inside the box; pin an entry
        with (0., 0.).
    scaler_mode : as in build_cp_motion_dv

    Returns a CPMotionDV with coord_idxs=None.
    """
    shape = (int(num_values),)
    coords = np.zeros((shape[0], 1))
    lower, upper = _resolve_direction_bounds(bounds, coords, 'sectional')
    lower, upper = np.array(lower), np.array(upper)

    if scaler_mode is None:
        scaler = None
    elif scaler_mode == 'bounds':
        half_range = np.maximum(np.abs(lower), np.abs(upper))
        scaler = np.divide(1.0, half_range, out=np.ones_like(half_range), where=half_range > 0.)
    else:
        raise ValueError("scaler_mode must be 'bounds' or None, got {!r}".format(scaler_mode))

    return CPMotionDV(coord_idxs=None, shape=shape, value=np.zeros(shape, dtype=np.double),
                      lower=lower, upper=upper, scaler=scaler)


class ShapeDVSet:
    """The ordered set of shape design variables of a driver.

    ``flat_variable`` concatenates them, in declaration order and each variable
    flattened in C order, into the one vector the model takes as its
    ``cp_motion_inputs``.
    """

    def __init__(self):
        self.names = []
        self.dvs = []
        self.variables = []

    def add(self, name, dv):
        """Create the csdl design variable for `dv` (a CPMotionDV) and return it."""
        import csdl_alpha as csdl
        if name in self.names:
            raise ValueError("shape design variable {!r} added twice".format(name))
        var = csdl.Variable(name=name, value=np.array(dv.value, dtype=np.double))
        var.set_as_design_variable(lower=dv.lower, upper=dv.upper, scaler=dv.scaler)
        self.names.append(name)
        self.dvs.append(dv)
        self.variables.append(var)
        return var

    def __getitem__(self, name):
        return self.variables[self.names.index(name)]

    def dv(self, name):
        return self.dvs[self.names.index(name)]

    @property
    def size(self):
        return int(sum(np.prod(dv.shape) for dv in self.dvs))

    @property
    def shape(self):
        return (self.size,)

    def flat_variable(self):
        """All shape variables as one csdl vector, in declaration order.

        This is what the model takes as its ``cp_motion_inputs``. A single
        variable is returned as is.
        """
        import csdl_alpha as csdl
        if not self.variables:
            raise ValueError("no shape design variables added")
        if len(self.variables) == 1:
            return self.variables[0]
        return csdl.concatenate([v.reshape((int(np.prod(v.shape)),)) for v in self.variables],
                                axis=0)
