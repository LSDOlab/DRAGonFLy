# Shape parameterization

The wall shape is controlled by a B-spline FFD block around the wall nodes. Design variables change the block's
control points; the wall moves with them, and the volume mesh follows through IDWarp.

All of this lives in one object, an `FFDShapeParameterization` (`dragonfly_sim.core.shape_design`), passed to
`DG_windtunnel_model`. The model does not know which design variables exist. In `set_up_sim()` it hands the wall
nodes to the parameterization, and `model.deform_mesh()` turns the resulting wall displacement into mesh node
motions:
```python
shape = FFDShapeParameterization(ffd_shape, ffd_degree, block_corner_list, layers=[...])
model = DG_windtunnel_model(mesh, boundary_dict, shape, aero_center, ...)
model.set_up_sim()
shape_parameters = shape.declare_design_variables()   # all shape DVs, with bounds, as one vector
alpha = ...                                           # declare alpha after the shape variables
u_vec = model.evaluate(model.deform_mesh(), shape_parameters, alpha)
```

## FFD block
`ffd_shape` sets the number of control points per parametric direction (2 entries in 2D, 3 in 3D) and
`ffd_degree` the B-spline degree. By default the block is the axis-aligned bounding box of the wall. For a
tapered or swept wing, pass a `block_corner_list`: two rectangular end surfaces (root and tip), each given by
two opposite corners, e.g.
`[[(-0.1, 0., 0.32), (5.1, 0., -0.3)], [(7.4, 14.1, 0.32), (9.1, 14.1, -0.3)]]`.
Without an explicit `mesh_inner_bdry_function`, the model uses the block corners to find the wall facets.

## Layers
The design variables come from the layers, which are applied in order and declared in order:

| Layer | Design variables |
|---|---|
| `ControlPointMotions(dv_spec)` | `cp_motions`: individual control-point motions |
| `SectionalVariables(principal_dim, {kind: bounds})` | sectional presets (camber, thickness, twist, ...) |
| `WingShape(planform={...}, sections={...})` | wing sweep, aspect ratio, root chord, span, taper ratio; chordwise thickness and camber modes per spanwise station |

The coefficients are built as the constant baseline, then one lsdo_geo `SectionalParameterization` holding
every layer's sectional operations, then additive motions (control-point motions, sectional modes) on top.
lsdo_geo freezes axes and stretch origins from the points' values when the graph is built, which is why all
sectional operations act on the constant baseline in one evaluation. Layers with sectional operations must
therefore share one principal direction. A new layer subclasses `ShapeLayer` (`setup`, `declare`,
`sectional`, `additive`, `outputs`).

### Control-point motions
`cp_motions` moves control points individually. Its directions and bounds come from one dictionary keyed by
spatial direction (0 = x, 1 = y, 2 = z):
```python
ControlPointMotions({1: 0.01})                     # 2D: vertical motions within +/-0.01
ControlPointMotions({0: 0.02, 2: (-0.05, 0.10)})   # 3D: x symmetric, z asymmetric
ControlPointMotions({2: spanwise_linear_bounds(base_bound=0.1, scale_at_root=1.0, scale_at_ref=0.5, y_ref=14.1)})
```
A bound is a symmetric half-range, a `(lower, upper)` pair, or a callable of the control-point coordinates such
as `spanwise_linear_bounds`, which varies the bound linearly along the span. Each direction's box is scaled to
$\pm 1$ in the optimizer's space.

### Sectional variables
`SectionalVariables(principal_dim, {kind: bounds})` views the block as a stack of sections: `principal_dim=0` in
2D (chordwise stations), `1` for a wing (spanwise stations). The kinds are:

| Kind | Operation | Dimensions |
|---|---|---|
| `camber` | translation along the vertical axis | 2D |
| `thickness` | stretch along the vertical axis | 2D, 3D |
| `twist` | rotation about the spanwise axis (radians) | 3D |
| `chord` | stretch along the chordwise axis | 3D |
| `sweep` | translation along x | 3D |
| `dihedral` | translation along z | 3D |
| `span` | translation along y | 3D |

A variable is a delta from the baseline and has one value per section. Its bounds are a scalar, a pair, or
per-section arrays; pin an entry with `(0., 0.)`. A dict spec, `{'bounds': ..., 'num_values': n, 'pivot': ...}`,
gives fewer values, which are then the B-spline coefficients of a profile over the sections, and/or a pivot.
```python
SectionalVariables(0, {'camber': 0.01, 'thickness': 0.01})                       # 2D airfoil
SectionalVariables(1, {'twist': {'bounds': np.radians(2.), 'pivot': [0.25, 0.5]}})   # 3D wing
```

### Wing planform and sections
`WingShape` parameterizes a straight-tapered half wing whose root lies on the $y = 0$ symmetry plane (x
chordwise, z up). It needs a lofted block with parametric axes (chord, span, vertical), which is what a
`block_corner_list` produces. The baseline planform is measured from the wall when the model is set up: straight
lines are fitted to the leading and trailing edges, and a wall that is not straight-tapered is rejected.
```python
WingShape(planform={'sweep': np.radians((20., 30.)),   # sweep of the quarter-chord line [rad]
                    'aspect_ratio': (7.5, 10.),        # full wing, b^2 / S
                    'span': (26., 30.),                # full wing; root chord follows
                    'taper_ratio': (0.2, 0.4)},        # tip chord / root chord
          sections={'thickness': {'bounds': 0.1, 'modes': 4},    # relative change of t/c
                    'camber': {'bounds': 0.01, 'modes': 3}},     # change of camber / local chord
          sweep_chord_fraction=0.25)
```
- **Planform variables** hold the quantity itself. Each starts at the baseline wing's value, and its
  `(lower, upper)` bounds must bracket it. A variable that is left out stays at its baseline value.
- **Size variables.** Aspect ratio, root chord and span $b = 2s$ are tied by $s = AR\, c_r (1 + \lambda) / 4$,
  so at most two of them are variables and the third follows (`derived`). By default the span follows; with
  `span` a variable, whichever of `aspect_ratio` and `root_chord` is not one follows (with `span` the only one,
  pass `derived`). A larger aspect ratio at a fixed root chord gives a longer wing with more area.
- **Section variables** are given at each FFD spanwise section, root first, with `modes` chordwise values per
  section (default 1; `{'bounds': ..., 'modes': n}`), section-major. They are deltas that start at zero. With
  $\xi$ the local chord fraction and $B_m$ the Bernstein polynomials of degree $n - 1$:
  - `thickness` scales the section about its chord plane by $1 + \sum_m \tau_m B_m(\xi)$. Chord changes
    scale the thickness along, so t/c is kept. Equal values give a uniform change.
  - `camber` adds $c \sum_m \kappa_m\, 4\xi(1-\xi) B_m(\xi)$, so the leading and trailing edges stay in
    place. Equal values give a parabolic camber line of maximum $\kappa$.

  The modes are least-squares fits on the block's chordwise basis. They are exact when that basis is a Bezier
  ($n_{chord} = $ degree + 1) of degree at least $n - 1$ for thickness and $n + 1$ for camber;
  `print_design_variables` reports the fit residual. Bounds may be per section, `(n_sections,)`, or per entry.

The map behind it works per FFD section $j$, at the baseline span fraction $\eta_j = y_j / s_0$:
1. The section moves to $y_j' = \eta_j s'$.
2. It is scaled by $c_j'/c_{j,0}$, in x about the reference chord line and in z about the chord plane.
3. Its reference-line point is placed at $x_{ref,root} + y_j' \tan\Lambda'$, so the root point of that line
   stays fixed.

These are lsdo_geo sectional stretches and translations; thickness and camber are additive sectional modes.

With the taper ratio fixed, the map is affine, so the deformed wall is exactly the trapezoid that the
variables describe. A taper change is exact at the FFD sections and blended between them by the spanwise
B-spline. Changing the taper ratio from 0.3 to 0.4 on the 5-section test block moves the chords up to 0.25% away
from the trapezoid; with 3 sections, up to 0.9% for taper ratios 0.2 to 0.5.

`shape.outputs()['wing']` holds the planform as CSDL variables, for use as constraints: `sweep`, `aspect_ratio`,
`root_chord`, `taper_ratio`, `tip_chord`, `semi_span`, `span`, `area` (full wing), and `section_chords`.
`wing.baseline` holds the measured baseline values as numbers, for constraint bounds.

## Design-variable set
`ShapeDVSet` (`shape.dvs`) collects all shape variables in declaration order. `shape.parameter_vector()` is the
vector passed to the mesh warper, flow model and postprocessor. `shape[name]` returns one variable, and
`shape.print_design_variables()` lists them with their bounds.

## Geometric constraints
`shape.geometry` (also `model.wall_geometry()`) is a `WallGeometry` for the deformed wall:

| Method | Quantity |
|---|---|
| `enclosed_measure(wall_displacement)` | enclosed area (2D) or volume (3D) |
| `add_thickness_stations(chord_fractions, span_stations=None)` then `thickness_ratio(coefficients, handle)` | deformed over baseline thickness at the stations |
| `add_planform_stations(span_stations)` then `planform(coefficients, handle)` | section chords, span and planform area (3D) |

```python
geometry = shape.geometry
area = geometry.enclosed_measure(shape.wall_displacement())
area.set_as_constraint(lower=geometry.baseline_measure)
stations = geometry.add_thickness_stations(np.linspace(0.1, 0.9, 9))
geometry.thickness_ratio(shape.coefficients(), stations).set_as_constraint(lower=0.9)
```
