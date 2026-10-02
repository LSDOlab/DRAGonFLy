# Shape parameterization

The wall shape is controlled by a B-spline FFD block around the wall nodes (`WallFFD`, created by
`DG_windtunnel_model.set_up_sim()` as `model.ffd`). Design variables change the block's control points; the wall
moves with them, and the volume mesh follows through IDWarp.

## FFD block
`ffd_shape` sets the number of control points per parametric direction (2 entries in 2D, 3 in 3D) and
`ffd_degree` the B-spline degree. By default the block is the axis-aligned bounding box of the wall. For a
tapered or swept wing, pass `ffd_block_corner_list` to `DG_windtunnel_model`: two rectangular end surfaces (root
and tip), each given by two opposite corners, e.g.
`[[(-0.1, 0., 0.32), (5.1, 0., -0.3)], [(7.4, 14.1, 0.32), (9.1, 14.1, -0.3)]]`.

## Control-point motions
`cp_motions` moves control points individually. Its directions and bounds come from one dictionary keyed by
spatial direction (0 = x, 1 = y, 2 = z):
```python
dv_spec = {1: 0.01}                     # 2D: vertical motions within +/-0.01
dv_spec = {0: 0.02, 2: (-0.05, 0.10)}   # 3D: x symmetric, z asymmetric
dv_spec = {2: spanwise_linear_bounds(base_bound=0.1, scale_at_root=1.0, scale_at_ref=0.5, y_ref=14.1)}
```
A bound is a symmetric half-range, a `(lower, upper)` pair, or a callable of the control-point coordinates such
as `spanwise_linear_bounds`, which varies the bound linearly along the span.

- `cp_dv_directions(dv_spec)` gives the `cp_coord_opt_idxs` that `DG_windtunnel_model` needs.
- `build_cp_motion_dv(model.ffd.baseline_coefficients, dv_spec)` builds the variable's value, bounds and scaling
  (each direction's box maps to $\pm 1$).
- `model.ffd.apply_cp_motions(coefficients, cp_motions, cp_coord_opt_idxs)` adds the motions to the control
  points.

## Sectional variables
`SectionalShape(model.ffd, principal_dim)` views the block as a stack of sections: `principal_dim=0` in 2D
(chordwise stations), `1` for a wing (spanwise stations). Variables are added with `add(kind, values)`:

| Kind | Operation | Dimensions |
|---|---|---|
| `camber` | translation along the vertical axis | 2D |
| `thickness` | stretch along the vertical axis | 2D, 3D |
| `twist` | rotation about the spanwise axis (radians) | 3D |
| `chord` | stretch along the chordwise axis | 3D |
| `sweep` | translation along x | 3D |
| `dihedral` | translation along z | 3D |
| `span` | translation along y | 3D |

A variable has one value per section, or fewer values, which are then B-spline coefficients of a profile over the
sections. Create its bounds with `build_sectional_dv(num_values, bounds)`.

Apply the sectional layer to the constant baseline first, then the control-point motions:
```python
sectional = SectionalShape(model.ffd, principal_dim=1)
twist = shape_dvs.add('twist', build_sectional_dv(5, np.radians(2.)))
sectional.add('twist', twist)
coefficients = sectional.apply(model.ffd.baseline_variable())
coefficients = model.ffd.apply_cp_motions(coefficients, cp_motions, cp_coord_opt_idxs)
```

## Design-variable set
`ShapeDVSet` collects all shape variables in declaration order; `shape_dvs.flat_variable()` is the vector
passed to the mesh warper, flow model and postprocessor.

## Geometric constraints
`model.wall_geometry()` returns a `WallGeometry` for the deformed wall:

| Method | Quantity |
|---|---|
| `enclosed_measure(wall_displacement)` | enclosed area (2D) or volume (3D) |
| `add_thickness_stations(chord_fractions, span_stations=None)` then `thickness_ratio(coefficients, handle)` | deformed over baseline thickness at the stations |
| `add_planform_stations(span_stations)` then `planform(coefficients, handle)` | section chords, span and planform area (3D) |

```python
geometry = model.wall_geometry()
area = geometry.enclosed_measure(model.ffd.wall_displacement(coefficients))
area.set_as_constraint(lower=geometry.baseline_measure)
stations = geometry.add_thickness_stations(np.linspace(0.1, 0.9, 9))
geometry.thickness_ratio(coefficients, stations).set_as_constraint(lower=0.9)
```
