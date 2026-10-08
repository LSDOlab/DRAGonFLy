# Examples

The `examples/` directory contains complete cases. They read their mesh from the repository's `meshes/`
directory, build the model as described in the [user guide](user_guide.md), and by default compare the
adjoint derivatives with finite differences (`sim.check_totals`). To optimize instead, uncomment the modOpt
block at the end of the script.

Run them from any directory, for example
```sh
$ OMP_NUM_THREADS=1 mpirun -n 4 python examples/airfoil_opt.py
```

## `airfoil_opt.py`: 2D airfoil
| | |
|---|---|
| Mesh | `naca0012_euler_mesh_quad_v2`: quadrilaterals around a NACA 0012 airfoil |
| Flow | $M_\infty = 0.8$, $\alpha = 2^\circ$ |
| Design variables | vertical motion of 5 x 3 FFD control points (bounds $\pm 0.01$); $\alpha \in [1.75^\circ, 2.25^\circ]$ |
| Objective, constraint | minimize drag $D$ subject to lift $L \ge 0.225$ |
| Optional (commented out) | camber and thickness variables; area and thickness constraints |

## `rans_airfoil_opt.py`: turbulent 2D airfoil
| | |
|---|---|
| Mesh | `naca0012_cgrid_rans_coarse`: 9,480-cell wall-resolved C-grid (first cell $1.5\times10^{-5}$ chords) |
| Flow | SA-neg RANS, $M_\infty = 0.7$, $\alpha = 2^\circ$, $Re = 6\times10^6$ |
| Design variables | as `airfoil_opt.py` |
| Purpose | `check_totals` of the RANS adjoint (through the reconstructed gradient and the wall distance) |

## `airfoil_analysis.py`: forward analysis through the windtunnel model
`DG_windtunnel_model` without an FFD block or mesh warper (Euler at $M = 0.8$, or `--model rans` at $M = 0.7$):
one plain `solve_forward`, then a CSDL graph with the angle of attack as its only input and `check_totals` of
$dD/d\alpha$ and $dL/d\alpha$.

## `flow_analysis.py`: steady and unsteady analysis
Euler, laminar or SA-neg RANS from any XDMF mesh or structured CGNS grid (`--mesh`), steady (PTC) or unsteady (`--unsteady`, BDF2, optional
`--dual-time`), with the split, `riemann` or `riemann2` far field. Writes `result.csv` or `history.csv`, wall
$C_p$/$c_f$ files, optional VTX fields, and a checkpoint for `--restart`. See `--help`.

## `wing_opt.py`: 3D wing
| | |
|---|---|
| Mesh | `wing_vol_L3.cgns`: 12 structured blocks around the Simple Transonic Wing (half-wing, symmetry plane at $y = 0$), read with PyVista into 181,152 hexahedra; `wing_vol_L3_xdmf_wallmerged` is the same mesh with merged wall layers (fewer cells, lower aspect ratio) |
| Flow | $M_\infty = 0.8$, $\alpha = 2^\circ$ |
| Design variables | `WingShape` on a 5 x 3 x 2 swept, tapered FFD block (quartic Bezier chordwise): quarter-chord sweep $\in [20^\circ, 30^\circ]$, aspect ratio $\in [7.5, 10]$, span $\in [26, 30]$, taper ratio $\in [0.2, 0.4]$ (root chord follows); at each of the 3 spanwise sections, 4 thickness modes ($\pm 10\%$) and 3 camber modes ($\pm 0.01$ chord); $\alpha \in [1.5^\circ, 2.5^\circ]$ |
| Objective, constraint | minimize drag $D$ subject to lift $L \ge 10$ |
| Optional (commented out) | raw control-point motions; planform-area, volume and thickness constraints |

## `benchmark_pod.py`: full-order against reduced-order solutions
The [POD reduced-order model](reduced_order_modeling.md) of the Euler and the RANS model at the same design
points. Each model trains its own `pod_global` and `pod_local` (local_weighted) ROMs on the full-order solutions
at a Latin-Hypercube sample of (FFD control-point motions, $\alpha$, $M_\infty$), then compares them with the
full-order solve at a separate random test set: state L2 errors, lift, drag and moment errors, the ROM residual
and the wall times, with a per-phase breakdown of every test solve (see Profiling in the [user guide](user_guide.md)). It
writes one `DataStore` per (model, variant), box plots with Euler and RANS side by side, and stacked bars of where
the full-order and reduced-order solves spend their time.

| | |
|---|---|
| Meshes | 2D (`--dimension 2D`): `naca0012_euler_mesh_quad_v2` (Euler), `naca0012_cgrid_rans_coarse` (RANS); 3D (default): `wing_vol_L3.cgns` for both |
| Design space | vertical FFD control-point motions (10 x 3 in 2D, $\pm 0.01$; 4 x 5 x 3 in 3D, spanwise-tapered), $\alpha \in [1.75^\circ, 2.25^\circ]$, $M_\infty \in [0.7, 0.9]$ |
| Defaults | 50 training and 25 test points, `rb_size` 10, Mach continuation for the full-order solves, ROMs started from the free stream |

Every setting is a command-line option (`--help`), e.g. a quick 2D run:
```sh
$ OMP_NUM_THREADS=1 mpirun -n 4 python examples/benchmark_pod.py --dimension 2D --n-train 10 --n-test 5
```

## Validation
Validation cases compare DRAGonFLy with experiments and other solvers, and test how the results depend on the
grid. They need meshes that are not in the repository; each page says where to get them.

- [ONERA OAT15A airfoil](validation_oat15a.md) (`examples/ONERA-OAT15A/`): mesh-sensitivity study of the OAT15A
  supercritical airfoil with SA-neg RANS at $M_\infty = 0.73$, $Re = 3\times10^6$, on the structured DPW8 Cadence
  grids.

```{toctree}
:maxdepth: 1

validation_oat15a
```
