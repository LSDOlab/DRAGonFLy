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

## `flow_analysis.py`: steady and unsteady analysis
Euler, laminar or SA-neg RANS from any mesh, steady (PTC) or unsteady (`--unsteady`, BDF2, optional
`--dual-time`), with the split, `riemann` or `riemann2` far field. Writes `result.csv` or `history.csv`, wall
$C_p$/$c_f$ files, optional VTX fields, and a checkpoint for `--restart`. See `--help`.

## `wing_opt.py`: 3D wing
| | |
|---|---|
| Mesh | `wing_vol_L3_xdmf`: hexahedra around the Simple Transonic Wing (half-wing, symmetry plane at $y = 0$); `wing_vol_L3_xdmf_wallmerged` is the same mesh with merged wall layers (fewer cells, lower aspect ratio) |
| Flow | $M_\infty = 0.8$, $\alpha = 2^\circ$ |
| Design variables | vertical motion of 3 x 5 x 2 control points of a swept, tapered FFD block (bounds $\pm 0.1$); $\alpha \in [1.5^\circ, 2.5^\circ]$ |
| Objective, constraint | minimize drag $D$ subject to lift $L \ge 10$ |
| Optional (commented out) | twist and chord variables; volume, thickness and planform-area constraints |
