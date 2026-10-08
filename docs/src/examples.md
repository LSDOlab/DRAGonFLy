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

## `ONERA-OAT15A/`: transonic RANS validation
A mesh-sensitivity study of the ONERA OAT15A supercritical airfoil, the conditions of the OAT15A buffet test case.
The results and figures are on the [OAT15A results page](_temp/examples/ONERA-OAT15A/README.md).

| | |
|---|---|
| Mesh | ONERA's structured multi-block Rizzi grids 1-7 (CGNS, one cell thick in span; 15,872 to 471,040 cells), reduced to 2D quadrilateral meshes in memory by `dragonfly_sim.utils.mesh_io_utils` |
| Flow | SA-neg RANS, $M_\infty = 0.73$, $Re = 3\times10^6$, $T_\infty = 300$ K; $\alpha$ = 1.36°, 1.50°, 2.50°, 3.00°, 3.10° |
| Solver | pseudo-transient continuation from the free stream to a residual of $10^{-9}$ |
| Outputs | $c_l$, $c_d$ (and its friction part), $c_m$, and wall time per grid and angle of attack (`results.csv`); wall $C_p$ and $c_f$ per case; figures of $c_l$-$\alpha$, $c_l$-$c_d$, $c_m$-$c_l$ and cost against degrees of freedom |
| Status | preliminary results on grids 1-4; comparisons with experiment and other solvers to follow |

```sh
$ cd examples/ONERA-OAT15A
$ OMP_NUM_THREADS=1 mpirun -n 8 python oat15a_analysis.py --out-dir <output folder> [--grids 1 2] [--alphas 2.5]
$ python oat15a_analysis.py --plot <run folder>/results.csv --figure-dir figures
```
The grids are read from `meshes/ONERA-ONERA-OAT15A-Rizzi/gridN/OAT15A_Rizzi_N.cgns` (or `--mesh-dir`). The script
reuses the model set-up, force coefficients and wall output of `flow_analysis.py`.

```{toctree}
:hidden:

_temp/examples/ONERA-OAT15A/README
```
