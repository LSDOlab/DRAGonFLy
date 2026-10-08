# ONERA OAT15A airfoil

Steady RANS of the ONERA OAT15A supercritical airfoil on structured grids, with DRAGonFLy's $p = 0$ solver. The
case compares lift, drag and moment coefficients across grid levels at the transonic conditions of the OAT15A
buffet test case, and the cost of each grid. The case lives in `examples/ONERA-OAT15A/`.

```{note}
The results on this page are **preliminary**. The case runs on the structured Cadence grids from the Eighth AIAA
Drag Prediction Workshop (DPW8). The full sweep on those grids has not been run yet; so far only level 1 at
$\alpha = 1.36^\circ$ and $1.50^\circ$ has been solved. The figures below come from an earlier study on the
coarser Rizzi grids 1–4. Comparisons with experimental data and other solvers will be added later.
```

## Case

| | |
|---|---|
| Geometry | ONERA OAT15A, chord 230 mm (scaled to 1 in the solver), blunt trailing edge |
| Mach number | $M_\infty = 0.73$ |
| Reynolds number | $Re = 3\times10^6$, based on the chord |
| Static free-stream temperature | 300 K (Sutherland's law) |
| Angles of attack | $\alpha$ = 1.36°, 1.50°, 2.50°, 3.00°, 3.10° |
| Model | SA-neg RANS, cell-centred finite volume ($p = 0$), MUSCL inviscid fluxes |
| Boundary conditions | adiabatic no-slip wall; subsonic inflow/outflow far field |
| Solver | pseudo-transient continuation from the free stream (CFL$_0$ = 5) to a residual of $10^{-9}$ |
| Moment reference | quarter chord, positive nose-up |

Every case starts from the same free stream. Cases run one at a time.

## Obtaining the meshes

The meshes are not stored in the repository. The case uses the **structured Cadence grids** from DPW8:

1. Download `Cadence-ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured.zip` (554 MB) from
   <https://dpw.larc.nasa.gov/DPW8/ONERA_OAT15A/Cadence_Grids.REV01/>. This directory can also be reached from the
   DPW8 grid page, <https://www.aiaa-dpw.org/grids.html>. The same directory has an `_Unstructured.zip`; the case
   does not use it.
2. Unzip it in the repository's `meshes/` directory:

   ```sh
   $ cd meshes
   $ unzip Cadence-ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured.zip
   ```

   This creates `meshes/Cadence-ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured/`. The case reads
   only the `.cgns` files in it, `ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured_Level-N.cgns` for
   N = 1 to 6. The `.ugrid`, `.mapbc` and `.pw` files, and the `__MACOSX` folder the archive may contain, are not
   needed.

To keep the meshes elsewhere, pass that directory with `--mesh-dir`. The earlier Rizzi grids are still supported
with `--mesh-family rizzi`; `meshes/README.md` says where to get them.

## Grids

Each Cadence grid is a structured multi-block CGNS grid: 6 blocks, 3D, and one cell thick in span.
`dragonfly_sim.utils.mesh_io_utils` reads it with PyVista and reduces it in memory to the 2D quadrilateral mesh
the solver uses: it keeps one span plane of every block and merges the nodes the blocks share. No converted mesh
file is written. At $p = 0$ with SA-neg, each cell carries 5 unknowns.

| Level | Cells | Degrees of freedom | Wall edges |
|---:|---:|---:|---:|
| 1 | 151,003 | 755,015 | 703 |
| 2 | 240,554 | 1,202,770 | 842 |
| 3 | 378,312 | 1,891,560 | 976 |
| 4 | 596,792 | 2,983,960 | 1,110 |
| 5 | 937,027 | 4,685,135 | 1,239 |
| 6 | 1,471,296 | 7,356,480 | 1,371 |

The first cell at the wall is about $7\times10^{-6}$ chords high on level 1 and $2\times10^{-6}$ chords on
level 6. The far field is about 150 chords from the airfoil.

## Results

### Cadence grids

Level 1, 8 MPI ranks of an Intel Core i7-10700 (8 cores), 8.8 GiB of memory in total:

| $\alpha$ | PTC steps | Solve wall time | $c_l$ | $c_d$ (friction) | $c_m$ |
|---:|---:|---:|---:|---:|---:|
| 1.36° | 415 | 1981 s | 0.7605 | 0.01363 (0.00648) | −0.1280 |
| 1.50° | 560 | 2996 s | 0.7901 | 0.01478 (0.00642) | −0.1288 |

On this grid the residual stalls for long stretches (around $10^{-3}$, then $10^{-5}$) before it converges, so
each case takes several hundred pseudo-time steps.

### Preliminary: Rizzi grids 1–4

These figures come from the earlier study on ONERA's structured Rizzi grids (15,872 to 100,352 cells), at the same
conditions. Each line is one grid's angle-of-attack sweep. Hollow markers would mark cases that did not converge;
every case shown converged.

#### Lift coefficient against angle of attack

![Lift coefficient against angle of attack, one line per grid](_temp/examples/ONERA-OAT15A/figures/cl_alpha.svg)

#### Lift coefficient against drag coefficient

![Lift coefficient against drag coefficient, one line per grid](_temp/examples/ONERA-OAT15A/figures/cl_cd.svg)

#### Moment coefficient against lift coefficient

![Moment coefficient against lift coefficient, one line per grid](_temp/examples/ONERA-OAT15A/figures/cm_cl.svg)

#### Cost: wall time against degrees of freedom

Wall time of the steady solve (pseudo-transient continuation only, without model set-up and output), on 8 MPI
ranks of an Intel Core i7-10700 (8 cores). One line per angle of attack.

![Wall time of the solve against degrees of freedom, one line per angle of attack](_temp/examples/ONERA-OAT15A/figures/dofs_wall.svg)

### Comparison with experiment and other solvers

To be added.

## Running

The full sweep, Cadence levels 1–6 at all five angles of attack, one case at a time:

```sh
$ cd examples/ONERA-OAT15A
$ OMP_NUM_THREADS=1 mpirun -n 8 python oat15a_analysis.py --out-dir <output folder>
```

Each run writes a new folder under `--out-dir` containing:

- `results.csv`: one row per case, rewritten after every case. Columns: grid, cells, dofs, alpha, converged,
  steps, residual, `wall_s` (the solve), `case_s` (model set-up + solve), $c_l$, $c_d$, $c_{d,\text{friction}}$,
  $c_m$.
- `wall_gridN_aA.csv`: wall $C_p$ and $c_f$ per wall facet.
- The four figures above, in that same folder.

Useful options:

- `--grids` and `--alphas` run a subset, for example `--grids 1 2 --alphas 2.5`.
- `--mesh-family rizzi` runs the Rizzi grids 1–7 instead.
- `--mesh-dir` points at meshes kept outside `meshes/`.

The SA-neg model, the mesh input and the force coefficients come from `examples/flow_analysis.py` and the
`dragonfly_sim` package.

```{warning}
The finer Cadence levels are expensive. On level 1, each case took 33–50 minutes on 8 ranks and 8.8 GiB of memory.
Memory grows roughly with the number of degrees of freedom, so level 6 (about 10 times level 1) is expected to need
more memory than a 46 GB workstation has. This is an estimate, not a measurement.
```

To redraw the figures from a results file:

```sh
$ python oat15a_analysis.py --plot <run folder>/results.csv --figure-dir figures
```

Figures are SVG by default. Use `--format pdf` (or `png`) for other uses.
