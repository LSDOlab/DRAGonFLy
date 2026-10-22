# ONERA OAT15A airfoil: mesh sensitivity study

Steady SA-neg RANS of the ONERA OAT15A supercritical airfoil at $M_\infty = 0.73$, $Re = 3\times10^6$ and
T = 300 K, for angles of attack 1.36°, 1.50°, 2.50°, 3.00° and 3.10°. It runs on the structured DPW8 Cadence grids
(levels 1–6) and compares lift, drag, moment and cost across grid levels.

The case description, results and figures are in the documentation, on the page **Examples > Validation >
ONERA OAT15A airfoil** (source: `docs/src/validation_oat15a.md`). Where to download the meshes is described there
and in `meshes/README.md`.

```
cd examples/ONERA-OAT15A
OMP_NUM_THREADS=1 mpirun -n 8 python oat15a_analysis.py --out-dir <output folder>
python oat15a_analysis.py --plot <run folder>/results.csv --figure-dir figures
```

`figures/rizzi/` holds the figures of the Rizzi-grid sweep shown on the documentation page, and
`data/rizzi_results.csv` its results (grids 1-7, all five angles of attack).

## Unsteady separated flow at high angle of attack

`oat15a_unsteady.py` runs an unsteady (URANS, SA-neg) forward analysis on a Rizzi grid at 12° (M 0.73,
Re 3e6 by default), where the flow is massively separated and has no steady state. It starts from a first-order
RANS solution, steps the second-order model in time with BDF2, and writes the flow fields at every time step
to `solution.bp` (open it in ParaView and animate over time), the force coefficients to `history.csv`, wall
C_p/c_f every `--wall-every` steps, and restart checkpoints (`--restart`). On grid 3 a frame takes ~9 MB, a
time step ~7 s on 4 ranks and the run ~2.5 GiB.

```
cd examples/ONERA-OAT15A
OMP_NUM_THREADS=1 mpirun -n 4 python oat15a_unsteady.py --grid 3 --steps 2000 --out-dir <output folder>
```
