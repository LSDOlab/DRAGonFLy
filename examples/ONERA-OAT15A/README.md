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

`figures/` holds the figures shown on the documentation page.
