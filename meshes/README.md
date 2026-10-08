# Meshes

The meshes are not stored in the repository. Place them in this directory; the examples read them from here.

## ONERA-OAT15A validation case
The ONERA-OAT15A case (`examples/ONERA-OAT15A/`) uses the structured Cadence grids from the Eighth AIAA Drag
Prediction Workshop (DPW8). Download
`Cadence-ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured.zip` (554 MB) from
<https://dpw.larc.nasa.gov/DPW8/ONERA_OAT15A/Cadence_Grids.REV01/>, which can be reached from the DPW8 grid page,
<https://www.aiaa-dpw.org/grids.html>, and unzip it here:

```sh
cd meshes
unzip Cadence-ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured.zip
```

The case expects the grids at
`meshes/Cadence-ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured/ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured_Level-N.cgns`,
for N = 1 to 6. Only the `.cgns` files are used; the `.ugrid`, `.mapbc` and `.pw` files are not needed. The
`_Unstructured.zip` in the same directory is not used.

The case can also run on ONERA's structured Rizzi grids (`--mesh-family rizzi`), available at
<https://dpw.larc.nasa.gov/DPW8/ONERA_OAT15A/ONERA-Rizzi_Grids.REV00/>. It expects them at
`meshes/ONERA-ONERA-OAT15A-Rizzi/gridN/OAT15A_Rizzi_N.cgns`, for N = 1 to 7.
