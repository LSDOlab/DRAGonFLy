# DRAGonFLy

[![Tests](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/actions.yml)
[![Coverage](https://lsdolab.github.io/DRAGonFLy/coverage/badge.svg)](https://lsdolab.github.io/DRAGonFLy/coverage/)
[![Docs](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/docs.yml/badge.svg)](https://lsdolab.github.io/DRAGonFLy)

Aerodynamic analysis and gradient-based shape optimization of airfoils and wings with a discontinuous Galerkin
(DG) compressible flow solver, built on FEniCSx and CSDL. It solves the Euler and the Reynolds-averaged
Navier-Stokes (RANS) equations, steady or unsteady. It is distributed as `dragonfly-sim` and imported as
`dragonfly_sim`.

**Documentation: <https://lsdolab.github.io/DRAGonFLy>**

## Capabilities
- **Flow models**: compressible Euler in 2D and 3D, and compressible RANS with the Spalart-Allmaras (SA-neg)
  turbulence model or laminar Navier-Stokes. Piecewise-constant (p=0) DG elements, i.e. a cell-centred
  finite-volume scheme. Euler uses HLL, HLLC or local Lax-Friedrichs fluxes. RANS adds a Green-Gauss gradient
  reconstruction, MUSCL inviscid fluxes and adiabatic no-slip walls. Turbulence closures are plug-ins
  (`TurbulenceModel`).
- **Boundary conditions**: subsonic inflow/outflow, a characteristic (Riemann) far field, and in 2D an unsteady
  far field with transverse terms that reduces reflections; slip and no-slip walls; symmetry planes.
- **Solvers**: Newton's method (PETSc SNES) with a positivity-preserving step limiter, and GMRES/FGMRES with
  additive-Schwarz/ILU preconditioning (or MUMPS). Steady solves can use pseudo-transient continuation with an
  adaptive CFL number (the default for RANS). Unsteady solves use BDF1/BDF2 with adaptive time steps, optionally
  with dual time stepping. Checkpoints restart on any number of MPI ranks.
- **Shape parameterization**: B-spline free-form deformation (FFD) of the wall, with design variables in
  composable layers: individual control-point motions, sectional variables (camber, thickness, twist, chord,
  ...) and wing planform variables (sweep, aspect ratio, root chord, taper ratio). Geometric constraints:
  enclosed area or volume, thickness at chosen stations, and planform.
- **Mesh deformation**: inverse-distance-weighting volume mesh warping (IDWarp), driven by the FFD wall motion.
- **Outputs and derivatives**: lift, drag (including friction drag for viscous models) and pitching moment, with
  adjoint-based derivatives with respect to the shape variables and the angle of attack, for both Euler and
  RANS. CSDL assembles them across the FFD, mesh warping, flow and force computations, for gradient-based
  optimization with modOpt (e.g. SLSQP). The force and moment coefficients and the pitching-moment slope are
  reported for monitoring.
- **Forward analysis**: the same flow model runs on a fixed mesh without any shape variables, with or without a
  CSDL graph.
- **Reduced-order modeling**: an optional POD reduced-order model of the steady flow solve, for Euler and RANS.
  Before every solve it tries a least-squares Petrov-Galerkin (LSPG) solution in a POD basis of the converged
  solutions so far, and falls back to the full-order solve (adding a snapshot) when the residual is too large.
- **Mesh input**: DOLFINx meshes from XDMF, and structured multi-block CGNS grids one cell thick (2D), which are
  converted to quadrilateral meshes in memory.
- **Output files**: flow solution, pressure and mesh deformation in ADIOS2/VTX (`.bp`) format for ParaView,
  written without seams at the MPI partition boundaries. The analysis examples also write the wall pressure and
  skin-friction coefficients as CSV.

## Examples
| Script | Case |
| --- | --- |
| `examples/airfoil_opt.py` | Euler drag minimization of a 2D airfoil at a lift constraint |
| `examples/rans_airfoil_opt.py` | RANS (SA-neg) airfoil: adjoint derivatives checked against finite differences |
| `examples/wing_opt.py` | Euler drag minimization of a 3D wing with planform and section variables |
| `examples/airfoil_analysis.py` | forward Euler or RANS analysis without shape variables |
| `examples/flow_analysis.py` | steady or unsteady Euler, laminar or RANS analysis of any mesh, from the command line |
| `examples/benchmark_pod.py` | POD reduced-order model against the full-order solve, Euler and RANS at the same design points |
| `examples/ONERA-OAT15A/` | RANS mesh-sensitivity study of the ONERA OAT15A transonic airfoil ([results](examples/ONERA-OAT15A/README.md)) |

## Installation
DRAGonFLy requires Python 3.12 or newer. Two of its dependencies, `fenics-dolfinx` 0.11.0 and `mpich`, are not
on PyPI and must come from conda-forge, so a conda environment is needed either way. `pyvista` and `networkx`
are also installed from conda-forge, to keep them under conda's control.

### Option 1: new conda environment (recommended)
`environment.yml` creates an environment with all dependencies and installs the package in editable mode,
together with the test and documentation tools:
```sh
git clone https://github.com/LSDOlab/DRAGonFLy.git
cd DRAGonFLy
conda env create -f environment.yml
conda activate dragonfly
```

### Option 2: pip install into an existing environment
In an existing conda environment that already provides the conda-forge dependencies, install the package
directly from GitHub; pip then installs the remaining dependencies automatically (it leaves the conda-installed
`pyvista` and `networkx` alone, since they already satisfy the requirements):
```sh
conda install -c conda-forge fenics-dolfinx=0.11.0 mpich pyvista networkx   # skip what is already installed
pip install git+https://github.com/LSDOlab/DRAGonFLy.git
```
To work on the code, clone the repository and run `pip install -e .[test,docs]` instead (the extras add `pytest`
and the documentation tools).

### Dependencies
| Installed with | Packages |
| --- | --- |
| conda-forge (required beforehand for Option 2) | `fenics-dolfinx` 0.11.0 (with `mpi4py`, `petsc4py`, `ufl` and `basix`), `mpich`, `pyvista`, `networkx` |
| pip, from PyPI | `numpy`, `scipy`, `matplotlib`, `jax` |
| pip, from the `main` branches of the LSDOlab repositories | [CSDL_alpha](https://github.com/LSDOlab/CSDL_alpha), [lsdo_function_spaces](https://github.com/LSDOlab/lsdo_function_spaces), [lsdo_geo](https://github.com/LSDOlab/lsdo_geo), [modopt](https://github.com/LSDOlab/modopt), [IDWarp-JAX](https://github.com/LSDOlab/IDWarp-JAX) |

All runtime dependencies are declared in `setup.py` (so `pyvista` and `networkx` are still listed there, and pip
would fetch them from PyPI if they were missing).

See the [documentation](https://lsdolab.github.io/DRAGonFLy/src/getting_started.html) for running the
examples and the tests.

## License
This project is licensed under the terms of the **MIT License**.
