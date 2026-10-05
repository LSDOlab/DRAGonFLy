# DRAGonFLy

[![Tests](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/actions.yml)
[![Coverage](https://lsdolab.github.io/DRAGonFLy/coverage/badge.svg)](https://lsdolab.github.io/DRAGonFLy/coverage/)
[![Docs](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/docs.yml/badge.svg)](https://lsdolab.github.io/DRAGonFLy)

Aerodynamic shape optimization with a discontinuous Galerkin (DG) compressible Euler solver, built on
FEniCSx and CSDL. It is distributed as `dragonfly-sim` and imported as `dragonfly_sim`.

**Documentation: <https://lsdolab.github.io/DRAGonFLy>**

## Capabilities
- **Flow solver**: steady compressible Euler equations in 2D and 3D, discretized with a discontinuous Galerkin
  method (piecewise-constant, i.e. p=0, elements in this release) and HLL, HLLC or local Lax-Friedrichs numerical
  fluxes. Subsonic inflow/outflow, slip-wall and symmetry-plane boundaries.
- **Nonlinear and linear solvers**: Newton's method (PETSc SNES) with a positivity-preserving step limiter on
  density and pressure, and GMRES with additive-Schwarz/ILU preconditioning. Runs in parallel with MPI.
- **Shape parameterization**: B-spline free-form deformation (FFD) of the wall, with individual control-point
  motions and sectional variables (camber, thickness, twist) as design variables, plus geometric constraints:
  enclosed area/volume, thickness at chosen stations and wing planform.
- **Mesh deformation**: inverse-distance-weighting volume mesh warping (IDWarp), driven by the FFD wall motion.
- **Outputs and derivatives**: lift, drag and pitching moment, with adjoint-based derivatives with respect to
  the shape variables and the angle of attack, assembled by CSDL across the FFD, mesh warping, flow and force
  computations, for gradient-based optimization with modOpt (e.g. SLSQP). The force and moment coefficients
  and the pitching-moment slope are reported for monitoring.
- **Output files**: flow solution, pressure and mesh deformation in ADIOS2/VTX (`.bp`) format for ParaView.

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
git lfs install
git lfs pull   # fetches the meshes, which are stored with Git LFS
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
would fetch them from PyPI if they were missing). `environment.yml` additionally installs `git-lfs`.

See the [documentation](https://lsdolab.github.io/DRAGonFLy/src/getting_started.html) for running the
examples and the tests.

## License
This project is licensed under the terms of the **MIT License**.
