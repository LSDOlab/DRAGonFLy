# DRAGonFLy

[![Tests](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOlab/DRAGonFLy/actions/workflows/actions.yml)
[![Coverage](https://codecov.io/gh/LSDOlab/DRAGonFLy/branch/main/graph/badge.svg)](https://codecov.io/gh/LSDOlab/DRAGonFLy)
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
`fenics-dolfinx` 0.11.0 and `mpich` come from conda-forge; `environment.yml` sets up everything:
```sh
git clone https://github.com/LSDOlab/DRAGonFLy.git
cd DRAGonFLy
conda env create -f environment.yml
conda activate dragonfly
```
See the [documentation](https://lsdolab.github.io/DRAGonFLy/src/getting_started.html) for details, running the
examples and the tests.

## License
This project is licensed under the terms of the **MIT License**.
