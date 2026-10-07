# Welcome to DRAGonFLy

![alt text](/src/images/lsdolab.png "LSDOlab")

DRAGonFLy performs aerodynamic analysis and gradient-based shape optimization of airfoils and wings with a
discontinuous Galerkin (DG) compressible flow solver for the Euler and Reynolds-averaged Navier-Stokes (RANS)
equations. The flow solver is built on [FEniCSx](https://fenicsproject.org), and the optimization model is
assembled with [CSDL](https://github.com/LSDOlab/CSDL_alpha), which computes total derivatives through the whole
chain from the shape design variables to lift, drag and pitching moment.

The Python package is distributed as `dragonfly-sim` and imported as `dragonfly_sim`. The source code is on
[GitHub](https://github.com/LSDOlab/DRAGonFLy).

## Capabilities
- Compressible Euler flow in 2D and 3D, and compressible RANS with the SA-neg turbulence model or laminar
  Navier-Stokes, discretized with piecewise-constant (p=0) DG elements (see
  [RANS, time integration and far fields](src/flow_models.md)).
- Steady solves with Newton's method and a positivity-preserving step limiter, optionally with pseudo-transient
  continuation; unsteady BDF1/BDF2 time stepping, with optional dual time stepping; checkpoint and restart.
- Subsonic inflow/outflow and characteristic far fields; slip and adiabatic no-slip walls.
- GMRES with additive-Schwarz/ILU preconditioning and parallel execution with MPI.
- Free-form deformation (FFD) shape parameterization built from layers: control-point motions, sectional
  variables (camber, thickness, twist, ...) and wing planform variables (sweep, aspect ratio, root chord, taper
  ratio), with geometric constraints (area/volume, thickness, planform).
- Volume mesh deformation by inverse-distance weighting (IDWarp).
- Adjoint-based derivatives of lift, drag and pitching moment with respect to the shape variables and the angle
  of attack, for Euler and RANS, for optimization with [modOpt](https://github.com/LSDOlab/modopt).
- Forward analysis on a fixed mesh, without shape variables.
- Mesh input from XDMF files or structured CGNS grids; VTX output for ParaView.

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/flow_models
src/user_guide
src/shape_parameterization
src/examples
src/api
src/contributing
src/release_notes
```
