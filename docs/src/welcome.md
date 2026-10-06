# Welcome to DRAGonFLy

![alt text](/src/images/lsdolab.png "LSDOlab")

DRAGonFLy performs gradient-based aerodynamic shape optimization of airfoils and wings with a discontinuous
Galerkin (DG) compressible Euler solver. The flow solver is built on [FEniCSx](https://fenicsproject.org), and
the optimization model is assembled with [CSDL](https://github.com/LSDOlab/CSDL_alpha), which computes total
derivatives through the whole chain from the shape design variables to lift, drag and pitching moment.

The Python package is distributed as `dragonfly-sim` and imported as `dragonfly_sim`. The source code is on
[GitHub](https://github.com/LSDOlab/DRAGonFLy).

## Capabilities
- Steady compressible Euler flow in 2D and 3D, discretized with piecewise-constant (p=0) DG elements and
  HLL, HLLC or local Lax-Friedrichs numerical fluxes.
- Newton solver with a positivity-preserving step limiter; GMRES with additive-Schwarz/ILU preconditioning;
  parallel execution with MPI.
- Free-form deformation (FFD) shape parameterization with control-point and sectional (camber, thickness,
  twist, ...) design variables, and geometric constraints (area/volume, thickness, planform).
- Volume mesh deformation by inverse-distance weighting (IDWarp).
- Adjoint-based derivatives of lift, drag and pitching moment with respect to the shape variables and the angle
  of attack, for optimization with [modOpt](https://github.com/LSDOlab/modopt).

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
