# Release notes

## 0.1.0 (unreleased)
First release.
- Steady compressible Euler solver with piecewise-constant (p=0) DG elements in 2D and 3D.
- FFD shape parameterization with control-point and sectional design variables, and geometric constraints.
- IDWarp volume mesh deformation.
- Adjoint-based derivatives of lift, drag and pitching moment with respect to the shape variables and the angle
  of attack.
- Examples: 2D airfoil (`airfoil_opt.py`) and 3D wing (`wing_opt.py`).
