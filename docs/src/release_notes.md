# Release notes

## 0.2.0 (unreleased)
- Compressible RANS at p = 0 (`CompressibleRANSModel`): Spalart-Allmaras (SA-neg) or laminar Navier-Stokes,
  Green-Gauss gradient reconstruction eliminated exactly from Newton, MUSCL inviscid flux, adiabatic no-slip
  walls. The turbulence closure is a plug-in (`TurbulenceModel`).
- The RANS model is built on the Euler model by composition; `DG_windtunnel_model(model_class=...)` runs it
  through CSDL with exact adjoints, including the chain terms through the reconstructed gradient, the cell
  centres and the wall distance.
- One time-integration module for steady pseudo-transient continuation, unsteady BDF1/BDF2 and dual time
  (`core/time_integration.py`); opt-in for the Euler model, PTC by default for RANS.
- Characteristic far field (`define_farfield_bc`, `DG_windtunnel_model(farfield="riemann")`) and its unsteady
  transverse refinement (`TransverseFarfield`, 2D).
- Partition-independent checkpoints; `examples/flow_analysis.py` and `examples/rans_airfoil_opt.py`.
- The default Euler path is unchanged (bit-identical residual, Jacobian, derivatives and Newton iterates).

## 0.1.0 (unreleased)
First release.
- Steady compressible Euler solver with piecewise-constant (p=0) DG elements in 2D and 3D.
- FFD shape parameterization with control-point and sectional design variables, and geometric constraints.
- IDWarp volume mesh deformation.
- Adjoint-based derivatives of lift, drag and pitching moment with respect to the shape variables and the angle
  of attack.
- Examples: 2D airfoil (`airfoil_opt.py`) and 3D wing (`wing_opt.py`).
