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
- Shape design variables as modular layers (`core/shape_design.py`): `DG_windtunnel_model` takes an
  `FFDShapeParameterization` (FFD block + layers) instead of `ffd_shape`, `ffd_degree`, `ffd_block_corner_list`
  and `cp_coord_opt_idxs`, and no longer depends on which variables exist; `model.deform_mesh()` gives the
  mesh motion. Layers: `ControlPointMotions`, `SectionalVariables`, and `WingShape`, which adds wing sweep,
  aspect ratio, root chord, span and taper ratio (absolute values with bounds), plus chordwise thickness and
  camber modes per spanwise station. `examples/wing_opt.py` uses `WingShape`. The model's and the warper's shape input is
  renamed from `cp_motion_inputs` to `shape_parameters`.
- `DG_windtunnel_model` without a shape parameterization or mesh warper: forward analysis on the fixed
  mesh, with `solve_forward()` outside CSDL or `evaluate(alpha=...)` in a graph; `DG_postprocessor` computes
  forces and coefficients without FFD definitions (`forces_and_coefficients`, `evaluate` without a mesh
  deformation).
- Partition-independent checkpoints; `examples/flow_analysis.py`, `examples/airfoil_analysis.py` and
  `examples/rans_airfoil_opt.py`.
- Mesh input from structured multi-block CGNS grids one cell thick (`utils/mesh_io_utils.py`:
  `load_dolfinx_mesh`, `cgns_to_dolfinx_mesh`), reduced to a 2D quadrilateral mesh in memory;
  `examples/flow_analysis.py --mesh` accepts them. 3D structured grids are read as hexahedra (`tdim=3`);
  `examples/wing_opt.py` reads `wing_vol_L3.cgns` this way.
- ONERA OAT15A validation case (`examples/ONERA-OAT15A/`): SA-neg RANS mesh-sensitivity study on the Rizzi grids
  at $M = 0.73$, $Re = 3\times10^6$ (preliminary results on grids 1-4).
- `FileWriter` gathers piecewise-constant and continuous fields to rank 0, so VTX output shows no seams at the
  MPI partition boundaries; new `mesh_deformation` and `time_dependent` keywords. The solution and pressure
  files of a shape optimization are written on the deformed mesh of each evaluation.
- The default Euler path is unchanged (bit-identical residual, Jacobian, derivatives and Newton iterates).

## 0.1.0 (unreleased)
First release.
- Steady compressible Euler solver with piecewise-constant (p=0) DG elements in 2D and 3D.
- FFD shape parameterization with control-point and sectional design variables, and geometric constraints.
- IDWarp volume mesh deformation.
- Adjoint-based derivatives of lift, drag and pitching moment with respect to the shape variables and the angle
  of attack.
- Examples: 2D airfoil (`airfoil_opt.py`) and 3D wing (`wing_opt.py`).
