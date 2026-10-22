# Release notes

## 0.2.0 (unreleased)
- Primitive unknowns and fluid materials: every flow model now solves for $(p', \mathbf{u}, T, \phi)$ with the
  conservative residual $\nabla\cdot\mathbf{F}(U(V))$, and the fluid is a material (`core/materials.py`:
  `IdealGas`, the default, and `ConstantDensity` for water), the only fluid-specific code. Quantities are scaled
  with the free stream ($\rho_\infty = |\mathbf{u}_\infty| = T_\infty = 1$, pressure in $\rho_\infty u_\infty^2$)
  and the energy row is weighted by $1/(c_p T_\infty)$, so the residual is the same discrete scheme as before
  (checked against the stored reference to 12 digits) with rows of one size at every Mach number. Optional local
  low-Mach (Weiss-Smith) preconditioning (`preconditioning=True`, with the free stream as a ghost far field) and
  the Barter-Darmofal artificial viscosity in its p = 0 form (`shock_capturing=True`). Breaking changes: the
  solution fields are `pressure` ($p'$), `velocity`, `temperature`, the turbulence variables (`nu_tilde`),
  `density` and `mach`; `farfield_conditions` is in the solver's units; `Euler_utils.pressure`,
  `energy_density`, `NS_utils.extended_*`, `primitive_gradients`, `sutherland_viscosity` and
  `utils/farfield_utils.py` are gone (the Riemann far field is `IdealGas.farfield_state`); the turbulence-closure
  methods take `(V, th, ...)`; `compute_positivity_preserving_theta` takes linear constraints; checkpoints carry
  their state layout and older (conservative) ones are refused; physical time steps are in chords per free-stream
  speed; the far-field Mach number must be positive.
- Output-sensitivity weights on the `local_weighted` POD snapshot distance: `utils/dv_sensitivity.py` reads the
  total derivatives of the objective and constraints from the optimizer (`attach_total_derivative_probe`) and
  `ReducedOrderModel(distance_weighting=...)` weights the distance with them. Unweighted unless a probe supplies
  them.
- Compressible RANS at p = 0 (`CompressibleRANSModel`): Spalart-Allmaras (SA-neg) or laminar Navier-Stokes,
  Green-Gauss gradient reconstruction eliminated exactly from Newton, MUSCL inviscid flux, adiabatic no-slip
  walls. The turbulence closure is a plug-in (`TurbulenceModel`).
- The RANS model is built on the Euler model by composition; `DG_windtunnel_model(model_class=...)` runs it
  through CSDL with exact adjoints, including the chain terms through the reconstructed gradient, the cell
  centres and the wall distance.
- One time-integration module for steady pseudo-transient continuation, unsteady BDF1/BDF2 and dual time
  (`core/time_integration.py`); opt-in for the Euler model, PTC by default for RANS.
- Characteristic far field (`define_farfield_bc`) and its unsteady transverse refinement (`TransverseFarfield`, 2D).
  It is the only far field of `DG_windtunnel_model` and the default of `examples/flow_analysis.py`. The inflow/outflow
  split is removed, and with it the 1 degree limit on changes of the angle of attack. Breaking changes: the
  boundary dictionary is the flat free-stream state `{'rho', 'p', 'M', 'alpha'}` (the `'inlet'`/`'outlet'` form
  raises); `DG_windtunnel_model(farfield=...)`, `define_subsonic_inflow_bc`, `define_subsonic_outflow_bc`,
  `subsonic_inflow_state`, `subsonic_outflow_state`, `INLET_TAG`/`OUTLET_TAG` and `alpha_tagging` are gone;
  `define_inlet_outlet_conditions`, `boundary_conditions['inlet'|'outlet']` and
  `compute_initial_conditions_from_inlet` became `define_farfield_conditions`, `farfield_conditions` (flat) and
  `compute_initial_conditions_from_farfield`. The far field is subsonic: `M >= 1` raises.
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
- Moments about any number of points and directions: `DG_windtunnel_model(moments=[(name, point, direction), ...])`
  and `DG_postprocessor(moments=...)` give CSDL outputs `M_<name>` (differentiated) and `c_m_<name>`, e.g. a
  wing's root bending moment and pitching moment in one simulation. **Breaking:** `aero_center` and the outputs
  `M`, `c_m` and `c_m_alpha` (with `log_cm_alpha`) are removed; the old pitching moment is
  `("pitch", aero_center, "pitch")`. The FOM/ROM coefficient histories hold one $c_m$ per moment.
- Partition-independent checkpoints; `examples/flow_analysis.py`, `examples/airfoil_analysis.py` and
  `examples/rans_airfoil_opt.py`.
- Boundary tags of the mesh can define the wall and symmetry planes: `load_dolfinx_mesh` reads and stores the facet
  tags of XDMF ("Facet tags"), gmsh `.msh` (physical groups, now readable) and planar CGNS (Wall, Farfield) meshes, and
  `DG_windtunnel_model(wall_tags=..., symmetry_tags=...)` uses them in place of `mesh_inner_bdry_function`; every other
  boundary is far field. The function is used if given, with a warning that lists the mesh's tags. `examples/flow_analysis.py`
  takes `--wall-tags` and `--symmetry-tags`.
- Mesh input from structured multi-block CGNS grids one cell thick (`utils/mesh_io_utils.py`:
  `load_dolfinx_mesh`, `cgns_to_dolfinx_mesh`), reduced to a 2D quadrilateral mesh in memory;
  `examples/flow_analysis.py --mesh` accepts them. 3D structured grids are read as hexahedra (`tdim=3`);
  `examples/wing_opt.py` reads `wing_vol_L3.cgns` this way.
- ONERA OAT15A validation case (`examples/ONERA-OAT15A/`): SA-neg RANS mesh-sensitivity study on the Rizzi grids
  at $M = 0.73$, $Re = 3\times10^6$ (preliminary results on grids 1-4).
- `FileWriter` gathers piecewise-constant and continuous fields to rank 0, so VTX output shows no seams at the
  MPI partition boundaries; new `mesh_deformation` and `time_dependent` keywords. The solution and pressure
  files of a shape optimization are written on the deformed mesh of each evaluation.
- POD reduced-order model for the steady forward solves of the Euler and RANS models
  (`DG_windtunnel_model(reduced_order_model=ReducedOrderModel(...))`, `core/reduced_order_model.py`,
  `core/pod.py`): LSPG on a global or locally weighted POD basis of converged solutions in a state-scaled
  inner product (Euclidean by default, optionally with the mass matrix), with the original POD benchmark's
  convergence criterion; accepted below a residual threshold, with a fallback to the full-order solve that adds
  a snapshot. The host gains `positivity_step_length`, `steady_residual()`, `state_scales()` and the
  `extra_scales` plug-in.
- The default Euler path is unchanged (bit-identical residual, Jacobian, derivatives and Newton iterates).

## 0.1.0 (unreleased)
First release.
- Steady compressible Euler solver with piecewise-constant (p=0) DG elements in 2D and 3D.
- FFD shape parameterization with control-point and sectional design variables, and geometric constraints.
- IDWarp volume mesh deformation.
- Adjoint-based derivatives of lift, drag and pitching moment with respect to the shape variables and the angle
  of attack.
- Examples: 2D airfoil (`airfoil_opt.py`) and 3D wing (`wing_opt.py`).
