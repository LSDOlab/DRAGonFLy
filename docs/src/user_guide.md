# User guide

A DRAGonFLy case is a CSDL model built from three operations: the mesh warper, the flow model
(`DG_windtunnel_model`) and the force postprocessor (`DG_postprocessor`). This page walks through setting
one up, in the order of the scripts in `examples/`. Run the script with MPI
(`OMP_NUM_THREADS=1 mpirun -n <ranks> python script.py`).

## 1. Mesh
Read a DOLFINx mesh, typically from XDMF:
```python
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "naca0012_euler_mesh_quad_v2.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
```
`dragonfly_sim.utils.mesh_io_utils.load_dolfinx_mesh(path)` reads either an XDMF file or a structured
multi-block CGNS grid, chosen by file extension, and returns `(mesh, info)`:
```python
from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh

mesh, info = load_dolfinx_mesh("OAT15A_Rizzi_1.cgns")   # info: chord, cell and boundary-edge counts
```
A CGNS grid of a 2D case must be one cell thick (or planar). Rank 0 reads it with PyVista, keeps one span plane
of every block, merges the nodes the blocks share by exact coordinate match, and checks that the boundary edges
are exactly those of the wall and far-field families (`wall_family="Wall"`, `farfield_family="Farfield"`). By
default the points are shifted and scaled so that the wall's leading edge is at $x = 0$ and its chord is 1
(`scale_to_chord`). DOLFINx then distributes the mesh; no converted file is written.

A 3D structured grid is read with `tdim=3`, e.g. the Simple Transonic Wing's
`load_dolfinx_mesh("wing_vol_L3.cgns", tdim=3)`: every block becomes hexahedra, the blocks' shared nodes are merged
by exact coordinate match, and the mesh is checked for inverted cells and unmerged block interfaces. Its boundary
conditions are not read (VTK skips them), so the solver finds the boundaries geometrically, as for an XDMF mesh.

Requirements:
- **Orientation**: the freestream flows in $+x$. In 2D, y is vertical; in 3D, y is spanwise and z vertical.
- **3D**: model a half-wing whose root lies on the symmetry plane $y = 0$.
- **Far field**: the outer boundary should be far from the body; it is split automatically into inflow and outflow.
- **Wall**: identified by a function `mesh_inner_bdry_function(x)` that takes facet midpoints of shape
  `(dim, n)` and returns a boolean mask. `dragonfly_sim.utils.mesh_manager_utils` provides
  `airfoil_inner_bdry_function` and `wing_inner_bdry_function` for the meshes used in the examples. Alternatively,
  give the shape parameterization a `block_corner_list`, and the wall facets inside that FFD block are used.

## 2. Flow conditions
```python
boundary_dict = {'inlet':  {'rho': 1.0, 'M': 0.8, 'p': 1.0, 'alpha': np.radians(2)},
                 'outlet': {'p': 1.0}}
```
Only subsonic inflow and outflow are supported. The inflow/outflow split of the far field is frozen at the initial
`alpha`; changing the angle of attack by more than 1 degree from it raises an error, so keep the design variable
bounds within that range.

## 3. Flow model
```python
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.shape_design import FFDShapeParameterization, ControlPointMotions
from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function

# 5 x 3 FFD control points; they move vertically, within +/-0.01
shape = FFDShapeParameterization([5, 3], [2, 2], layers=[ControlPointMotions({1: 0.01})])
model = DG_windtunnel_model(mesh, boundary_dict, shape, aero_center=np.array([0.25, 0.]),
                            mesh_inner_bdry_function=airfoil_inner_bdry_function,
                            poly_order=0, filename_suffix="my_case", asm_overlap=1, ilu_levels=1)
model.set_up_sim()
```
`poly_order=0` is required in this release. The shape parameterization holds the FFD block and the design
variables with their bounds (see [Shape parameterization](shape_parameterization.md)); the model does not depend
on which variables they are. `set_up_sim()` tags the boundaries, builds the weak form and the mesh warper
(`model.mesh_warper`), and sets up the shape parameterization's FFD block (`model.ffd`).

Solver settings are attributes; change them before `set_up_sim()`:

| Attribute | Default | Meaning |
|---|---|---|
| `euler_residual_conv_limit` | `1e-9` | residual norm at which the flow solve has converged |
| `max_newton_iterations` | `100` | total Newton step budget per flow solve |
| `linear_solver_rtol`, `linear_solver_atol` | `1e-20`, `1e-11` | adjoint and tangent linear solves |
| `linear_solver_max_it` | `1000` | iteration limit of those solves |
| `log_cm_alpha` | `True` | print $dc_m/d\alpha$ after each solve (one extra linear solve) |

Constructor options select the flow model and the far field:

| Argument | Default | Meaning |
|---|---|---|
| `model_class`, `model_kwargs` | `CompressibleEulerModel`, `{}` | e.g. `CompressibleRANSModel`, `{'Re': 6e6}` for SA-neg RANS (see [RANS, time integration and far fields](flow_models.md)); the body becomes an adiabatic no-slip wall |
| `farfield` | `"split"` | `"riemann"`: characteristic far field on the whole outer boundary instead of the inflow/outflow split |
| `reduced_order_model` | `None` | a `ReducedOrderModel` to try before every flow solve (see [Reduced-order modeling](reduced_order_modeling.md)) |

RANS solves use pseudo-transient continuation, so `max_newton_iterations` counts pseudo steps (~60 from the free
stream on the coarse NACA0012 C-grid at $M = 0.7$); raise it to a few hundred.

## 4. Postprocessor
```python
from dragonfly_sim.core.postprocessor import DG_postprocessor

post = DG_postprocessor(model.mesh, model.sim_model, model.WALL_TAG, np.array([0.25, 0.]), p_inf_dim=101325.)
```
The aerodynamic centre argument may be left out (quarter chord). `ref_area_dim` and `ref_chord_dim` override the reference area and chord of the coefficients. `p_inf_dim` and
`L_ref_dim` only scale the dimensional forces printed to the console.

## 5. Design variables and model graph
Build the CSDL graph between `recorder.start()` and `recorder.stop()`:
```python
recorder = csdl.Recorder(inline=False)
recorder.start()
# ... construct model and post (steps 3 and 4) ...

params = shape.declare_design_variables()   # every shape variable, with its bounds

alpha = csdl.Variable(name='alpha', value=np.array([np.radians(2)]))
alpha.set_as_design_variable(lower=np.radians(1.75), upper=np.radians(2.25), scaler=1/np.radians(1))

mesh_motion = model.deform_mesh()
u = model.evaluate(mesh_motion, params, alpha)
out = post.evaluate(u, mesh_motion, alpha, params)

out.D.set_as_objective()
out.L.set_as_constraint(lower=0.225)
recorder.stop()
```
`post.evaluate` returns `L`, `D` and `M`, which have derivatives and may be used as objectives and constraints,
and the coefficients `c_l`, `c_d`, `c_m` and `c_m_alpha`, which are for monitoring only. The flattened shape
parameter vector `params` is passed along for bookkeeping; it carries no derivatives.

## 6. Running
```python
sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
sim.run()                          # one analysis
sim.check_totals(step_size=1e-5)   # compare derivatives with finite differences

from modopt import CSDLAlphaProblem, SLSQP
prob = CSDLAlphaProblem(problem_name='shape_opt', simulator=sim)
SLSQP(prob, solver_options={'ftol': 1e-8, 'maxiter': 100}).solve()
```
Each flow solve starts from the previous converged solution.

## Forward analysis (no shape variables)
Leave out the shape parameterization: `set_up_sim()` then builds no FFD block and no mesh warper
(`model.ffd` and `model.mesh_warper` stay `None`), the mesh stays fixed, and the angle of attack is the only
input.
```python
model = DG_windtunnel_model(mesh, boundary_dict, mesh_inner_bdry_function=airfoil_inner_bdry_function,
                            poly_order=0)
model.set_up_sim()

# one flow solve, no CSDL graph: the state, a convergence flag, and the forces
u, converged, forces = model.solve_forward(np.radians(2))
print(forces["c_l"], forces["c_d"], forces["c_m"])

# or a CSDL graph with alpha as its only input
u = model.evaluate(alpha=alpha)
out = model.postprocessor.evaluate(u, alpha=alpha)
```
`DG_postprocessor` needs no FFD definitions either: `post.forces_and_coefficients(u, alpha)` returns `L`, `D`,
`M`, `c_l`, `c_d`, `c_m` (and, for a viscous model, the friction drag `D_friction`, `c_d_friction`) of any state,
outside CSDL, and `post.evaluate(u, alpha=alpha)` leaves out the mesh deformation. `aero_center` defaults to
the quarter chord, $(0.25, 0[, 0])$. See `examples/airfoil_analysis.py`. Time-dependent runs are set up directly
from the model classes, as in `examples/flow_analysis.py`.

## Reduced-order model
Pass a `ReducedOrderModel` to try a POD reduced-order model before every flow solve, in `solve_forward` and in
the CSDL graph alike:
```python
from dragonfly_sim.core.reduced_order_model import ReducedOrderModel

rom = ReducedOrderModel(rb_size=20, eta_threshold=1e-10)
model = DG_windtunnel_model(mesh, boundary_dict, ..., reduced_order_model=rom)
```
Once two converged solutions exist, each evaluation first solves the least-squares Petrov-Galerkin problem in
their POD basis. Its solution is used if $\eta = \|R\|/\|u\| <$ `eta_threshold`; otherwise the full-order
solve runs as usual and its solution is added to the basis. `run_fom_every_evaluation=True` runs the full-order
solve every time, to compare the two. Basis, inner product and stopping rules are described in
[Reduced-order modeling](reduced_order_modeling.md).

## Output files
Written to the working directory, named with `filename_suffix`:
- `FOM_solution_*.bp`, `mesh_deformation_*.bp`: solution and mesh deformation per evaluation, in ADIOS2/VTX
  format (open in ParaView). The solution file holds separate fields: `density`, `momentum` (vector), `energy`
  ($\rho E$), each transported turbulence variable (e.g. `rho_nu_tilde`), `velocity` (vector) and `pressure`.
  With a shape parameterization, the solution is written on the deformed mesh it was solved on, and the mesh
  deformation on the baseline mesh (apply it with *Warp By Vector*); frame $k$ of each file belongs to the same
  evaluation. `write_mesh_deformation = False` leaves out the mesh deformation file, and
  `write_once_per_design = True` writes one frame per design when the optimizer evaluates a design twice;
- `output_meshtags.xdmf`: the boundary tags, to check the inflow/outflow/wall/symmetry split;
- `memory_logs/`: peak memory per MPI rank.

`model.data_store` collects per-evaluation results (with a reduced-order model also its wall times, $\eta$,
coefficients, acceptance, errors against the full-order solve, and the snapshot singular values: the `ROM_*` and
`rom_*` entries);
`model.data_store.write_store_to_numpy_file(save_folder, save_filename)` saves them to an existing folder.

To write other fields, use `dragonfly_sim.utils.filewriter.FileWriter(filename, comm, function_space,
mesh_deformation=False, time_dependent=False)` and call `interpolate_and_write(f, write_counter=...)`, or
`interpolate_and_write(f, time=t)` for a time-dependent writer. With `mesh_deformation=True` each frame is
written on the mesh geometry at the time of the write. Piecewise-constant and continuous fields are gathered to
rank 0 and written as one block, so ParaView shows no cracks at the MPI partition boundaries.
