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
Requirements:
- **Orientation**: the freestream flows in $+x$. In 2D, y is vertical; in 3D, y is spanwise and z vertical.
- **3D**: model a half-wing whose root lies on the symmetry plane $y = 0$.
- **Far field**: the outer boundary should be far from the body; it is split automatically into inflow and outflow.
- **Wall**: identified by a function `mesh_inner_bdry_function(x)` that takes facet midpoints of shape
  `(dim, n)` and returns a boolean mask. `dragonfly_sim.utils.mesh_manager_utils` provides
  `airfoil_inner_bdry_function` and `wing_inner_bdry_function` for the meshes used in the examples. Alternatively,
  pass `ffd_block_corner_list`, and the wall facets inside that FFD block are used.

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
from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function
from dragonfly_sim.utils.ffd_dv_utils import cp_dv_directions

dv_spec = {1: 0.01}   # control points move vertically, within +/-0.01
model = DG_windtunnel_model(mesh, boundary_dict, ffd_shape=[5, 3], aero_center=np.array([0.25, 0.]),
                            mesh_inner_bdry_function=airfoil_inner_bdry_function,
                            cp_coord_opt_idxs=cp_dv_directions(dv_spec),
                            poly_order=0, ffd_degree=[2, 2],
                            filename_suffix="my_case", asm_overlap=1, ilu_levels=1)
model.set_up_sim()
```
`poly_order=0` is required in this release. `cp_coord_opt_idxs` lists the spatial directions in which control
points may move (see [Shape parameterization](shape_parameterization.md)). `set_up_sim()` tags the
boundaries, builds the weak form, the FFD block (`model.ffd`) and the mesh warper (`model.mesh_warper`).

Solver settings are attributes; change them before `set_up_sim()`:

| Attribute | Default | Meaning |
|---|---|---|
| `euler_residual_conv_limit` | `1e-9` | residual norm at which the flow solve has converged |
| `max_newton_iterations` | `100` | total Newton step budget per flow solve |
| `linear_solver_rtol`, `linear_solver_atol` | `1e-20`, `1e-11` | adjoint and tangent linear solves |
| `linear_solver_max_it` | `1000` | iteration limit of those solves |
| `log_cm_alpha` | `True` | print $dc_m/d\alpha$ after each solve (one extra linear solve) |

## 4. Postprocessor
```python
from dragonfly_sim.core.postprocessor import DG_postprocessor

post = DG_postprocessor(model.mesh, model.sim_model, model.WALL_TAG, np.array([0.25, 0.]), p_inf_dim=101325.)
```
`ref_area_dim` and `ref_chord_dim` override the reference area and chord of the coefficients. `p_inf_dim` and
`L_ref_dim` only scale the dimensional forces printed to the console.

## 5. Design variables and model graph
Build the CSDL graph between `recorder.start()` and `recorder.stop()`:
```python
from dragonfly_sim.utils.ffd_dv_utils import ShapeDVSet, build_cp_motion_dv

recorder = csdl.Recorder(inline=False)
recorder.start()
# ... construct model and post (steps 3 and 4) ...

shape_dvs = ShapeDVSet()
cp_motions = shape_dvs.add('cp_motions', build_cp_motion_dv(model.ffd.baseline_coefficients, dv_spec))
coefficients = model.ffd.apply_cp_motions(model.ffd.baseline_variable(), cp_motions,
                                          cp_dv_directions(dv_spec))

alpha = csdl.Variable(name='alpha', value=np.array([np.radians(2)]))
alpha.set_as_design_variable(lower=np.radians(1.75), upper=np.radians(2.25), scaler=1/np.radians(1))

params = shape_dvs.flat_variable()
mesh_motion = model.mesh_warper.evaluate(model.ffd.wall_displacement(coefficients), params)
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

## Output files
Written to the working directory, named with `filename_suffix`:
- `FOM_solution_*.bp`, `FOM_pressure_*.bp`, `mesh_deformation_*.bp`: solution, pressure and mesh deformation per
  evaluation, in ADIOS2/VTX format (open in ParaView);
- `output_meshtags.xdmf`: the boundary tags, to check the inflow/outflow/wall/symmetry split;
- `memory_logs/`: peak memory per MPI rank.

`model.data_store` collects per-evaluation results;
`model.data_store.write_store_to_numpy_file(save_folder, save_filename)` saves them to an existing folder.
