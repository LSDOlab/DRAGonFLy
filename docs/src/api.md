# API reference
Auto-generated from the docstrings of the `dragonfly_sim` package.

The main entry points:
- `dragonfly_sim.core.windtunnel_model.DG_windtunnel_model`: flow model (CSDL implicit operation)
- `dragonfly_sim.core.postprocessor.DG_postprocessor`: lift, drag and pitching moment
- `dragonfly_sim.core.meshwarping.IDWarp_jax`: volume mesh deformation
- `dragonfly_sim.core.shape_design`: the shape design variables (`FFDShapeParameterization` and its layers
  `ControlPointMotions`, `SectionalVariables`, `WingShape`)
- `dragonfly_sim.core.shape_parameterization`: `WallFFD`, `SectionalShape`, `WallGeometry`
- `dragonfly_sim.utils.ffd_dv_utils`: design-variable construction (`build_cp_motion_dv`, `build_sectional_dv`,
  `build_planform_dv`, `ShapeDVSet`)
- `dragonfly_sim.core.Euler_model.CompressibleEulerModel` and `dragonfly_sim.core.mesh_manager.Mesh`: the
  underlying flow discretization and mesh handling
- `dragonfly_sim.core.RANS_model.CompressibleRANSModel` and `dragonfly_sim.core.turbulence_models`: RANS and
  the turbulence closures (see [RANS, time integration and far fields](flow_models.md))
- `dragonfly_sim.core.time_integration`: `TimeIntegrator`, `CFLController`, `StepSizeController`
- `dragonfly_sim.core.farfield.TransverseFarfield`: the transverse (riemann2) far field
- `dragonfly_sim.core.reduced_order_model`: `ReducedOrderModel`, `LSPGSolver` and `dragonfly_sim.core.pod`:
  `SnapshotMatrix`, `PODBasis` (see [Reduced-order modeling](reduced_order_modeling.md))
- `dragonfly_sim.utils.mesh_io_utils`: mesh input from XDMF or structured CGNS grids (`load_dolfinx_mesh`,
  `cgns_to_dolfinx_mesh`, `read_cgns_planar`)
- `dragonfly_sim.utils.filewriter.FileWriter`: VTX output for ParaView
- `dragonfly_sim.utils.checkpoint`: partition-independent checkpoints (`save_checkpoint`, `load_checkpoint`)

```{toctree}
:maxdepth: 2

autoapi/dragonfly_sim/index
```
