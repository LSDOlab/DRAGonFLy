# API reference
Auto-generated from the docstrings of the `dragonfly_sim` package.

The main entry points:
- `dragonfly_sim.core.windtunnel_model.DG_windtunnel_model`: flow model (CSDL implicit operation)
- `dragonfly_sim.core.postprocessor.DG_postprocessor`: lift, drag and pitching moment
- `dragonfly_sim.core.meshwarping.IDWarp_jax`: volume mesh deformation
- `dragonfly_sim.core.shape_parameterization`: `WallFFD`, `SectionalShape`, `WallGeometry`
- `dragonfly_sim.utils.ffd_dv_utils`: design-variable construction (`build_cp_motion_dv`, `build_sectional_dv`,
  `ShapeDVSet`)
- `dragonfly_sim.core.Euler_model.CompressibleEulerModel` and `dragonfly_sim.core.mesh_manager.Mesh`: the
  underlying flow discretization and mesh handling

```{toctree}
:maxdepth: 2

autoapi/dragonfly_sim/index
```
