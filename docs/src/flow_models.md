---
title: RANS, time integration and far fields
---

The package's flow models share one host, `CompressibleEulerModel`. Everything numerical lives there:

- function spaces and boundary conditions;
- the Newton solver and positivity limiter;
- pseudo-time and physical time stepping;
- the derivative assembly used by the adjoint.

Other models plug their physics into it. `CompressibleRANSModel` does this for Reynolds-averaged
Navier-Stokes.

## RANS at p = 0

`CompressibleRANSModel(mesh, Re, turbulence_model=SpalartAllmarasNeg())` owns a `CompressibleEulerModel`
(`model.flow`) whose state carries the closure's transported scalars after the mean flow, e.g.
$(\rho, \rho\mathbf{u}, \rho E, \rho\tilde\nu)$ for Spalart-Allmaras. It adds the viscous physics through the
host's plug-in points. Attributes and methods it does not define itself are forwarded to `model.flow`. Both
`DG_windtunnel_model` and `DG_postprocessor` therefore treat it exactly like the Euler model.

The discretization is a cell-centred finite-volume scheme.

**Gradient reconstruction.** A piecewise-constant field has no gradient inside a cell, so a gradient $G$ is
reconstructed by Green-Gauss over each cell's facets: facet averages inside, the boundary condition's state on
the boundary. $G$ is linear in the state and is eliminated from the Newton unknowns exactly, so Newton sees the
true Jacobian $\partial R/\partial U = A_{UU} + A_{UG} M^{-1} B_{GU}$.

**Inviscid flux.** HLL with MUSCL-reconstructed facet traces $U_K + G_K\cdot(\mathbf{x}_f - \mathbf{x}_K)$,
optionally limited (`limiter="vanalbada"`). Without MUSCL, first-order dissipation on the tangential momentum
exceeds the turbulent viscosity in the boundary layer.

**Viscous flux.** $\mathbf{F}_v(U_f, \nabla U_f)\cdot\mathbf{n}$ through each facet. $\nabla U_f$ is the
average of the two cells' $G$, with its component along the line between the cell centres replaced by the
direct difference. On a boundary facet, the normal component is replaced by $(U_b - U)/d_n$.

**Turbulence.** The closure's sources use $G$ and the distance to the wall.

**Boundary conditions.** The extra variables take the closure's free-stream values at inflow, the interior
values at outflow and on slip walls, and the closure's wall values on the adiabatic no-slip wall. On a
characteristic far field they take free-stream values where the flow enters and interior values where it leaves.

### Turbulence closures
A closure subclasses `dragonfly_sim.core.turbulence_models.TurbulenceModel`:

| Member | Meaning |
|---|---|
| `n_vars`, `names` | number and names of the transported scalars $\rho\phi_k$ |
| `needs_wall_distance` | whether the sources use the wall distance |
| `eddy_viscosity(U, mu, n_mean)` | $\mu_t$ |
| `diffusive_fluxes(U, mu, n_mean)` | `f(grad_phi, grad_rho)`: one viscous-flux row per scalar |
| `sources(U, grad_U, mu, d, n_mean, gamma)` | one source term per scalar |
| `freestream_values(fs)`, `wall_values(U, n_mean)` | free-stream and no-slip-wall values; `fs` holds $\rho_\infty$, $p_\infty$, $c_\infty$, $M_\infty$, speed, $\mu_\infty$ |
| `limiter_scales(fs)` | magnitude per scalar for the MUSCL limiter (default $10^3\times$ free stream) |

`Laminar` (no scalars: laminar Navier-Stokes) and `SpalartAllmarasNeg` (SA-neg, Allmaras, Johnson and Spalart
2012) are provided. Convection, the state size, boundary values, the Jacobian and the adjoint follow from the
interface, so a k-$\omega$ model is a new subclass with `n_vars = 2`.

### Linear solver
`jacobian_mode` sets the Newton linear solve:

| Mode | Behaviour |
|---|---|
| `"compact_pc"` (default) | FGMRES on the exact condensed Jacobian, preconditioned with ASM/ILU of the compact part $A_{UU}$ (gradients frozen), as finite-volume codes precondition with their first-order Jacobian. On a 181k-hex wing it converges with about 15 Krylov iterations per step, where exact subdomain factorizations run out of memory. |
| `"exact"` | ASM/ILU of the full condensed Jacobian. |
| `"compact"` | Approximate Newton on $A_{UU}$ alone. |

`direct_linear_solver = True` uses MUMPS on the exact Jacobian instead.

### Derivatives
The reconstructed gradient, the cell centres and the wall distance are *derived fields*
(`dragonfly_sim.utils.derived_fields`): functions of the state, the angle of attack and the mesh coordinates
that the residual reads. Every derivative the adjoint needs includes their chain-rule terms:

- $\partial R/\partial U$
- $\partial R/\partial\alpha$
- $\partial R/\partial\mathbf{x}_{mesh}$, applied matrix-free
- the force functionals' state and mesh derivatives

The wall distance has an analytic mesh derivative, taken with respect to the node and the vertices of its
nearest wall facet. The derived fields follow the mesh through every node motion.

## Time integration

One residual serves steady and unsteady solves:

$$
R^*(U) = R(U) + \frac{a_0 U + a_1 U^n + a_2 U^{n-1}}{\Delta t}
+ \frac{w_K}{\mathrm{CFL}}\,(U - U_{old}),
$$

with $w_K = \lambda_K/h_K$ (plus a viscous $2\nu/h^2$) the cell's inverse local time step. Both time terms live
only in the Newton forms (`newton_forms()`), so `model.F` stays the steady residual that the adjoint
differentiates. A model that uses neither builds neither, and the plain Newton solve is unchanged.

| Mode | How to select | What it does |
|---|---|---|
| Steady PTC | `model.use_ptc = True` before `set_up_solver()` | Physical term off. One Newton step per pseudo step; the CFL law adapts the step from the steady residual. `solve_system()` runs it. |
| Unsteady BDF | `integrator = model.enable_time_stepping()` | BDF1 start, then variable-step BDF2. Newton per step to the residual criterion; a time-step ramp and halving on failure (`StepSizeController`). |
| Dual time | `model.enable_time_stepping(dual_time=True)` | As unsteady, with PTC on $R^*$ inside every physical step. |

The CFL law (`CFLController`) is switched evolution relaxation with these safeguards:

- **Growth caps.** Growth per step is capped.
- **Rejection.** A step that raises $|R|$ tenfold is undone.
- **Minimum growth on progress.** CFL grows by at least a minimum factor after a step that makes progress.
- **Limited steps.** A step the positivity limiter scales below $\theta = 0.1$ cuts CFL, down to $10^{-3}$, below
  the usual floor of 1.
- **Flat residual.** Inside a $-0.1\%$ to $+5\%$ band, CFL grows gently on clean steps.

The last two keep transonic RANS from freezing in the positivity limiter, and keep slowly creeping residuals from
pinning CFL. The RANS model uses PTC by default (CFL$_0$ = 5).

Checkpoints (`dragonfly_sim.utils.checkpoint`) are keyed by input-mesh cell index, so a run can restart on any
number of ranks.

## Far fields

| Option | Boundary condition |
|---|---|
| Inflow/outflow split | Subsonic inflow and outflow states, split by a plane normal to the free stream. The default, and what `DG_windtunnel_model(farfield="split")` uses. Fixed states reflect outgoing acoustic waves. |
| `define_farfield_bc(tag)` (`"riemann"`) | The characteristic state from the Riemann invariants on the whole outer boundary. Outgoing waves leave exactly at normal incidence. The free stream is preserved exactly. Usable steady and unsteady, and with the windtunnel (`farfield="riemann"`). |
| `TransverseFarfield` (`"riemann2"`, 2D, unsteady) | Adds the transverse terms of the incoming characteristic (Engquist-Majda / Higdon type, $\beta = 1/2$). They are integrated explicitly with the time integrator's BDF coefficients. At p = 0 the tangential derivatives come from a Green-Gauss gradient. |

On the NACA 0012 quadrilateral mesh (far field 50 chords out, Euler, $M = 0.6$, p = 0), `riemann2` cuts the
c_l oscillation from the first returning reflections about fivefold compared with `riemann`. It also adds a
small steady offset, about $10^{-3}$ in c_l by $t = 150$, from its relaxation term. Prefer `riemann` for runs
that are averaged over long times or run to a steady state.

## Running

`examples/flow_analysis.py` runs steady or unsteady Euler, laminar or SA-neg cases from a mesh file, with any of
the far fields, history and wall output, and checkpoint/restart. It builds the model directly from the package's
classes, with no FFD block, mesh warping or CSDL graph: its `build_model` tags the boundaries, builds the
measures and function spaces, defines the boundary conditions and the weak form, in the order
`DG_windtunnel_model.set_up_sim` uses. Copy it as the starting point for other forward analyses.
`examples/rans_airfoil_opt.py` checks the RANS adjoint totals through CSDL.
