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
$(p', \mathbf{u}, T, \tilde\nu)$ for Spalart-Allmaras (the equations stay in density-weighted conservative
form). It adds the viscous physics through the
host's plug-in points. Attributes and methods it does not define itself are forwarded to `model.flow`. Both
`DG_windtunnel_model` and `DG_postprocessor` therefore treat it exactly like the Euler model.

The discretization is a cell-centred finite-volume scheme.

**Gradient reconstruction.** A piecewise-constant field has no gradient inside a cell, so a gradient $G$ is
reconstructed by Green-Gauss over each cell's facets: facet averages inside, the boundary condition's state on
the boundary. $G$ is the gradient of the unknowns $V$, so $\nabla\mathbf{u}$, $\nabla T$ and $\nabla\tilde\nu$ are
read off it directly. It is linear in the state and is eliminated from the Newton unknowns exactly, so Newton
sees the true Jacobian $\partial R/\partial V = A_{VV} + A_{VG} M^{-1} B_{GV}$.

**Inviscid flux.** HLL with MUSCL-reconstructed facet traces $V_K + G_K\cdot(\mathbf{x}_f - \mathbf{x}_K)$ of the
primitive variables, optionally limited (`limiter="vanalbada"`). Without MUSCL, first-order dissipation on the tangential momentum
exceeds the turbulent viscosity in the boundary layer.

**Viscous flux.** $\mathbf{F}_v(V_f, \nabla V_f)\cdot\mathbf{n}$ through each facet, with the heat flux
$-(k + h_T\mu_t/Pr_t)\nabla T$ and the material's $\mu(T)$ and $k(T)$. $\nabla V_f$ is the average of the two
cells' $G$, with its component along the line between the cell centres replaced by the direct difference. On a
boundary facet, the normal component is replaced by $(V_b - V)/d_n$.

**Turbulence.** The closure's sources use $G$ and the distance to the wall.

**Boundary conditions.** The extra variables take the closure's free-stream values at inflow, the interior
values at outflow and on slip walls, and the closure's wall values on the adiabatic no-slip wall. On a
characteristic far field they take free-stream values where the flow enters and interior values where it leaves.

### Turbulence closures
A closure subclasses `dragonfly_sim.core.turbulence_models.TurbulenceModel`:

| Member | Meaning |
|---|---|
| `n_vars`, `names` | number and names of the transported scalars $\phi_k$ (conserved as $\rho\phi_k$) |
| `needs_wall_distance` | whether the sources use the wall distance |
| `eddy_viscosity(V, th, mu, n_mean)` | $\mu_t$; `th` is the state's `ThermoState` (density, its derivatives) |
| `diffusive_fluxes(V, th, mu, n_mean)` | `f(G)`: one viscous-flux row per scalar, from a gradient $G$ of $V$ |
| `sources(V, th, G, mu, d, n_mean)` | one source term per scalar |
| `freestream_values(fs)`, `wall_values(V, n_mean)` | free-stream and no-slip-wall values of $\phi_k$; `fs` holds $\rho_\infty$, the speed and $\mu_\infty$ in the solver's units |
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
R^*(V) = R(V) + \frac{a_0 U(V) + a_1 U(V^n) + a_2 U(V^{n-1})}{\Delta t}
+ \frac{w_K}{\mathrm{CFL}}\,\Gamma(V_{old})\,(V - V_{old}),
$$

with $w_K = \lambda_K/h_K$ (plus a viscous $2\nu/h^2$) the cell's inverse local time step. The physical term is
the BDF derivative of the conserved state, so the scheme is conservative in time whatever the unknowns.
$\Gamma = \partial U/\partial V$, or with low-Mach preconditioning its Weiss-Smith form (see below), and $w_K$
then uses the preconditioned wave speed. Both time terms live
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

## Fluids and variables

The unknowns are the primitive variables $(p', \mathbf{u}, T, \phi)$ and the fluid is a *material*
(`dragonfly_sim.core.materials`), the only fluid-specific code: four functions $\rho(p', T)$, $h(p', T)$,
$\mu(T)$, $k(T)$ and their constants. `thermo()` derives $\rho_p$, $\rho_T$, $h_p$, $h_T$ and
$1/c^2 = \rho_p + \rho_T(1 - \rho h_p)/(\rho h_T)$ by `ufl.diff`; every flux, boundary condition, the
preconditioning and the artificial viscosity use these and never $\gamma$, $R$ or $c$.

| Material | Far-field dict | Notes |
|---|---|---|
| `IdealGas(gamma=1.4, Pr=0.72, viscosity_law="sutherland")` (default) | `{'rho', 'p', 'M', 'alpha'}` | $\rho = p/(RT)$, $h = c_pT$; in the solver's units $p_{ref} = R = 1/(\gamma M^2)$ |
| `ConstantDensity(rho0=998.2, c=4182, Pr=7)` (water) | `{'U', 'T', 'p', 'alpha'}` | $\rho = \rho_0$, gauge enthalpy $h = cT + p'/\rho_0$, $1/c^2 = 0$: always preconditioned, no shock capturing |

Pass a material as `material=` to `CompressibleEulerModel`, `MUSCLEulerModel` or `CompressibleRANSModel` (through
`model_kwargs` of `DG_windtunnel_model`). Air at low Mach number and water at the same Reynolds number give the
same nondimensional flow to $O(M^2)$ (`tests/test_materials.py`).

## Low-Mach preconditioning

`preconditioning=True` (the default for an incompressible material, off for air) applies the local
Weiss-Smith preconditioning. The reference velocity

$$
\frac{1}{U_r^2} = \max\left(\frac{1}{c^2},\; \frac{1}{\max(|\mathbf{u}|,\ \varepsilon u_\infty,\
\sqrt{|\Delta p'|/\rho},\ \nu/h)^2}\right)
$$

(smooth maxima; the pressure-jump term at facets, the viscous term for RANS) caps itself at $c$, so the
preconditioning switches off wherever the flow is sonic or faster. The floor $\varepsilon$ =
`preconditioning_floor` = 1, i.e. $U_r \ge u_\infty$: smaller floors (0.1, 0.3) slowed the transonic RANS start
from the free stream on the Rizzi grids twofold or stalled it, and $U_r = O(u_\infty)$ keeps the low-Mach scaling.
It changes

- the flux dissipation: wave speeds $u'_n \pm c'$ and $U_m - U_p \to \Gamma\,(V_m - V_p)$, whose mass row carries
  the pressure-jump dissipation $\Delta p'/U_r$ (HLL and LLF; at $U_r = c$ they reduce exactly to the plain fluxes);
- the pseudo-time term, $\Gamma\,\Delta V$.

The physical time derivative is never preconditioned. The steady solution then has the incompressible scaling
($p'$ converges to its $M \to 0$ limit at $O(M^2)$ instead of $O(M)$) and PTC needs the same number of steps at
every Mach number; without it the solve stalls by $M = 0.001$.

With preconditioning the far field is the free stream as a ghost state through the (preconditioned) numerical
flux instead of the Riemann-invariant state: the Riemann invariants belong to the unpreconditioned system, and at
low Mach number they leave the pressure level of a preconditioned solution almost undetermined.

## Artificial viscosity

`shock_capturing=True` (or a dict of options) adds the Barter-Darmofal artificial viscosity of the
project-sandbox Euler code in its p = 0 form (`core/shock_capturing.py`): a viscosity field $\sigma \in [0, 1]$,
an auxiliary unknown solved with the flow, obeys a screened-Poisson equation driven by the normalised pressure
jumps $|\Delta p'|/(\rho c^2)$ through $\tanh(s/\eta)$ (times the Ducros dilatation switch when the model has a
reconstructed gradient). The jumps are those the MUSCL reconstruction leaves between its facet traces: $O(h^2)$ in
smooth flow, $O(\Delta p)$ at a shock. (The jumps of the cell values are $O(h)$ in smooth flow; with them the
sensor fired across the whole suction peak of the OAT15A on the coarse Rizzi grid and cost 12-20% of the lift.)
The viscosity adds $C\lambda\,\sigma\,[\![U]\!]$ to the facet fluxes: a smooth blend towards
first-order dissipation at shocks. The Jacobian, the adjoint and the shape derivatives stay exact. The smoothing
length is per facet (the cell extent across it), so in stretched boundary-layer cells $\sigma$ spreads a few
cells, not $\sqrt{\text{aspect ratio}}$ cells, and the Ducros switch keeps it out of boundary layers.

## Far fields

| Option | Boundary condition |
|---|---|
| `define_farfield_bc(tag)` (`"riemann"`) | The characteristic state from the Riemann invariants on the whole outer boundary: the far field of `DG_windtunnel_model` and the default of `examples/flow_analysis.py`. It takes the free-stream state `{'rho', 'p', 'M', 'alpha'}` only; no inflow/outflow division is made, so the angle of attack can change freely. Outgoing waves leave exactly at normal incidence. The free stream is preserved exactly. Usable steady and unsteady. |
| ghost far field (automatic with low-Mach preconditioning or an incompressible material) | The free stream as the ghost state of the numerical flux; also no inflow/outflow division. |
| `TransverseFarfield` (`"riemann2"`, 2D, unsteady, ideal gas, unpreconditioned) | Adds the transverse terms of the incoming characteristic (Engquist-Majda / Higdon type, $\beta = 1/2$). They are integrated explicitly with the time integrator's BDF coefficients. At p = 0 the tangential derivatives come from a Green-Gauss gradient. |

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
