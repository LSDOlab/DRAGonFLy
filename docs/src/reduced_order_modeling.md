---
title: Reduced-order modeling (POD)
---

A design optimization or a parameter sweep solves the same flow problem many times at nearby inputs.
`DG_windtunnel_model` can try a reduced-order model (ROM) before every flow solve. The ROM is a
least-squares Petrov-Galerkin (LSPG) solve {cite:p}`carlberg2011efficient` on a proper orthogonal decomposition
(POD) basis built from the converged solutions computed so far. If the ROM solution is accurate enough, it replaces the full-order (FOM)
solve. Otherwise the FOM runs as usual, and its solution is added to the basis.

```python
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.reduced_order_model import ReducedOrderModel

rom = ReducedOrderModel(rb_size=20, eta_threshold=1e-10)
model = DG_windtunnel_model(mesh, boundary_dict, ..., reduced_order_model=rom)
model.set_up_sim()
for alpha in alphas:
    u, converged, forces = model.solve_forward(alpha)
```

The same applies inside a CSDL graph (`evaluate`), where the snapshot parameters are the shape parameters and
the angle of attack. Where the ROM solution is accepted, the derivatives use the adjoint of the reduced
equations (see Derivatives below). Elsewhere they use the full-order adjoint.

## One evaluation

1. Once at least `min_snapshots` (default 2) snapshots exist, build the POD basis $V$ for the current
   parameter vector, and project the warm start (the previous accepted solution) onto it.
2. Solve the LSPG problem in the span of $V$ (below), and compute the full steady residual of its solution.
3. Accept the ROM solution if $\eta = \|R\|/\|u\| <$ `eta_threshold`.
4. Otherwise, run the FOM from the warm start exactly as without a ROM. If it converges, its solution becomes a
   snapshot at the current parameter vector.

With `run_fom_every_evaluation=True` the FOM also runs when the ROM is accepted, for benchmarking. The output
is still the ROM solution. The data store records the ROM wall time, $\eta$, the ROM coefficients, whether
each ROM solve was accepted, the ROM-minus-FOM coefficient errors, and the snapshot singular values
(`ROM_*` and `rom_*` entries of `DataStore`).

## Snapshots and basis

`core/pod.py:SnapshotMatrix` keeps the snapshot matrix only as an incrementally updated SVD. It is distributed
over MPI with PETSc, and `max_rank` (default `3 * rb_size`) caps the stored rank. A snapshot whose parameter
vector repeats an earlier one adds no column. A snapshot that lies in the span of the stored ones, up to a
relative residual `qr_rank_tol` (default $10^{-10}$, in the inner product below), adds no basis direction; it
still counts in the singular values and in the weights of `local_weighted`. `PODBasis` builds the basis in one of two ways:

- `basis_mode="global"`: the leading `rb_size` left-singular vectors of all snapshots.
- `basis_mode="local_weighted"`: the leading left-singular vectors of the snapshots weighted by their distance
  to the current parameter vector. The weights form a partition of unity. The default weight function,
  `"Cubic"`, is a cubic with compact support, which vanishes at its support radius. By default the support
  covers the `n_nonzero_weights` nearest snapshots (default `rb_size`). With `cubic_cutoff=c` in $[0, 1)$ the
  radius is

  $$
  r = \max\left(c\,d_\text{min} + (1 - c)\,d_\text{max},\; r_\text{rb}\right),
  $$

  where $r_\text{rb}$ is the smallest radius that leaves the `rb_size` nearest snapshots a nonzero weight: the
  next distance beyond the `rb_size`-th smallest, and at least a margin of $10^{-3}(d_\text{max} - d_\text{min})$
  beyond it. The basis therefore always has `rb_size` columns (as long as the snapshots span that many
  directions), and a smaller $c$ lets more snapshots contribute to it. Only snapshots with a basis column count
  towards `rb_size`, not repeated parameter vectors. `parameter_scales` weights the parameters in the distance.

The `local_weighted` basis can be centred on an offset $c$ in the span of the snapshots, so that the reduced
state is $u = c + V a$ (`ReducedOrderModel(centering=...)`):

- `centering="warm_start"`, the default for `local_weighted`: $c$ is the warm start of the solve, projected onto
  the snapshot span. In a transonic sweep the snapshot mean smears the shock over its range of positions; the
  warm start has a sharp shock near the right place, so the truncated basis spends its modes on deviations from
  an admissible state.
- `centering="mean"`: the weighted mean of the snapshots, with `mean_weighting="snapshot_weights"` (the
  distance weights, default) or `"uniform"`.
- `centering=None`: no offset. This is the only choice for `basis_mode="global"`.

The snapshots exist only as their SVD $S = U\Sigma V_s^T$, so the weighted snapshots are $S D = U K$ with the
small factor $K = \Sigma V_s^T D$, $D$ the diagonal of snapshot weights, and an offset in the snapshot span is
$c = U g$. Subtracting it from every weighted snapshot is the rank-one update $K - g\,\mathbf{1}^T D$. For the
mean, $g = \Sigma V_s^T x$ with the (bounded) mean weights $x$. For the warm start $u_0$, $g = U^T W u_0$, so $c$
is the $W$-projection of $u_0$ onto the snapshot span; this projection is the only full-order work. Its
snapshot least-squares fit $x$ (the minimum-norm solution of $\Sigma V_s^T x = g$) is logged but not used: $x$
grows like $1/\sigma_\text{min}$ when $u_0$ is not a well-scaled combination of the snapshots, and centring
through it would lose about $\varepsilon\,\mathrm{cond}(\Sigma)$ in relative accuracy. When $c$ is an affine
combination of the snapshots (the mean; a warm start that is a snapshot or an earlier ROM solution), the
centred snapshots lose a direction, and the basis stops at their numerical rank. A warm start further than
$10^{-6}$ (relative, in the $W$ norm) from the snapshot span is reported; only its projection enters the
offset. The LSPG solve starts at $a = 0$, that is at the offset: exactly at the warm start whenever it lies in
the snapshot span, as a snapshot or an earlier ROM solution does.

### Output-sensitivity weights in an optimization

Shape variables and the angle of attack differ in units and in how strongly the outputs respond to them, so the
plain Euclidean distance is dominated by whichever parameter has the largest range. Inside an optimization the
adjoint already gives the total derivatives of the objective and constraints, $\mathrm{d}F/\mathrm{d}p_j$.
`utils/dv_sensitivity.py` reads them without extra solves and `ReducedOrderModel.set_distance_sensitivities`
turns them into one weight per parameter,

$$
s_j = \lVert [\partial F_1/\partial p_j, \dots, \partial F_m/\partial p_j] \rVert_2,
$$

normalized to mean 1 and multiplied into the distance (and with `parameter_scales`). `distance_weighting` selects
`'total_derivative'` (default; each output divided by its own magnitude), `'total_derivative_raw'` or `None`.
The weights come from the design variables of the CSDL problem, which must be the snapshot parameters (the shape
variables, then alpha).

```python
from dragonfly_sim.utils.dv_sensitivity import attach_total_derivative_probe

sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
attach_total_derivative_probe(sim, printer=PETSc.Sys.Print,
                              on_norms=model.reduced_order_model.set_distance_sensitivities)
```

Without the probe there is no information and the distance is left unweighted, also for
`basis_mode="local_weighted"`. The same holds before the first gradient evaluation, and when the weights do not fit
the parameter vector. The weights come from the previous design point, so they lag the basis by one evaluation.

## Inner product

By default (`inner_product="euclidean"`, `state_scaling="freestream"`), snapshots and basis use the
state-scaled Euclidean inner product

$$
(u, v)_W = \sum_i \frac{u_i v_i}{s_{k(i)}^2}, \qquad W = D^{-2},
$$

with $s_k$ a typical magnitude of state component $k$: the free-stream $\rho$, $\rho U$ and
$p/(\gamma-1) + \rho U^2/2$, and for RANS the turbulence model's scale of its variable. Without $D$, the
Spalart-Allmaras variable $\rho\tilde\nu$ ($\sim 10^{-4}$ at $Re = 6\times10^6$) would carry no weight next to
$\rho E \sim 2$, and a truncated basis would drop it.

`inner_product="l2"` adds the DG0 mass matrix, $W = M D^{-2}$, i.e. $\int_\Omega \sum_k u_k v_k / s_k^2\,
d\mathbf{x}$. On NACA0012 angle-of-attack sweeps (Euler at $M = 0.8$, SA RANS at $M = 0.7$) it was 10 to 100
times less accurate than the Euclidean weight. The mass matrix gives the large far-field cells, where the free
stream rotates with $\alpha$, most of the weight, at the expense of the region near the body.
`state_scaling=None` drops $D$. $W$ is assembled once, on the undeformed mesh, because the incremental SVD
needs a fixed inner product.

## LSPG solve

With $u = V a$ and $V^T W V = I$, the ROM solves the least-squares Petrov-Galerkin problem
$\min_a \|S R(Va)\|$ by Gauss-Newton:

$$
(S A V)^T (S A V)\, \Delta a = -(S A V)^T S\, r, \qquad a \leftarrow a + \theta\, \Delta a,
$$

where $A$ is the exact Jacobian and $S = D^{-1}$ ($D^{-1} M^{-1/2}$ on the current mesh with
`inner_product="l2"`) is the residual norm matching the inner product. $\theta$ is the Newton solver's
positivity-preserving step length, halved until $\tfrac12\|S r\|^2$ decreases sufficiently (Armijo,
`line_search=True`); if no halving does, the positivity-limited step is taken. The reduced system is small and
is solved on every rank.

The stopping rules are those of the original POD benchmark (`newton_*`, `stall_*`, `max_outer_iterations`,
`g_conv_limit`; defaults in brackets):

- Each pass runs at most `newton_max_it` [15] iterations. It ends when
  $\|g\| = \|(SAV)^T S r\|$ falls below $\max(\texttt{newton\_atol}, \texttt{newton\_rtol}\,\|g_0\|)$
  [1e-12, 1e-10], with $g_0$ from the first iteration of the first pass (a restart does not reset it); when
  $\|\Delta a\| < \max(\texttt{newton\_atol}, \texttt{newton\_rtol} \max(\|a\|, 1))$; or when $\|g\|$ has not
  decreased over `stall_window` [3] iterations (relative `stall_rtol` [1e-10]).
- Up to `max_outer_iterations` [3] passes run, each restarted from the current state. The solve has converged
  when a pass ends with $\|g\| < \max(\texttt{newton\_atol}, \texttt{g\_conv\_limit}\,\|g_0\|)$
  [1e-10], the same $g_0$; `g_conv_limit` = 0 never accepts. Keep `g_conv_limit` $\ge$ `newton_rtol`, or the
  passes stop before the acceptance target.

$\|g\|$ is the LSPG stationarity residual. The projected residual $V^T r$ does not vanish at the LSPG
solution, so it is not tested. Acceptance is decided separately, by $\eta$ against `eta_threshold`.

## Derivatives

The LSPG solution solves the $k$ projected residual equations

$$
\Phi^T R(V a; x) = 0, \qquad \Phi = S^T S A V,
$$

the stationarity condition of $\min_a \|S R(Va; x)\|$, with $x$ the shape parameters and the angle of attack.
When this solution is accepted, the adjoint of an output $f$ is the adjoint of these equations, with $V$, $S$
and $\Phi$ frozen at the solution:

$$
(S A V)^T (S A V)\, \lambda = V^T g, \qquad \psi = \Phi \lambda = S^T S A V \lambda, \qquad
\frac{df}{dx} = \frac{\partial f}{\partial x} - \psi^T \frac{\partial R}{\partial x},
$$

with $g = (\partial f/\partial u)^T$. The reduced matrix is the Gauss-Newton matrix of the forward solve. The
lifted adjoint $\psi$ takes the place of the full-order adjoint $A^{-T} g$, and the rest of the derivative
($\partial R/\partial \alpha$ and $\partial R/\partial x_\text{mesh}$, then the mesh warping) is unchanged.
$\psi$ is the Petrov-Galerkin solution of $A^T \psi = g$ in the span of $\Phi$, tested with $V$:
$V^T (A^T \psi - g) = 0$. A Galerkin ROM ($\Phi = V$) would give $V^T A^T V \lambda = V^T g$ and $\psi = V \lambda$
instead. `DG_windtunnel_model.apply_inverse_jacobian` computes $\psi = S Q y$ from a distributed thin QR
$S A V = Q R$ and $R^T y = V^T g$, so it never forms the normal matrix, whose condition number is squared.
This costs one Jacobian assembly and $k$ products with it, and no Krylov solve. CSDL passes one right-hand
side per output, and every output after the first reuses the factorization. `DataStore.adjoint_solves`
records which adjoint each derivative used.

This is the derivative of the reduced equations with the basis held fixed. Two kinds of terms are left out:

- The exact linearization of the stationarity condition also has second-derivative terms of $R$ contracted
  with $z = S^T S R(u_r)$, the weighted residual at the ROM solution:
  $V^T \big(\sum_i z_i \nabla^2_{uu} R_i\big) V$ in the reduced matrix, and
  $\partial_x \big(z^T A(u; x) V\lambda\big)$ in the derivative. With `inner_product="l2"`, $S$ also depends
  on the mesh. These terms are linear in the residual of the accepted solution, so they are of relative size
  $O(\eta)$.
- For `basis_mode="local_weighted"` the basis depends on $x$ through the snapshot weights, and that dependence
  is not differentiated. Neither is the offset's dependence on $x$ (through the weights for the mean) or on the
  evaluation history (the warm start). For `"global"` the basis is fixed.

CSDL re-runs the model before every derivative evaluation, so each design is evaluated at least twice. A
re-evaluation of the design that the full-order model solved last skips the ROM. The full-order solve restarts
from its own converged solution, so it takes no Newton steps, and the derivative is the full-order adjoint. By
then that solution is a snapshot, and the ROM would reproduce it and be accepted. A design the ROM solved keeps
the ROM and its reduced adjoint.

### Adjoint errors

The reduced adjoint is kept (`ReducedOrderModel.adjoint`, with `adjoint.accepted`) for every ROM solve that
built a basis, accepted or not. Only an accepted solve's adjoint is used for the derivative. With
`DG_windtunnel_model(compute_all_adjoint_errors=True)` the error of the reduced adjoint against the full-order
one is also recorded, for every output row of the derivative, in `data_store.adjoint_errors`. The adjoint that
goes back to the optimizer is the same with and without the flag.

- For an accepted solve the reduced adjoint is the optimizer's, and a full-order adjoint is solved at the same
  state and right-hand side as the reference.
- For a rejected solve the full-order adjoint is the optimizer's, and the reduced adjoint of the rejected solve
  is evaluated at that full-order state: it needs the Jacobian there and the basis and weights of the rejected
  solve. The basis predates the design's own snapshot, which the full-order solve adds afterwards, so the error
  is that of a basis that has not seen the design. The adjoint of the rejected solve survives the re-run of the
  design that CSDL does before the derivative.

Both are therefore errors of the adjoint approximation alone: state and right-hand side are the same for the two
adjoints. The distance between the vectors overstates what reaches the derivative: the reduced adjoint
lives in span($S^2 A V$), a few directions built from state snapshots, which cannot hold the full-order adjoint
field (on the test cases the best vector in that span is ~98 % away from it), while its contribution to the
derivative, $(\partial R/\partial x)^T \psi$, is accurate to ~$10^{-6}$. `data_store.adjoint_derivative_errors`
records that contribution's error for every input of the model (`alpha`, and `mesh_node_motions` in the mesh
coordinate space, before the mesh warping and shape parameterization), next to each `adjoint_errors` entry; with
`profile_phases=True` its cost is the `adjoint_contributions` record. The extra solve costs about one adjoint per output row, so the flag is a diagnostic. With
`profile_phases=True` the phases of these reference solves are recorded as `adjoint_fom_reference` and
`adjoint_rom_reference`, apart from the `adjoint_fom` and `adjoint_rom` records of the adjoint the optimizer used.

## Hyper-reduction (comparison)

The LSPG solve above assembles the full residual and the full Jacobian $A$ at every iteration, so its cost per
iteration scales with the mesh. Hyper-reduction approximates those operators from a few sampled rows or
entries. `DG_windtunnel_model(hyper_reduction=...)` runs DEIM {cite:p}`chaturantabut2010nonlinear` and/or MDEIM
{cite:p}`negri2015efficient` **beside** the ROM, to measure how they would perform in an optimization. They solve
every design the ROM solves, and the adjoint of every derivative, and are recorded against the ROM and the FOM.
The optimizer's solutions and adjoints are those without them.

```python
from dragonfly_sim.core.hyper_reduction import HyperReductionComparison

rom = ReducedOrderModel(rb_size=10, basis_mode='local_weighted')
hr = HyperReductionComparison(methods=('deim', 'mdeim'), deim_n_modes=100, mdeim_n_modes=10)
model = DG_windtunnel_model(mesh, boundary_dict, ..., reduced_order_model=rom, hyper_reduction=hr)
```

**What is approximated.** The residual stays exact; residual DEIM was measured in the project sandbox and does
not work on these problems. The residual hardly compresses, and interpolating it moved the solution by
5–15 %. Only $S A V$, the weighted Jacobian on the POD basis, is approximated:

- **DEIM**: a POD basis $U$ of Jacobian actions $S A v_j$ along the POD basis, with greedy points $P$ (rows).
  $S A V \approx U C$, with $C = (P^T U)^+ P^T S A V$, which needs only the rows $P$ of $A$.
- **MDEIM**: a POD basis $\{\Phi_l\}$ of the scaled Jacobians $\mathrm{vec}(S A)$, with points $P$ (entries).
  $S A \approx \sum_l c_l \Phi_l$, with $c = (P^T \Phi)^+ P^T \mathrm{vec}(S A)$. The products $\Phi_l V$ and
  their Gram matrices are formed once per design, so the reduced system costs no product with $A$.

The snapshots are scaled by the LSPG residual weight $S$ (rows). Without it, the basis would neglect the rows of
small components, such as the SA variable. The sampling stays exact, since $(SA)_{ij} = S_i A_{ij}$.

$S A V$ is also LSPG's test basis. The hyper-reduced solve therefore stops where $\widetilde{(SAV)}^T S r = 0$,
which differs from the ROM's solution by as much as the approximation tilts the test space. On the test cases
MDEIM tracks the ROM to $10^{-9}$–$10^{-12}$ and DEIM is far less accurate, in line with the sandbox's
measurements. The adjoint is the LSPG adjoint above, with $S A V$ replaced by its approximation at the
optimizer's state.

**Sampled assembly.** A sampled row belongs to a cell. Its row is exact when that cell and every facet of it are
integrated, including facets shared with unsampled cells and facets owned by another rank.
`FormManager.restrict` binds the full-order kernels over such an entity set; `entity_set_for_cells` builds it.
With a reconstructed gradient (MUSCL, RANS), $A = A_{UU} + A_{UG} M^{-1} B_{GU}$. The sampled rows then read the
gradient of the sampled cells and of their facet neighbours. $A_{UU}$ and $A_{UG}$ are integrated over the
sampled cells $C_1$, and $B_{GU}$ over $C_2 = C_1$ plus one facet ring (`cell_neighbours`), so the product's
rows on $C_1$ are exact too.

**Training.** During every FOM solve of a new design, the steady Jacobian at up to
`*_max_snapshots_per_solve` Newton iterates is harvested (`CompressibleEulerModel.record_fom_jacobians`), along
with the Jacobian at the converged state. With pseudo-time continuation that costs one extra assembly per
harvested iterate. The snapshots are added after the comparison of their own design, so that comparison stays
out of sample, and only where the ROM keeps a snapshot of the design.

**Bases follow the ROM.** The DEIM and MDEIM bases use the ROM's `basis_mode`. For a `local_weighted` ROM, every
snapshot of a design carries the weight that design has in the ROM's basis
(`ReducedOrderModel.basis_settings`).

**Records** (`data_store`, per method `<M>` = DEIM/MDEIM): `<M>_walltime` (online and offline), the relative
residual, iterations, coefficients and outputs (`<M>_outputs`, with
`output_errors(name, kind='MDEIM', reference='ROM')`), the coefficient and L2 state errors against the FOM and
the ROM, the operator sizes and conditioning (`<m>_operator_info`), the adjoint errors (`<m>_adjoint_errors`)
and those of the adjoints' contributions to the derivative (`<m>_adjoint_derivative_errors`), and the snapshot
counts. With `profile_phases=True` the phases are recorded as `<m>_offline`, `<m>` and
`adjoint_<m>`; the sampled assembly appears as `assemble:jacobian_sampled`. `validate=True` adds, at the
optimizer's solution, the sampled assembly's exactness and the Jacobian approximation error, at the cost of
full assemblies. `examples/hyperreduction_comparison.py` summarizes a saved DataStore, and
`examples/wing_opt_rans_pod.py --hyper-reduction both` runs the comparison in the RANS wing optimization.

Note that the `forward` phase record of an evaluation then includes the comparison's solves and harvesting.

## What to expect

The ROM pays off when the solutions over the parameter range lie close to a low-dimensional linear subspace
(a small Kolmogorov $n$-width). Shocks that move with the parameters are the hard case, the more so at p = 0,
where they are smeared over a few cells. Angle-of-attack sweeps over $1.4^\circ$ to $2.8^\circ$ on the NACA0012
meshes in `meshes/`, with the default settings (`rb_size=20`), 10 evaluations and the full-order solve run every
time for comparison, gave:

| Case | $\eta$ of the ROM, 2 → 9 snapshots | $c_l$ error | wall time, ROM / full order |
|---|---|---|---|
| Euler, $M = 0.8$, 240,752 dofs | $2.4\times10^{-5}$ → $2.0\times10^{-6}$ | $2\times10^{-2}$ → $10^{-4}$ to $10^{-3}$ | 1–5 s / 26–84 s |
| SA-neg RANS, $M = 0.7$, $Re = 6\times10^6$, 47,400 dofs | $1.9\times10^{-5}$ → $1.1\times10^{-7}$ | $10^{-2}$ → $10^{-5}$ to $10^{-3}$ | 1–4 s / 17–40 s |

All ROM solves converged, but none met the default `eta_threshold` of $10^{-10}$. On these cases every
evaluation fell back to the full-order solve, and the ROM added a few seconds per evaluation. Raise
`eta_threshold` only where the coefficient errors at that $\eta$ are acceptable; `run_fom_every_evaluation=True`
measures them.

## Flow models

The ROM uses only the host interface of `CompressibleEulerModel`, which `CompressibleRANSModel` forwards:

| | |
|---|---|
| `u_vec` | the state, whose layout the basis rows follow |
| `problem.F`, `problem.J` | residual and exact Jacobian; for RANS the condensed-gradient problem, which updates the reconstructed gradient |
| `steady_residual()` | switches the pseudo-time (PTC) and physical time terms off for the reduced solve |
| `positivity_step_length(x, dx)` | the Newton step limiter's step length |
| `state_scales()` | the magnitudes $s_k$ (a new flow model with extra variables supplies them through the `extra_scales` plug-in) |

There is therefore no model-specific ROM code. The reduced solve is plain damped Gauss-Newton, also for the
RANS model, whose full-order steady solve uses pseudo-transient continuation. This is a sound choice because
the ROM starts from the projected previous solution.
