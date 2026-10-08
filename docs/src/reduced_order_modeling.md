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
the angle of attack. Only the forward solve is reduced. The adjoint is the full-order adjoint at the accepted
state.

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
  [1e-12, 1e-10], with $g_0$ from the pass's first iteration; when
  $\|\Delta a\| < \max(\texttt{newton\_atol}, \texttt{newton\_rtol} \max(\|a\|, 1))$; or when $\|g\|$ has not
  decreased over `stall_window` [3] iterations (relative `stall_rtol` [1e-10]).
- Up to `max_outer_iterations` [3] passes run, each restarted from the current state. The solve has converged
  when a pass ends with $\|g\| <$ `g_conv_limit` [1e-12].

$\|g\|$ is the LSPG stationarity residual. The projected residual $V^T r$ does not vanish at the LSPG
solution, so it is not tested. Acceptance is decided separately, by $\eta$ against `eta_threshold`.

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
