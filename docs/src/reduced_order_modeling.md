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

### Adjoint snapshots

The reduced adjoint lies in span($S^2 A V$): directions built from state snapshots, which need not hold the
adjoint field. `ReducedOrderModel(adjoint_snapshots=...)` adds the full-order adjoints the optimizer solves, one per
output row at every design the FOM solved and converged, as snapshots (`data_store.adjoint_snapshots` records them):

- `'augment'`: into the state snapshot matrix, scaled to the $W$-norm of the design's state snapshot (adjoints have
  other units) and tagged, so that they are not taken for duplicates of the state and are not centred. The one
  basis $V$ then holds them, for the primal solve and the LSPG adjoint alike. `rb_size` must leave room for the
  states: with `rb_size=5` and two adjoints per design, the ROM was never accepted on the test cases.
- `'separate'`: into a snapshot matrix of their own (inner product $W^{-1}$, the dual of the state's), from which
  every solve builds an adjoint basis $\Psi$ (`adjoint_rb_size`, weighted per design like the state basis). The
  adjoint is then the least-squares solution of the adjoint equations over span$[S^2 A V, \Psi]$,
  $\min \| W^{-1/2} (A^T \psi - g) \|$. That space contains the LSPG adjoint's, so in this norm the adjoint is
  never worse. It is no longer the exact derivative of the ROM's outputs, but closer to the full-order one where
  $\Psi$ holds the adjoint.

- `'concatenate'` (recommended): into a snapshot matrix of their own, as `'separate'`. The basis of every solve is
  then a state basis of $\lceil$`rb_size`/2$\rceil$ vectors, as the POD gives it, followed by an adjoint basis $\Psi$
  of $\lfloor$`rb_size`/2$\rfloor$ orthogonalized against it, both weighted per design: the two blocks are
  $W$-orthonormal and mutually $W$-orthogonal. Slots $\Psi$ cannot fill, e.g. before
  there are adjoint snapshots, go to the state basis. The states keep their own SVD, so the adjoints cannot
  crowd them out as with `'augment'`. The primal solve and the adjoint both use the concatenated $V$.

`adjoint_solve` chooses the reduced adjoint: `'lspg'` (above; the exact derivative of the ROM's outputs),
`'galerkin'` ($\psi = V z$ with $V^T A^T V z = V^T g$, so the adjoint basis is in its trial space, which
span($S^2 A V$) is not) or `'least_squares'`. The defaults are `'least_squares'` for `'separate'` and
`'lspg'` otherwise, which keeps the derivative consistent with the forward (LSPG) solve.

`test_space='hybrid'` (`'concatenate'` only) keeps LSPG for the states and changes the test basis of the forward
solve to $\Phi = [S^2 A V_s, \Psi]$: the state part includes the Jacobian, the adjoint part is the adjoint basis
itself. The reduced equations $\Phi^T R(V a) = 0$ are solved by Newton with $\Phi$ frozen per iteration, as LSPG
does, and their adjoint $(\Phi^T A V)^T \lambda = V^T g$, $\psi = \Phi \lambda$ is the derivative of the forward
solve (checked against finite differences of the ROM's solves) and contains $\Psi$. On the test cases it gave the
most accurate derivatives (table), with the ROM states as accurate as before for Euler and within
$1.5\times10^{-8}$ for RANS; the residual is slightly larger ($\eta$ up to $5\times10^{-9}$ for RANS against
$2\times10^{-10}$), since the forward solve no longer minimizes $\|S R\|$, which matters for a tight
`eta_threshold`.

Why the LSPG adjoint gains little from adjoint snapshots in $V$: it lies in span($S^2 A V$), so an adjoint $\psi$ in
$V$ adds $S^2 A \psi$ to its space, not $\psi$. If instead the full-order adjoint lies in span($S^2 A V$), the LSPG
adjoint equations recover it exactly. That is the case for the preimage $\chi = A^{-1} S^{-2} \psi$
($S^2 A \chi = \psi$), one solve with the forward Jacobian per adjoint at the state it was computed at. An option
that put the preimages into the trial basis in place of the adjoint basis brought the LSPG adjoint to the Galerkin
one's accuracy on the test cases (table), consistent with the forward solve, but it did not extrapolate on a real
shape optimization (below) and was removed.

**Recommended: `adjoint_snapshots='concatenate'`, `test_space='hybrid'`, `hybrid_trial='paired'`.** The table
above is from small cases with one design variable, where every variant looks good. On a transonic RANS shape
optimization (OAT15A, Rizzi grid 2 cut at 200 chords, 15 FFD control points and $\alpha$, three outputs:
examples/airfoil_opt_rans_pod.py) they behave very differently. Errors are relative and in the dual norm
$W^{-1} = D^2$; "space" is the distance of $\psi_\text{FOM}$ to the space the reduced adjoint lies in, the
best any reduced adjoint there can do.

| | space | reduced adjoint | why |
|---|---|---|---|
| `test_space='hybrid'`, `hybrid_trial='adjoint'` | $0.2$–$1\%$ | $1.0$ | $\Phi^T A V$ nearly singular: amplification 40–180 |
| `test_space='lspg'`, adjoints as snapshots | $1.0$ | $1.0$ | the space span($S^2 A V$) does not hold $\psi$ |
| `test_space='lspg'`, preimages $\chi$ in the trial basis (removed) | $1.0$ | $1.0$ | preimages do not extrapolate (below) |
| `test_space='hybrid'`, `hybrid_trial='paired'` | $0.2$–$4\%$ | $4$–$20\%$ | paired: amplification 2–20 |

With the paired hybrid the mesh-motion contributions of the reduced adjoint came within $1$–$6\%$ of the
full-order ones for a design $10^{-2}$ steps from a snapshot design, against $120$–$180\%$ unpaired; the
$\alpha$ contributions, which come from the far-field rows where the adjoint is small, within $8$–$30\%$.

Why. An adjoint $A^{-T} g$ lies mostly along the weakest directions of $A$: $A^T\psi = g$ is small next to
$\psi$. Testing with the adjoint basis $\Psi$ makes the rows $\Psi^T A V = (A^T\Psi)^T V$ small for every
trial direction, and the reduced matrix gets about one near-zero singular value per output (between
dual-orthonormal test and $W$-orthonormal trial bases: $\sim 10^{-1}$ down to three of $10^{-8}$ for three
outputs; a property of the spans, not the basis: `hybrid_state_test='orthonormal'` gave identical results).
The trial direction that pairs best with a test direction $\psi$ maximizes $\psi^T A v / \|v\|_W$:
$v = W^{-1} A^T \psi$. `hybrid_trial='paired'` uses $Z = W^{-1} A_0^T \Psi$, $A_0$ the Jacobian at the solve's
warm start, for the adjoint part of the trial basis; $\Psi^T A Z$ is then the Gram matrix of $A^T\Psi$, and
the smallest singular value rose to $\sim 5\times10^{-5}$. It costs one Jacobian assembly and $k$ products with
$A^T$ per ROM solve, and its adjoint does as well as a minimum-residual adjoint on the same space.

LSPG avoids the pairing problem by construction: its adjoint is the best approximation of $\psi_\text{FOM}$
in its space in the dual norm (with $S A V = Q R$ and $D = S^{-1}$, $D\psi = Q Q^T D\psi_\text{FOM}$; on the
OAT15A case its amplification was exactly 1). But its space is span($S^2 A V$), which holds $\psi$ only if
$V$ holds the preimage $\chi = A^{-1} S^{-2}\psi$. The preimages work at the Jacobian they were computed with
(the table above), not on the OAT15A case. $\chi$ is dominated by $A$'s weakest direction ($\sim 10^3$ times
$\psi$), so $S^2 (A_\text{new} - A_\text{old}) \chi_\text{old}$ is large for a small change of $A$:
$S^2 A_\text{new} \chi_\text{old}$ missed $\psi_\text{old}$ by a factor 5 over a design step of 0.009 (of a
snapshot spacing of 1; shape steps of up to 0.2% chord and $0.05^\circ$ per unit) and by a factor 50 over 0.09.
Two versions were tried: a POD of the $\chi$ in the $W$-norm, which keeps the wrong directions (they all look
like the weakest mode), and the adjoint basis chosen from the $\psi$ in the dual norm with the same combinations
of the $\chi$ in the trial basis, which avoids that but not the drift. Distance of $\psi_\text{FOM}$ to the LSPG
adjoint's space (grid 1; the reduced adjoint's error is the same, LSPG's amplification being 1), against the
space of all earlier preimages untruncated and against the old adjoints themselves:

| step from the nearest snapshot design | 0.1 | 0.01 | 0.001 |
|---|---|---|---|
| preimages, 5 adjoint directions ($W$-norm POD of $\chi$) | $1.0$ | $0.995$ | $0.83$–$0.99$ |
| preimages, 5 adjoint directions (chosen from the $\psi$) | $0.99$–$1.0$ | $0.98$–$0.99$ | $0.90$–$0.99$ |
| span($S^2 A_\text{new}\,[\chi]$), all 15 earlier preimages | $0.79$–$1.0$ | $0.45$–$0.62$ | $0.08$–$0.10$ |
| span of the earlier $\psi$ of one design | $5\%$ | $0.4\%$ | $0.04\%$ |
| paired hybrid: space / reduced adjoint | $0.6$–$0.9\%$ / $4$–$12\%$ | $0.2\%$ / $5\%$ | |

**Dual-corrected LSPG (`adjoint_snapshots='separate'`, `adjoint_solve='dual_corrected'`).** The non-conforming
dual approach of Keil, Mechelli, Ohlberger, Schindler and Volkwein (ESAIM: M2AN, 2021), adapted to LSPG. The
forward solve is plain LSPG on the state basis alone. A dual ROM on the adjoint basis $\Psi$ gives
$\psi_d = \Psi z$, $z$ minimizing $\|W^{-1/2}(A^T \Psi z - g)\|$, and corrects an output to
$f_c = f(u_r) - \psi_d^T R(u_r)$, which leaves only $(\psi_\text{FOM} - \psi_d)^T R(u_r)$ of the error. With the
bases, the weights and $\psi_d$'s dependence on the design frozen and the terms proportional to $R(u_r)$ dropped
(as for the LSPG adjoint), the derivative of $f_c$ uses
$$\psi = \psi_d + S Q R^{-T} V^T (g - A^T \psi_d), \qquad S A V = Q R,$$
the LSPG adjoint of what the dual ROM leaves of the adjoint equations; with $D = S^{-1}$,
$D\psi = D\psi_\text{LSPG} + (I - QQ^T) D\psi_d$. The forward solve does not see $\Psi$. The output correction is
not yet part of the ROM's outputs; `data_store.output_corrections` records it. On the test cases the derivative
matches central differences of the ROM's solves to $2$–$4\times10^{-8}$, and the correction takes the ROM's
output error from $1.7\times10^{-6}$ to $\sim10^{-11}$.

On the OAT15A optimization (Rizzi grid 2, 6 SLSQP iterations, 15 reduced adjoints, the ROM never accepted; the
dual-corrected ROM with a state basis of 10 and the paired hybrid's 5 adjoint vectors):

| | dual-corrected | paired hybrid | LSPG (concatenated, 5 + 5) |
|---|---|---|---|
| reduced adjoint, dual norm: median / max | $6.5\%$ / $28\%$ | $10\%$ / $35\%$ | $1.0$ |
| mesh-motion contributions, median | $6.7\%$ | $12\%$ | $1.0$ |
| $\alpha$ contributions, median | $29\%$ | $47\%$ | $1.07$ |
| forward $\eta$ | $2\times10^{-7}$–$3\times10^{-6}$ | $4\times10^{-6}$–$3\times10^{-4}$ | $2\times10^{-7}$–$3\times10^{-6}$ |
| DEIM / MDEIM forward $\eta$ | up to $10^{-4}$ | up to $5\times10^{-3}$ | up to $4\times10^{-6}$ |

The DEIM and MDEIM adjoints stay plain (hyper-reduced) LSPG, error $1.0$: the dual ROM uses the full $A^T$. The
output correction reduces the ROM's first-order output error to a median of $32\%$ of itself, about as much as
the full-order adjoint does ($47\%$): the limit is the nonlinearity of $R$ between $u_r$ and $u_\text{FOM}$ (the
shock), not the dual ROM.

With `basis_mode='global'` instead of `'local_weighted'` (same case; the state bases hold every state snapshot in
both modes, so only the adjoint basis, 5 of up to 24 adjoint snapshots, differs):

| | dual-corrected, local / global | paired hybrid, local / global |
|---|---|---|
| reduced adjoint, dual norm: median | $6.5\%$ / $17.5\%$ | $10\%$ / $28\%$ |
| distance of $\psi_\text{FOM}$ to the adjoint space, median | $1.2\%$ / $1.7\%$ | $1.0\%$ / $1.7\%$ |
| mesh-motion / $\alpha$ contributions, median | $6.7\%$ / $29\%$ ; $10.5\%$ / $46\%$ | $12\%$ / $47\%$ ; $19\%$ / $99\%$ |
| $c_l$ error, median | $1.7\times10^{-3}$ / $1.7\times10^{-3}$ | $4\times10^{-3}$ / $1.7\times10^{-2}$ |

The weighting toward nearby designs keeps the adjoint basis's 5 vectors on the adjoints that matter: global
bases make both reduced adjoints about $2.5$ times worse, and the paired hybrid's forward solve too (its trial
basis holds the adjoint directions). Most of the dual-corrected adjoint's error is the dual ROM's solve, a
minimum residual, not the space: $4$ (local) to $10$ (global) times the distance to the space.

**First-order model management** (`data_store.first_order_corrections`, a diagnostic): a trust-region method
corrects its model's gradient at the centre $\mu_0$, $\nabla J_r(\mu) + (\nabla J_\text{FOM} - \nabla J_r)(\mu_0)$.
Recorded with $\mu_0$ the previous design (every design has a full-order derivative here), against the
previous design's full-order derivative reused as it is. Medians on the same case, over the 12 (design, output)
pairs from the second rejected design on:

| contributions: mesh motions / $\alpha$ | ROM | ROM, corrected | previous design's FOM derivative |
|---|---|---|---|
| LSPG | $1.0$ / $1.0$ | $7.5\%$ / $20\%$ | $7.5\%$ / $20\%$ |
| paired hybrid, local_weighted | $4.3\%$ / $57\%$ | $6.6\%$ / $71\%$ | |
| paired hybrid, global | $18.5\%$ / $130\%$ | $14.7\%$ / $92\%$ | |
| dual-corrected, local_weighted | $7.4\%$ / $46\%$ | $11.5\%$ / $39\%$ | |
| dual-corrected, global | $14.6\%$ / $58\%$ | $11.0\%$ / $72\%$ | |

The LSPG adjoint contributes nothing (it is nearly orthogonal to the full-order one), so its corrected gradient
is the previous design's, to four digits. For the other ROMs the correction helps where the ROM is poor (global
bases) and hurts where it is good (local_weighted): the previous design's error does not carry over. On these
steps (the first SLSQP iterations) the previous design's full-order derivative is a hard baseline: the
local_weighted ROMs match or beat it in the mesh-motion contributions (in $58$–$67\%$ of the pairs), but not in
$\alpha$ ($46$–$57\%$ against $20\%$), whose contributions come from the far-field rows where the adjoint is small.

With `compute_all_adjoint_errors`, `data_store.adjoint_space_errors` records for every reduced adjoint the
distance of the full-order adjoint to the reduced adjoint's space, to the adjoint basis and to the stored adjoint
snapshots, the reduced adjoint's error, the condition number, the contributions of the best approximation and of
the minimum-residual adjoint on the same space, in the 2-norm and in the dual norm $W^{-1} = D^2$. Use the dual
norm: in the 2-norm a RANS adjoint is all $\tilde\nu$ (its residual scale is ~$10^{-3}$, so its adjoint is
~$10^3$ larger).

On the angle-of-attack test cases (three outputs, three FOM designs, seven designs), the relative errors of the
optimizer's derivatives at the ROM-accepted designs against full-order-only derivatives were:

| | Euler | RANS (Re 200) |
|---|---|---|
| state snapshots only | $2$–$6\times10^{-6}$ | $2$–$5\times10^{-6}$ |
| `'augment'`, `rb_size=12` | $0.5$–$1.3\times10^{-6}$ | $0.8$–$2.2\times10^{-6}$ |
| `'separate'`, `rb_size=5`, `adjoint_rb_size=9` | $0.8$–$4.6\times10^{-7}$ | $2.6$–$4.4\times10^{-6}$ |
| `'concatenate'`, `rb_size=5` (3 + 2), `'lspg'` | $1.4$–$3.7\times10^{-6}$ | $1.4$–$3.7\times10^{-6}$ |
| `'concatenate'`, `rb_size=5` (3 + 2), `'galerkin'` | $1.1$–$1.4\times10^{-8}$ | $2.2$–$5.7\times10^{-6}$ |
| `'concatenate'`, `rb_size=8` (4 + 4), `'galerkin'` | $0.9$–$1.1\times10^{-8}$ | $0.6$–$1.7\times10^{-8}$ |
| `'concatenate'`, `rb_size=8` (4 + 4), `'lspg'`, preimages $\chi = A^{-1} S^{-2} \psi$ (removed) | $0.6$–$1.2\times10^{-8}$ | $1.8$–$2.9\times10^{-8}$ |
| `'concatenate'`, `rb_size=5` (3 + 2), `test_space='hybrid'` | $1.0$–$3.7\times10^{-8}$ | $1.4$–$3.8\times10^{-6}$ |
| `'concatenate'`, `rb_size=8` (4 + 4), `test_space='hybrid'` | $0.5$–$1.4\times10^{-10}$ | $0.8$–$3.8\times10^{-8}$ |
| `'concatenate'`, `rb_size=8` (4 + 4), `'least_squares'` | $1$–$3\times10^{-5}$ | $0.5$–$1\times10^{-3}$ |

The ROM states stayed as accurate as with the state basis alone (relative $L^2$ error ~$3\times10^{-9}$). With
`'galerkin'` the derivatives come down to the level of that state error, once the adjoint basis can hold the
outputs' adjoints: for RANS two adjoint vectors for three outputs were too few. The least-squares adjoint needs
all of them in $\Psi$ (as with `'separate'` and `adjoint_rb_size=9`); with fewer it is worse than the LSPG adjoint.

With `'separate'` (and `adjoint_rb_size=9`) the adjoint vector itself is within $10^{-7}$ (Euler) and $2\times10^{-6}$ (RANS) of the
full-order one, against ~100 % for the other two. For RANS the derivative error stays at the level of the ROM's
state error, which no adjoint can remove. These one-parameter cases have small derivative errors to begin with;
the shape-optimization cases, where the errors are large, are what the options are for.

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
measurements. With the ROM's hybrid test space (`test_space='hybrid'`), DEIM and MDEIM solve the same hybrid
equations, the state part of the test basis from their approximation of $S A V$.

Their adjoints are consistent with their own solves. The residual of those solves is exact, so their solutions
follow the design through the true Jacobian: with the test basis $\tilde\Phi$ frozen, the adjoint is
$(\tilde\Phi^T A V)^T \lambda = V^T g$, $\psi = \tilde\Phi \lambda$, linearized about the method's own solution.
It matches finite differences of the hyper-reduced solves (to $10^{-9}$–$10^{-6}$ on the test cases), and costs
one full Jacobian assembly per design, which the `adjoint_<m>` phase records show. Its derivative contributions
are evaluated at that solution too, against the full-order and ROM adjoints' at the optimizer's state.

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
