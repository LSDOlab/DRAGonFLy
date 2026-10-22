---
title: Background
---

This page summarizes the formulation and the numerical methods of the default, steady Euler model. The RANS
model, time integration and the characteristic far fields build on it and are described in
[RANS, time integration and far fields](flow_models.md). For how to use them, see the
[user guide](user_guide.md).

## Governing equations
The flow is governed by the steady compressible Euler equations in conservation form. Their unknowns are the
primitive variables $V = (p', \mathbf{u}, T)$, with $p' = p - p_{ref}$ the gauge pressure; the conserved state is
a function of them:

$$
\nabla \cdot \mathbf{F}(U(V)) = 0, \qquad
U = \begin{pmatrix} \rho \\ \rho\mathbf{u} \\ \rho H - p' \end{pmatrix}, \qquad
\mathbf{F} = \begin{pmatrix} \rho\mathbf{u} \\ \rho\mathbf{u}\otimes\mathbf{u} + p'\mathbf{I} \\ \rho H\,\mathbf{u} \end{pmatrix},
$$

with $H = h + \tfrac{1}{2}|\mathbf{u}|^2$. The fluid enters only through its *material*: $\rho(p', T)$, $h(p', T)$,
$\mu(T)$ and $k(T)$ (an ideal gas, $\rho = p/(RT)$ and $h = c_p T$, by default; see
[fluids and low-Mach flow](flow_models.md#fluids-and-variables)). The residual is the conservative one, so shocks
keep their jump conditions; the primitive unknowns stay well defined at any Mach number, also for a
constant-density liquid where pressure is not a function of $(\rho, \rho\mathbf{u}, \rho E)$.

All quantities are scaled with the free stream: $\rho_\infty = |\mathbf{u}_\infty| = T_\infty = 1$, pressure in
units of $\rho_\infty u_\infty^2$ (so $C_p = 2p'$) and lengths in mesh units. For an ideal gas the reference
pressure is then $p_{ref} = 1/(\gamma M_\infty^2)$, which enters only the equation of state.

The freestream flows in the $+x$ direction at angle of attack $\alpha$: its direction is
$(\cos\alpha, \sin\alpha)$ in 2D (x chordwise, y vertical) and $(\cos\alpha, 0, \sin\alpha)$ in 3D
(x chordwise, y spanwise, z vertical). For air the freestream state is set by the user's $\rho_\infty$,
$p_\infty$ and Mach number $M_\infty$ (in any consistent units: only $M_\infty$ survives the scaling).

## Discretization
The unknowns are approximated by piecewise-constant (p=0) DG functions. For every test function $v$ the residual is

$$
R(V_h; v) = -\sum_K \int_K \nabla v \cdot \mathbf{F}(U(V_h))\,dx
+ \sum_{f \in \mathcal{F}_{int}} \int_f \hat{H}(V_h^+, V_h^-, \mathbf{n}^+)\,[\![v]\!]\,ds
+ \sum_{f \in \mathcal{F}_{bdry}} \int_f \hat{H}_b(V_h, V_b, \mathbf{n})\,v\,ds = 0 ,
$$

with the energy equation's test function weighted by $1/(h_T T_{ref})$, which keeps the residual rows of one size
at every Mach number. At p=0 the volume term vanishes and the scheme is a cell-centered finite-volume method.
The interior numerical flux $\hat{H}$ is HLL {cite:p}`harten1983upstream` by default; HLLC
{cite:p}`toro1994restoration` and local Lax-Friedrichs are available (`flux_function` of
`CompressibleEulerModel`). Every flux is evaluated on the conserved state and physical flux of its traces, so it
is the same discrete scheme as with conservative unknowns. The integrals are assembled with FEniCSx, and the
Jacobian $\partial R/\partial V$ is obtained by automatic differentiation of the weak form.

## Boundary conditions
Boundary conditions are imposed weakly. The boundary flux is the normal flux of the interior state plus a local
Lax-Friedrichs penalty toward a boundary state $V_b$:
$\hat{H}_b = \mathbf{F}(U)\cdot\mathbf{n} + \lambda\,(U - U(V_b))$.

| Boundary | $V_b$ |
|---|---|
| Far field | characteristic state from the Riemann invariants of the interior and the freestream (see [far fields](flow_models.md)) |
| Slip wall | interior state with the normal velocity reflected, $\mathbf{u}_b = \mathbf{u} - 2(\mathbf{u}\cdot\mathbf{n})\mathbf{n}$ |
| Symmetry plane (3D, $y=0$) | as the slip wall |

The whole outer boundary is one far field; no inflow/outflow division is made, so the angle of attack can change freely.

Viscous (RANS) models, pseudo-transient continuation and unsteady time stepping, and the characteristic far
field are described in [RANS, time integration and far fields](flow_models.md).

## Nonlinear and linear solvers
$R(V) = 0$ is solved with Newton's method (PETSc SNES, full steps). Each update $\delta V$ is scaled by a
step length $\theta \le 1$ that keeps the absolute pressure and the temperature above $10^{-4}$ times their
free-stream values in every cell (the material's linear positivity constraints), and changes neither by more
than 50% (`max_relative_change`: an update of the primitive unknowns misses the quadratic kinetic-energy term an
update of $\rho E$ carries, which overheats cells in the large first steps of an impulsive start): $\theta$ starts at
$\min\left(1, (\|\delta V\| / \max(\|V\|, 1))^{-1/2}\right)$ and is halved until the updated state is
admissible on all MPI ranks. The solve stops when the residual norm drops below `euler_residual_conv_limit`.

Each Newton system is solved inexactly with GMRES (relative tolerance $10^{-3}$), preconditioned with additive
Schwarz (overlap `asm_overlap`) and ILU(`ilu_levels`) on the subdomains. The adjoint and tangent systems use the
same preconditioner with tight tolerances.

Optionally, a reduced-order model replaces the full-order solve when it is accurate enough: a least-squares
Petrov-Galerkin solve {cite:p}`carlberg2011efficient` in a POD basis of earlier solutions, see
[Reduced-order modeling](reduced_order_modeling.md).

## Shape parameterization
The wall is embedded in a B-spline free-form deformation (FFD) block {cite:p}`sederberg1986free`, built with
lsdo_geo. Wall node positions are a linear B-spline map of the FFD control points, so design variables that move
control points, either individually or through sectional operations, deform the wall smoothly. See
[Shape parameterization](shape_parameterization.md).

## Mesh deformation
The volume mesh follows the wall with IDWarp {cite:p}`luke2012fast`: each volume node moves by an
inverse-distance-weighted average of the surface node displacements and rotations. The far-field
boundary is included as a fixed surface, so the far field stays in place. The weighted sums are evaluated with
a kd-tree (the `idwarp_jax` package) and pruned with a relative tolerance `err_tol` (default $10^{-6}$).

## Outputs
The pressure force on the wall is $\mathbf{F}_p = \int_{wall} p'\,\mathbf{n}\,ds$ (the reference pressure
integrates to zero over a closed body, and over a half-model only adds a spurious roll and yaw moment). Drag $D$ and lift $L$ are its
components along and normal to the freestream, and each moment $M = \mathbf{d}\cdot(\mathbf{r}\times\mathbf{F}_p)$ is taken about a user-given point along a
user-given direction `moments=[(name, point, direction), ...]` (the pitching moment about a point, positive
nose-up, is direction `"pitch"`; the root bending moment of a half-wing is direction $(1, 0, 0)$ about a point
on the root). The coefficients divide these by $\tfrac{1}{2}\rho_\infty|\mathbf{u}_\infty|^2 S_{ref}$ (and a
reference length for the moments, the reference chord by default), with $S_{ref} = 1$ in 2D and the planform area of the half-wing in 3D, unless set by
the user.

## Derivatives
CSDL composes the model as

$$
\text{design variables} \rightarrow \text{FFD coefficients} \rightarrow \text{wall displacement}
\rightarrow \text{volume mesh motion} \rightarrow V \;(R(V; \mathbf{x}_{mesh}, \alpha) = 0) \rightarrow L, D, M
$$

and computes total derivatives in reverse mode. The flow solve is an implicit operation: for an output $f$ it
solves the adjoint system $(\partial R/\partial V)^T \psi = (\partial f/\partial V)^T$ and propagates
$-\psi^T \partial R/\partial \mathbf{x}_{mesh}$ and $-\psi^T \partial R/\partial \alpha$ upstream through the
mesh deformation and the FFD. Derivatives are provided for $L$, $D$ and $M$; the coefficients are monitoring
outputs only.

## Bibliography

```{bibliography} references.bib
```
