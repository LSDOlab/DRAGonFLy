---
title: Background
---

This page summarizes the formulation and the numerical methods. For how to use them, see the
[user guide](user_guide.md).

## Governing equations
The flow is governed by the steady compressible Euler equations for the conservative state
$U = (\rho, \rho\mathbf{u}, \rho E)$:

$$
\nabla \cdot \mathbf{F}(U) = 0, \qquad
\mathbf{F}(U) = \begin{pmatrix} \rho\mathbf{u} \\ \rho\mathbf{u}\otimes\mathbf{u} + p\mathbf{I} \\ (\rho E + p)\,\mathbf{u} \end{pmatrix},
\qquad p = (\gamma - 1)\left(\rho E - \tfrac{1}{2}\rho|\mathbf{u}|^2\right).
$$

The freestream flows in the $+x$ direction at angle of attack $\alpha$: its direction is
$(\cos\alpha, \sin\alpha)$ in 2D (x chordwise, y vertical) and $(\cos\alpha, 0, \sin\alpha)$ in 3D
(x chordwise, y spanwise, z vertical). The freestream state is set by the user's $\rho_\infty$, $p_\infty$ and
Mach number $M_\infty$.

## Discretization
The state is approximated by piecewise-constant (p=0) DG functions. For every test function $v$ the residual is

$$
R(U_h; v) = -\sum_K \int_K \nabla v \cdot \mathbf{F}(U_h)\,dx
+ \sum_{f \in \mathcal{F}_{int}} \int_f \hat{H}(U_h^+, U_h^-, \mathbf{n}^+)\,[\![v]\!]\,ds
+ \sum_{f \in \mathcal{F}_{bdry}} \int_f \hat{H}_b(U_h, U_b, \mathbf{n})\,v\,ds = 0 .
$$

At p=0 the volume term vanishes and the scheme is a cell-centered finite-volume method.
The interior numerical flux $\hat{H}$ is HLL {cite:p}`harten1983upstream` by default; HLLC
{cite:p}`toro1994restoration` and local Lax-Friedrichs are available (`flux_function` of
`CompressibleEulerModel`). The integrals are assembled with FEniCSx, and the Jacobian
$\partial R/\partial U$ is obtained by automatic differentiation of the weak form.

## Boundary conditions
Boundary conditions are imposed weakly. The boundary flux is the normal flux of the interior state plus a local
Lax-Friedrichs penalty toward a boundary state $U_b$:
$\hat{H}_b = \mathbf{F}(U)\cdot\mathbf{n} + \lambda\,(U - U_b)$.

| Boundary | $U_b$ |
|---|---|
| Subsonic inflow | $\rho_\infty$ and freestream velocity prescribed, pressure from the interior |
| Subsonic outflow | $p_\infty$ prescribed, density and velocity from the interior |
| Slip wall | interior state with the normal velocity reflected, $\mathbf{u}_b = \mathbf{u} - 2(\mathbf{u}\cdot\mathbf{n})\mathbf{n}$ |
| Symmetry plane (3D, $y=0$) | as the slip wall |

The far-field boundary is split into inflow and outflow by a plane through the middle of the domain,
normal to the freestream direction at the initial angle of attack.

Viscous (RANS) models, pseudo-transient continuation and unsteady time stepping, and the characteristic far
field are described in [RANS, time integration and far fields](flow_models.md).

## Nonlinear and linear solvers
$R(U) = 0$ is solved with Newton's method (PETSc SNES, full steps). Each update $\delta U$ is scaled by a
step length $\theta \le 1$ that keeps density and pressure above a floor ($10^{-4}$) in every cell: $\theta$
starts at $\min\left(1, (\|\delta U\| / \max(\|U\|, 1))^{-1/2}\right)$ and is halved until the updated state is
admissible on all MPI ranks. The solve stops when the residual norm drops below `euler_residual_conv_limit`.

Each Newton system is solved inexactly with GMRES (relative tolerance $10^{-3}$), preconditioned with additive
Schwarz (overlap `asm_overlap`) and ILU(`ilu_levels`) on the subdomains. The adjoint and tangent systems use the
same preconditioner with tight tolerances.

## Shape parameterization
The wall is embedded in a B-spline free-form deformation (FFD) block {cite:p}`sederberg1986free`, built with
lsdo_geo. Wall node positions are a linear B-spline map of the FFD control points, so design variables that move
control points, either individually or through sectional operations, deform the wall smoothly. See
[Shape parameterization](shape_parameterization.md).

## Mesh deformation
The volume mesh follows the wall with IDWarp {cite:p}`luke2012fast`: each volume node moves by an
inverse-distance-weighted average of the surface node displacements and rotations. The inflow and outflow
boundaries are included as fixed surfaces, so the far field stays in place. The weighted sums are evaluated with
a kd-tree (the `idwarp_jax` package) and pruned with a relative tolerance `err_tol` (default $10^{-6}$).

## Outputs
The pressure force on the wall is $\mathbf{F}_p = \int_{wall} p\,\mathbf{n}\,ds$. Drag $D$ and lift $L$ are its
components along and normal to the freestream, and the pitching moment $M$ is taken about `aero_center`,
positive nose-up. The coefficients divide these by $\tfrac{1}{2}\rho_\infty|\mathbf{u}_\infty|^2 S_{ref}$ (and the
reference chord for $c_m$), with $S_{ref} = 1$ in 2D and the planform area of the half-wing in 3D, unless set by
the user.

## Derivatives
CSDL composes the model as

$$
\text{design variables} \rightarrow \text{FFD coefficients} \rightarrow \text{wall displacement}
\rightarrow \text{volume mesh motion} \rightarrow U \;(R(U; \mathbf{x}_{mesh}, \alpha) = 0) \rightarrow L, D, M
$$

and computes total derivatives in reverse mode. The flow solve is an implicit operation: for an output $f$ it
solves the adjoint system $(\partial R/\partial U)^T \psi = (\partial f/\partial U)^T$ and propagates
$-\psi^T \partial R/\partial \mathbf{x}_{mesh}$ and $-\psi^T \partial R/\partial \alpha$ upstream through the
mesh deformation and the FFD. Derivatives are provided for $L$, $D$ and $M$; the coefficients are monitoring
outputs only.

## Bibliography

```{bibliography} references.bib
```
