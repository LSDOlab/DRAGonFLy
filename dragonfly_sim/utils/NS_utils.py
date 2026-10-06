"""
Navier-Stokes / RANS building blocks on top of Euler_utils (used by
core/RANS_model.py and by CompressibleEulerModel whenever the state carries
extra transported scalars).

The state is the conservative mean flow U_m = (rho, rho*u, rho*E), n_mean =
2 + dim components, optionally followed by conservative transported scalars
(rho*phi_k, e.g. rho*nu_tilde for Spalart-Allmaras). Euler_utils only ever
sees the mean-flow slice; the scalars are convected passively here, with the
mean flow's wave speeds.

Every flux is written against an explicit n_mean, and reduces to the
corresponding Euler_utils expression when there are no extra components.

Nondimensionalization is the solver's: rho_inf = p_inf = 1, reference length
1, so T is only ever needed as the ratio (p/rho)/(p_inf/rho_inf). The heat
flux is written in terms of p/rho, which removes the gas constant:

    q = -kappa grad T = -(gamma/(gamma-1)) (mu/Pr + mu_t/Pr_t) grad(p/rho)
"""
import ufl

from dragonfly_sim.utils.Euler_utils import (primitives, flux,
                               signed_wave_speeds, llf_penalty_term)


def mean_flow(U, n_mean):
    """The mean-flow slice of the state expression U."""
    if U.ufl_shape[0] == n_mean:
        return U
    return ufl.as_vector([U[i] for i in range(n_mean)])


def extra_components(U, n_mean):
    """The transported scalars rho*phi_k of U, as a list."""
    return [U[i] for i in range(n_mean, U.ufl_shape[0])]


def concat_state(U_mean, extras):
    """A full state from a mean-flow vector and a list of scalars."""
    if len(extras) == 0:
        return U_mean
    return ufl.as_vector([U_mean[i] for i in range(U_mean.ufl_shape[0])] + list(extras))


# ----------------------------------------------------------------------
# Inviscid fluxes
# ----------------------------------------------------------------------
def extended_flux(U, gamma, n_mean):
    """The inviscid flux tensor, (n_state, dim): Euler rows, then rho*phi*u."""
    if U.ufl_shape[0] == n_mean:
        return flux(U, gamma)
    Um = mean_flow(U, n_mean)
    _, u, _ = primitives(Um)
    Fm = flux(Um, gamma)
    rows = [Fm[i, :] for i in range(n_mean)]
    rows += [s * u for s in extra_components(U, n_mean)]
    return ufl.as_tensor(rows)


def extended_normal_flux(U, n, gamma, n_mean):
    return ufl.dot(extended_flux(U, gamma, n_mean), n)


def extended_hll_flux(U_p, U_m, n, gamma, n_mean):
    """
    hll_flux on the full state. The wave speeds come from the mean flow
    alone, and the same single-intermediate-state formula is applied to
    every component, so the mean-flow rows are exactly hll_flux and each
    scalar is upwinded with the fan of the flow carrying it.
    """
    sl_p, sr_p = signed_wave_speeds(mean_flow(U_p, n_mean), n, gamma)
    sl_m, sr_m = signed_wave_speeds(mean_flow(U_m, n_mean), n, gamma)

    SL_minus = ufl.min_value(ufl.min_value(sl_p, sl_m), 0.0)
    SR_plus = ufl.max_value(ufl.max_value(sr_p, sr_m), 0.0)
    denom = ufl.max_value(SR_plus - SL_minus, 1e-8)

    return (
        SR_plus * extended_normal_flux(U_p, n, gamma, n_mean)
        - SL_minus * extended_normal_flux(U_m, n, gamma, n_mean)
        + SL_minus * SR_plus * (U_m - U_p)
    ) / denom


def extended_boundary_flux(U, U_ext, n, gamma, n_mean):
    """Euler_utils.boundary_flux on the full state."""
    alpha = llf_penalty_term(mean_flow(U, n_mean), mean_flow(U_ext, n_mean), n, gamma)
    return extended_normal_flux(U, n, gamma, n_mean) + alpha * (U - U_ext)


# ----------------------------------------------------------------------
# Gradients of the primitive quantities
# ----------------------------------------------------------------------
def primitive_gradients(U, G, gamma, n_mean):
    """
    grad(u), grad(p/rho) and grad(phi_k) from the state U and a tensor G
    standing in for grad(U), shape (n_state, dim).

    G is an argument rather than ufl.grad(U) because the interior-penalty
    terms evaluate the viscous flux with the jump tensor [U] (x) n in
    place of the gradient. Everything returned is linear in G.
    """
    d = n_mean - 2
    rho = U[0]
    u = ufl.as_vector([U[1 + i] / rho for i in range(d)])
    rhoE = U[1 + d]
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * ufl.dot(u, u))

    grad_rho = ufl.as_vector([G[0, j] for j in range(d)])
    grad_m = [ufl.as_vector([G[1 + i, j] for j in range(d)]) for i in range(d)]
    grad_rhoE = ufl.as_vector([G[1 + d, j] for j in range(d)])

    grad_u = ufl.as_tensor([(grad_m[i] - u[i] * grad_rho) / rho for i in range(d)])
    grad_p = (gamma - 1.0) * (grad_rhoE - sum(u[i] * grad_m[i] for i in range(d))
                              + 0.5 * ufl.dot(u, u) * grad_rho)
    grad_p_over_rho = (grad_p - (p / rho) * grad_rho) / rho

    grad_phi = []
    for k in range(n_mean, U.ufl_shape[0]):
        phi = U[k] / rho
        grad_s = ufl.as_vector([G[k, j] for j in range(d)])
        grad_phi.append((grad_s - phi * grad_rho) / rho)

    return grad_u, grad_p_over_rho, grad_phi, grad_rho


def vorticity_magnitude(grad_u, eps=1e-16):
    """
    sqrt(2 W:W), W the antisymmetric part of grad(u). Regularized by eps
    inside the root so its derivative stays finite where the flow is
    irrotational (the whole free stream).
    """
    W = 0.5 * (grad_u - grad_u.T)
    return ufl.sqrt(2.0 * ufl.inner(W, W) + eps)


# ----------------------------------------------------------------------
# Viscous flux
# ----------------------------------------------------------------------
def viscous_stress(grad_u, mu_eff):
    d = grad_u.ufl_shape[0]
    return mu_eff * (grad_u + grad_u.T - (2.0 / 3.0) * ufl.tr(grad_u) * ufl.Identity(d))


def viscous_flux(U, G, gamma, n_mean, mu, mu_t, Pr, Pr_t,
                 extra_fluxes=None, heat_flux=True):
    """
    The viscous flux tensor F_v(U, G), shape (n_state, dim), linear in G:

        mass       0
        momentum   tau = (mu + mu_t)(grad u + grad u^T - 2/3 div u I)
        energy     tau u - q
        scalars    extra_fluxes(grad_phi, grad_rho), one row per scalar

    `mu` and `mu_t` are expressions of the state (not of G). heat_flux=False
    drops the conduction term, which is how the adiabatic wall imposes
    q.n = 0.
    """
    grad_u, grad_T, grad_phi, grad_rho = primitive_gradients(U, G, gamma, n_mean)
    d = n_mean - 2
    rho = U[0]
    u = ufl.as_vector([U[1 + i] / rho for i in range(d)])

    tau = viscous_stress(grad_u, mu + mu_t)
    energy = ufl.dot(tau, u)
    if heat_flux:
        energy = energy + gamma / (gamma - 1.0) * (mu / Pr + mu_t / Pr_t) * grad_T

    rows = [ufl.as_vector([0.0] * d)]
    rows += [ufl.as_vector([tau[i, j] for j in range(d)]) for i in range(d)]
    rows += [energy]
    if U.ufl_shape[0] > n_mean:
        rows += list(extra_fluxes(grad_phi, grad_rho))
    return ufl.as_tensor(rows)


def sutherland_viscosity(U, gamma, n_mean, mu_inf, T_ratio_inf, S_ratio):
    """
    Sutherland's law in the solver's units:

        mu/mu_inf = theta^(3/2) (1 + S) / (theta + S),
        theta = T/T_inf = (p/rho) / (p_inf/rho_inf),  S = S_dim / T_inf_dim

    `T_ratio_inf` is p_inf/rho_inf. theta is floored so a transiently
    negative pressure cannot hand theta^(3/2) a negative base.
    """
    Um = mean_flow(U, n_mean)
    rho, u, E = primitives(Um)
    p = (gamma - 1.0) * (E - 0.5 * rho * ufl.dot(u, u))
    theta = ufl.max_value((p / rho) / T_ratio_inf, 1e-8)
    return mu_inf * theta ** 1.5 * (1.0 + S_ratio) / (theta + S_ratio)
