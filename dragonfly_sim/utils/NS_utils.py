"""
Navier-Stokes / RANS building blocks (used by core/RANS_model.py and the
turbulence closures).

The state is V = (p', u, T, phi): the primitive mean flow, n_mean = 2 + dim
components, optionally followed by transported scalars phi_k (nu_tilde for
Spalart-Allmaras). Gradients are gradients of V -- a reconstructed G, or the
jump tensor of an interior-penalty term -- so grad u, grad T and grad phi
are read off directly.
"""
import ufl


def mean_flow(U, n_mean):
    """The mean-flow slice of the state expression U."""
    if U.ufl_shape[0] == n_mean:
        return U
    return ufl.as_vector([U[i] for i in range(n_mean)])


def extra_components(U, n_mean):
    """The transported scalars phi_k of U, as a list."""
    return [U[i] for i in range(n_mean, U.ufl_shape[0])]


def concat_state(U_mean, extras):
    """A full state from a mean-flow vector and a list of scalars."""
    if len(extras) == 0:
        return U_mean
    return ufl.as_vector([U_mean[i] for i in range(U_mean.ufl_shape[0])] + list(extras))


def gradient_rows(G, rows):
    """The rows `rows` of a gradient tensor G, as a tensor (or a vector for one int)."""
    d = G.ufl_shape[1]
    if isinstance(rows, int):
        return ufl.as_vector([G[rows, j] for j in range(d)])
    return ufl.as_tensor([[G[i, j] for j in range(d)] for i in rows])


def velocity_gradient(G, n_mean):
    """grad u (d x d, grad(u)[i, j] = du_i/dx_j) from a gradient G of V."""
    return gradient_rows(G, range(1, n_mean - 1))


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


def viscous_flux(th, G, n_mean, mu, mu_t, k, Pr_t, extra_fluxes=None, heat_flux=True):
    """
    The viscous flux tensor F_v(V, G), shape (n_state, dim), linear in G:

        mass       0
        momentum   tau = (mu + mu_t)(grad u + grad u^T - 2/3 div u I)
        energy     tau u - q,   q = -(k + h_T mu_t / Pr_t) grad T
        scalars    extra_fluxes(G), one row per scalar

    `th` is the ThermoState of the state the coefficients are evaluated at;
    `mu`, `mu_t` and the conductivity `k` are expressions of it (not of G).
    heat_flux=False drops the conduction term, which is how the adiabatic
    wall imposes q.n = 0.
    """
    d = n_mean - 2
    tau = viscous_stress(velocity_gradient(G, n_mean), mu + mu_t)
    energy = ufl.dot(tau, th.u)
    if heat_flux:
        energy = energy + (k + th.h_T * mu_t / Pr_t) * gradient_rows(G, n_mean - 1)

    rows = [ufl.as_vector([0.0] * d)]
    rows += [ufl.as_vector([tau[i, j] for j in range(d)]) for i in range(d)]
    rows += [energy]
    if G.ufl_shape[0] > n_mean:
        rows += list(extra_fluxes(G))
    return ufl.as_tensor(rows)
