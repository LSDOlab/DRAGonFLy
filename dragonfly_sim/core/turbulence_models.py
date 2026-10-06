"""
Turbulence closures for CompressibleRANSModel.

A model contributes n_vars conservative transported scalars rho*phi_k. They
are appended after the mean flow in the state vector and solved fully
coupled with it. The RANS model asks the closure for the following, all UFL
expressions of the state U (mean flow in U[:n_mean], scalars after it):

    eddy_viscosity(U, mu, n_mean)                  mu_t
    diffusive_fluxes(U, mu, n_mean)                -> f(grad_phi, grad_rho), one
                                                   viscous-flux row per scalar
    sources(U, grad_U, mu, d, n_mean, gamma)       one source per scalar (right-hand side)
    freestream_values(fs)                          conservative rho*phi_inf per scalar
    wall_values(U, n_mean)                         conservative values on a no-slip wall
    limiter_scales(fs)                             a magnitude per scalar, for the MUSCL
                                                   limiter's smoothing parameter

`fs` is a dict with the free stream: rho, p, c, M, speed (M c) and mu (a
dolfinx Constant: mu_inf follows set_mach). Convection needs nothing from
the closure: every scalar is carried with the mean flow, rho*phi*u
(NS_utils.extended_flux), and upwinded with the mean flow's wave speeds. A
closure whose sources need the distance to the wall sets
needs_wall_distance.

Laminar is the zero-equation case (pure Navier-Stokes). Adding a closure,
k-omega SST say, is a new subclass with n_vars = 2 and names
("rho_k", "rho_omega"). Nothing in CompressibleRANSModel changes: the state
size, boundary values, limiter scales, the Jacobian and the adjoint all
follow from this interface.
"""
import ufl

from dragonfly_sim.utils.NS_utils import primitive_gradients, vorticity_magnitude


class TurbulenceModel:
    """Base class and interface (see the module docstring)."""
    n_vars = 0
    names = ()
    needs_wall_distance = False

    def eddy_viscosity(self, U, mu, n_mean):
        return 0.0

    def diffusive_fluxes(self, U, mu, n_mean):
        return lambda grad_phi, grad_rho: []

    def sources(self, U, grad_U, mu, d, n_mean, gamma):
        return []

    def freestream_values(self, fs):
        return []

    def wall_values(self, U, n_mean):
        return []

    def limiter_scales(self, fs):
        """Default: 1000x the free-stream value, i.e. the scalar's size in a
        boundary layer whose eddy viscosity is ~1e3 times the laminar one."""
        return [1e3 * v for v in self.freestream_values(fs)]


class Laminar(TurbulenceModel):
    """No turbulence model: the laminar compressible Navier-Stokes equations."""


class SpalartAllmarasNeg(TurbulenceModel):
    """
    The negative Spalart-Allmaras model, SA-neg (Allmaras, Johnson & Spalart,
    ICCFD7-1902, 2012), in the conservative compressible form

        div(rho nu~ u) = rho (P - D)
                         + (1/sigma) div[(mu + rho nu~ f_n) grad nu~]
                         + (c_b2/sigma) rho |grad nu~|^2
                         - (1/sigma) (nu + nu~ f_n) grad rho . grad nu~

    For nu~ >= 0 this is standard SA (without the f_t2 term unless ft2=True),
    with the S~ modification that keeps S~ positive. For nu~ < 0 it switches
    to the negative-branch production, destruction and diffusion, which make
    a negative nu~ decay instead of growing. Under-resolved boundary-layer
    edges routinely undershoot below zero, which is why SA-neg is the usual
    choice. mu_t is zero there.

    The two branches meet continuously at nu~ = 0 (production and
    destruction both vanish), so the conditionals add kinks, not jumps.

    Free stream: nu~_inf = chi_inf nu_inf, chi_inf = 3 (NASA TMR). Wall:
    nu~ = 0.
    """
    n_vars = 1
    names = ("rho_nu_tilde",)
    needs_wall_distance = True

    cb1 = 0.1355
    sigma = 2.0 / 3.0
    cb2 = 0.622
    kappa = 0.41
    cw2 = 0.3
    cw3 = 2.0
    cv1 = 7.1
    ct3 = 1.2
    ct4 = 0.5
    cn1 = 16.0
    cv2 = 0.7
    cv3 = 0.9
    r_lim = 10.0

    def __init__(self, chi_inf=3.0, ft2=False):
        self.chi_inf = float(chi_inf)
        self.ft2 = bool(ft2)
        self.cw1 = self.cb1 / self.kappa**2 + (1.0 + self.cb2) / self.sigma

    # --- pointwise closure functions (also what the unit tests check) ----
    def fv1(self, chi):
        chi3 = chi**3
        return chi3 / (chi3 + self.cv1**3)

    def fv2(self, chi):
        return 1.0 - chi / (1.0 + chi * self.fv1(chi))

    def ft2_fn(self, chi):
        return self.ct3 * ufl.exp(-self.ct4 * chi**2) if self.ft2 else 0.0

    def fn(self, chi):
        chi3 = chi**3
        return ufl.conditional(ufl.ge(chi, 0.0), 1.0,
                               (self.cn1 + chi3) / (self.cn1 - chi3))

    def s_tilde(self, Omega, nu_t, chi, d):
        S_bar = nu_t * self.fv2(chi) / (self.kappa**2 * d**2)
        modified = Omega + Omega * (self.cv2**2 * Omega + self.cv3 * S_bar) / (
            (self.cv3 - 2.0 * self.cv2) * Omega - S_bar)
        return ufl.conditional(ufl.ge(S_bar, -self.cv2 * Omega), Omega + S_bar, modified)

    def fw(self, r):
        g = r + self.cw2 * (r**6 - r)
        return g * ((1.0 + self.cw3**6) / (g**6 + self.cw3**6))**(1.0 / 6.0)

    def production_destruction(self, nu_t, nu, Omega, d):
        """
        (P, D) per unit density, with both SA-neg branches. S~ is floored
        before it divides r, which only matters on the unselected branch.
        """
        chi = nu_t / nu
        St = self.s_tilde(Omega, nu_t, chi, d)
        r = ufl.min_value(nu_t / ufl.max_value(St * self.kappa**2 * d**2, 1e-30), self.r_lim)
        ft2 = self.ft2_fn(chi)
        P_pos = self.cb1 * (1.0 - ft2) * St * nu_t
        D_pos = (self.cw1 * self.fw(r) - self.cb1 * ft2 / self.kappa**2) * (nu_t / d)**2
        P_neg = self.cb1 * (1.0 - self.ct3) * Omega * nu_t
        D_neg = -self.cw1 * (nu_t / d)**2
        positive = ufl.ge(nu_t, 0.0)
        return (ufl.conditional(positive, P_pos, P_neg),
                ufl.conditional(positive, D_pos, D_neg))

    # --- the TurbulenceModel interface -----------------------------------
    def _nu_tilde(self, U, n_mean):
        return U[n_mean] / U[0]

    def eddy_viscosity(self, U, mu, n_mean):
        rho = U[0]
        nu_t = self._nu_tilde(U, n_mean)
        chi = nu_t * rho / mu
        return ufl.conditional(ufl.ge(nu_t, 0.0), rho * nu_t * self.fv1(chi), 0.0)

    def diffusive_fluxes(self, U, mu, n_mean):
        rho = U[0]
        nu_t = self._nu_tilde(U, n_mean)
        coeff = (mu + rho * nu_t * self.fn(nu_t * rho / mu)) / self.sigma
        return lambda grad_phi, grad_rho: [coeff * grad_phi[0]]

    def sources(self, U, grad_U, mu, d, n_mean, gamma):
        rho = U[0]
        nu = mu / rho
        nu_t = self._nu_tilde(U, n_mean)
        grad_u, _, grad_phi, grad_rho = primitive_gradients(U, grad_U, gamma, n_mean)
        grad_nt = grad_phi[0]
        Omega = vorticity_magnitude(grad_u)
        # d is zero only on the wall itself; quadrature points never sit there
        d_safe = ufl.max_value(d, 1e-14)
        P, D = self.production_destruction(nu_t, nu, Omega, d_safe)
        fn = self.fn(nu_t / nu)
        return [rho * (P - D)
                + self.cb2 / self.sigma * rho * ufl.dot(grad_nt, grad_nt)
                - (nu + nu_t * fn) / self.sigma * ufl.dot(grad_rho, grad_nt)]

    def freestream_values(self, fs):
        return [self.chi_inf * fs['mu']]   # rho_inf * (chi_inf * mu_inf / rho_inf)

    def wall_values(self, U, n_mean):
        return [0.0]
