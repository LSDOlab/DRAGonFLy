"""
Fluid materials, and the thermodynamic state the flow models build from them.

The unknowns of every flow model are the primitive variables

    V = (p', u, T, phi),    p' = p - p_ref (gauge pressure),

with phi the turbulence variables, while the residual stays the conservative
one, div F(U(V)) = 0. A material is the ONLY code that knows which fluid is
solved: four functions of (p', T) returning UFL expressions,

    rho(p', T)   density            h(p', T)   specific enthalpy
    mu(T)        dynamic viscosity  k(T)       thermal conductivity

plus its constants (dolfinx Constants, so a new Mach number does not
recompile anything). thermo() derives everything else with ufl.diff: rho_p,
rho_T, h_p, h_T and the inverse squared sound speed

    1/c^2 = rho_p + rho_T (1 - rho h_p) / (rho h_T),

which is 1/(gamma R T) for an ideal gas and identically zero (UFL Zero) for a
constant-density fluid. Downstream code only ever uses 1/c^2, never c, so
constant density needs no special case.

Nondimensionalization: every quantity is scaled with the far-field state the
material is bound to (bind()),

    rho_ref = rho_inf,  u_ref = |u_inf|,  T_ref = T_inf,  p' / (rho_inf u_inf^2),
    length = the mesh unit,  time = length / u_inf,  mu_ref = rho u L / Re,

so p', u and T are O(1) at every Mach number. For an ideal gas the absolute
reference pressure is then p_ref = 1/(gamma M^2), which enters only the
equation of state.
"""
from dataclasses import dataclass, field

import numpy as np
import ufl
import dolfinx


@dataclass
class ThermoState:
    """
    The state V = (p', u, T, phi) of one evaluation point (a cell, or one
    facet trace) with its thermodynamics. Every field is a UFL expression.
    inv_Ur2 is 1/U_r^2 of the low-Mach preconditioning, None without it.
    """
    p: object
    u: object
    T: object
    phi: list
    rho: object
    rho_p: object
    rho_T: object
    h: object
    h_p: object
    h_T: object
    inv_c2: object
    inv_Ur2: object = None
    material: object = field(default=None, repr=False)

    @property
    def H(self):
        """Total enthalpy h + |u|^2/2."""
        return self.h + 0.5 * ufl.dot(self.u, self.u)

    @property
    def U(self):
        """The conserved state (rho, rho u, rho E', rho phi), rho E' = rho H - p'."""
        rho, d = self.rho, self.u.ufl_shape[0]
        return ufl.as_vector([rho] + [rho * self.u[i] for i in range(d)]
                             + [rho * self.H - self.p] + [rho * s for s in self.phi])

    def flux(self):
        """The inviscid flux tensor, (n_state, dim): rho u, rho u (x) u + p' I,
        rho u H, rho u phi."""
        rho, u, d = self.rho, self.u, self.u.ufl_shape[0]
        I = ufl.Identity(d)
        rows = [rho * u] + [rho * u[i] * u + self.p * I[i, :] for i in range(d)]
        rows += [rho * self.H * u] + [rho * s * u for s in self.phi]
        return ufl.as_tensor(rows)

    def normal_flux(self, n):
        return ufl.dot(self.flux(), n)

    def grad_rho(self, G):
        """grad(rho) from a gradient G of V (shape (n_state, dim))."""
        d = self.u.ufl_shape[0]
        return self.rho_p * G[0, :] + self.rho_T * G[1 + d, :]


def split_state(V, n_mean, n_phi=None):
    """(p', u, T, phi) of a state vector V; phi are the n_phi components
    after the mean flow (all of them by default)."""
    d = n_mean - 2
    n_phi = V.ufl_shape[0] - n_mean if n_phi is None else n_phi
    return (V[0], ufl.as_vector([V[1 + i] for i in range(d)]), V[1 + d],
            [V[k] for k in range(n_mean, n_mean + n_phi)])


def thermo(material, V, n_mean, n_phi=None, inv_Ur2=None):
    """The ThermoState of V; rho_p, rho_T, h_p, h_T by ufl.diff. Components
    beyond the n_phi transported scalars (auxiliary unknowns) are ignored."""
    p, u, T, phi = split_state(V, n_mean, n_phi)
    pv, Tv = ufl.variable(p), ufl.variable(T)
    rho, h = material.rho(pv, Tv), material.h(pv, Tv)
    rho_p, rho_T = ufl.diff(rho, pv), ufl.diff(rho, Tv)
    h_p, h_T = ufl.diff(h, pv), ufl.diff(h, Tv)
    inv_c2 = rho_p + rho_T * (1.0 - rho * h_p) / (rho * h_T)
    return ThermoState(p, u, T, phi, rho, rho_p, rho_T, h, h_p, h_T, inv_c2, inv_Ur2, material)


def state_vector(p, u, T, phi=()):
    """V = (p', u, T, phi) as one UFL vector."""
    return ufl.as_vector([p] + [u[i] for i in range(u.ufl_shape[0])] + [T] + list(phi))


# ----------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------
class IdealGas:
    """
    Calorically perfect gas: rho = p/(R T), h = c_p T, Sutherland's (or a
    constant) viscosity and k = c_p mu / Pr.

    Bound to the far field {'rho', 'p', 'M', 'alpha'} (any consistent units:
    only the Mach number survives the nondimensionalization), the constants
    are p_ref = R = 1/(gamma M^2) and c_p = gamma R/(gamma - 1); mu_ref =
    1/Re is set by the viscous models. 0 < M < 1: the characteristic far
    field (farfield_state) is the subsonic one.
    """
    incompressible = False
    farfield_keys = ('rho', 'p', 'M', 'alpha')
    # positivity margin of the Newton step limiter, relative to the far field
    positivity_floor = 1e-4

    def __init__(self, gamma=1.4, Pr=0.72, viscosity_law="sutherland", S_ratio=110.4 / 300.0):
        if viscosity_law not in ("sutherland", "constant"):
            raise ValueError("viscosity_law must be 'sutherland' or 'constant'")
        self.gamma = float(gamma)
        self.Pr = float(Pr)
        self.viscosity_law = viscosity_law
        self.S_ratio = float(S_ratio)
        self.p_ref = self.R = self.cp = self.mu_ref = None

    def bind(self, mesh, farfield):
        """Create the constants for the far field; returns the far-field
        quantities in the solver's units (rho, p, c, speed, M)."""
        _check_keys(farfield, self.farfield_keys)
        rho, p, M = (float(farfield[k]) for k in ('rho', 'p', 'M'))
        if rho <= 0.0 or p <= 0.0:
            raise ValueError("far-field density and pressure must be positive")
        if not 0.0 < M < 1.0:
            raise NotImplementedError(
                "far-field Mach number {} outside (0, 1): the solver's units are "
                "built on the free-stream speed, and only the subsonic "
                "characteristic far field is implemented".format(M))
        self.p_ref = dolfinx.fem.Constant(mesh, 0.0)
        self.R = dolfinx.fem.Constant(mesh, 0.0)
        self.cp = dolfinx.fem.Constant(mesh, 0.0)
        self.mu_ref = dolfinx.fem.Constant(mesh, 0.0)
        # the dimensional free stream, for dimensional outputs
        c = (self.gamma * p / rho) ** 0.5
        self.dimensional = {'rho': rho, 'p': p, 'c': c, 'speed': M * c}
        return self.set_mach(M)

    def set_mach(self, M):
        M = float(M)
        g = self.gamma
        self.p_ref.value = self.R.value = 1.0 / (g * M * M)
        self.cp.value = g / (g - 1.0) * self.R.value
        return {'rho': 1.0, 'p': float(self.p_ref.value), 'c': 1.0 / M, 'speed': 1.0, 'M': M}

    # --- the four material functions -------------------------------------
    def rho(self, p, T):
        return (self.p_ref + p) / (self.R * T)

    def h(self, p, T):
        return self.cp * T

    def mu(self, T):
        if self.viscosity_law == "constant":
            return self.mu_ref
        # T is T/T_inf; floored so a transiently negative T cannot hand
        # theta^(3/2) a negative base
        theta = ufl.max_value(T, 1e-8)
        S = self.S_ratio
        return self.mu_ref * theta ** 1.5 * (1.0 + S) / (theta + S)

    def k(self, T):
        return self.cp * self.mu(T) / self.Pr

    # --- traits and scales ---------------------------------------------
    def energy_scale(self):
        """h_T T_ref, the size of the energy equation relative to the others;
        its inverse weights the energy row of the residual."""
        return self.cp

    def viscous_time_factor(self):
        """Ratio of the largest diffusivity (thermal) to mu/rho."""
        return self.gamma / self.Pr

    def positivity_constraints(self, n_mean):
        """[(component, lower bound)]: absolute pressure and temperature stay
        above positivity_floor times their far-field values."""
        f = self.positivity_floor
        return [(0, -(1.0 - f) * float(self.p_ref.value)), (n_mean - 1, f)]

    def farfield_state(self, th, n, u_inf, R_in=None):
        """
        Subsonic characteristic far-field state from the Riemann invariants,
        and the inflow flag. With n the outward normal, R+ = u.n + 2c/(g-1)
        comes from the interior and R- = u.n - 2c/(g-1) from the free stream
        (or R_in when given):

            u.n_b = (R+ + R-)/2,    c_b = (g-1)(R+ - R-)/4

        Entropy p/rho^g and the tangential velocity are taken from the free
        stream where u.n_b < 0 (inflow) and from the interior otherwise. A
        uniform free-stream interior reproduces the free stream exactly.
        Exact for waves at normal incidence; oblique waves are partly
        reflected (core/farfield.py adds the transverse correction).

        Returns the mean-flow state (p', u, T) and the inflow flag.
        """
        g, p_inf = self.gamma, self.p_ref
        p_i = self.p_ref + th.p
        c_i = ufl.sqrt(g * p_i / th.rho)
        un_i = ufl.dot(th.u, n)
        R_out = un_i + 2.0 * c_i / (g - 1.0)
        if R_in is None:
            R_in = self.incoming_invariant(n, u_inf)
        un_b = 0.5 * (R_out + R_in)
        c_b = 0.25 * (g - 1.0) * (R_out - R_in)

        inflow = ufl.lt(un_b, 0.0)
        un_inf = ufl.dot(u_inf, n)
        s_b = ufl.conditional(inflow, p_inf, p_i / th.rho ** g)       # rho_inf = 1
        ut_b = ufl.conditional(inflow, u_inf - un_inf * n, th.u - un_i * n)

        rho_b = (c_b ** 2 / (g * s_b)) ** (1.0 / (g - 1.0))
        p_b = rho_b * c_b ** 2 / g
        return state_vector(p_b - self.p_ref, ut_b + un_b * n, p_b / (rho_b * self.R)), inflow

    def incoming_invariant(self, n, u_inf):
        """R- = u_inf.n - 2 c_inf/(g-1) of the free stream."""
        g = self.gamma
        return ufl.dot(u_inf, n) - 2.0 / (g - 1.0) * ufl.sqrt(g * self.p_ref)


def _check_keys(farfield, keys):
    if not isinstance(farfield, dict) or set(farfield) != set(keys):
        raise ValueError(
            "The far-field conditions are a flat dict with exactly the keys {}; got {}. "
            "The former {{'inlet': {{...}}, 'outlet': {{...}}}} form is gone: pass the "
            "contents of 'inlet' and drop 'outlet'.".format(
                sorted(keys), sorted(farfield) if isinstance(farfield, dict)
                else type(farfield).__name__))


class ConstantDensity:
    """
    A constant-density liquid (water by default): rho = rho_0, gauge enthalpy
    h = c T + p'/rho_0, constant viscosity and k = c mu / Pr.

    The enthalpy is the gauge one so the reference pressure stays out of the
    energy flux: with h = c T + p/rho_0, p_ref div(u_h) would appear as
    spurious heating wherever the discrete velocity is not exactly
    divergence free. 1/c^2 = 0 (UFL Zero): the model must be preconditioned
    (artificial compressibility, Gamma's pressure column 1/U_r^2), and the
    far field is the free stream as a ghost state. Gravity is left out of
    the flow (p' is the hydrodynamic pressure).

    Bound to the far field {'U', 'T', 'p', 'alpha'} -- free-stream speed
    [m/s], temperature [K], absolute pressure [Pa] (for dimensional output
    only) and angle of attack -- in the solver's units rho_0 = U = T = 1, so
    c becomes c T / U^2 (the inverse Eckert number) and mu_ref = 1/Re is
    set by the viscous models.
    """
    incompressible = True
    farfield_keys = ('U', 'T', 'p', 'alpha')
    positivity_floor = 1e-4

    def __init__(self, rho0=998.2, c=4182.0, Pr=7.0):
        self.rho0, self.c_dim, self.Pr = float(rho0), float(c), float(Pr)
        self.rho_hat = self.c = self.mu_ref = None

    def bind(self, mesh, farfield):
        _check_keys(farfield, self.farfield_keys)
        U, T, p = (float(farfield[k]) for k in ('U', 'T', 'p'))
        if U <= 0.0 or T <= 0.0:
            raise ValueError("free-stream speed and temperature must be positive")
        self.rho_hat = dolfinx.fem.Constant(mesh, 1.0)
        self.c = dolfinx.fem.Constant(mesh, self.c_dim * T / U ** 2)
        self.mu_ref = dolfinx.fem.Constant(mesh, 0.0)
        q = self.rho0 * U ** 2
        self.dimensional = {'rho': self.rho0, 'p': p, 'speed': U, 'T': T}
        return {'rho': 1.0, 'p': p / q, 'c': np.inf, 'speed': 1.0, 'M': 0.0}

    def rho(self, p, T):
        return self.rho_hat

    def h(self, p, T):
        return self.c * T + p / self.rho_hat

    def mu(self, T):
        return self.mu_ref

    def k(self, T):
        return self.c * self.mu(T) / self.Pr

    def energy_scale(self):
        return self.c

    def viscous_time_factor(self):
        # momentum diffuses faster than heat (Pr > 1)
        return max(1.0, 1.0 / self.Pr)

    def positivity_constraints(self, n_mean):
        return [(n_mean - 1, self.positivity_floor)]
