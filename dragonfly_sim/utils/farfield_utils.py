"""
Characteristic (Riemann-invariant) far-field state.

Used by CompressibleEulerModel.define_farfield_bc. It returns the inflow
switch along with the state, so the caller can upwind any further
transported scalars (turbulence variables) with it. Unlike the fixed
inflow/outflow states, it lets outgoing acoustic waves leave the domain:
exact at normal incidence, partly reflecting at oblique incidence (see
core/farfield.py for the transverse correction).
"""
import ufl

from dragonfly_sim.utils.Euler_utils import pressure, primitives


def riemann_farfield_state(U, n, rho_inf, u_inf, p_inf, gamma, R_in=None):
    """
    Subsonic far-field state from the Riemann invariants, and the inflow flag.

    With n the outward normal, R+ = u.n + 2c/(g-1) comes from the interior
    and R- = u.n - 2c/(g-1) from the free stream (or R_in when given):

        u.n_b = (R+ + R-)/2,    c_b = (g-1)(R+ - R-)/4

    Entropy p/rho^g and the tangential velocity are taken from the free
    stream where u.n_b < 0 (inflow) and from the interior otherwise. A
    uniform free-stream interior reproduces U_inf exactly. Exact for waves at
    normal incidence; oblique waves are partly reflected.

    U is the mean flow (rho, rho u, rho E). Returns (U_b, inflow).
    """
    rho_i, u_i, _ = primitives(U)
    p_i = pressure(U, gamma)
    c_i = ufl.sqrt(gamma * p_i / rho_i)
    c_inf = (gamma * p_inf / rho_inf) ** 0.5
    un_i = ufl.dot(u_i, n)
    un_inf = ufl.dot(u_inf, n)

    R_out = un_i + 2.0 * c_i / (gamma - 1.0)
    if R_in is None:
        R_in = un_inf - 2.0 * c_inf / (gamma - 1.0)
    un_b = 0.5 * (R_out + R_in)
    c_b = 0.25 * (gamma - 1.0) * (R_out - R_in)

    inflow = ufl.lt(un_b, 0.0)
    s_b = ufl.conditional(inflow, p_inf / rho_inf ** gamma, p_i / rho_i ** gamma)
    ut_b = ufl.conditional(inflow, u_inf - un_inf * n, u_i - un_i * n)

    rho_b = (c_b ** 2 / (gamma * s_b)) ** (1.0 / (gamma - 1.0))
    p_b = rho_b * c_b ** 2 / gamma
    u_b = ut_b + un_b * n
    d = u_b.ufl_shape[0]
    U_b = ufl.as_vector([rho_b, *[rho_b * u_b[i] for i in range(d)],
                         p_b / (gamma - 1.0) + 0.5 * rho_b * ufl.dot(u_b, u_b)])
    return U_b, inflow
