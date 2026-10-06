"""
Pointwise checks of the inviscid building blocks: the extended-state fluxes
reduce to the Euler ones, the characteristic far-field state against an
independent numpy transcription, and the positivity limiter ignoring the
transported scalars.
"""
import numpy as np
import pytest
import ufl
import dolfinx
from mpi4py import MPI

from dragonfly_sim.utils.Euler_utils import hll_flux, flux, boundary_flux
from dragonfly_sim.utils.NS_utils import extended_hll_flux, extended_flux, extended_boundary_flux
from dragonfly_sim.utils.farfield_utils import riemann_farfield_state
from dragonfly_sim.utils.solver_utils import (compute_positivity_preserving_theta,
                                              locate_positivity_limiting_nodes)

GAMMA = 1.4
_MESH = None


def _mesh():
    global _MESH
    if _MESH is None:
        _MESH = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1, dolfinx.mesh.CellType.quadrilateral)
    return _MESH


def evaluate(expr):
    """Value of a spatially constant UFL expression."""
    msh = _mesh()
    e = dolfinx.fem.Expression(expr, np.array([[0.5, 0.5]]), comm=MPI.COMM_SELF)
    return np.asarray(e.eval(msh, np.array([0], dtype=np.int32))).ravel()


def const(value):
    return dolfinx.fem.Constant(_mesh(), np.asarray(value, dtype=np.float64))


def state(rho, u, v, p, *extras):
    return [rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v), *extras]


STATES = [(state(1.0, 0.6, 0.1, 1.0), state(0.9, 0.5, -0.05, 0.85)),      # subsonic
          (state(1.0, 2.0, 0.0, 0.5), state(1.1, 1.9, 0.1, 0.6)),          # supersonic +
          (state(1.0, -1.8, 0.2, 0.6), state(0.95, -2.0, 0.1, 0.55))]      # supersonic -


@pytest.mark.parametrize("UL, UR", STATES)
def test_extended_hll_reduces_to_hll(UL, UR):
    n = const([0.6, 0.8])
    a, b = const(UL), const(UR)
    np.testing.assert_allclose(evaluate(extended_hll_flux(a, b, n, GAMMA, 4)),
                               evaluate(hll_flux(a, b, n, GAMMA)), rtol=1e-14, atol=1e-14)
    # an extra scalar is upwinded with the mean flow's fan; the mean rows are unchanged
    a5, b5 = const(UL + [0.3]), const(UR + [0.7])
    F5 = evaluate(extended_hll_flux(a5, b5, n, GAMMA, 4))
    np.testing.assert_allclose(F5[:4], evaluate(hll_flux(a, b, n, GAMMA)), rtol=1e-14, atol=1e-14)
    un_L, un_R = (UL[1] * 0.6 + UL[2] * 0.8) / UL[0], (UR[1] * 0.6 + UR[2] * 0.8) / UR[0]
    if un_L > 1.5 and un_R > 1.5:   # fully upwind from the left
        assert abs(F5[4] - 0.3 * un_L) < 1e-12


def test_extended_fluxes_without_extras_are_euler_fluxes():
    U, Ue, n = const(STATES[0][0]), const(STATES[0][1]), const([0.0, 1.0])
    np.testing.assert_allclose(evaluate(extended_flux(U, GAMMA, 4)), evaluate(flux(U, GAMMA)), rtol=1e-15)
    np.testing.assert_allclose(evaluate(extended_boundary_flux(U, Ue, n, GAMMA, 4)),
                               evaluate(boundary_flux(U, Ue, n, GAMMA)), rtol=1e-15)


def np_riemann_state(U, n, rho_inf, u_inf, p_inf, g):
    """Independent numpy transcription of the Riemann-invariant far field."""
    rho, mx, my, E = U
    u = np.array([mx, my]) / rho
    p = (g - 1) * (E - 0.5 * rho * u @ u)
    c, c_inf = np.sqrt(g * p / rho), np.sqrt(g * p_inf / rho_inf)
    R_out = u @ n + 2 * c / (g - 1)
    R_in = u_inf @ n - 2 * c_inf / (g - 1)
    un, cb = 0.5 * (R_out + R_in), 0.25 * (g - 1) * (R_out - R_in)
    if un < 0:
        s, ut = p_inf / rho_inf**g, u_inf - (u_inf @ n) * n
    else:
        s, ut = p / rho**g, u - (u @ n) * n
    rb = (cb**2 / (g * s))**(1 / (g - 1))
    pb = rb * cb**2 / g
    ub = ut + un * n
    return np.array([rb, rb * ub[0], rb * ub[1], pb / (g - 1) + 0.5 * rb * ub @ ub]), un < 0


def test_riemann_state_matches_numpy_and_preserves_freestream():
    rng = np.random.default_rng(2)
    M, alpha = 0.6, np.radians(3.0)
    u_inf = M * np.sqrt(GAMMA) * np.array([np.cos(alpha), np.sin(alpha)])
    U_inf = np.array(state(1.0, *u_inf, 1.0))
    for theta in np.linspace(0, 2 * np.pi, 13):
        n = np.array([np.cos(theta), np.sin(theta)])
        # the free stream maps to itself
        Ub, _ = riemann_farfield_state(const(U_inf), const(n), 1.0, const(u_inf), 1.0, GAMMA)
        np.testing.assert_allclose(evaluate(Ub), U_inf, rtol=1e-13)
        for _ in range(3):
            U = np.array(state(1 + 0.1 * rng.standard_normal(), *(u_inf + 0.1 * rng.standard_normal(2)),
                               1 + 0.1 * rng.standard_normal()))
            Ub, inflow = riemann_farfield_state(const(U), const(n), 1.0, const(u_inf), 1.0, GAMMA)
            ref, ref_in = np_riemann_state(U, n, 1.0, u_inf, 1.0, GAMMA)
            np.testing.assert_allclose(evaluate(Ub), ref, rtol=1e-12)


def test_positivity_limiter_ignores_extra_components():
    rng = np.random.default_rng(0)
    nb = 50
    x = np.column_stack([1 + 0.1 * rng.random(nb), rng.random(nb), rng.random(nb), 3 + rng.random(nb)])
    dx = 0.8 * rng.standard_normal((nb, 4))
    x5 = np.hstack([x, -rng.random((nb, 1))]).ravel()          # negative nu_tilde must not matter
    dx5 = np.hstack([dx, 5 * rng.standard_normal((nb, 1))]).ravel()
    t4 = compute_positivity_preserving_theta(MPI.COMM_SELF, x.ravel(), dx.ravel(), block_size=4)
    t5 = compute_positivity_preserving_theta(MPI.COMM_SELF, x5, dx5, block_size=5, n_mean=4)
    assert t4 == t5 and t4 < 1.0
    r4 = locate_positivity_limiting_nodes(x.ravel(), dx.ravel(), 2 * t4, block_size=4)
    r5 = locate_positivity_limiting_nodes(x5, dx5, 2 * t4, block_size=5, n_mean=4)
    assert r4 == r5
