"""
Pointwise checks of the inviscid building blocks on primitive states: the
fluxes are consistent and match an independent numpy transcription of the
conservative HLL flux, transported scalars are upwinded with the mean flow,
the characteristic far-field state matches a numpy transcription and keeps
the free stream, and the positivity limiter ignores the transported scalars.
"""
import numpy as np
import pytest
import dolfinx
from mpi4py import MPI

from dragonfly_sim.core.materials import IdealGas, thermo
from dragonfly_sim.utils.Euler_utils import hll_flux, llf_flux, hllc_flux, boundary_flux
from dragonfly_sim.utils.solver_utils import (compute_positivity_preserving_theta,
                                              locate_positivity_limiting_nodes)

GAMMA = 1.4
FARFIELD = {'rho': 1.0, 'p': 1.0, 'M': 0.5, 'alpha': 0.0}
_MESH = None
_GAS = None


def _mesh():
    global _MESH
    if _MESH is None:
        _MESH = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1, dolfinx.mesh.CellType.quadrilateral)
    return _MESH


def gas():
    """An ideal gas bound to FARFIELD: units rho_inf = 1, u_inf = M sqrt(gamma)."""
    global _GAS
    if _GAS is None:
        _GAS = IdealGas(GAMMA)
        _GAS.bind(_mesh(), FARFIELD)
    return _GAS


def evaluate(expr):
    """Value of a spatially constant UFL expression."""
    e = dolfinx.fem.Expression(expr, np.array([[0.5, 0.5]]), comm=MPI.COMM_SELF)
    return np.asarray(e.eval(_mesh(), np.array([0], dtype=np.int32))).ravel()


def const(value):
    return dolfinx.fem.Constant(_mesh(), np.asarray(value, dtype=np.float64))


def U_SPEED():
    return FARFIELD['M'] * np.sqrt(GAMMA * FARFIELD['p'] / FARFIELD['rho'])


def primitive(rho, u, v, p, *extras):
    """V = (p', u, v, T, extras) of a state in FARFIELD's units (rho_inf = p_inf = 1)."""
    s = U_SPEED()
    return [(p - 1.0) / s**2, u / s, v / s, p / rho, *extras]


def conservative(rho, u, v, p, *extras):
    return np.array([rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v),
                     *[rho * e for e in extras]])


def row_scaling(n_extra=0):
    """Flux in FARFIELD's units = D_R x the solver's flux (unweighted energy row)."""
    s = U_SPEED()
    return np.array([s, s**2, s**2, s**3] + [s] * n_extra)


def th(V):
    return thermo(gas(), const(V), 4)


# (rho, u, v, p) in FARFIELD's units
STATES = [((1.0, 0.6, 0.1, 1.0), (0.9, 0.5, -0.05, 0.85)),       # subsonic
          ((1.0, 2.0, 0.0, 0.5), (1.1, 1.9, 0.1, 0.6)),           # supersonic +
          ((1.0, -1.8, 0.2, 0.6), (0.95, -2.0, 0.1, 0.55))]       # supersonic -


def np_hll(QL, QR, n):
    """Independent numpy transcription of the conservative HLL flux (Davis speeds)."""
    def parts(q):
        rho, u, v, p = q
        un, c = u * n[0] + v * n[1], np.sqrt(GAMMA * p / rho)
        U = conservative(*q)
        F = np.array([rho * un, rho * u * un + p * n[0], rho * v * un + p * n[1],
                      (U[3] + p) * un])
        return U, F, un - c, un + c
    UL, FL, sl_l, sr_l = parts(QL)
    UR, FR, sl_r, sr_r = parts(QR)
    SL, SR = min(sl_l, sl_r, 0.0), max(sr_l, sr_r, 0.0)
    return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)


@pytest.mark.parametrize("QL, QR", STATES)
@pytest.mark.verification
def test_hll_matches_conservative_transcription(QL, QR):
    n = np.array([0.6, 0.8])
    F = evaluate(hll_flux(th(primitive(*QL)), th(primitive(*QR)), const(n)))
    # the momentum flux carries the gauge pressure: p_inf n less than with p
    gauge = FARFIELD['p'] * np.array([0.0, n[0], n[1], 0.0])
    np.testing.assert_allclose(F * row_scaling() + gauge, np_hll(QL, QR, n), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("QL, QR", STATES)
@pytest.mark.verification
def test_fluxes_are_consistent_and_upwind_scalars(QL, QR):
    n = const([0.6, 0.8])
    a, b = th(primitive(*QL, 0.3)), th(primitive(*QR, 0.7))
    exact = evaluate(a.normal_flux(n))
    for flux in (hll_flux, llf_flux, hllc_flux):
        np.testing.assert_allclose(evaluate(flux(a, a, n)), exact, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(evaluate(boundary_flux(a, a, n)), exact, rtol=1e-13, atol=1e-13)
    # supersonic left to right: fully upwind, the scalar rides with the mass flux
    un_L, un_R = (QL[1] * 0.6 + QL[2] * 0.8), (QR[1] * 0.6 + QR[2] * 0.8)
    if un_L > 1.5 and un_R > 1.5:
        F = evaluate(hll_flux(a, b, n))
        np.testing.assert_allclose(F, exact, rtol=1e-13)
        assert abs(F[4] - 0.3 * F[0]) < 1e-13


def np_riemann_state(q, n, u_inf, g):
    """Independent numpy transcription of the Riemann-invariant far field
    (rho_inf = p_inf = 1); returns (rho, u, v, p)."""
    rho, ux, uy, p = q
    u = np.array([ux, uy])
    c, c_inf = np.sqrt(g * p / rho), np.sqrt(g)
    R_out = u @ n + 2 * c / (g - 1)
    R_in = u_inf @ n - 2 * c_inf / (g - 1)
    un, cb = 0.5 * (R_out + R_in), 0.25 * (g - 1) * (R_out - R_in)
    if un < 0:
        s, ut = 1.0, u_inf - (u_inf @ n) * n
    else:
        s, ut = p / rho**g, u - (u @ n) * n
    rb = (cb**2 / (g * s))**(1 / (g - 1))
    ub = ut + un * n
    return rb, ub[0], ub[1], rb * cb**2 / g


@pytest.mark.verification
def test_riemann_state_matches_numpy_and_preserves_freestream():
    rng = np.random.default_rng(2)
    alpha = np.radians(3.0)
    u_dir = np.array([np.cos(alpha), np.sin(alpha)])
    u_inf = U_SPEED() * u_dir
    q_inf = (1.0, *u_inf, 1.0)
    for theta in np.linspace(0, 2 * np.pi, 13):
        n = np.array([np.cos(theta), np.sin(theta)])
        # the free stream maps to itself
        Vb, _ = gas().farfield_state(th(primitive(*q_inf)), const(n), const(u_dir))
        np.testing.assert_allclose(evaluate(Vb), primitive(*q_inf), rtol=1e-13, atol=1e-13)
        for _ in range(3):
            q = (1 + 0.1 * rng.standard_normal(), *(u_inf + 0.1 * rng.standard_normal(2)),
                 1 + 0.1 * rng.standard_normal())
            Vb, _ = gas().farfield_state(th(primitive(*q)), const(n), const(u_dir))
            np.testing.assert_allclose(evaluate(Vb), primitive(*np_riemann_state(q, n, u_inf, GAMMA)),
                                       rtol=1e-12, atol=1e-12)


@pytest.mark.unit_test
def test_positivity_limiter_ignores_extra_components():
    rng = np.random.default_rng(0)
    nb = 50
    x = np.column_stack([rng.random(nb), rng.random(nb), rng.random(nb), 1 + 0.1 * rng.random(nb)])
    dx = 0.8 * rng.standard_normal((nb, 4))
    dx[:, 0] *= 5.0
    constraints = [(0, -0.9), (3, 1e-4)]
    x5 = np.hstack([x, -rng.random((nb, 1))]).ravel()          # negative nu_tilde must not matter
    dx5 = np.hstack([dx, 5 * rng.standard_normal((nb, 1))]).ravel()
    t4 = compute_positivity_preserving_theta(MPI.COMM_SELF, x.ravel(), dx.ravel(), constraints, block_size=4)
    t5 = compute_positivity_preserving_theta(MPI.COMM_SELF, x5, dx5, constraints, block_size=5)
    assert t4 == t5 and t4 < 1.0
    # the step is feasible and twice it is not
    trial = x - t4 * dx
    assert np.all(trial[:, 0] > -0.9) and np.all(trial[:, 3] > 1e-4)
    r4 = locate_positivity_limiting_nodes(x.ravel(), dx.ravel(), 2 * t4, constraints, block_size=4)
    r5 = locate_positivity_limiting_nodes(x5, dx5, 2 * t4, constraints, block_size=5)
    assert r4 == r5 and r4["block"] is not None and sum(r4["counts"].values()) > 0


@pytest.mark.unit_test
def test_relative_change_limit():
    """With max_change the constrained components change by at most that
    fraction of their distance from the bound (T: of its value)."""
    x = np.array([[0.0, 1.0, 0.0, 1.0], [0.1, 1.0, 0.0, 2.0]])
    dx = np.array([[0.0, 0.0, 0.0, -1.6], [0.0, 3.0, 0.0, 0.1]])    # T 1 -> 2.6 at theta = 1
    constraints = [(0, -1.0), (3, 0.0)]
    t = compute_positivity_preserving_theta(MPI.COMM_SELF, x.ravel(), dx.ravel(), constraints, block_size=4)
    assert t == 1.0                                          # positive everywhere
    t = compute_positivity_preserving_theta(MPI.COMM_SELF, x.ravel(), dx.ravel(), constraints,
                                            block_size=4, max_change=0.5)
    assert t == 0.25 and 0.25 * 1.6 <= 0.5                   # the largest halving within 50 %
