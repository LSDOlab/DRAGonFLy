"""
Turbulence closures: SA-neg pointwise against an independent numpy
transcription of the published formulas (Allmaras, Johnson & Spalart 2012;
NASA TMR), and the closure interface's modularity -- a different closure
with two transported scalars plugs into CompressibleRANSModel unchanged.
"""
import numpy as np
import pytest
import ufl

from dragonfly_sim.core.turbulence_models import TurbulenceModel, Laminar, SpalartAllmarasNeg

from cases import rans_square_model
from test_fluxes_and_states import evaluate, const

SA = dict(cb1=0.1355, sigma=2 / 3, cb2=0.622, kappa=0.41, cw2=0.3, cw3=2.0,
          cv1=7.1, ct3=1.2, ct4=0.5, cn1=16.0, cv2=0.7, cv3=0.9)
SA["cw1"] = SA["cb1"] / SA["kappa"]**2 + (1 + SA["cb2"]) / SA["sigma"]


def np_sa_pd(nu_t, nu, Omega, d, ft2=False):
    c = SA
    chi = nu_t / nu
    if nu_t < 0:
        return c["cb1"] * (1 - c["ct3"]) * Omega * nu_t, -c["cw1"] * (nu_t / d)**2
    fv1 = chi**3 / (chi**3 + c["cv1"]**3)
    fv2 = 1 - chi / (1 + chi * fv1)
    S_bar = nu_t * fv2 / (c["kappa"]**2 * d**2)
    if S_bar >= -c["cv2"] * Omega:
        St = Omega + S_bar
    else:
        St = Omega + Omega * (c["cv2"]**2 * Omega + c["cv3"] * S_bar) / ((c["cv3"] - 2 * c["cv2"]) * Omega - S_bar)
    r = min(nu_t / (St * c["kappa"]**2 * d**2), 10.0)
    g = r + c["cw2"] * (r**6 - r)
    fw = g * ((1 + c["cw3"]**6) / (g**6 + c["cw3"]**6))**(1 / 6)
    f_t2 = c["ct3"] * np.exp(-c["ct4"] * chi**2) if ft2 else 0.0
    return (c["cb1"] * (1 - f_t2) * St * nu_t,
            (c["cw1"] * fw - c["cb1"] * f_t2 / c["kappa"]**2) * (nu_t / d)**2)


@pytest.mark.parametrize("nu_t, nu, Omega, d", [
    (3e-5, 1e-5, 50.0, 1e-2),      # boundary layer
    (3e-5, 1e-5, 1e-3, 1e-1),      # near-irrotational: the S~ modification
    (5e-4, 1e-5, 2.0, 0.5),        # large chi
    (-2e-5, 1e-5, 30.0, 1e-2),     # negative branch
])
@pytest.mark.parametrize("ft2", [False, True])
def test_sa_neg_production_destruction(nu_t, nu, Omega, d, ft2):
    sa = SpalartAllmarasNeg(ft2=ft2)
    P, D = sa.production_destruction(const(nu_t), const(nu), const(Omega), const(d))
    np.testing.assert_allclose([evaluate(P)[0], evaluate(D)[0]], np_sa_pd(nu_t, nu, Omega, d, ft2),
                               rtol=1e-12)


def test_sa_neg_eddy_viscosity_vanishes_for_negative_nu_tilde():
    sa = SpalartAllmarasNeg()
    U = const([1.0, 0.5, 0.0, 2.5, -1e-5])
    assert evaluate(sa.eddy_viscosity(U, const(1e-5), 4))[0] == 0.0
    U = const([1.2, 0.5, 0.0, 2.5, 3.6e-5])
    chi = 3.6e-5 / 1e-5
    np.testing.assert_allclose(evaluate(sa.eddy_viscosity(U, const(1e-5), 4))[0],
                               3.6e-5 * chi**3 / (chi**3 + 7.1**3), rtol=1e-13)


class TwoPassiveScalars(TurbulenceModel):
    """A toy closure: two transported scalars, a constant eddy viscosity and a
    linear relaxation source -- only there to exercise the interface."""
    n_vars = 2
    names = ("rho_a", "rho_b")
    needs_wall_distance = False

    def eddy_viscosity(self, U, mu, n_mean):
        return 2.0 * mu

    def diffusive_fluxes(self, U, mu, n_mean):
        return lambda grad_phi, grad_rho: [mu * grad_phi[0], 0.5 * mu * grad_phi[1]]

    def sources(self, U, grad_U, mu, d, n_mean, gamma):
        return [-(U[n_mean] - 0.2 * U[0]), -(U[n_mean + 1] - 0.4 * U[0])]

    def freestream_values(self, fs):
        return [0.2 * fs["rho"], 0.4 * fs["rho"]]

    def wall_values(self, U, n_mean):
        return [0.2 * U[0], 0.4 * U[0]]


@pytest.mark.parametrize("closure", [Laminar(), TwoPassiveScalars()], ids=["laminar", "two-scalar"])
def test_closures_plug_in(closure):
    """A uniform free stream with a characteristic far field all round is an
    exact discrete solution for any closure whose free-stream values make
    its sources vanish."""
    model = rans_square_model(closure, n=4, jitter=0.0)
    assert model.n_state == 4 + closure.n_vars
    model.compute_initial_conditions_from_inlet()
    model.interpolate_solution_vector()
    model.define_farfield_bc(1)
    model.compute_weakform()
    model._ensure_residual_vector()
    r = model._assemble_physical_residual()
    assert r < 1e-12, r
