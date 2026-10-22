"""
The material layer and the low-Mach preconditioning, pointwise: the
thermodynamic state derived by ufl.diff against the closed forms, the
preconditioned wave speeds against the eigenvalues of Gamma^-1 A, and the
preconditioned fluxes reducing to the plain ones where U_r = c.
"""
import numpy as np
import pytest
import ufl

from dragonfly_sim.core.materials import thermo
from dragonfly_sim.utils.Euler_utils import hll_flux, llf_flux, boundary_flux, wave_speeds

from test_fluxes_and_states import evaluate, const, gas, GAMMA, FARFIELD

V_STATES = [[0.3, 0.8, -0.1, 1.05],       # subsonic, M ~ 0.4
            [-0.2, 0.05, 0.02, 0.98],     # near stagnation
            [0.0, 2.6, 0.3, 1.0]]         # supersonic (|u| > c = 2 in these units)


@pytest.mark.parametrize("V", V_STATES)
@pytest.mark.verification
def test_ideal_gas_thermodynamics(V):
    g = gas()
    th = thermo(g, const(V), 4)
    p_ref, R = float(g.p_ref.value), float(g.R.value)
    p, T = p_ref + V[0], V[3]
    rho = p / (R * T)
    np.testing.assert_allclose(evaluate(th.rho), rho, rtol=1e-14)
    np.testing.assert_allclose(evaluate(th.rho_p), 1 / (R * T), rtol=1e-14)
    np.testing.assert_allclose(evaluate(th.rho_T), -rho / T, rtol=1e-14)
    np.testing.assert_allclose(evaluate(th.h_T), GAMMA / (GAMMA - 1) * R, rtol=1e-14)
    # 1/c^2 = 1/(gamma R T); with the far field's units c_inf = 1/M
    np.testing.assert_allclose(evaluate(th.inv_c2), 1 / (GAMMA * R * T), rtol=1e-13)
    np.testing.assert_allclose(np.sqrt(GAMMA * R), 1 / FARFIELD["M"], rtol=1e-14)
    # rho E' = rho H - p' = p/(gamma - 1) + rho |u|^2/2 + p_ref: the energy shifted by p_ref
    U = evaluate(th.U)
    u2 = V[1] ** 2 + V[2] ** 2
    np.testing.assert_allclose(U[3], p / (GAMMA - 1) + 0.5 * rho * u2 + p_ref, rtol=1e-14)


def _gamma_and_jacobian(V, n, inv_Ur2):
    """Gamma (dU/dV with the Weiss-Smith pressure column) and A = d(F.n)/dV at V."""
    Vc = const(V)
    Vv = ufl.variable(Vc)
    th = thermo(gas(), Vv, 4)
    dUdV = evaluate(ufl.diff(th.U, Vv)).reshape(4, 4)
    A = evaluate(ufl.diff(th.normal_flux(const(n)), Vv)).reshape(4, 4)
    th0 = thermo(gas(), Vc, 4)
    w = evaluate(ufl.as_vector([1.0, th0.u[0], th0.u[1], th0.H]))
    Gamma = dUdV.copy()
    Gamma[:, 0] += (inv_Ur2 - evaluate(th0.inv_c2)[0]) * w
    return Gamma, A, th0


@pytest.mark.parametrize("V", V_STATES)
@pytest.mark.parametrize("Ur_over_u", [1.0, 0.5])
@pytest.mark.verification
def test_preconditioned_wave_speeds_are_eigenvalues_of_gamma_inv_A(V, Ur_over_u):
    n = np.array([0.6, 0.8])
    speed = np.hypot(V[1], V[2])
    c = 1 / np.sqrt(GAMMA * float(gas().R.value) * V[3])
    Ur = min(c, max(Ur_over_u * speed, 0.05))
    Gamma, A, th = _gamma_and_jacobian(V, n, 1 / Ur ** 2)
    eig = np.sort(np.linalg.eigvals(np.linalg.solve(Gamma, A)).real)
    th.inv_Ur2 = const(1 / Ur ** 2)
    sl, sr = (evaluate(s)[0] for s in wave_speeds(th, const(n)))
    un = V[1] * n[0] + V[2] * n[1]
    np.testing.assert_allclose(eig, np.sort([sl, un, un, sr]), rtol=1e-10, atol=1e-12)


@pytest.mark.verification
def test_preconditioned_fluxes_reduce_to_plain_ones_at_ur_equal_c():
    n = const([0.6, 0.8])
    Va, Vb = V_STATES[0], [0.1, 0.7, 0.05, 1.02]
    for flux in (hll_flux, llf_flux, boundary_flux):
        plain = evaluate(flux(thermo(gas(), const(Va), 4), thermo(gas(), const(Vb), 4), n))
        a, b = thermo(gas(), const(Va), 4), thermo(gas(), const(Vb), 4)
        a.inv_Ur2, b.inv_Ur2 = a.inv_c2, b.inv_c2
        np.testing.assert_allclose(evaluate(flux(a, b, n)), plain, rtol=1e-13, atol=1e-14)


@pytest.mark.verification
def test_preconditioned_mass_dissipation_scales_with_pressure_jump_over_ur():
    """At M = 0.01 the plain LLF mass dissipation of a pure pressure jump is
    c drho ~ dp'/c = O(M dp'), while the preconditioned one is ~ dp'/U_r =
    O(dp') (the Rhie-Chow-like term that couples pressure and velocity)."""
    from dragonfly_sim.core.materials import IdealGas
    from test_fluxes_and_states import _mesh
    low = IdealGas(GAMMA)
    low.bind(_mesh(), dict(FARFIELD, M=0.01))
    n = const([1.0, 0.0])
    Va, Vb = [0.0, 1.0, 0.0, 1.0], [0.1, 1.0, 0.0, 1.0]
    a, b = thermo(low, const(Va), 4), thermo(low, const(Vb), 4)
    central = 0.5 * (evaluate(a.normal_flux(n)) + evaluate(b.normal_flux(n)))
    plain = evaluate(llf_flux(a, b, n)) - central
    a.inv_Ur2 = b.inv_Ur2 = const(1.0)       # U_r = |u| = u_inf
    pre = evaluate(llf_flux(a, b, n)) - central
    assert abs(plain[0]) < 0.05 * 0.1                  # O(M dp')
    assert 0.05 < abs(pre[0]) < 0.2                    # O(dp')


def _channel_pressure(M, preconditioning, n=12):
    """Converged p' of the channel case (5 degrees inflow onto slip walls) and
    the PTC steps it took."""
    import numpy as np
    from dragonfly_sim.core.mesh_manager import Mesh
    from dragonfly_sim.core.Euler_model import CompressibleEulerModel
    from cases import jittered_square, channel_tags, FARFIELD_TAG, WALL_TAG
    mesh_obj = Mesh(jittered_square(n), lambda pts: np.zeros(pts.shape[1], dtype=bool))
    ff, wall = channel_tags(mesh_obj)
    mesh_obj.tag_boundary_facets([(ff, FARFIELD_TAG), (wall, WALL_TAG)])
    meta = {'quadrature_degree': 1}
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=meta)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(measure_metadata=meta)
    m = CompressibleEulerModel(mesh_obj, preconditioning=preconditioning)
    m.report_positivity_limiter = False
    m.use_ptc, m.fom_newton_inner_max_it = True, 1
    m.max_newton_iterations, m.residual_conv_limit = 60, 1e-11
    m.define_farfield_conditions({'rho': 1., 'p': 1., 'M': M, 'alpha': np.radians(5.)})
    m.define_elements_functionspaces()
    m.define_trial_testfunctions()
    m.compute_initial_conditions_from_farfield()
    m.interpolate_solution_vector()
    m.define_farfield_bc(FARFIELD_TAG)
    m.define_slipwall_bc(WALL_TAG)
    m.compute_weakform()
    m.set_up_solver()
    m.solve_system()
    return m.u_vec.x.array.reshape(-1, 4)[:, 0].copy(), m.time_integrator.last_steady_steps, \
        bool(m.solver_converged)


@pytest.mark.skipif(__import__("mpi4py").MPI.COMM_WORLD.size > 1, reason="serial comparison of fields")
@pytest.mark.verification
def test_preconditioned_solution_has_a_mach_independent_low_mach_limit():
    """With preconditioning the pressure p'/(rho u^2) converges to its
    incompressible limit at O(M^2) and PTC needs the same steps at every
    Mach number; without it the error is O(M) (Guillard & Viozat)."""
    p2, steps2, ok2 = _channel_pressure(0.01, True)
    p3, steps3, ok3 = _channel_pressure(0.001, True)
    assert ok2 and ok3 and max(steps2, steps3) <= 20, (steps2, steps3)
    assert np.abs(p2 - p3).max() < 1e-4
    q2, _, _ = _channel_pressure(0.1, False)
    q3, _, _ = _channel_pressure(0.01, False)
    p1, _, _ = _channel_pressure(0.1, True)
    assert np.abs(q2 - q3).max() > 10 * np.abs(p1 - p2).max()


def _plate(fluid, turbulence_model, n=10, Re=200.0):
    """Converged state of a laminar/RANS flow over a no-slip bottom wall."""
    from cases import rans_square_model, WALL_TAG
    from dragonfly_sim.core.materials import ConstantDensity
    if fluid == "air":
        kw = dict(farfield={'rho': 1.0, 'p': 1.0, 'M': 0.01, 'alpha': 0.0}, preconditioning=True)
    else:
        kw = dict(farfield={'U': 5.0, 'T': 293.0, 'p': 101325.0, 'alpha': 0.0},
                  material=ConstantDensity())
    m = rans_square_model(turbulence_model, n=n, Re=Re, wall_bottom=True, jitter=0.0, **kw)
    m.report_positivity_limiter = False
    m.max_newton_iterations, m.residual_conv_limit = 100, 1e-10
    m.compute_initial_conditions_from_farfield()
    m.interpolate_solution_vector()
    m.define_farfield_bc(1)
    m.define_wall_bc(WALL_TAG)
    m.compute_weakform()
    m.set_up_solver()
    m.solve_system()
    return m.u_vec.x.array.reshape(-1, m.n_state).copy(), bool(m.solver_converged)


@pytest.mark.skipif(__import__("mpi4py").MPI.COMM_WORLD.size > 1, reason="serial comparison of fields")
@pytest.mark.parametrize("closure", ["laminar", "sa"])
@pytest.mark.verification
def test_air_at_low_mach_and_water_give_the_same_nondimensional_flow(closure):
    """The same code with only the material swapped: at equal Reynolds
    number, air at M = 0.01 and water agree to O(M^2) in p'/(rho u^2), u and
    the turbulence variable (only the passive temperature field may differ,
    through the Prandtl number)."""
    from dragonfly_sim.core.turbulence_models import Laminar, SpalartAllmarasNeg
    tm = (lambda: Laminar()) if closure == "laminar" else (lambda: SpalartAllmarasNeg())
    air, ok_a = _plate("air", tm())
    water, ok_w = _plate("water", tm())
    assert ok_a and ok_w
    flow = [0, 1, 2] + list(range(4, air.shape[1]))
    np.testing.assert_allclose(water[:, flow], air[:, flow], rtol=0, atol=1e-4)
