"""
Inviscid numerical fluxes on thermodynamic states.

Every flux takes its two facet traces as core/materials.py:ThermoStates,
which carry the conserved state U(V), the physical flux, 1/c^2 and, with
low-Mach preconditioning, 1/U_r^2. The residual is therefore conservative in
U whatever the unknowns are, and no flux knows which fluid it is evaluating.
Orientation: n is the '+' side's outward normal, so th_p is the upstream
state for u.n > 0. Transported scalars (rho phi) are part of U and of the
physical flux, so every flux carries them with the mean flow's waves.

Low-Mach preconditioning (Weiss & Smith) enters in two places only:

  wave speeds   u'_n +- c',  u' = u_n (1 - a),  c' = sqrt(a^2 u_n^2 + U_r^2),
                a = (1 - U_r^2 / c^2) / 2
  dissipation   U_m - U_p  ->  Gamma (V_m - V_p)
                = U_m - U_p + (1/U_r^2 - 1/c^2) (p'_m - p'_p) (1, u, H, phi)

(dissipation_jump; the second line is exact to first order in the jump).
At U_r = c both reduce to the unpreconditioned flux, which is what a state
without inv_Ur2 evaluates.
"""
import ufl


def smooth_max(a, b, kappa=0.1):
    """
    A smooth max of two non-negative expressions,

        (a + b + sqrt((a - b)^2 + kappa^2 (a^2 + b^2))) / 2,

    scale-invariant, within kappa/sqrt(2) of max(a, b) and differentiable
    everywhere (ufl.max_value's kink stalls Newton).
    """
    return 0.5 * (a + b + ufl.sqrt((a - b) ** 2 + kappa ** 2 * (a * a + b * b)))


def sound_speed_squared(th):
    """c^2 of a compressible state, floored so a transiently non-physical
    state (negative T) gives a tiny sound speed instead of a NaN."""
    return ufl.max_value(1.0 / th.inv_c2, 1e-12)


def wave_speeds(th, n):
    """
    The signed extremal wave speeds (u'_n - c', u'_n + c') of the
    (preconditioned) normal flux Jacobian, with n the normal they are taken
    along. Without preconditioning these are u.n -+ c.
    """
    un = ufl.dot(th.u, n)
    if th.inv_Ur2 is None:
        c = ufl.sqrt(sound_speed_squared(th))
        return un - c, un + c
    a = 0.5 * (1.0 - th.inv_c2 / th.inv_Ur2)
    up = un * (1.0 - a)
    cp = ufl.sqrt(a * a * un * un + 1.0 / th.inv_Ur2)
    return up - cp, up + cp


def spectral_radius(th):
    """The largest wave speed |u| + c (|u'| + c' with preconditioning), the
    inverse time scale of the pseudo-time step at h = 1."""
    speed = ufl.sqrt(ufl.dot(th.u, th.u))
    if th.inv_Ur2 is None:
        return speed + ufl.sqrt(sound_speed_squared(th))
    a = 0.5 * (1.0 - th.inv_c2 / th.inv_Ur2)
    return speed * (1.0 - a) + ufl.sqrt(a * a * speed * speed + 1.0 / th.inv_Ur2)


def dissipation_jump(th_p, th_m):
    """U_m - U_p, or its preconditioned form Gamma (V_m - V_p) (module
    docstring), with the preconditioning coefficients of the facet average."""
    dU = th_m.U - th_p.U
    if th_p.inv_Ur2 is None:
        return dU
    avg = lambda a, b: 0.5 * (a + b)
    u, d = avg(th_p.u, th_m.u), th_p.u.ufl_shape[0]
    w = ufl.as_vector([1.0] + [u[i] for i in range(d)] + [avg(th_p.H, th_m.H)]
                      + [avg(a, b) for a, b in zip(th_p.phi, th_m.phi)])
    coeff = avg(th_p.inv_Ur2, th_m.inv_Ur2) - avg(th_p.inv_c2, th_m.inv_c2)
    return dU + coeff * (th_m.p - th_p.p) * w


def slip_wall_state(V, n, n_mean):
    """The mirror state of V: normal velocity reversed, everything else kept."""
    d = n_mean - 2
    u = ufl.as_vector([V[1 + i] for i in range(d)])
    u_ref = u - 2.0 * ufl.dot(u, n) * n
    return ufl.as_vector([V[0]] + [u_ref[i] for i in range(d)] + [V[1 + d]])


def llf_penalty_term(th_p, th_m, n):
    """Half the largest wave speed magnitude of the two traces."""
    def largest(th):
        sl, sr = wave_speeds(th, n)
        return ufl.max_value(abs(sl), abs(sr))
    return ufl.max_value(largest(th_p), largest(th_m)) / 2.0


def llf_flux(th_p, th_m, n):
    """
    Local Lax-Friedrichs (Rusanov) flux, as implemented in dolfin_dg: the
    central flux plus the dissipation of the fastest wave,

        0.5 (F(U_p) + F(U_m)).n + alpha (U_p - U_m),  alpha = max|lambda| / 2.
    """
    alpha = llf_penalty_term(th_p, th_m, n)
    return 0.5 * (th_p.normal_flux(n) + th_m.normal_flux(n)) - alpha * dissipation_jump(th_p, th_m)


def hll_flux(th_p, th_m, n):
    """
    HLL (Harten-Lax-van Leer) approximate Riemann solver, the default flux.

    The Riemann fan is bounded by two waves with Davis speed estimates

        SL = min(sl_p, sl_m),  SR = max(sr_p, sr_m)      (wave_speeds)

    and the flux is the corresponding single averaged intermediate state

        SL >= 0 : F(U_p).n                              (fully upwind)
        SR <= 0 : F(U_m).n                              (fully upwind)
        else    : (SR F(U_p).n - SL F(U_m).n + SL SR (U_m - U_p)) / (SR - SL)

    Written branchlessly via SL_m = min(SL, 0), SR_p = max(SR, 0), which
    reproduces all three cases exactly with min/max only, so the expression
    is continuous across the sonic lines of a transonic solution and the
    Jacobian (which also feeds the shape-optimization adjoint) stays
    consistent with the residual.

    HLL is positivity preserving with sound wave-speed estimates, admits
    neither carbuncles nor expansion shocks, needs no entropy fix, and is
    less dissipative than llf_flux (which upper-bounds this family). With
    preconditioning the wave speeds are the preconditioned ones and the
    jump U_m - U_p is Gamma (V_m - V_p); the mass equation then carries the
    pressure-jump dissipation (p'_m - p'_p) / U_r that keeps low-Mach
    pressure fields free of checkerboards.
    """
    sl_p, sr_p = wave_speeds(th_p, n)
    sl_m, sr_m = wave_speeds(th_m, n)
    SL_minus = ufl.min_value(ufl.min_value(sl_p, sl_m), 0.0)
    SR_plus = ufl.max_value(ufl.max_value(sr_p, sr_m), 0.0)
    # zero only in the degenerate SL = SR = 0 case
    denom = ufl.max_value(SR_plus - SL_minus, 1e-8)
    return (SR_plus * th_p.normal_flux(n) - SL_minus * th_m.normal_flux(n)
            + SL_minus * SR_plus * dissipation_jump(th_p, th_m)) / denom


def hllc_flux(th_p, th_m, n):
    """
    HLLC approximate Riemann solver: HLL with the contact wave restored
    (Toro, ch. 10), unpreconditioned only. Wave speeds SL/SR are hll_flux's
    Davis estimates; the contact speed is

        S* = (p_m - p_p + rho_p un_p (SL - un_p) - rho_m un_m (SR - un_m))
             / (rho_p (SL - un_p) - rho_m (SR - un_m))

    with the star state on side K (E' = rho E' the volumetric total energy;
    gauge pressure p' and E' = E + p_ref leave the scheme unchanged)

        U*_K = rho_K (S_K - un_K)/(S_K - S*) *
               [1, u_K + (S* - un_K) n, E'_K/rho_K + (S* - un_K)(S* + p'_K/(rho_K (S_K - un_K))), phi_K]

    The two outer branches are absorbed into min(SL, 0) / max(SR, 0), which
    leaves one select on the sign of S*; it is continuous (the two star
    fluxes agree at S* = 0), so it adds no jump to the Jacobian. HLLC
    resolves stationary contacts exactly (sharper shear and boundary
    layers) at about 1.5x hll_flux's arithmetic, and reintroduces the
    carbuncle susceptibility HLL's contact smearing avoids.
    """
    if th_p.inv_Ur2 is not None:
        raise NotImplementedError("hllc_flux has no preconditioned form; use hll_flux or llf_flux")
    sl_p, sr_p = wave_speeds(th_p, n)
    sl_m, sr_m = wave_speeds(th_m, n)
    SL = ufl.min_value(sl_p, sl_m)
    SR = ufl.max_value(sr_p, sr_m)
    un_p, un_m = ufl.dot(th_p.u, n), ufl.dot(th_m.u, n)
    rho_p, rho_m = th_p.rho, th_m.rho

    # strictly negative analytically: clamp away from zero on its own side,
    # so a degenerate state cannot flip the sign of S*
    S_star = (th_m.p - th_p.p + rho_p * un_p * (SL - un_p) - rho_m * un_m * (SR - un_m)) \
        / ufl.min_value(rho_p * (SL - un_p) - rho_m * (SR - un_m), -1e-8)

    def star_state(th, un, S_K, denom):
        U = th.U
        d = th.u.ufl_shape[0]
        factor = th.rho * (S_K - un) / denom
        momentum = th.u + (S_star - un) * n
        e_star = U[1 + d] / th.rho + (S_star - un) * (S_star + th.p / (th.rho * (S_K - un)))
        return ufl.as_vector([factor] + [factor * momentum[i] for i in range(d)]
                             + [factor * e_star] + [factor * s for s in th.phi])

    U_star_p = star_state(th_p, un_p, SL, ufl.min_value(SL - S_star, -1e-8))
    U_star_m = star_state(th_m, un_m, SR, ufl.max_value(SR - S_star, 1e-8))
    F_star_p = th_p.normal_flux(n) + ufl.min_value(SL, 0.0) * (U_star_p - th_p.U)
    F_star_m = th_m.normal_flux(n) + ufl.max_value(SR, 0.0) * (U_star_m - th_m.U)
    return ufl.conditional(ufl.ge(S_star, 0.0), F_star_p, F_star_m)


def boundary_flux(th, th_b, n):
    """One-sided Lax-Friedrichs flux against the boundary state th_b:
    F(U).n + alpha (U - U_b)."""
    alpha = llf_penalty_term(th, th_b, n)
    return th.normal_flux(n) - alpha * dissipation_jump(th, th_b)
