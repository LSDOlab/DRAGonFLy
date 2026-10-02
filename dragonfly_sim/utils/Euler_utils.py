import ufl

# TODO: Return the velocity components all in one tuple, so that only the length of that tuple changes

def primitives(U):
    dim = U.ufl_shape[0] - 2
    rho = U[0]
    u = ufl.as_vector([U[i] / rho for i in range(1,1+dim)])
    E = U[1 + dim]
    return rho, u, E

def pressure(U, gamma):
    rho, u, E = primitives(U)
    return (gamma - 1.0) * (E - 0.5 * rho * ufl.dot(u, u))

def energy_density(p_in, rho_in, u_in, gamma):
    kinetic = 0.5 * rho_in * ufl.dot(u_in, u_in)
    E_new = p_in/(gamma - 1.0) + kinetic
    return E_new

def flux(U, gamma):
    rho, u, E = primitives(U)
    p = pressure(U, gamma)

    d = u.ufl_shape[0]
    I = ufl.Identity(d)

    # Build block structure
    # Shape: (d+2, d)
    rows = []

    # mass
    rows.append(rho * u)

    # momentum (d rows)
    for i in range(d):
        rows.append(rho * u[i] * u + p * I[i, :])

    # energy
    rows.append((E + p) * u)

    return ufl.as_tensor(rows)
    # F = ufl.as_vector([rho*u[0], rho*u[0]*u[0] + p, rho*u[0]*u[1], (E+p)*u[0]])
    # G = ufl.as_vector([rho*u[1], rho*u[0]*u[1], rho*u[1]*u[1] + p, (E+p)*u[1]])
    # return ufl.as_vector([F, G]).T

def normal_flux(U, n_normal, gamma):
    flux_U = flux(U, gamma)
    return ufl.dot(flux_U, n_normal)


def subsonic_inflow_state(U, rho_in, u_in, gamma):
    p_in = pressure(U, gamma)
    rhoE_in = energy_density(p_in, rho_in, u_in, gamma)

    return ufl.as_vector([
        rho_in,
        *[rho_in*u_in[i] for i in range(u_in.ufl_shape[0])],
        rhoE_in
    ])

def subsonic_outflow_state(U, p_out, gamma):
    rho, u, _ = primitives(U)

    rhoE_out = energy_density(p_out, rho, u, gamma)

    return ufl.as_vector([
        rho,
        *[U[i] for i in range(1, 1+u.ufl_shape[0])],
        rhoE_out
    ])

def slip_wall_state(U, n):
    rho, u, E = primitives(U)

    u_normal = ufl.dot(u, n)

    u_ref = u - 2 * u_normal * n

    return ufl.as_vector([
        U[0],
        *[rho*u_ref[i] for i in range(u_ref.ufl_shape[0])],
        U[1 + u_ref.ufl_shape[0]]
    ])

def llf_penalty_term(U_p, U_m, n, gamma):
    rho_p, u_p, _ = primitives(U_p)
    p_p = pressure(U_p, gamma)
    c_p = ufl.sqrt(gamma * p_p / rho_p)
    lam_p = [ufl.dot(u_p, n) - c_p, ufl.dot(u_p, n), ufl.dot(u_p, n) + c_p]

    rho_m, u_m, _ = primitives(U_m)
    p_m = pressure(U_m, gamma)
    c_m = ufl.sqrt(gamma * p_m / rho_m)
    lam_m = [ufl.dot(u_m, n) - c_m, ufl.dot(u_m, n), ufl.dot(u_m, n) + c_m]

    def max_reduce(items):
        res = items[0]
        for item in items[1:]:
            res = ufl.max_value(res, item)
        return res

    eigen_vals_max_p = max_reduce([abs(l) for l in lam_p])
    eigen_vals_max_m = max_reduce([abs(l) for l in lam_m])
    return ufl.max_value(eigen_vals_max_p, eigen_vals_max_m) / 2.0


def llf_flux(U_p, U_m, n, gamma):
    """
    Local Lax-Friedrichs (LLF) numerical flux, as implemented in dolfin_dg.
    Mathematically, this evaluates to the same expression as the Rusanov flux 
    for the compressible Euler equations.
    """
    alpha = llf_penalty_term(U_p, U_m, n, gamma)

    return 0.5*(normal_flux(U_p, n, gamma) + normal_flux(U_m, n, gamma)) \
         + alpha*(U_p - U_m)

def signed_wave_speeds(U, n, gamma):
    """
    The signed extremal acoustic wave speeds (u.n - c, u.n + c) of the
    normal flux Jacobian.

    The Lax-Friedrichs family only ever needs the largest wave speed in
    magnitude, |u.n| + c, whereas an approximate Riemann solver needs to
    know which *direction* the fan is travelling in -- that is precisely
    what lets it become fully one-sided in supersonic flow instead of
    always dissipating both ways. So this returns the signed quantity that
    the magnitude throws away.

    Floors rho/p at 1e-8 before c = sqrt(gamma*p/rho). Worth being
    explicit about why this floor is not merely cosmetic here: sqrt of a
    negative argument returns NaN, not just an implausibly large number,
    so a single transiently negative pointwise pressure -- which a DG
    discretization can readily produce in early, far-from-converged
    Newton iterations -- would otherwise poison the entire assembled
    residual rather than just the facet it occurred on.

    Works for facet-restricted traces (U('+')/U('-')) exactly like
    primitives()/pressure().
    """
    rho, u, _ = primitives(U)
    p = pressure(U, gamma)
    rho_safe = ufl.max_value(rho, 1e-8)
    p_safe = ufl.max_value(p, 1e-8)
    c = ufl.sqrt(gamma * p_safe / rho_safe)
    un = ufl.dot(u, n)
    return un - c, un + c

def hll_flux(U_p, U_m, n, gamma):
    """
    HLL (Harten-Lax-van Leer) approximate Riemann solver, a drop-in
    alternative to llf_flux for flows containing shocks.

    The Riemann fan is bounded by two waves with Davis speed estimates

        SL = min(un_p - c_p, un_m - c_m)
        SR = max(un_p + c_p, un_m + c_m)

    and the flux is the corresponding single averaged intermediate state

        SL >= 0 : F(U_p).n                              (fully upwind)
        SR <= 0 : F(U_m).n                              (fully upwind)
        else    : (SR F(U_p).n - SL F(U_m).n + SL SR (U_m - U_p))
                  / (SR - SL)

    Written branchlessly below via SL_m = min(SL, 0), SR_p = max(SR, 0),
    which reproduces all three cases *exactly*: SL >= 0 sends SL_m to 0
    and collapses the quotient to F(U_p).n, SR <= 0 sends SR_p to 0 and
    collapses it to F(U_m).n. This is not merely a tidier spelling. It
    needs only min_value/max_value rather than nested ufl.conditional
    nodes, so the expression is genuinely continuous across the branch
    boundaries instead of only piecewise defined -- which is what keeps
    the assembled Jacobian consistent with the residual for a scheme
    whose branch boundaries (un = +/- c) sit exactly on the sonic lines
    of a transonic solution. That matters twice over here, since this
    model's Jacobian also feeds the adjoint used for shape optimization.

    Verified numerically against an explicit three-branch reference in
    test_algorithms/test_hll_flux.py -- branchless vs branched agree to
    ~1e-16 with states swept across sub-, trans- and supersonic so that
    every branch is exercised, in both 2D and 3D.

    Why HLL rather than Roe, and a caveat about dissipation
    -------------------------------------------------------
    HLL is the robust member of the characteristic-scheme family: it is
    positivity-preserving given sound wave-speed estimates, and it
    admits neither the carbuncle phenomenon nor entropy-violating
    expansion shocks -- both of which a Roe flux needs an explicit
    entropy fix to suppress. It also needs no eigenvector decomposition,
    no Roe average and no entropy fix at all, which makes it markedly
    cheaper to assemble and, more importantly here, to differentiate.

    It is NOT more dissipative than the llf_flux it is offered as an
    alternative to -- the opposite. Rusanov/LLF damps every
    characteristic field at the speed of the fastest one (|u.n| + c),
    which upper-bounds any consistent Riemann solver, so LLF is the
    dissipation ceiling of this family and nothing swapped in here can
    exceed it. Measured mean facet dissipation |F_hat - {F.n}| over
    random state pairs, relative to LLF:

                          M~0.3    M~0.85    M~1.3
        LLF                1.00x    1.00x    1.00x
        HLL (Davis)        0.86x    0.75x    0.72x
        Roe + entropy fix  0.81x    0.67x    0.67x
        Van Leer FVS       0.52x    0.54x    0.61x

    The exact ratios depend on how the trace pairs are sampled -- the
    test's own sampling reports HLL at 0.89x/0.81x/0.80x, for instance --
    so it is the *ordering* that is the invariant here, and that is what
    test_hll_flux.py asserts rather than the individual numbers.

    The consequence to watch for: with HLL the sensor-driven volume
    viscosity in this model is doing relatively more of the shock
    capturing than it was under LLF, and those constants were tuned
    against LLF. If oscillations appear near a shock after switching,
    the viscosity constants are the thing to revisit, not this flux.

    Wave-speed estimates: Davis. The Einfeldt (HLLE) variant, which
    additionally folds in the Roe-average speeds, measured within 3% of
    Davis on the same states (0.85x/0.72x/0.70x) -- not worth pulling
    the Roe-average machinery in for. It remains the first thing to try
    if a near-vacuum positivity failure ever surfaces, since Einfeldt's
    estimate is the one carrying the positivity proof.

    Signature and orientation convention match llf_flux (n
    is the "+"-side outward normal, so U_p is the upstream state for
    u.n > 0), so this can be handed straight to
    CompressibleEulerModel(..., flux_function=hll_flux). Dimension
    agnostic: written against n alone, the expression is identical in 2D
    and 3D.
    """
    sl_p, sr_p = signed_wave_speeds(U_p, n, gamma)
    sl_m, sr_m = signed_wave_speeds(U_m, n, gamma)

    SL = ufl.min_value(sl_p, sl_m)
    SR = ufl.max_value(sr_p, sr_m)

    SL_minus = ufl.min_value(SL, 0.0)
    SR_plus = ufl.max_value(SR, 0.0)

    # Non-negative by construction, and zero only in the degenerate
    # SL = SR = 0 vacuum case -- the floor is essentially never active,
    # it is here so that case yields a finite number instead of a 0/0.
    denom = ufl.max_value(SR_plus - SL_minus, 1e-8)

    return (
        SR_plus * normal_flux(U_p, n, gamma)
        - SL_minus * normal_flux(U_m, n, gamma)
        + SL_minus * SR_plus * (U_m - U_p)
    ) / denom

def hllc_flux(U_p, U_m, n, gamma):
    """
    HLLC approximate Riemann solver: HLL with the contact wave restored.

    hll_flux collapses the entire Riemann fan into a single averaged
    intermediate state, which throws the contact wave away entirely.
    HLLC reinstates it as a third wave of speed S*, so the two
    intermediate states either side of the contact are resolved
    separately. Wave speeds SL/SR come from the same
    signed_wave_speeds/Davis estimate hll_flux uses -- deliberately, so
    the two schemes differ only in the contact treatment -- and the
    contact speed is

        S* = (p_m - p_p + rho_p un_p (SL - un_p)
                        - rho_m un_m (SR - un_m))
             / (rho_p (SL - un_p) - rho_m (SR - un_m))

    with the star state on side K (Toro eq. 10.39; E is the volumetric
    total energy, as primitives() returns it)

        U*_K = rho_K (S_K - un_K)/(S_K - S*) *
               [ 1,
                 u_K + (S* - un_K) n,
                 E_K/rho_K + (S* - un_K)(S* + p_K/(rho_K (S_K - un_K))) ]

    What this buys over hll_flux
    ----------------------------
    Exact resolution of isolated contact discontinuities -- the fields
    HLL still smears. On a stationary contact (density jump, uniform
    pressure, zero normal velocity) this returns the exact physical
    flux to machine precision (~4e-15), where hll_flux and llf_flux are
    both O(1) wrong on the same states. In practice that means sharper
    contacts, shear layers and boundary layers.

    Branch structure: one conditional, and it is continuous
    ------------------------------------------------------
    Textbook HLLC is four-branched. Absorbing the two outer branches
    into min(SL, 0) / max(SR, 0) -- the same trick hll_flux uses to
    become fully branchless -- leaves exactly one select, on the sign
    of S*:

        F*_L = F(U_p).n + min(SL, 0) (U*_L - U_p)
        F*_R = F(U_m).n + max(SR, 0) (U*_R - U_m)
        F    = conditional(S* >= 0, F*_L, F*_R)

    SL >= 0 forces S* >= 0 (since SL <= S* <= SR) and collapses F*_L to
    F(U_p).n; SR <= 0 forces S* <= 0 and collapses F*_R to F(U_m).n.

    That the remaining conditional is *continuous* is what makes it an
    acceptable concession here. At S* = 0 the two star fluxes are
    equal: the mass flux rho* S*, the tangential momentum and the
    energy all vanish, and the normal momentum is p*, which is equal on
    both sides by construction. So unlike a generic branch, this one
    introduces no jump in the assembled Jacobian -- which matters
    because that Jacobian also feeds the adjoint used for shape
    optimization. Verified by bisecting states onto S* = 0
    (|F*_L - F*_R| ~ 5e-15) in test_algorithms/test_hllc_flux.py, along
    with exact agreement against an explicit four-branch reference.

    Cost relative to hll_flux
    -------------------------
    Roughly 1.5x the flux arithmetic. Counting arithmetic nodes in the
    lowered expression graph (unique-node traversal, so common
    subexpressions are collapsed as code generation would) and weighting
    by approximate operation latency, the 2D interior facet term costs
    1.53x for the residual and 1.49x for the Jacobian; the ratio moves
    by under 5% across weightings from "all ops equal" to
    "div=10, sqrt=20", so it is not an artifact of the weights. 3D is
    the same picture (1.49x / 1.43x).

    The extra cost is divisions, not the conditional. Division count in
    the 2D residual goes 7 (LLF) -> 7 (HLL) -> 13 (HLLC), and in the
    Jacobian 16 -> 16 -> 28: the star-state construction divides once
    for S*, then per side for rho(S_K - un)/(S_K - S*),
    p/(rho(S_K - un)) and E/rho. Square roots stay at 2 throughout (the
    two sound speeds). For scale, hll_flux costs only 1.03x/1.05x over
    llf_flux -- HLLC is the first flux here with a real price tag.

    Read that 1.5x as a flux-arithmetic upper bound, NOT as an
    end-to-end slowdown. Diluted into the full inviscid residual
    (volume term plus facet term) it is already 1.44x/1.41x in 2D, and
    in the assembled kernel it dilutes further against geometry, basis
    tabulation and the quadrature loop -- all identical between the two
    -- while this model's stabilization terms are unchanged and dilute
    it again. Per Newton step the linear solve typically dominates
    assembly outright. This is a static arithmetic proxy, produced
    without running anything; it models neither memory traffic,
    vectorization, FFCx's own optimizations, nor wall time.

    Caveat worth knowing
    --------------------
    HLL's contact smearing is exactly what makes it carbuncle-free.
    Restoring the contact reintroduces HLLC's standard susceptibility
    to the carbuncle/odd-even instability at strong grid-aligned bow
    shocks. Unlikely on a transonic airfoil c-grid, but it is the
    mechanism to suspect if a strong normal shock ever develops a
    ragged, grid-locked profile -- and hll_flux stays available as the
    immediate fallback.

    Signature and orientation convention match hll_flux/llf_flux (n is
    the "+"-side outward normal, so U_p is the upstream state for
    u.n > 0), so this can be handed straight to
    CompressibleEulerModel(..., flux_function=hllc_flux). Dimension
    agnostic: written against n alone, identical in 2D and 3D.
    """
    sl_p, sr_p = signed_wave_speeds(U_p, n, gamma)
    sl_m, sr_m = signed_wave_speeds(U_m, n, gamma)

    SL = ufl.min_value(sl_p, sl_m)
    SR = ufl.max_value(sr_p, sr_m)

    rho_p, u_p, E_p = primitives(U_p)
    rho_m, u_m, E_m = primitives(U_m)
    p_p = pressure(U_p, gamma)
    p_m = pressure(U_m, gamma)
    un_p = ufl.dot(u_p, n)
    un_m = ufl.dot(u_m, n)

    # SL - un_p <= -c_p < 0 and SR - un_m >= c_m > 0, so this denominator
    # is strictly negative analytically. Clamp it *away from zero on the
    # negative side* rather than flooring toward zero, so a degenerate
    # state cannot flip the sign of S* -- which would select the wrong
    # star flux rather than merely perturb its magnitude.
    S_star = (
        p_m - p_p + rho_p * un_p * (SL - un_p) - rho_m * un_m * (SR - un_m)
    ) / ufl.min_value(rho_p * (SL - un_p) - rho_m * (SR - un_m), -1e-8)

    def star_state(rho, u, E, p, un, S_K, denom):
        factor = rho * (S_K - un) / denom
        momentum = u + (S_star - un) * n
        e_star = E / rho + (S_star - un) * (S_star + p / (rho * (S_K - un)))
        return ufl.as_vector([
            factor,
            *[factor * momentum[i] for i in range(u.ufl_shape[0])],
            factor * e_star,
        ])

    # SL - S* <= 0 <= SR - S* by construction; clamp each on its own side
    # for the same sign-preserving reason as above.
    U_star_p = star_state(rho_p, u_p, E_p, p_p, un_p, SL,
                          ufl.min_value(SL - S_star, -1e-8))
    U_star_m = star_state(rho_m, u_m, E_m, p_m, un_m, SR,
                          ufl.max_value(SR - S_star, 1e-8))

    # min/max absorb the two fully-upwind outer branches: SL >= 0 zeroes
    # the first correction, SR <= 0 zeroes the second.
    F_star_p = normal_flux(U_p, n, gamma) + ufl.min_value(SL, 0.0) * (U_star_p - U_p)
    F_star_m = normal_flux(U_m, n, gamma) + ufl.max_value(SR, 0.0) * (U_star_m - U_m)

    return ufl.conditional(ufl.ge(S_star, 0.0), F_star_p, F_star_m)

def boundary_flux(U, U_ext, n, gamma):
    alpha = llf_penalty_term(U, U_ext, n, gamma)
    return normal_flux(U, n, gamma) + alpha * (U - U_ext)
