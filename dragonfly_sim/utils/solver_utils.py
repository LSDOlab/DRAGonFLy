from petsc4py import PETSc
import numpy as np


_KSP_REASON_NAMES = None


_KSP_NORM_TYPE_NAMES = None


def _enum_name_map(enum_cls):
    """
    Reverse map {code: name} for a petsc4py enum-like class.

    petsc4py exposes most codes under several aliases -- PC.Side has both "L"
    and "LEFT" for 0, KSP.ConvergedReason both "DIVERGED_PC_FAILED" and
    "DIVERGED_PCSETUP_FAILED" for -11 -- so the map has to choose. Longest name
    first, ties broken alphabetically: longest picks the descriptive alias over
    the terse one (which is the whole point of printing a name instead of the
    integer), and the tie-break makes the result identical on every run rather
    than dependent on `dir()` ordering.
    """
    names = {}
    for name in sorted((n for n in dir(enum_cls) if not n.startswith("_")),
                       key=lambda n: (-len(n), n)):
        value = getattr(enum_cls, name)
        if isinstance(value, int):
            names.setdefault(value, name)
    return names


def ksp_converged_reason_name(reason):
    """
    Human-readable name for a PETSc KSP convergence reason code.

    Worth printing next to the iteration count because the count alone cannot
    distinguish the two failures that call for opposite responses:
    DIVERGED_MAX_IT means a hard but well-posed operator that simply exhausted
    its Krylov budget (tune the solver), while DIVERGED_BREAKDOWN or
    DIVERGED_PCSETUP_FAILED means the Arnoldi process or the preconditioner
    itself failed (the operator or the coarse space is broken). Both show up as
    "ran to max_it" in a log that reports only iterations.

    Unknown codes come back as "REASON(<n>)" rather than raising -- this is a
    diagnostic and must never be the thing that breaks a solve.
    """
    global _KSP_REASON_NAMES
    if _KSP_REASON_NAMES is None:
        _KSP_REASON_NAMES = _enum_name_map(PETSc.KSP.ConvergedReason)
    return _KSP_REASON_NAMES.get(int(reason), "REASON({})".format(reason))


def ksp_norm_type_name(norm_type):
    """
    Human-readable name for a PETSc KSP norm type.

    This is what says whether a reported residual norm is the true ||b - Ax||
    or the preconditioned one, which decides whether a norm LARGER than ||b||
    is impossible (UNPRECONDITIONED, as fgmres defaults to) or unremarkable
    (PRECONDITIONED, as gmres defaults to). Without it the number cannot be
    read at all.
    """
    global _KSP_NORM_TYPE_NAMES
    if _KSP_NORM_TYPE_NAMES is None:
        _KSP_NORM_TYPE_NAMES = _enum_name_map(PETSc.KSP.NormType)
    return _KSP_NORM_TYPE_NAMES.get(int(norm_type), "NORM({})".format(norm_type))


def ksp_pc_side_name(pc_side):
    """Human-readable name for a PETSc preconditioner side (LEFT/RIGHT/SYMMETRIC)."""
    return _enum_name_map(PETSc.PC.Side).get(
        int(pc_side), "SIDE({})".format(pc_side))


# Floor on density and pressure enforced by the positivity limiter below.
# Shared with locate_positivity_limiting_nodes so the diagnostic reports on
# exactly the constraint the limiter applied -- a diagnostic running against a
# floor of its own would name the wrong nodes as soon as the two drifted.
# Lower bound density and pressure are held above. A state is REJECTED at
# p <= POSITIVITY_FLOOR, so this is not merely a guard against negative
# pressure -- it reserves a margin, and a solve whose transient legitimately
# passes through a low-pressure state is blocked by it.
POSITIVITY_FLOOR = 1e-4


# Smallest step length the halving search will accept. Below this the search
# gives up and returns theta = 0.0, freezing the state: see the tail of
# compute_positivity_preserving_theta. Module-level rather than a keyword
# default because no call site passes min_theta -- every path takes the
# default -- so this is the only place it can be varied.
MIN_POSITIVITY_THETA = 1e-6


def _mean_flow_blocks(x_arr, dx_arr, block_size, n_mean):
    """
    Drop the trailing transported scalars (turbulence variables) from every
    block, so the density/pressure checks below see (rho, rho u, rho E) only.
    Returns (x, dx, block size); a no-op when n_mean is None or the blocks
    are already mean flow.
    """
    if n_mean is None or n_mean >= block_size:
        return x_arr, dx_arr, block_size
    n_blocks = len(x_arr) // block_size
    x = x_arr[:n_blocks * block_size].reshape(-1, block_size)[:, :n_mean].ravel()
    dx = dx_arr[:n_blocks * block_size].reshape(-1, block_size)[:, :n_mean].ravel()
    return x, dx, n_mean


def compute_positivity_preserving_theta(comm, x_arr, dx_arr, gamma=1.4, initial_theta=1.0, min_theta=None, block_size=4,
                                        n_mean=None):
    """
    Computes the maximum theta <= initial_theta such that x_new = x_arr - theta * dx_arr
    has positive mass density and positive pressure everywhere.

    Returns the step length only. To find out WHICH nodes bounded it -- the
    question a collapsed theta raises -- pass twice the returned value to
    locate_positivity_limiting_nodes.

    With n_mean given, each block of block_size entries is (rho, rho u, rho E)
    followed by transported scalars, which are not limited.
    """
    import numpy as np
    if min_theta is None:
        min_theta = MIN_POSITIVITY_THETA
    x_arr, dx_arr, block_size = _mean_flow_blocks(x_arr, dx_arr, block_size, n_mean)
    theta = initial_theta

    # Determine the number of valid full blocks
    n_blocks = len(x_arr) // block_size
    if n_blocks == 0:
        # No local degrees of freedom
        local_theta = theta
    else:
        x_nodes = x_arr[:n_blocks*block_size].reshape(-1, block_size)
        dx_nodes = dx_arr[:n_blocks*block_size].reshape(-1, block_size)

        rho = x_nodes[:, 0]
        rhou = x_nodes[:, 1:block_size-1]
        rhoE = x_nodes[:, block_size-1]

        d_rho = dx_nodes[:, 0]
        d_rhou = dx_nodes[:, 1:block_size-1]
        d_rhoE = dx_nodes[:, block_size-1]

        local_theta = theta
        while local_theta >= min_theta:
            rho_new = rho - local_theta * d_rho
            rhou_new = rhou - local_theta * d_rhou
            rhoE_new = rhoE - local_theta * d_rhoE

            # Check density positivity
            if np.any(rho_new <= POSITIVITY_FLOOR):
                local_theta *= 0.5
                continue

            # Check pressure positivity
            kin_energy_new = 0.5 * np.sum(rhou_new**2, axis=1) / rho_new
            p_new = (gamma - 1.0) * (rhoE_new - kin_energy_new)

            if np.any(p_new <= POSITIVITY_FLOOR):
                local_theta *= 0.5
                continue

            break

        if local_theta < min_theta:
            local_theta = 0.0

    if comm.Get_size() > 1:
        from mpi4py import MPI
        global_theta = comm.allreduce(local_theta, op=MPI.MIN)
    else:
        global_theta = local_theta

    return global_theta


def locate_positivity_limiting_nodes(x_arr, dx_arr, theta_probe, gamma=1.4,
                                     block_size=4, floor=POSITIVITY_FLOOR, n_mean=None):
    """
    Find which local nodes make x_arr - theta_probe*dx_arr unphysical.

    Diagnostic companion to compute_positivity_preserving_theta, which returns
    the surviving step length and says nothing about what bounded it. Pass
    theta_probe = twice the accepted theta -- the smallest step that bisection
    actually rejected -- and the nodes reported here are exactly the ones that
    forced the final halving.

    Rank-local and pure numpy: the caller owns the MPI reduction and the
    dof-coordinate lookup. `block` is an index into the BLOCKS of x_arr, which
    for a blocked function space is also a row index into that space's local
    dof coordinates.

    Returns a dict:
        n_density, n_pressure, n_nonfinite
            local counts of nodes violating each way, for the caller to sum
            across ranks. One violating node out of millions and a whole
            region going unphysical at once are different problems, and the
            count is what tells them apart.
        block
            block index of this rank's WORST violator, or None if this rank
            has none -- in which case the keys below are absent.
        severity
            trial value over current value at that node, so that a density
            and a pressure violation are comparable and one argmin ranks the
            whole set; more negative is a worse overshoot, -inf for a
            non-finite trial state.
        constraint
            "density", "pressure" or "non-finite".
        rho, p, rho_trial, p_trial, d_rho, d_rhou_norm, d_rhoE
            the state, the trial state and the update at that node.
    """
    x_arr, dx_arr, block_size = _mean_flow_blocks(x_arr, dx_arr, block_size, n_mean)
    n_blocks = len(x_arr) // block_size
    empty = {"n_density": 0, "n_pressure": 0, "n_nonfinite": 0, "block": None}
    if n_blocks == 0:
        # No local degrees of freedom.
        return empty

    x_nodes = x_arr[:n_blocks*block_size].reshape(-1, block_size)
    dx_nodes = dx_arr[:n_blocks*block_size].reshape(-1, block_size)
    trial = x_nodes - theta_probe * dx_nodes

    rho = x_nodes[:, 0]
    rhou = x_nodes[:, 1:block_size-1]
    rhoE = x_nodes[:, block_size-1]

    rho_trial = trial[:, 0]
    rhou_trial = trial[:, 1:block_size-1]
    rhoE_trial = trial[:, block_size-1]

    # Checked first and excluded from the two comparisons below, because a NaN
    # fails `<= floor` and would otherwise be silently counted as healthy --
    # the one state that most needs reporting.
    nonfinite = ~(np.isfinite(rho_trial) & np.isfinite(rhoE_trial)
                  & np.all(np.isfinite(rhou_trial), axis=1))

    bad_rho = (~nonfinite) & (rho_trial <= floor)

    # Pressure is only meaningful where the trial density survived; where it
    # did not, the density violation is the one worth reporting anyway.
    rho_trial_safe = np.where(rho_trial > floor, rho_trial, 1.0)
    p_trial = (gamma - 1.0) * (rhoE_trial
                               - 0.5*np.sum(rhou_trial**2, axis=1) / rho_trial_safe)
    bad_p = (~nonfinite) & (~bad_rho) & (p_trial <= floor)

    result = dict(empty)
    result["n_density"] = int(np.count_nonzero(bad_rho))
    result["n_pressure"] = int(np.count_nonzero(bad_p))
    result["n_nonfinite"] = int(np.count_nonzero(nonfinite))

    violating = np.flatnonzero(nonfinite | bad_rho | bad_p)
    if violating.size == 0:
        return result

    # The incoming state has already passed the limiter, so rho > floor there;
    # the guard is only so a caller handing in a blown-up state still gets a
    # report rather than a divide-by-zero.
    rho_safe = np.where(rho > floor, rho, 1.0)
    p = (gamma - 1.0) * (rhoE - 0.5*np.sum(rhou**2, axis=1) / rho_safe)

    reference = np.where(bad_rho, np.abs(rho), np.abs(p))
    reference = np.where(reference > 0.0, reference, 1.0)
    severity = np.where(bad_rho, rho_trial, p_trial) / reference
    severity = np.where(nonfinite, -np.inf, severity)

    worst = int(violating[np.argmin(severity[violating])])

    result["block"] = worst
    result["severity"] = float(severity[worst])
    result["constraint"] = ("non-finite" if nonfinite[worst]
                            else "density" if bad_rho[worst] else "pressure")
    result["rho"] = float(rho[worst])
    result["p"] = float(p[worst])
    result["rho_trial"] = float(rho_trial[worst])
    result["p_trial"] = float(p_trial[worst])
    result["d_rho"] = float(dx_nodes[worst, 0])
    result["d_rhou_norm"] = float(np.linalg.norm(dx_nodes[worst, 1:block_size-1]))
    result["d_rhoE"] = float(dx_nodes[worst, block_size-1])
    return result
