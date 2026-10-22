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


# The positivity limiter holds the unknowns above linear lower bounds,
# x_k > bound_k for the constrained components k of every block. With the
# primitive unknowns (p', u, T) these are the absolute pressure and the
# temperature (the material's positivity_constraints, which reserve a
# margin above zero), so no nonlinear state-to-pressure map is needed.
# locate_positivity_limiting_nodes reports on exactly the constraints the
# limiter applied.


# Smallest step length the halving search will accept. Below this the search
# gives up and returns theta = 0.0, freezing the state: see the tail of
# compute_positivity_preserving_theta. Module-level rather than a keyword
# default because no call site passes min_theta -- every path takes the
# default -- so this is the only place it can be varied.
MIN_POSITIVITY_THETA = 1e-6


def _blocks(arr, block_size):
    n_blocks = len(arr) // block_size
    return arr[:n_blocks * block_size].reshape(-1, block_size)


def _feasible(x_nodes, dx_nodes, theta, constraints, max_change=None):
    """True where every constrained component of x - theta dx is above its
    bound and, with max_change, changes by at most max_change times its
    distance x - bound from it."""
    ok = np.ones(x_nodes.shape[0], dtype=bool)
    for k, bound in constraints:
        ok &= x_nodes[:, k] - theta * dx_nodes[:, k] > bound
        if max_change is not None:
            ok &= theta * np.abs(dx_nodes[:, k]) <= max_change * (x_nodes[:, k] - bound)
    return ok


def compute_positivity_preserving_theta(comm, x_arr, dx_arr, constraints, initial_theta=1.0,
                                        min_theta=None, block_size=4, max_change=None):
    """
    The largest theta = initial_theta / 2^k such that x_new = x_arr - theta *
    dx_arr satisfies every constraint (component k, lower bound) in every
    block of block_size entries and, with max_change, changes every
    constrained component by at most max_change times its distance from the
    bound (a relative-change limit: for the absolute pressure and the
    temperature, at most that fraction of their value). Below min_theta it
    gives up and returns 0.

    Returns the step length only. To find out WHICH nodes bounded it -- the
    question a collapsed theta raises -- pass twice the returned value to
    locate_positivity_limiting_nodes.
    """
    if min_theta is None:
        min_theta = MIN_POSITIVITY_THETA
    x_nodes, dx_nodes = _blocks(x_arr, block_size), _blocks(dx_arr, block_size)
    local_theta = initial_theta
    if x_nodes.shape[0]:
        while local_theta >= min_theta and not np.all(_feasible(x_nodes, dx_nodes, local_theta,
                                                                 constraints, max_change)):
            local_theta *= 0.5
        if local_theta < min_theta:
            local_theta = 0.0
    if comm.Get_size() > 1:
        from mpi4py import MPI
        return comm.allreduce(local_theta, op=MPI.MIN)
    return local_theta


def locate_positivity_limiting_nodes(x_arr, dx_arr, theta_probe, constraints, block_size=4,
                                     names=None):
    """
    Find which local nodes make x_arr - theta_probe*dx_arr violate the
    constraints (component k, lower bound).

    Diagnostic companion to compute_positivity_preserving_theta, which returns
    the surviving step length and says nothing about what bounded it. Pass
    theta_probe = twice the accepted theta -- the smallest step that bisection
    actually rejected -- and the nodes reported here are exactly the ones that
    forced the final halving.

    Rank-local and pure numpy: the caller owns the MPI reduction and the
    dof-coordinate lookup. `block` is an index into the BLOCKS of x_arr, which
    for a blocked function space is also a row index into that space's local
    dof coordinates. `names` labels the constrained components.

    Returns a dict:
        counts
            {name: local number of nodes violating that constraint}, plus
            "non-finite", for the caller to sum across ranks. One violating
            node out of millions and a whole region going unphysical at once
            are different problems, and the count is what tells them apart.
        block
            block index of this rank's WORST violator, or None if this rank
            has none -- in which case the keys below are absent.
        severity
            margin of the trial value above its bound over the current
            margin, so violations of different constraints are comparable;
            more negative is a worse overshoot, -inf for a non-finite trial
            state.
        constraint
            the violated constraint's name, or "non-finite".
        value, trial, update
            the constrained component's value, trial value and update there.
    """
    names = list(names) if names is not None else ["x[{}]".format(k) for k, _ in constraints]
    x_nodes, dx_nodes = _blocks(x_arr, block_size), _blocks(dx_arr, block_size)
    result = {"counts": {name: 0 for name in names + ["non-finite"]}, "block": None}
    if x_nodes.shape[0] == 0:
        return result
    trial = x_nodes - theta_probe * dx_nodes
    # checked first and excluded below: a NaN fails every comparison and
    # would otherwise be silently counted as healthy
    nonfinite = ~np.all(np.isfinite(trial), axis=1)
    result["counts"]["non-finite"] = int(np.count_nonzero(nonfinite))
    severity = np.full(x_nodes.shape[0], np.inf)
    which = np.full(x_nodes.shape[0], -1)
    for i, (k, bound) in enumerate(constraints):
        bad = (~nonfinite) & (trial[:, k] <= bound)
        result["counts"][names[i]] = int(np.count_nonzero(bad))
        margin = x_nodes[:, k] - bound
        sev = np.where(bad, (trial[:, k] - bound) / np.where(margin > 0.0, margin, 1.0), np.inf)
        worse = sev < severity
        severity, which = np.where(worse, sev, severity), np.where(worse, i, which)
    severity = np.where(nonfinite, -np.inf, severity)
    violating = np.flatnonzero(nonfinite | (which >= 0))
    if violating.size == 0:
        return result
    worst = int(violating[np.argmin(severity[violating])])
    k = constraints[which[worst]][0] if which[worst] >= 0 else 0
    result.update({"block": worst, "severity": float(severity[worst]),
                   "constraint": "non-finite" if nonfinite[worst] else names[which[worst]],
                   "value": float(x_nodes[worst, k]), "trial": float(trial[worst, k]),
                   "update": float(dx_nodes[worst, k])})
    return result
