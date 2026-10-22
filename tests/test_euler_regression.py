"""
The default Euler path must stay the discretization it was before the RANS,
time integration, far-field and primitive-variable work: the same residual,
Jacobian, dR/dalpha and dR/dx, and the same converged solution.

The reference (tests/data/euler_reference.npz) was generated on main @ ccca26b
before those changes by tests/data/make_euler_reference.py, and regenerated
once on branch Riemann_bc_default when the channel's subsonic inflow/outflow
became the characteristic far field. Only the 16 cells on the far-field
sides differed in the residual, the Jacobian action and dR/dalpha (checked
before regenerating); the interior discretization is untouched.

Since the unknowns became the primitive (p', u, T) scaled with the free
stream, the quantities are mapped onto the reference's conservative
unknowns before comparing (euler_reference_quantities.py): an exact change
of variables and row scaling, so they still agree to round-off. The Newton
iterates depend on the unknowns and are no longer compared; the converged
solution is, against tests/data/euler_reference_converged.npz (the
conservative code at b67683c, tests/data/make_euler_reference_converged.py).

Assembly quantities are rank-independent (per-cell values ordered by
original cell index, matrices through their action on fixed smooth fields)
and compared on any rank count; the Newton solve depends on the ASM
partitioning only through round-off and is compared serially.
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

from euler_reference_quantities import euler_reference_quantities

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REFERENCE = os.path.join(DATA, "euler_reference.npz")
CONVERGED = os.path.join(DATA, "euler_reference_converged.npz")


pytestmark = [pytest.mark.verification]


@pytest.fixture(scope="module")
def quantities():
    return euler_reference_quantities(), dict(np.load(REFERENCE), **np.load(CONVERGED))


@pytest.mark.parametrize("key", ["residual", "jacobian_norm", "jacobian_action", "dRdalpha",
                                 "dRdx_norm", "dRdx_action"])
def test_assembly_matches_reference(quantities, key):
    q, ref = quantities
    serial = MPI.COMM_WORLD.size == 1
    np.testing.assert_almost_equal(q[key], ref[key], decimal=12)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="the Newton solve depends on the partitioning")
def test_newton_converges_to_reference(quantities):
    q, ref = quantities
    assert q["converged_history"][-1] < 1e-13
    np.testing.assert_almost_equal(q["converged_state"], ref["converged_state"], decimal=11)
