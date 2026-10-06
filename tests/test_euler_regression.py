"""
The default Euler path must stay what it was before the RANS, time
integration and far-field work: the same residual, Jacobian, dR/dalpha and
dR/dx, and the same Newton iterates.

The reference (tests/data/euler_reference.npz) was generated on main @ ccca26b
before those changes by tests/data/make_euler_reference.py. Assembly
quantities are rank-independent (per-cell values ordered by original cell
index, matrices through their action on fixed smooth fields) and compared on
any rank count; the Newton history depends on the ASM partitioning and is
compared serially, where it is bit-identical.
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

from euler_reference_quantities import euler_reference_quantities

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "euler_reference.npz")


@pytest.fixture(scope="module")
def quantities():
    return euler_reference_quantities(), np.load(REFERENCE)


@pytest.mark.parametrize("key", ["residual", "jacobian_norm", "jacobian_action", "dRdalpha",
                                 "dRdx_norm", "dRdx_action"])
def test_assembly_matches_reference(quantities, key):
    q, ref = quantities
    serial = MPI.COMM_WORLD.size == 1
    tol = 0.0 if serial else 1e-12 * max(np.abs(ref[key]).max(), 1.0)
    np.testing.assert_allclose(q[key], ref[key], rtol=0, atol=tol)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="the Newton iterates depend on the partitioning")
@pytest.mark.parametrize("key", ["newton_history", "newton_state"])
def test_newton_matches_reference(quantities, key):
    q, ref = quantities
    np.testing.assert_array_equal(q[key], ref[key])
