"""
Regenerate euler_reference.npz, the bit-level guard for the default Euler path.

The reference was generated on branch main @ ccca26b, BEFORE the RANS / time
integration / far-field work, and must not be regenerated casually: the
point of tests/test_euler_regression.py is that the default Euler forms stay
the same expressions. Only regenerate after a deliberate, reviewed change
to the Euler discretization.

    python tests/data/make_euler_reference.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from euler_reference_quantities import euler_reference_quantities  # noqa: E402

if __name__ == "__main__":
    q = euler_reference_quantities()
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "euler_reference.npz")
        np.savez(out, **q)
        for k, v in q.items():
            print(k, np.shape(v), float(np.linalg.norm(v)))
