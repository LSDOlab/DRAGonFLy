import numpy as np


def set_petsc_vec_array(petsc_vec, np_array):
    """Set the locally-owned entries of a PETSc Vec from a numpy array.

    Works for both sequential and distributed (MPI) Vecs.  For a
    distributed Vec, ``np_array`` must have exactly as many elements as
    the local ownership range of ``petsc_vec`` on this rank, and values
    are placed at the correct *global* indices — not always starting at 0.

    Parameters
    ----------
    petsc_vec : PETSc.Vec
        Target Vec.  May be sequential or MPI-distributed.
    np_array : np.ndarray, 1-D
        Values to write.  Length must equal the local ownership size.
    """
    # getOwnershipRange() returns the half-open global index interval
    # [rstart, rend) owned by this rank.  For a sequential Vec this is
    # always [0, n), so the call is safe in both serial and parallel.
    rstart, rend = petsc_vec.getOwnershipRange()
    global_indices = np.arange(rstart, rend, dtype=np.int32)
    petsc_vec.setValues(global_indices, np_array)
    petsc_vec.assemble()
