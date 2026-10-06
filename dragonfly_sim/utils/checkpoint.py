"""
Partition-independent checkpoints for time-dependent runs.

A checkpoint is one .npz written on rank 0. Every DG field is stored per cell,
keyed by the cell's index in the INPUT mesh (topology.original_cell_index)
and the cell-local dof order, so a run may be restarted on any number of
ranks. The dof coordinates are stored as well and checked on load.
"""
import json

import numpy as np
from mpi4py import MPI


def _owned_cell_dofs(V):
    """(owned cells' input-mesh indices, their dof blocks, the dof coordinates)."""
    mesh = V.mesh
    n_owned = mesh.topology.index_map(mesh.topology.dim).size_local
    dofs = V.dofmap.list[:n_owned]
    orig = np.asarray(mesh.topology.original_cell_index[:n_owned], dtype=np.int64)
    return orig, dofs, V.tabulate_dof_coordinates()[dofs]


def save_checkpoint(path, functions, meta):
    """
    Write `functions` (DG Functions on one mesh, distinct names) and the JSON
    serializable dict `meta` to `path`. The dof coordinates are stored as
    "coords" for the first function's space and "<name>__coords" for any
    function on another space. Collective.
    """
    comm = functions[0].function_space.mesh.comm
    V0 = functions[0].function_space
    local = {}
    for f in functions:
        V = f.function_space
        orig, dofs, coords = _owned_cell_dofs(V)
        local["orig"] = orig  # the same owned cells for every space
        local["coords" if V == V0 else f.name + "__coords"] = coords
        local[f.name] = f.x.array.reshape(-1, V.dofmap.index_map_bs)[dofs]
    gathered = comm.gather(local, root=0)
    if comm.rank == 0:
        orig_all = np.concatenate([g["orig"] for g in gathered])
        order = np.argsort(orig_all)
        assert np.array_equal(orig_all[order], np.arange(orig_all.size)), \
            "owned cells do not tile the input mesh exactly once"
        np.savez(path, meta=json.dumps(meta),
                 **{k: np.concatenate([g[k] for g in gathered])[order]
                    for k in local if k != "orig"})
    comm.Barrier()


def load_checkpoint(path, functions, coord_tol=1e-10, optional=False):
    """
    Fill `functions` from a save_checkpoint file, on any rank count, and
    return its meta dict. With optional=True a field missing from the file is
    left untouched instead of raising (e.g. far-field history when restarting
    a riemann2 run from a plain riemann checkpoint).
    """
    comm = functions[0].function_space.mesh.comm
    data = np.load(path)
    for f in functions:
        if f.name not in data.files:
            if optional:
                continue
            raise KeyError("{} has no field {!r}".format(path, f.name))
        V = f.function_space
        orig, dofs, coords = _owned_cell_dofs(V)
        ckey = f.name + "__coords" if f.name + "__coords" in data.files else "coords"
        ref = data[ckey]
        if ref.shape[0] != comm.allreduce(orig.size, MPI.SUM) or ref.shape[1:] != coords.shape[1:]:
            raise ValueError("checkpoint field {} does not match this mesh/space".format(f.name))
        err = comm.allreduce(float(np.abs(ref[orig] - coords).max(initial=0.0)), MPI.MAX)
        if err > coord_tol:
            raise ValueError("checkpoint dof coordinates of {} are off by {:.2e}".format(f.name, err))
        f.x.array.reshape(-1, V.dofmap.index_map_bs)[dofs] = data[f.name][orig]
        f.x.scatter_forward()
    return json.loads(str(data["meta"]))
