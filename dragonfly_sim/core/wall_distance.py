"""
Distance to the wall, as a CG1 DerivedField with an exact mesh derivative.

d at every CG1 node is the distance to the nearest wall facet: the exact
point-to-segment distance for the straight 2D wall facets, the distance to
the nearest wall vertex in 3D (an over-estimate between vertices). The
nearest facets are found with a KD-tree over the facet midpoints (2D) or
the wall vertices (3D).

mesh_jacobian() is the analytic derivative of every nodal distance with
respect to the coordinates of the node itself and of the wall vertices of
its nearest facet. With P the nearest point on segment AB, t its parameter
and r = (x - P)/d,

    dd/dx = r,    dd/dA = -(1 - t) r,    dd/dB = -t r

(t is stationary at an interior foot point, so it does not move to first
order; at a clipped end, t = 0 or 1, the same formula holds). On the wall
itself, d = 0 and the node moves with the wall, so the row is zero. The
derivative is exact wherever the nearest facet does not switch, i.e.
almost everywhere.
"""
import numpy as np
import scipy.spatial
from petsc4py import PETSc
import dolfinx

from dragonfly_sim.utils.derived_fields import DerivedField


def _nearest_segments(x, A, B, k=16):
    """Index of the nearest segment, foot parameter t and foot point, per point."""
    tree = scipy.spatial.cKDTree(0.5 * (A + B))
    k = min(k, len(A))
    _, idx = tree.query(x, k=k)
    idx = idx.reshape(len(x), k)
    Ak, Bk = A[idx], B[idx]
    AB = Bk - Ak
    t = np.clip(np.einsum("nkj,nkj->nk", x[:, None, :] - Ak, AB)
                / np.maximum(np.einsum("nkj,nkj->nk", AB, AB), 1e-300), 0.0, 1.0)
    P = Ak + t[..., None] * AB
    dist = np.linalg.norm(x[:, None, :] - P, axis=2)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(x))
    return idx[rows, j], t[rows, j], P[rows, j], dist[rows, j]


class WallDistance(DerivedField):

    def __init__(self, mesh_obj, wall_tags):
        self.mesh_obj = mesh_obj
        self.wall_tags = list(wall_tags)
        msh = mesh_obj.mesh
        V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
        Vx = mesh_obj.coordinate_space
        # The derivative maps d's dofs onto coordinate_space's: they must be
        # the same nodes in the same numbering.
        if not (np.array_equal(V.dofmap.list, Vx.dofmap.list)
                and V.dofmap.index_map.size_local == Vx.dofmap.index_map.size_local):
            raise RuntimeError("wall distance: CG1 dofmap differs from the coordinate space's")
        self.function = dolfinx.fem.Function(V, name="wall_distance")
        self.update_geometry()

    def _wall_geometry(self):
        """Wall segment endpoints (2D) or vertices (3D) on every rank, with the
        global coordinate_space block index of each endpoint."""
        msh = self.mesh_obj.mesh
        comm = msh.comm
        gdim = msh.geometry.dim
        fdim = msh.topology.dim - 1
        tags = self.mesh_obj.meshtags
        facets = tags.indices[np.isin(tags.values, self.wall_tags)]
        facets = facets[facets < msh.topology.index_map(fdim).size_local]
        Vx = self.mesh_obj.coordinate_space
        imap = Vx.dofmap.index_map
        # vertex -> coordinate_space block: through the cells' local numbering
        geom = dolfinx.mesh.entities_to_geometry(msh, fdim, facets, False)
        node_to_dof = self._geometry_node_to_dof()
        coords = msh.geometry.x[geom][:, :, :gdim] if len(facets) else np.zeros((0, 2, gdim))
        gids = (imap.local_to_global(node_to_dof[geom.ravel()].astype(np.int32)).reshape(geom.shape)
                if len(facets) else np.zeros((0, 2), dtype=np.int64))
        coords = np.concatenate([c for c in comm.allgather(coords) if c.size] or [np.zeros((0, 2, gdim))])
        gids = np.concatenate([g for g in comm.allgather(gids) if g.size] or [np.zeros((0, 2), np.int64)])
        if coords.shape[1] > 2 and msh.topology.dim == 2:
            raise NotImplementedError("wall distance for facets with {} geometry nodes".format(coords.shape[1]))
        return coords, gids

    def _geometry_node_to_dof(self):
        """Local geometry node -> local coordinate_space block (both ghosted)."""
        msh = self.mesh_obj.mesh
        Vx = self.mesh_obj.coordinate_space
        g = msh.geometry.dofmaps[0]
        d = Vx.dofmap.list
        out = np.full(msh.geometry.x.shape[0], -1, dtype=np.int64)
        out[g.ravel()] = d.ravel()
        return out

    def update_geometry(self):
        msh = self.mesh_obj.mesh
        gdim = msh.geometry.dim
        x = self.function.function_space.tabulate_dof_coordinates()[:, :gdim]
        segs, gids = self._wall_geometry()
        self._segs, self._gids = segs, gids
        if gdim == 3:
            verts = segs.reshape(-1, 3)
            vgids = gids.reshape(-1)
            tree = scipy.spatial.cKDTree(verts)
            dist, j = tree.query(x)
            self._nearest = ("vertex", j, vgids, verts)
        else:
            A, B = segs[:, 0, :], segs[:, -1, :]
            j, t, P, dist = _nearest_segments(x, A, B)
            self._nearest = ("segment", j, t, P)
        self._x = x
        self._dist = dist
        self.function.x.array[:] = dist
        self.function.x.scatter_forward()

    def mesh_jacobian(self):
        msh = self.mesh_obj.mesh
        gdim = msh.geometry.dim
        V = self.function.function_space
        Vx = self.mesh_obj.coordinate_space
        imap = V.dofmap.index_map
        n_own = imap.size_local
        rows_g = imap.local_to_global(np.arange(n_own, dtype=np.int32))
        D = PETSc.Mat().createAIJ(
            ((n_own, imap.size_global), (n_own * gdim, imap.size_global * gdim)), comm=msh.comm)
        D.setPreallocationNNZ((3 * gdim, 3 * gdim))
        D.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        x, dist = self._x[:n_own], self._dist[:n_own]
        if self._nearest[0] == "segment":
            _, j, t, P = self._nearest
            j, t, P = j[:n_own], t[:n_own], P[:n_own]
        else:
            _, j, vgids, verts = self._nearest
            j = j[:n_own]
        for i in range(n_own):
            if dist[i] <= 1e-14:
                continue
            if self._nearest[0] == "segment":
                r = (x[i] - P[i]) / dist[i]
                ga, gb = self._gids[j[i], 0], self._gids[j[i], -1]
                entries = [(rows_g[i], r), (ga, -(1.0 - t[i]) * r), (gb, -t[i] * r)]
            else:
                r = (x[i] - verts[j[i]]) / dist[i]
                entries = [(rows_g[i], r), (vgids[j[i]], -r)]
            for block, val in entries:
                D.setValues(np.array([rows_g[i]], dtype=PETSc.IntType),
                            np.array([block * gdim + k for k in range(gdim)], dtype=PETSc.IntType),
                            val, addv=True)
        D.assemble()
        return D
