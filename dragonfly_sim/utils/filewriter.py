import dolfinx
import numpy as np


class FileWriter:
    """Writes dolfinx Functions to VTX (.bp) files for ParaView.

    Ghost cells are excluded from the output by creating a submesh of
    only the locally-owned cells.  This avoids the overlapping-cell
    artefacts that appear in ParaView when multiple MPI ranks write
    their ghost layers to the same file.
    """

    def __init__(self, filename, comm_obj, func_space):
        self.filename = filename
        self.comm = comm_obj
        self.functionspace = func_space

        self.construct_function_and_writer()

    def construct_function_and_writer(self):
        mesh = self.functionspace.mesh
        tdim = mesh.topology.dim
        num_cells_owned = mesh.topology.index_map(tdim).size_local

        # Build a submesh that contains only the locally-owned cells
        owned_cells = np.arange(num_cells_owned, dtype=np.int32)
        self.submesh, self.entity_map, _, _ = dolfinx.mesh.create_submesh(
            mesh, tdim, owned_cells
        )

        # Mirror the original function space on the ghost-free submesh
        element = self.functionspace.ufl_element()
        self.submesh_funcspace = dolfinx.fem.functionspace(self.submesh, element)

        self.function = dolfinx.fem.Function(
            self.submesh_funcspace, name="solution"
        )

        # Pre-compute the DOF index mapping (parent -> submesh) so that
        # value transfer in interpolate_and_write is a single array op.
        bs = self.functionspace.dofmap.bs
        parent_dofmap = self.functionspace.dofmap
        sub_dofmap = self.submesh_funcspace.dofmap

        # dolfinx >= 0.10 returns an EntityMap object rather than the
        # sub-to-parent cell index array; recover that array from it.
        sub_cell_map = self.submesh.topology.index_map(tdim)
        num_sub_cells = sub_cell_map.size_local + sub_cell_map.num_ghosts
        sub_to_parent = self.entity_map.sub_topology_to_topology(
            np.arange(num_sub_cells, dtype=np.int32), False)

        parent_idx = []
        sub_idx = []
        for i_sub in range(num_sub_cells):
            i_parent = sub_to_parent[i_sub]
            p_dofs = parent_dofmap.cell_dofs(i_parent)
            s_dofs = sub_dofmap.cell_dofs(i_sub)
            for pd, sd in zip(p_dofs, s_dofs):
                for c in range(bs):
                    parent_idx.append(pd * bs + c)
                    sub_idx.append(sd * bs + c)

        self.parent_idx = np.array(parent_idx, dtype=np.intc)
        self.sub_idx = np.array(sub_idx, dtype=np.intc)

        self.writer = dolfinx.io.VTXWriter(
            self.comm,
            "{}.bp".format(self.filename),
            self.function,
            engine="BP5",
        )
        self.write_counter = 0

    def interpolate_and_write(self, u_interp, write_counter=None):
        if isinstance(u_interp, dolfinx.fem.Function):
            # Fast path: copy DOF values from the parent function
            self.function.x.array[self.sub_idx] = u_interp.x.array[self.parent_idx]
        else:
            # UFL expression – interpolate directly on the submesh
            self.function.interpolate(
                dolfinx.fem.Expression(
                    u_interp,
                    self.submesh_funcspace.element.interpolation_points,
                )
            )

        # Only advance the internal counter when the caller did not supply an
        # index. Bumping it unconditionally meant an explicit write_counter
        # still perturbed writer state, so callers that pin the index (the
        # aligned per-design-evaluation write) could not actually pin it.
        if write_counter is None:
            write_counter = self.write_counter
            self.write_counter += 1

        self.writer.write(write_counter)
