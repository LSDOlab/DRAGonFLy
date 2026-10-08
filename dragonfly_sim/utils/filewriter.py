import weakref

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI


# Rank-0 serial copies of each parent mesh, keyed by mesh_deformation: the
# copy shared by deforming writers follows the parent geometry, the other one
# keeps the geometry the parent had when it was built.
_serial_meshes = weakref.WeakKeyDictionary()


def _gather_rows(comm, rows):
    """Gather the (n_local, k) arrays of all ranks to rank 0, stacked in rank order."""
    rows = np.ascontiguousarray(rows)
    ncols = rows.shape[1]
    counts = comm.gather(rows.size, root=0)
    recv = None
    if comm.rank == 0:
        recv = (np.empty(sum(counts), dtype=rows.dtype), counts)
    comm.Gatherv(rows, recv, root=0)
    return recv[0].reshape(-1, ncols) if comm.rank == 0 else None


def _gather_geometry(mesh):
    """Owned geometry nodes of all ranks on rank 0, in global node order. Collective."""
    return _gather_rows(mesh.comm, mesh.geometry.x[:mesh.geometry.index_map().size_local])


def _serial_mesh(mesh, mesh_deformation):
    """Return the ghost-free serial copy of `mesh` on rank 0 (None elsewhere).

    Owned cells and owned geometry nodes are numbered contiguously per rank in
    global numbering, so gathering them in rank order yields the cells and
    nodes in global order, each exactly once. Collective.
    """
    copies = _serial_meshes.setdefault(mesh, {})
    if mesh_deformation in copies:
        return copies[mesh_deformation]
    comm = mesh.comm
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    num_cells_owned = mesh.topology.index_map(tdim).size_local
    x_map = mesh.geometry.index_map()

    cell_nodes = mesh.geometry.dofmaps[0][:num_cells_owned]
    cells_global = x_map.local_to_global(cell_nodes.ravel()).reshape(cell_nodes.shape)

    x_all = _gather_geometry(mesh)
    cells_all = _gather_rows(comm, cells_global.astype(np.int64))
    serial = None
    if comm.rank == 0:
        domain = ufl.Mesh(mesh.ufl_domain().ufl_coordinate_element())
        serial = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells_all, domain, x_all[:, :gdim])
    copies[mesh_deformation] = serial
    return serial


def _cell_dof_indices(V, cells):
    """(len(cells), dofs_per_cell * bs) dof-array indices of `cells`, cell-major."""
    bs = V.dofmap.bs
    cell_dofs = V.dofmap.list[cells]
    return (cell_dofs[:, :, None] * bs + np.arange(bs)).reshape(
        len(cells), cell_dofs.shape[1] * bs).astype(np.int32)


class _Field:
    """One named field of a FileWriter: its parent space and output function."""

    def __init__(self, name, V):
        self.name = name
        self.V = V
        tdim = V.mesh.topology.dim
        owned_cells = np.arange(V.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        # Parent dof-array indices of each owned cell, one row per cell
        self.parent_idx = _cell_dof_indices(V, owned_cells)
        # Target for UFL expressions, which are interpolated on the parent mesh
        self.parent_function = dolfinx.fem.Function(V)
        self.gather = not (V.ufl_element().discontinuous and V.dofmap.dof_layout.num_dofs > 1)
        self.target_funcspace = None
        self.function = None
        self.target_idx = None


class FileWriter:
    """Writes dolfinx Functions to VTX (.bp) files for ParaView.

    One writer holds one or more named fields on the same mesh, written to a
    single file: ``func_space`` is either a function space (one field, named
    "solution") or a dict ``{name: function space}``, e.g. density, momentum
    (a vector field), energy and pressure. All fields of a writer must share
    the mesh and be either all gathered or all per-rank (see below); the mesh
    is then stored once per frame for all of them.

    ParaView's VTX reader concatenates the per-rank blocks of a parallel write
    without merging the points the blocks share. Where neighbouring cells
    share points within a rank -- DG0 fields, written as cell data on the
    mesh geometry, and continuous fields -- the MPI partition boundaries then
    show up as cracks (exterior faces, boundary edges, jumps in
    point-interpolated cell data). Those spaces are gathered to rank 0 and
    written as a single block on a serial copy of the mesh.

    Note on the DG0 gather: rank 0 holds the whole mesh (built once per parent
    mesh and shared by the writers on it) plus one frame of the fields, and
    every frame passes through rank 0. Per-frame data is small (one value per
    cell and component); the serial mesh is what bounds the mesh size. Past
    that point, the parallel alternatives are writing DG0 as per-cell-node DG
    data of the geometry degree (flat cells, no smoothing across cells) or
    dolfinx's XDMFFile.

    Discontinuous spaces with more than one dof per cell (DG p >= 1) are
    written by VTX with separate points for every cell, so every cell face is
    a crack in serial as well and partition boundaries add none. Each rank
    writes its own owned cells in parallel; ghost cells are left out because
    they would be written twice.

    Keywords, set per writer to match the simulation and the fields:

    mesh_deformation
        False: frames are written on the geometry the mesh had when the
        writer was created. True: each frame is written on the mesh geometry
        at the time of the write (e.g. a shape-optimisation design
        evaluation, or a moving mesh).
    time_dependent
        False: frames are labelled by a counter (``write_counter``, e.g. the
        design-evaluation index). True: frames are labelled by the physical
        time passed as ``time=``, which must increase from frame to frame.
    """

    def __init__(self, filename, comm_obj, func_space, mesh_deformation=False,
                 time_dependent=False):
        self.filename = filename
        self.comm = comm_obj
        self.single_field = not isinstance(func_space, dict)
        spaces = {"solution": func_space} if self.single_field else dict(func_space)
        if not spaces:
            raise ValueError("FileWriter {}: no fields given".format(filename))
        self.fields = [_Field(name, V) for name, V in spaces.items()]
        # the first field's space; the mesh and the gather mode are shared
        self.functionspace = self.fields[0].V
        self.mesh_deformation = mesh_deformation
        self.time_dependent = time_dependent

        self.construct_function_and_writer()

    @property
    def field_names(self):
        return [f.name for f in self.fields]

    # The single-field attributes (the first field's), kept for callers and
    # tests that write one field
    @property
    def function(self):
        return self.fields[0].function

    @property
    def target_funcspace(self):
        return self.fields[0].target_funcspace

    @property
    def parent_idx(self):
        return self.fields[0].parent_idx

    def construct_function_and_writer(self):
        mesh = self.functionspace.mesh
        tdim = mesh.topology.dim
        if any(f.V.mesh is not mesh for f in self.fields):
            raise ValueError("FileWriter {}: all fields must be on the same mesh".format(
                self.filename))
        gather = {f.gather for f in self.fields}
        if len(gather) != 1:
            raise ValueError(
                "FileWriter {}: fields mix gathered (DG0, continuous) and per-rank (DG p >= 1) "
                "spaces; write them to separate files".format(self.filename))
        self.gather = gather.pop()

        if self.gather:
            # Serial cell s is input cell original_cell_index[s], which is
            # global parent cell original_cell_index[s] (the gather order);
            # likewise for the geometry nodes and input_global_indices
            target_mesh = _serial_mesh(mesh, self.mesh_deformation)
            writer_comm = MPI.COMM_SELF
            if target_mesh is not None:
                self.target_rows = np.asarray(target_mesh.topology.original_cell_index)
                self.target_nodes = np.asarray(target_mesh.geometry.input_global_indices)
        else:
            owned_cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
            target_mesh, entity_map, _, self.target_nodes = dolfinx.mesh.create_submesh(
                mesh, tdim, owned_cells)
            writer_comm = self.comm
            sub_cells = target_mesh.topology.index_map(tdim)
            self.target_rows = entity_map.sub_topology_to_topology(
                np.arange(sub_cells.size_local + sub_cells.num_ghosts, dtype=np.int32), False)
        self.target_mesh = target_mesh

        self.writer = None
        if target_mesh is not None:
            for f in self.fields:
                f.target_funcspace = dolfinx.fem.functionspace(target_mesh, f.V.ufl_element())
                f.function = dolfinx.fem.Function(f.target_funcspace, name=f.name)
                f.target_idx = _cell_dof_indices(
                    f.target_funcspace, np.arange(len(self.target_rows), dtype=np.int32))
            self.writer = dolfinx.io.VTXWriter(
                writer_comm,
                "{}.bp".format(self.filename),
                [f.function for f in self.fields],
                engine="BP5",
            )
        self._check_dof_mapping()
        self.write_counter = 0
        self.last_time = None

    def _copy_cell_values(self, array, field=None):
        """Copy a parent dof array into a field's output function. Collective."""
        field = self.fields[0] if field is None else field
        values = array[field.parent_idx]
        if self.gather:
            values = _gather_rows(self.comm, values)
        if self.writer is not None:
            field.function.x.array[field.target_idx] = values[self.target_rows]

    def _sync_geometry(self):
        """Copy the current parent node coordinates onto the output mesh. Collective."""
        x_parent = self.functionspace.mesh.geometry.x
        if self.gather:
            x_parent = _gather_geometry(self.functionspace.mesh)
        if self.writer is not None:
            self.target_mesh.geometry.x[:] = x_parent[self.target_nodes]

    def _check_dof_mapping(self):
        """Verify that mapped dofs sit at the same coordinates on both meshes.

        The cell-local dof order is assumed equal on the parent and output
        meshes; this holds for the Lagrange/DG spaces written here but would
        break for spaces whose entity dofs depend on cell orientation.
        Collective.
        """
        ok = True
        for f in self.fields:
            bs = f.V.dofmap.bs
            coords = np.repeat(f.V.tabulate_dof_coordinates(), bs, axis=0)
            for d in range(f.V.mesh.geometry.dim):
                self._copy_cell_values(coords[:, d], f)
                if self.writer is not None:
                    target_coords = np.repeat(
                        f.target_funcspace.tabulate_dof_coordinates(), bs, axis=0)[:, d]
                    ok &= np.allclose(f.function.x.array, target_coords, rtol=0, atol=1e-10)
        if not self.comm.allreduce(ok, op=MPI.LAND):
            raise RuntimeError(
                "FileWriter: dof mapping onto the output mesh failed for {}".format(
                    self.filename))

    def _parent_array(self, field, u_interp):
        if isinstance(u_interp, dolfinx.fem.Function):
            return u_interp.x.array
        # UFL expression - interpolate on the parent mesh, then copy
        field.parent_function.interpolate(
            dolfinx.fem.Expression(u_interp, field.V.element.interpolation_points))
        return field.parent_function.x.array

    def interpolate_and_write(self, u_interp, write_counter=None, time=None):
        """Write one frame. ``u_interp`` is, for a single-field writer, a
        Function on the writer's space or a UFL expression; for a multi-field
        writer, a dict ``{name: Function or UFL expression}`` with one entry
        per field.

        A counter-labelled writer (time_dependent=False) takes an optional
        ``write_counter``; a time-dependent writer requires ``time``.
        Collective.
        """
        if self.time_dependent:
            if time is None or write_counter is not None:
                raise ValueError(
                    "FileWriter {}: time-dependent output is labelled by time=, "
                    "not write_counter".format(self.filename))
            if self.last_time is not None and not time > self.last_time:
                raise ValueError(
                    "FileWriter {}: time {} does not increase past the last written "
                    "time {}".format(self.filename, time, self.last_time))
        elif time is not None:
            raise ValueError(
                "FileWriter {}: time= needs time_dependent=True".format(self.filename))

        if self.single_field:
            values = {self.fields[0].name: u_interp}
        else:
            values = dict(u_interp)
            if set(values) != set(self.field_names):
                raise ValueError("FileWriter {}: expected the fields {}, got {}".format(
                    self.filename, self.field_names, sorted(values)))
        for f in self.fields:
            self._copy_cell_values(self._parent_array(f, values[f.name]), f)
        if self.mesh_deformation:
            self._sync_geometry()

        if self.time_dependent:
            label = self.last_time = float(time)
        else:
            # Only advance the internal counter when the caller did not supply an
            # index. Bumping it unconditionally meant an explicit write_counter
            # still perturbed writer state, so callers that pin the index (the
            # aligned per-design-evaluation write) could not actually pin it.
            if write_counter is None:
                write_counter = self.write_counter
                self.write_counter += 1
            label = write_counter

        if self.writer is not None:
            self.writer.write(label)
