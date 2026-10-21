from dataclasses import dataclass
import warnings

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import ufl.classes
from scipy.spatial import KDTree
import basix.ufl
import dolfinx
import dolfinx.fem.petsc

from dragonfly_sim.utils.mpi_utils import allgather_numpy, assign_global_dof_offset
from dragonfly_sim.utils.meshwarping_utils import find_local_surface_nodes
from dragonfly_sim.core.form_manager import FormManager
from dragonfly_sim.utils.filewriter import FileWriter
from dragonfly_sim.utils.solver_profiling import PROFILER as _PROF


@dataclass(frozen=True)
class BoundaryNodeSet:
    """Owned nodes on tagged boundary facets, numbered globally in rank-block order.

    Built by ``Mesh.boundary_node_set``. ``local_idx`` are this rank's OWNED
    geometry-node indices; they carry the global ids ``global_idx`` =
    ``offset + arange(n_local)``, so the ranks' blocks tile ``[0, n_global)``
    in rank order. ``coords`` and ``geom_gids`` (dolfinx global geometry ids)
    are replicated on every rank, row for row in that same order.
    shape_parameterization.WallFFD relies on this numbering.
    """
    tags: tuple
    local_idx: np.ndarray
    offset: int
    n_global: int
    coords: np.ndarray
    geom_gids: np.ndarray

    @property
    def n_local(self):
        return int(self.local_idx.size)

    @property
    def global_idx(self):
        return np.arange(self.offset, self.offset + self.n_local, dtype=np.int64)


class Mesh():
    # DOLFINx returns quadrilateral entity nodes in tensor ("Z") order, which
    # traverses the quad as a bowtie; this permutation makes it cyclic.
    _QUAD_CYCLIC_PERM = np.array([0, 1, 3, 2], dtype=np.int64)

    def __init__(self, inp_mesh, inner_bdry_function=None, node_match_tol=1e-10,
                 wall_tags=None, symmetry_tags=None, facet_tags=None):
        """
        The wall (the body) is found in one of two ways:

        inner_bdry_function  a function of the boundary facet midpoints, shape
            (dim, n), returning a boolean mask: True for the wall. Used if given.
        wall_tags            the tags of the mesh's own boundary facet tags that
            are the wall (a tag value or name, or a sequence of them), and
            symmetry_tags those that are symmetry planes. Used if there is no
            function. Every other boundary facet is far field.

        The facet tags are `facet_tags` (a dolfinx MeshTags of the facets) or,
        if None, `inp_mesh.facet_tags`, which utils/mesh_io_utils.py:load_dolfinx_mesh
        sets (with `inp_mesh.facet_tag_names`, a {name: value} dict). If the
        function is used and the mesh also has tags, a UserWarning lists them.
        With the function, symmetry planes are not marked here (3D: y = 0 in
        DG_windtunnel_model); `symmetry_facet_mask` is None.
        """
        # define input Dolfinx mesh object
        self.mesh = inp_mesh

        self.baseline_nodes = self.mesh.geometry.x.copy()
        self.n_owned_nodes = int(self.mesh.geometry.index_map().size_local)
        self.node_match_tol = node_match_tol
        self.is_valid = True
        # design-parameter vector the current node coordinates were built for
        # (see apply_node_motions); None at baseline or when it is unknown
        self._deformation_design = None
        self._coordinate_space = None
        self._dof_to_node_map = None
        self._nodal_function = None
        self._quality = None
        self._node_sets = {}
        self.output_suffix = None
        self._deformation_writer = None
        self._quality_writers = None
        # callables run after every change of the node coordinates (see
        # add_geometry_listener)
        self._geometry_listeners = []

        self.n = ufl.FacetNormal(self.mesh)  # cell boundary normal vector

        # Volume-equivalent cell length |K|^(1/d). ufl.CellVolume is not
        # lowered for quads/hexes and FFCx cannot compile it, so it is written
        # pointwise from the Jacobian instead: exact on parallelogram cells,
        # the pointwise density elsewhere. Being a plain Jacobian expression,
        # it also differentiates w.r.t. the mesh coordinates, which an
        # assembled DG0 field would not.
        tdim = self.mesh.topology.dim
        self.cell_volume = (abs(ufl.JacobianDeterminant(self.mesh))
                            * ufl.classes.ReferenceCellVolume(self.mesh))
        self.h_vol = self.cell_volume ** (1.0 / tdim)
        # Shortest edge, for the local pseudo-time step of PTC
        # (core/time_integration.py); never part of the residual.
        self.h_min = ufl.MinCellEdgeLength(self.mesh)

        # The boundary queries below, and the wall-node lookups done later on
        # (entities_to_geometry) need these to exist already: the facet
        # entities, facet-to-cell connectivity (which also stores its
        # transpose), facet-to-vertex and vertex-to-cell connectivity
        self.mesh.topology.create_entities(tdim - 1)
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        self.mesh.topology.create_connectivity(tdim - 1, 0)
        self.mesh.topology.create_connectivity(0, tdim)

        # we compute which facets form the mesh boundaries and their midpoints
        self.bdry_facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology) 
        self.bdry_midpoints = dolfinx.mesh.compute_midpoints(
            self.mesh, self.mesh.topology.dim-1, self.bdry_facets)

        # from all boundary facets, isolate the boundary of the wing/airfoil/etc.
        self.inner_bdry_facet_mask, self.symmetry_facet_mask = self._wall_and_symmetry_masks(
            inner_bdry_function, wall_tags, symmetry_tags, facet_tags)

        # Initialize the meshtags object; used to define integration domains for boundary terms
        self.meshtags = None

        # Instantiate FormManager for handling UFL forms and integration measures
        self.form_manager = FormManager(self)

    # ------------------------------------------------------------------
    # Wall and symmetry boundaries: from a function or from the mesh's tags
    # ------------------------------------------------------------------
    def _tag_summary(self, facet_tags, names):
        """
        '{value: (name, number of boundary facets)}' lines for the tags on the
        exterior facets, the same on every rank. Owned facets only, so a facet
        shared between ranks is counted once.
        """
        comm = self.mesh.comm
        n_owned = self.mesh.topology.index_map(self.mesh.topology.dim - 1).size_local
        on_bdry = np.isin(facet_tags.indices, self.bdry_facets) & (facet_tags.indices < n_owned)
        values, counts = np.unique(facet_tags.values[on_bdry], return_counts=True)
        total = {}
        for v_c in comm.allgather(dict(zip(values.tolist(), counts.tolist()))):
            for v, c in v_c.items():
                total[v] = total.get(v, 0) + c
        by_value = {v: n for n, v in names.items()}
        return {v: (by_value.get(v), total[v]) for v in sorted(total)}

    @staticmethod
    def _describe(summary):
        return ", ".join("{}{} ({} facets)".format(v, " '{}'".format(n) if n else "", c)
                         for v, (n, c) in summary.items())

    def _resolve_tags(self, tags, summary, names, what):
        """Tag values from a value, a name, or a sequence of them; checked to be present."""
        if tags is None:
            return set()
        tags = [tags] if isinstance(tags, (int, np.integer, str)) else list(tags)
        values = set()
        for t in tags:
            if isinstance(t, str):
                if t not in names:
                    raise ValueError("{}: the mesh has no boundary tag named '{}' (named tags: {})".format(
                        what, t, sorted(names) or "none"))
                t = names[t]
            t = int(t)
            if t not in summary:
                raise ValueError("{}: tag {} is not on the mesh's boundary facets (tags present: {})".format(
                    what, t, self._describe(summary) or "none"))
            values.add(t)
        return values

    def _wall_and_symmetry_masks(self, inner_bdry_function, wall_tags, symmetry_tags, facet_tags):
        """(wall mask, symmetry mask or None) over self.bdry_facets; see __init__."""
        facet_tags = facet_tags if facet_tags is not None else getattr(self.mesh, "facet_tags", None)
        names = getattr(self.mesh, "facet_tag_names", None) or {}
        summary = self._tag_summary(facet_tags, names) if facet_tags is not None else {}

        if inner_bdry_function is not None:
            if self.mesh.comm.rank == 0 and (summary or wall_tags is not None
                                             or symmetry_tags is not None):
                if summary:
                    message = ("The input mesh has boundary facet tags: {}. The boundary function "
                               "is used to find the wall, so the boundary facet tags are "
                               "currently unused.{} To use the tags "
                               "instead, leave out the boundary function and pass wall_tags (the "
                               "tags of the body) and, if there is a symmetry plane, "
                               "symmetry_tags; all other boundaries are far field.").format(
                        self._describe(summary),
                        " wall_tags and symmetry_tags are ignored." if (
                            wall_tags is not None or symmetry_tags is not None) else "")
                else:
                    message = ("wall_tags and symmetry_tags are ignored: the boundary function "
                               "defines the wall.")
                warnings.warn(message, UserWarning, stacklevel=4)
            return np.asarray(inner_bdry_function(self.bdry_midpoints.T), dtype=bool), None

        if wall_tags is None:
            if summary:
                raise ValueError(
                    "No boundary function was given, and wall_tags was not set. The mesh has "
                    "boundary facet tags: {}. Pass wall_tags (and symmetry_tags for symmetry "
                    "planes); the other boundaries are far field.".format(self._describe(summary)))
            raise ValueError("Provide either a boundary function (facet midpoints -> wall mask) "
                             "or wall_tags with a mesh that has boundary facet tags.")
        if facet_tags is None:
            raise ValueError("wall_tags was given but the mesh has no boundary facet tags (read it "
                             "with load_dolfinx_mesh, or pass facet_tags)")
        wall = self._resolve_tags(wall_tags, summary, names, "wall_tags")
        sym = self._resolve_tags(symmetry_tags, summary, names, "symmetry_tags")
        if not wall:
            raise ValueError("wall_tags is empty")
        if wall & sym:
            raise ValueError("tags {} are both wall and symmetry".format(sorted(wall & sym)))

        # tag value of every boundary facet of this rank (0: untagged)
        n_facets = self.mesh.topology.index_map(self.mesh.topology.dim - 1)
        value = np.zeros(n_facets.size_local + n_facets.num_ghosts, dtype=np.int64)
        value[facet_tags.indices] = facet_tags.values
        value = value[self.bdry_facets]
        wall_mask = np.isin(value, list(wall))
        sym_mask = np.isin(value, list(sym)) if sym else None
        return wall_mask, sym_mask

    def tag_boundary_facets(self, bdry_facet_value_tuples):
        # Construct the boundary mesh tags from a list of tuples (boundary facets, domain_boundary_ID)
        values = np.zeros(self.bdry_midpoints.shape[0], dtype=np.int32)
        for facets, value in bdry_facet_value_tuples:
            values[facets] = value

        self.meshtags = dolfinx.mesh.meshtags(
            self.mesh, self.mesh.topology.dim - 1, self.bdry_facets, values)
        return self.meshtags

    def _wall_boundary_node_coords(self, wall_tag_value):
        if self.meshtags is None:
            raise RuntimeError("Mesh tags not yet constructed. "
                               "Call tag_boundary_facets first.")

        wall_facet_indices = self.meshtags.find(wall_tag_value)

        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, 0)
        facet_to_node = self.mesh.topology.connectivity(self.mesh.topology.dim - 1, 0)

        if len(wall_facet_indices) > 0:
            wall_vertex_topo_idxs = np.unique(np.concatenate(
                [facet_to_node.links(facet) for facet in wall_facet_indices]))
            wall_node_geom_idxs = dolfinx.cpp.mesh.entities_to_geometry(
                self.mesh._cpp_object, 0, wall_vertex_topo_idxs, False)
            return self.mesh.geometry.x[wall_node_geom_idxs[:, 0], :self.mesh.topology.dim]

        return np.empty(shape=(0, self.mesh.topology.dim))

    def _global_bounding_box(self, coords):
        if coords.shape[0] > 0:
            local_mins = coords.min(axis=0)
            local_maxs = coords.max(axis=0)
        else:
            local_mins = np.full(self.mesh.topology.dim, np.inf)
            local_maxs = np.full(self.mesh.topology.dim, -np.inf)

        return [(self.mesh.comm.allreduce(local_mins[i], op=MPI.MIN),
                 self.mesh.comm.allreduce(local_maxs[i], op=MPI.MAX))
                for i in range(self.mesh.topology.dim)]

    def compute_wall_bounding_box(self, wall_tag_value):
        # Safe to call every iteration on the current (possibly deformed) mesh.
        coords = self._wall_boundary_node_coords(wall_tag_value)
        return self._global_bounding_box(coords)


    # ==================================================================
    # Geometry nodes and the CG1 coordinate space
    # ==================================================================

    @property
    def coordinate_space(self):
        """Vector CG1 space whose dofs coincide with the geometry nodes.

        The space mesh-coordinate derivatives (dR/dx_mesh) are taken in.
        Built on first use.
        """
        if self._coordinate_space is None:
            element = basix.ufl.element(
                "Lagrange", self.mesh.basix_cell(), 1,
                shape=(self.mesh.geometry.dim,), dtype=dolfinx.default_real_type)
            self._coordinate_space = dolfinx.fem.functionspace(self.mesh, element)
        return self._coordinate_space

    @property
    def dof_to_node_map(self):
        """(owned CG1 dofs) x (owned geometry nodes) 0/1 permutation matrix.

        The CG1 vector dofs correspond 1:1 with the geometry nodes but are
        numbered differently: ``dof_to_node_map @ node_values`` reorders an
        (n_owned_nodes, gdim) nodal array into dof order, and its transpose
        maps dof-ordered derivatives back to geometry-node order. Matched by
        coordinates within ``node_match_tol`` -- an exact-match test with room
        for round-off. Built on first use.
        """
        if self._dof_to_node_map is None:
            n = self.n_owned_nodes
            dof_tree = KDTree(self.coordinate_space.tabulate_dof_coordinates()[:n, :])
            node_tree = KDTree(self.mesh.geometry.x[:n, :])
            perm = dof_tree.sparse_distance_matrix(
                node_tree, max_distance=self.node_match_tol).tocsr()
            perm.data[:] = 1.
            self._dof_to_node_map = perm
        return self._dof_to_node_map

    def nodal_function(self, node_values):
        """An (n_owned_nodes, gdim) nodal array as a Function on coordinate_space.

        The Function is reused between calls.
        """
        V = self.coordinate_space
        if self._nodal_function is None:
            self._nodal_function = dolfinx.fem.Function(V)
        n_local_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        self._nodal_function.x.array[:n_local_dofs] = \
            (self.dof_to_node_map @ node_values).reshape(-1)
        self._nodal_function.x.scatter_forward()
        return self._nodal_function

    # ==================================================================
    # Deformation
    # ==================================================================

    def add_geometry_listener(self, callback):
        """Run `callback()` (collective) after every reset_nodes and every
        apply_node_motions that moves the nodes (not one skipped for an
        unchanged design), e.g. to refresh fields computed from the node
        coordinates (cell centres, wall distance)."""
        self._geometry_listeners.append(callback)

    def _notify_geometry_listeners(self):
        for callback in self._geometry_listeners:
            callback()

    @property
    def deformation_design(self):
        """Copy of the design vector the mesh is currently deformed for, or None."""
        return None if self._deformation_design is None else self._deformation_design.copy()

    def reset_nodes(self, notify=True):
        """Put every local node, ghosts included, back at its baseline position.

        The callbacks no longer call this: the mesh stays deformed until
        apply_node_motions is given a different design.
        """
        with _PROF.phase("mesh:reset"):
            gdim = self.mesh.geometry.dim
            self.mesh.geometry.x[:, :gdim] = self.baseline_nodes[:, :gdim]
            self._deformation_design = None
            if notify:
                self._notify_geometry_listeners()

    def _is_deformed_for(self, design):
        """True on every rank if the mesh already holds the deformation of ``design``.

        ``design`` is replicated across ranks, but the comparison is reduced
        anyway: apply_node_motions is collective, so the ranks must agree on
        whether to skip it.
        """
        stored = self._deformation_design
        same = (stored is not None and stored.shape == design.shape
                and bool(np.array_equal(stored, design)))
        return bool(self.mesh.comm.allreduce(same, op=MPI.MIN))

    def apply_node_motions(self, node_motions, design=None):
        """Move the mesh to baseline + ``node_motions`` and re-check its cells.

        ``node_motions`` is (n_owned_nodes, gdim) in geometry-node order.
        Starting from the baseline makes the result depend on this call only,
        so repeated calls do not accumulate. Ghost nodes are updated from
        their owners. Sets ``is_valid`` (no inverted or zero-volume cell) and
        returns it. Collective.

        ``design`` is the (replicated) design-parameter vector that
        ``node_motions`` was computed from. The mesh stays deformed after this
        call and remembers ``design``; a later call with an equal ``design``
        returns the current ``is_valid`` without touching the mesh, so the
        deformation is built once per design. A different ``design`` resets
        the nodes and warps again. ``design=None`` always warps.
        """
        if design is not None:
            design = np.array(design, dtype=np.float64, copy=True).ravel()
            if self._is_deformed_for(design):
                PETSc.Sys.Print("Mesh already deformed for this design: reusing it")
                return self.is_valid

        # unknown until the new deformation is complete, so a failure part-way
        # never leaves a key that matches a half-moved mesh
        self._deformation_design = None
        # geometry writes, the ghost scatter, the quality metrics and the
        # geometry listeners (derived fields of a RANS model): per-evaluation
        # cost in its own "mesh" phase group
        with _PROF.phase("mesh:deform"):
            self.reset_nodes(notify=False)
            n, gdim = self.n_owned_nodes, self.mesh.geometry.dim
            x = self.mesh.geometry.x
            x[:n, :gdim] += node_motions[:, :gdim]

            # geometry.x has no scatter_forward of its own: route the owned
            # coordinates through a dolfinx.la.Vector on the geometry index map.
            geom_vec = dolfinx.la.vector(self.mesh.geometry.index_map(), bs=3)
            geom_vec.array[:n * 3] = x[:n, :].ravel()
            geom_vec.scatter_forward()
            x[:] = geom_vec.array.reshape(-1, 3)

            self._update_quality()
            self._notify_geometry_listeners()
        self._deformation_design = design
        return self.is_valid

    # ==================================================================
    # Mesh quality and output files
    # ==================================================================

    def _setup_quality(self):
        """DG0 fields for the quality metrics apply_node_motions reports."""
        if self._quality is not None:
            return self._quality
        msh = self.mesh
        V0 = dolfinx.fem.functionspace(msh, ("DG", 0))
        pts = V0.element.interpolation_points
        q = {"V": V0,
             "vol_form": dolfinx.fem.form(ufl.TestFunction(V0) * ufl.dx),
             "vol": dolfinx.fem.Function(V0)}
        # Per-cell shape metric, as distinct from the whole-mesh max/min edge
        # ratio also reported: it is the per-cell value that says how badly an
        # isotropic length scale misrepresents a cell.
        for name, expr in (("jac", ufl.JacobianDeterminant(msh)),
                           ("min_edge", ufl.MinCellEdgeLength(msh)),
                           ("max_edge", ufl.MaxCellEdgeLength(msh)),
                           ("cell_aspect_ratio",
                            ufl.MaxCellEdgeLength(msh) / ufl.MinCellEdgeLength(msh))):
            q[name + "_expr"] = dolfinx.fem.Expression(expr, pts)
            q[name] = dolfinx.fem.Function(V0)
        self._quality = q
        return q

    def _update_quality(self):
        """Evaluate, print and (if configured) write the quality metrics; set is_valid."""
        q = self._setup_quality()
        comm = self.mesh.comm
        with q["vol"].x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(q["vol"].x.petsc_vec, q["vol_form"])
        q["vol"].x.scatter_forward()
        for name in ("jac", "min_edge", "max_edge", "cell_aspect_ratio"):
            q[name].interpolate(q[name + "_expr"])

        if self._quality_writers is not None:
            for name, writer in self._quality_writers.items():
                writer.interpolate_and_write(q[name])

        def global_stat(name, fn, op, empty):
            a = q[name].x.array
            return comm.allreduce(fn(a) if a.size > 0 else empty, op=op)

        min_vol = global_stat("vol", np.min, MPI.MIN, np.inf)
        max_vol = global_stat("vol", np.max, MPI.MAX, -np.inf)
        min_jac = global_stat("jac", np.min, MPI.MIN, np.inf)
        max_jac = global_stat("jac", np.max, MPI.MAX, -np.inf)
        min_edge = global_stat("min_edge", np.min, MPI.MIN, np.inf)
        max_edge = global_stat("max_edge", np.max, MPI.MAX, -np.inf)
        max_cell_ar = global_stat("cell_aspect_ratio", np.max, MPI.MAX, -np.inf)
        cell_ar_sum = global_stat("cell_aspect_ratio", np.sum, MPI.SUM, 0.0)
        n_cells = comm.allreduce(q["cell_aspect_ratio"].x.array.size, op=MPI.SUM)
        mean_cell_ar = cell_ar_sum / n_cells if n_cells > 0 else np.inf
        aspect_ratio = max_edge / min_edge if min_edge > 0 else np.inf

        PETSc.Sys.Print("--- Mesh Quality Metrics ---")
        PETSc.Sys.Print("Min cell volume: {:.6e}".format(min_vol))
        PETSc.Sys.Print("Max cell volume: {:.6e}".format(max_vol))
        PETSc.Sys.Print("Min Jacobian determinant: {:.6e}".format(min_jac))
        PETSc.Sys.Print("Max Jacobian determinant: {:.6e}".format(max_jac))
        PETSc.Sys.Print("Min edge length: {:.6e}".format(min_edge))
        PETSc.Sys.Print("Max edge length: {:.6e}".format(max_edge))
        PETSc.Sys.Print("Max per-cell aspect ratio: {:.6e}".format(max_cell_ar))
        PETSc.Sys.Print("Mean per-cell aspect ratio: {:.6e}".format(mean_cell_ar))
        PETSc.Sys.Print("Max Global Aspect Ratio proxy (max_edge/min_edge over whole mesh): {:.6e}".format(aspect_ratio))
        self.is_valid = bool(min_jac > 0)
        if not self.is_valid:
            PETSc.Sys.Print("WARNING: Mesh has inverted or zero-volume cells!")
        PETSc.Sys.Print("----------------------------")

    def configure_output(self, suffix, deformation=True, quality=False):
        """Enable the per-evaluation mesh output files, named with ``suffix``.

        ``deformation`` writes mesh_deformation_<suffix>.bp (through
        write_deformation_output) on the baseline mesh, so ParaView's Warp By
        Vector with it reproduces the deformed mesh; ``quality`` writes the DG0
        quality fields mesh_{vol,jac,min_edge,max_edge,cell_aspect_ratio}_<suffix>.bp
        on the deformed mesh on every apply_node_motions. ``suffix`` also names
        the memory logs of the classes that use this mesh. Collective.
        """
        comm = self.mesh.comm
        self.output_suffix = suffix
        self._deformation_writer = (
            FileWriter("mesh_deformation_{}".format(suffix), comm, self.coordinate_space,
                       mesh_deformation=False)
            if deformation else None)
        self._quality_writers = None
        if quality:
            V0 = self._setup_quality()["V"]
            self._quality_writers = {
                name: FileWriter("mesh_{}_{}".format(name, suffix), comm, V0, mesh_deformation=True)
                for name in ("vol", "jac", "min_edge", "max_edge", "cell_aspect_ratio")}

    def write_deformation_output(self, node_motions, write_counter):
        """Write (n_owned_nodes, gdim) node motions to mesh_deformation_<suffix>.bp.

        A no-op unless configure_output enabled it. Called once per design
        evaluation with write_counter = eval_idx, so this file and the solution
        file share one index axis.
        """
        if self._deformation_writer is None:
            return
        self._deformation_writer.interpolate_and_write(
            self.nodal_function(node_motions), write_counter=write_counter)

    # ==================================================================
    # Tagged boundary node sets and facets
    # ==================================================================

    def boundary_node_set(self, tags, exclude=None):
        """Owned nodes on facets tagged with any of ``tags``, numbered globally.

        ``exclude`` (another BoundaryNodeSet) removes its nodes -- e.g. the
        wall from the farfield anchors, so a node shared by both stays with the
        wall. See BoundaryNodeSet for the numbering. Cached per (tags,
        exclude), so every caller gets the same numbering. Collective on first
        call; requires tag_boundary_facets.
        """
        if self.meshtags is None:
            raise RuntimeError("Mesh tags not yet constructed. "
                               "Call tag_boundary_facets first.")
        tags = tuple(int(t) for t in tags)
        key = (tags, None if exclude is None else exclude.tags)
        if key in self._node_sets:
            return self._node_sets[key]

        msh = self.mesh
        comm = msh.comm
        gdim = msh.geometry.dim
        self._check_vertex_geometry_numbering()

        parts = [find_local_surface_nodes(msh, self.meshtags, tag) for tag in tags]
        local_idx = (np.unique(np.concatenate(parts)) if parts
                     else np.zeros((0,), dtype=np.int64))
        if exclude is not None:
            local_idx = np.setdiff1d(local_idx, exclude.local_idx)
        local_idx = local_idx.astype(np.int64)

        offset, n_global = assign_global_dof_offset(comm, local_idx.size)
        coords = allgather_numpy(
            comm, self.baseline_nodes[:self.n_owned_nodes, :gdim][local_idx].ravel(),
            n_global * gdim).reshape(n_global, gdim)
        # comm.allgather rather than allgather_numpy: these are int64 global
        # ids and allgather_numpy is float64-only.
        gids = msh.geometry.index_map().local_to_global(
            local_idx.astype(np.int32)).astype(np.int64)
        geom_gids = np.concatenate(comm.allgather(gids) + [np.zeros(0, dtype=np.int64)])
        if np.unique(geom_gids).size != geom_gids.size:
            raise RuntimeError(
                "Mesh.boundary_node_set{}: a geometry node appears twice; each "
                "node should be owned by exactly one rank.".format(tags))

        node_set = BoundaryNodeSet(tags=tags, local_idx=local_idx, offset=int(offset),
                                   n_global=int(n_global), coords=coords,
                                   geom_gids=geom_gids)
        self._node_sets[key] = node_set
        return node_set

    def _check_vertex_geometry_numbering(self):
        """Owned vertex indices must equal owned geometry-node indices.

        find_local_surface_nodes works in topology vertex indices, the facet
        connectivity in geometry node indices. For first-order meshes the two
        coincide over the OWNED range, and only that range is used, so the node
        sets and the connectivity stay consistent.

        They do NOT coincide over the ghost range -- every rank has a few
        hundred ghosts numbered differently in the two index spaces. That is
        fine: ghosts only enter through the facet connectivity, which is
        converted to global geometry ids via geometry.index_map().local_to_global
        before being used, and that map is valid for ghosts.
        """
        msh = self.mesh
        n_owned_verts = msh.topology.index_map(0).size_local
        msh.topology.create_connectivity(0, msh.topology.dim)
        vertex_to_geom = dolfinx.mesh.entities_to_geometry(
            msh, 0, np.arange(n_owned_verts, dtype=np.int32)).reshape(-1)
        if not np.array_equal(vertex_to_geom, np.arange(n_owned_verts, dtype=np.int32)):
            raise RuntimeError(
                "Mesh: owned vertex indices and owned geometry node indices do "
                "not coincide (they do for first-order dolfinx meshes), so "
                "boundary node sets and facet connectivity would be indexed "
                "inconsistently.")

    def tagged_facets(self, tags, node_gids):
        """This rank's OWNED facets tagged with any of ``tags``, as node-id rows.

        A node id is the node's position in ``node_gids``, a replicated array
        of global geometry ids (a BoundaryNodeSet's ``geom_gids``, or several
        concatenated). Quad facets come back in cyclic vertex order. Only owned
        facets are taken: a facet shared between two ranks appears in both
        ranks' meshtags, and counting it twice would double it.

        Returns ``(conn, facets)``: (n_facets, nodes_per_facet) int64 node ids
        and the local facet indices.
        """
        msh = self.mesh
        fdim = msh.topology.dim - 1
        msh.topology.create_entities(fdim)
        msh.topology.create_connectivity(fdim, msh.topology.dim)
        msh.topology.create_connectivity(fdim, 0)
        empty = (np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.int32))

        parts = [self.meshtags.find(int(tag)) for tag in tags]
        if not parts:
            return empty
        facets = np.unique(np.concatenate(parts)).astype(np.int32)
        facets = facets[facets < msh.topology.index_map(fdim).size_local]
        if facets.size == 0:
            return empty

        facet_geom = dolfinx.mesh.entities_to_geometry(msh, fdim, facets, False)
        gids = msh.geometry.index_map().local_to_global(
            facet_geom.reshape(-1).astype(np.int32))

        node_gids = np.asarray(node_gids, dtype=np.int64)
        order = np.argsort(node_gids)
        sorted_gids = node_gids[order]
        if np.unique(sorted_gids).size != sorted_gids.size:
            raise RuntimeError("Mesh.tagged_facets: node_gids contains duplicates.")
        pos = np.clip(np.searchsorted(sorted_gids, gids), 0, max(sorted_gids.size - 1, 0))
        found = sorted_gids.size > 0 and np.all(sorted_gids[pos] == gids)
        if not found:
            raise RuntimeError(
                "Mesh.tagged_facets: {} facet node(s) are not in node_gids. This "
                "happens when a rank owns a tagged facet whose node is owned by a "
                "rank that saw no facet with that tag.".format(
                    int((sorted_gids[pos] != gids).sum()) if sorted_gids.size else gids.size))
        conn = order[pos].reshape(facet_geom.shape).astype(np.int64)
        if fdim == 2 and conn.shape[1] == 4:
            conn = conn[:, self._QUAD_CYCLIC_PERM]
        return conn, facets

    def facet_cell_midpoints(self, facets):
        """(n_facets, 3) baseline midpoint of the single cell attached to each exterior facet."""
        msh = self.mesh
        tdim = msh.topology.dim
        f_to_c = msh.topology.connectivity(tdim - 1, tdim)
        c_to_g = msh.geometry.dofmap
        x = self.baseline_nodes
        mids = np.empty((facets.size, 3), dtype=np.float64)
        for k, f in enumerate(facets):
            mids[k] = x[c_to_g[f_to_c.links(int(f))[0]]].mean(axis=0)
        return mids

    @staticmethod
    def newell_normals(face_coords):
        """Newell's method: (n_faces, k, 3) -> (n_faces, 3), twice the area vector."""
        rolled = np.roll(face_coords, -1, axis=1)
        return np.cross(face_coords, rolled).sum(axis=1)

    @staticmethod
    def orient_facets(conn, coords, cell_mids):
        """Reverse the traversal of faces whose Newell normal points into the fluid.

        After this every face normal points out of its adjacent fluid cell.
        DOLFINx orders facet vertices from the sorted global vertex numbering,
        which fixes neither the traversal order nor a consistent winding.
        Measured on the 3D wing mesh: without this, only ~51% of facets come
        out consistently oriented. Returns ``(conn, n_flipped)``.
        """
        face_coords = coords[conn]
        normals = Mesh.newell_normals(face_coords)
        outward = np.einsum("ij,ij->i", normals, face_coords.mean(axis=1) - cell_mids)
        flip = outward < 0.0
        conn = conn.copy()
        conn[flip] = conn[flip][:, ::-1]
        return conn, int(flip.sum())

    def wall_facets(self, wall_tag):
        """Replicated wall facets as rows of wall node ids in ``[0, n_global)``.

        Node ids are the ``boundary_node_set((wall_tag,))`` numbering. Segments
        in 2D, triangles or quads (cyclic vertex order) in 3D, in rank order, so
        every rank holds the same array. Consistently wound: the facet normal --
        Newell's in 3D, ``(dy, -dx)`` of the traversal in 2D -- points out of
        the adjacent fluid cell, i.e. into the body. Collective.
        """
        comm = self.mesh.comm
        gdim = self.mesh.geometry.dim
        wall = self.boundary_node_set((wall_tag,))
        local_faces, local_facets = self.tagged_facets((wall_tag,), wall.geom_gids)
        if local_faces.size:
            coords = np.zeros((wall.n_global, 3), dtype=np.float64)
            coords[:, :gdim] = wall.coords
            cell_mids = self.facet_cell_midpoints(local_facets)
            if gdim == 2:
                a, b = local_faces[:, 0], local_faces[:, 1]
                tangent = coords[b] - coords[a]
                normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
                to_face = 0.5 * (coords[a] + coords[b]) - cell_mids
                flip = np.einsum("ij,ij->i", normal, to_face[:, :2]) < 0.0
                conn = local_faces.copy()
                conn[flip] = conn[flip][:, ::-1]
            else:
                conn, _ = self.orient_facets(local_faces, coords, cell_mids)
        else:
            conn = np.zeros((0, 0), dtype=np.int64)
        gathered = [g for g in comm.allgather(np.ascontiguousarray(conn, dtype=np.int64))
                    if g.size]
        if not gathered:
            raise RuntimeError("Mesh.wall_facets: no wall facets found")
        return np.concatenate(gathered, axis=0)
