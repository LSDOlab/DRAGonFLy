import numpy as np
import ufl
import dolfinx
import dolfinx.fem.petsc
from dolfinx.fem import IntegralType

# Subdomain IDs of the tagged integration measures. Tagged rather than the
# default id so that a restricted twin (FormManager.restrict) can bind the
# same compiled kernel over a subset of the entities: create_form applies
# entity lists only to integrals with a non-default subdomain id.
CELL_TAG = 1
INTERIOR_FACET_TAG = 1

def _all_entities_meshtags(mesh, dim, value):
    # Apply the meshtag `value` to all local + ghost entities of dimension `dim`
    imap = mesh.topology.index_map(dim)
    if imap is None:
        mesh.topology.create_entities(dim)
        imap = mesh.topology.index_map(dim)

    n_entities = imap.size_local + imap.num_ghosts
    indices = np.arange(n_entities, dtype=np.int32)
    values = np.full(n_entities, value, dtype=np.int32)
    return dolfinx.mesh.meshtags(mesh, dim, indices, values)


# ufcx's integral-type order (ufcx.h: cell, exterior_facet, interior_facet,
# vertex, ridge), which is the order of ufcx_form.form_integral_offsets.
_UFCX_INTEGRAL_TYPES = (IntegralType.cell, IntegralType.exterior_facet,
                        IntegralType.interior_facet, IntegralType.vertex,
                        IntegralType.ridge)


def _pad_integral_ids(compiled_form, subdomains):
    """
    Give create_form an entry for EVERY subdomain id the compiled form
    integrates over, with an empty entity array for any id this rank has no
    entities for.

    dolfinx 0.11 needs this: with ids {2, 3, 4} in the form, supplying only
    [3] (anything but a leading run of the sorted ids) makes create_matrix
    raise "Requested domain for argument not found". A restricted EntitySet
    lists only the tags it has sampled facets for, and that differs by rank,
    so the failure is rank-local and the other ranks deadlock in the
    collective sparsity build.
    """
    ufcx = compiled_form.ufcx_form
    offsets = [ufcx.form_integral_offsets[k]
               for k in range(len(_UFCX_INTEGRAL_TYPES) + 1)]
    padded = {itype: list(entries) for itype, entries in subdomains.items()}
    for k, itype in enumerate(_UFCX_INTEGRAL_TYPES):
        ids = {int(ufcx.form_integral_ids[j])
               for j in range(offsets[k], offsets[k + 1])}
        present = {int(i) for i, _ in padded.get(itype, [])}
        # -1 is the "everywhere" id: dolfinx integrates it over the whole
        # domain whatever is supplied for it, so it needs no padding.
        for i in sorted(ids - present - {-1}):
            padded.setdefault(itype, []).append(
                (i, np.zeros(0, dtype=np.int32)))
    return padded


class EntitySet:
    """
    The mesh entities a form is integrated over: cells, interior facets and
    the exterior facets of each boundary tag (process-local indices). For a
    full-order assembly that is every entity; a hyper-reduced assembly binds
    the same compiled form over a small subset (FormManager.restrict).
    """

    def __init__(self, mesh, cells=None, interior_facets=None,
                 exterior_facets_by_tag=None):
        self.mesh = mesh
        tdim = mesh.topology.dim

        self.cells = (np.asarray(cells, dtype=np.int32)
                      if cells is not None else
                      self._owned_indices(tdim))
        self.interior_facets = (np.asarray(interior_facets, dtype=np.int32)
                                if interior_facets is not None else
                                self._owned_indices(tdim - 1))
        self.exterior_facets_by_tag = dict(exterior_facets_by_tag or {})

    def _owned_indices(self, dim):
        imap = self.mesh.topology.index_map(dim)
        if imap is None:
            self.mesh.topology.create_entities(dim)
            imap = self.mesh.topology.index_map(dim)
        return np.arange(imap.size_local, dtype=np.int32)

    @classmethod
    def all(cls, mesh, facet_tags=None, tags=None):
        """Every owned cell and interior facet, and the boundary facets of
        `tags` in `facet_tags`."""
        mesh.topology.create_connectivity(mesh.topology.dim - 1,
                                          mesh.topology.dim)
        exterior = {}
        if facet_tags is not None and tags is not None:
            for tag in tags:
                exterior[int(tag)] = facet_tags.find(int(tag))
        return cls(mesh, exterior_facets_by_tag=exterior)

    def integration_domains(self):
        """
        `{IntegralType: [(subdomain_id, entities)]}` for create_form.

        compute_integration_domains returns the flat entity array directly
        (since dolfinx 0.10 it takes no `dim`: the entity dimension follows
        from the integral type), and drops every entity this rank does not
        own.
        """
        topology = self.mesh.topology
        tdim = topology.dim
        topology.create_connectivity(tdim - 1, tdim)
        topology.create_connectivity(tdim, tdim - 1)

        def domains(integral_type, entities):
            computed = dolfinx.fem.compute_integration_domains(
                integral_type, topology, np.asarray(entities, dtype=np.int32))
            return np.asarray(computed, dtype=np.int32).flatten()

        subdomains = {
            IntegralType.cell: [
                (CELL_TAG, domains(IntegralType.cell, self.cells))],
            IntegralType.interior_facet: [
                (INTERIOR_FACET_TAG,
                 domains(IntegralType.interior_facet, self.interior_facets))],
        }
        if self.exterior_facets_by_tag:
            subdomains[IntegralType.exterior_facet] = [
                (int(tag), domains(IntegralType.exterior_facet, facets))
                for tag, facets in sorted(self.exterior_facets_by_tag.items())]
        return subdomains

    def __repr__(self):
        return ("EntitySet(cells={}, interior_facets={}, "
                "exterior_facets={})".format(
                    self.cells.size, self.interior_facets.size,
                    {t: f.size for t, f in self.exterior_facets_by_tag.items()}))


class FormManager:
    """
    Owns the integration measures and the form graph.

    Instances come in two flavours, because the two things have different
    lifetimes:

    - MESH-LEVEL (`model=None`, reached as `mesh_obj.form_manager`): owns the
      measures. There is one per Mesh, so every model built on that Mesh
      integrates against the same measures.

      Mesh itself exposes no dx/dS/ds. It did briefly, as delegating
      properties, and that was worse than either alternative: a measure
      reached through the Mesh reads as though the Mesh owned it, which is
      the exact misreading that let the old construct_* methods overwrite
      measures in place underneath compiled forms and live postprocessor
      reads. Naming the owner at every use site is the point.
    - MODEL-LEVEL (`model=...`, `measure_source=<the mesh-level one>`): owns
      one model's residual/Jacobian expressions and its compiled forms, and
      reads the measures through the mesh-level instance.

    `restrict()` produces a third flavour, for hyper-reduction: a model-level
    manager over a sampled EntitySet, sharing the parent's UFL expressions and
    compiled kernels, so that its forms are the same forms bound to fewer
    entities.

    The measures are built incrementally, because the two construction calls
    are independent and the existing call order varies: set_up_sim
    builds the boundary measure first (it needs the facet split to tag the
    wall before the mesh warper is constructed) and dx/dS later, while
    some tests build dx/dS and never build ds at all.
    """

    def __init__(self, mesh_obj, model=None, measure_source=None,
                 entity_set=None, kernels=None):
        self.mesh_obj = mesh_obj
        self.mesh = mesh_obj.mesh
        self.model = model
        self._measure_source = measure_source

        self._dx = None
        self._dS = None
        self._ds = None

        self._cell_tags = None
        self._interior_facet_tags = None

        self.cell_metadata = None
        self.boundary_metadata = None

        # None: this manager binds its forms over the whole domain, through
        # dolfinx.fem.form. An EntitySet: over those entities, through
        # create_form (a restricted twin, see restrict()).
        self._entity_set = entity_set

        # {key: {"ufl": ufl.Form, "kernel": CompiledForm, "jit_options": dict}}
        # -- the UFL expressions registered on this manager and their
        # JIT-compiled kernels. Shared by reference with every twin produced by
        # restrict(), so a twin never recompiles and cannot bind a different
        # expression than its parent.
        self._kernels = kernels if kernels is not None else {}

        # {key: dolfinx.fem.Form} -- this manager's bindings of those
        # expressions, cached so each form is bound at most once. Not shared:
        # two managers bind the same kernel over different entity sets.
        self._compiled = {}

    # ------------------------------------------------------------------
    # Measures
    # ------------------------------------------------------------------
    def build_cell_and_interior_facet_measures(self, measure_metadata=None):
        """
        dx and dS, tagged and backed by all-ones MeshTags.

        Both come back already restricted to their tag, so a call site writes
        `* forms.dx` and gets dx(CELL_TAG).
        """
        self.cell_metadata = measure_metadata
        tdim = self.mesh.topology.dim

        self._cell_tags = _all_entities_meshtags(self.mesh, tdim, CELL_TAG)
        self._interior_facet_tags = _all_entities_meshtags(
            self.mesh, tdim - 1, INTERIOR_FACET_TAG)

        kwargs = {} if measure_metadata is None else {"metadata": measure_metadata}
        self._dx = ufl.Measure(
            "dx", domain=self.mesh, subdomain_data=self._cell_tags,
            **kwargs)(CELL_TAG)
        self._dS = ufl.Measure(
            "dS", domain=self.mesh, subdomain_data=self._interior_facet_tags,
            **kwargs)(INTERIOR_FACET_TAG)

    def build_boundary_measure(self, measure_metadata=None):
        """
        ds, over the boundary facet tagging already on the Mesh.

        The tagging itself stays Mesh.tag_boundary_facets' job and is NOT
        built here. It is mesh topology, not a measure: the mesh warper, the
        FFD wall-node selection and the meshtags XDMF export all read
        Mesh.meshtags without ever touching a form. Only the measure derived
        from it belongs to this class.

        Unlike dx/dS this measure comes back UNrestricted, because every call
        site applies its own tag: ds(WALL_TAG), ds(FARFIELD_TAG), ...
        """
        if self.mesh_obj.meshtags is None:
            raise RuntimeError(
                "The boundary facets have not been tagged yet. Call "
                "Mesh.tag_boundary_facets() before build_boundary_measure().")

        self.boundary_metadata = measure_metadata

        kwargs = {} if measure_metadata is None else {"metadata": measure_metadata}
        self._ds = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=self.mesh_obj.meshtags,
            **kwargs)

        return self._ds

    def _measures(self):
        """The manager that actually holds the measures (self, or the mesh-level one)."""
        return self if self._measure_source is None else self._measure_source

    @property
    def dx(self):
        src = self._measures()
        if src is not self:
            return src.dx
        if self._dx is None:
            raise RuntimeError(
                "dx has not been built yet. Call "
                "form_manager.build_cell_and_interior_facet_measures() "
                "before building any form.")
        return self._dx

    @property
    def dS(self):
        src = self._measures()
        if src is not self:
            return src.dS
        if self._dS is None:
            raise RuntimeError(
                "dS has not been built yet. Call "
                "form_manager.build_cell_and_interior_facet_measures() "
                "before building any form.")
        return self._dS

    @property
    def ds(self):
        src = self._measures()
        if src is not self:
            return src.ds
        if self._ds is None:
            raise RuntimeError(
                "ds has not been built yet. Call Mesh.tag_boundary_facets() "
                "and then form_manager.build_boundary_measure() before "
                "building any boundary form.")
        return self._ds

    @property
    def facet_tags(self):
        """The boundary facet tagging ds is built over. Owned by the Mesh."""
        return self.mesh_obj.meshtags

    # ------------------------------------------------------------------
    # The form graph
    # ------------------------------------------------------------------
    def build_residual(self, U_coeff=None):
        """
        The physical residual: interior + boundary flux terms.

        `U_coeff` is accepted for backward compatibility but has no effect.

        The physics itself stays on CompressibleEulerModel; this composes it.
        """
        model = self.model
        if model is None:
            raise RuntimeError(
                "build_residual() needs a model-level FormManager; this one "
                "owns measures only.")

        F = model.compute_interior_integral_terms()
        for F_bc in model.bcs:
            F += F_bc

        return F

    def ufl_form(self, key, factory=None, jit_options=None):
        """The UFL expression registered under `key`, building it on demand.
        `jit_options` (CFFI JIT options, e.g. the optimization flags of the
        Newton problem's forms) are kept for every binding of the expression."""
        entry = self._kernels.setdefault(key, {})
        if "ufl" not in entry:
            if factory is None:
                raise KeyError(
                    "No UFL form registered under {!r} and no factory given "
                    "to build one.".format(key))
            entry["ufl"] = factory()
            entry["jit_options"] = jit_options
        return entry["ufl"]

    def compiled(self, key, factory=None, jit_options=None):
        """
        This manager's binding of the form registered under `key`, bound on
        first request and cached: over the whole domain (dolfinx.fem.form) for
        an ordinary manager, over its EntitySet (create_form, with the shared
        kernel) for a restricted twin.
        """
        if key in self._compiled:
            return self._compiled[key]
        form_ufl = self.ufl_form(key, factory, jit_options=jit_options)
        if self._entity_set is None:
            options = self._kernels[key].get("jit_options")
            self._compiled[key] = (dolfinx.fem.form(form_ufl) if options is None
                                   else dolfinx.fem.form(form_ufl, jit_options=options))
        else:
            self._compiled[key] = self._bind_restricted(key, form_ufl)
        return self._compiled[key]

    def _kernel(self, key, form_ufl):
        """The JIT-compiled FFCx kernel for `key`, compiled at most once (and
        found in the FFCx cache when the full-order binding compiled it)."""
        entry = self._kernels.setdefault(key, {})
        if "kernel" not in entry:
            entry["kernel"] = dolfinx.fem.compile_form(
                self.mesh.comm, form_ufl, jit_options=entry.get("jit_options"))
        return entry["kernel"]

    def _bind_restricted(self, key, form_ufl):
        """
        create_form over this manager's EntitySet. The coefficient and
        constant maps are identities: the model's Functions and Constants are
        the data the bound form reads, so the restricted form follows the same
        live state as the full one.
        """
        spaces = [arg.ufl_function_space() for arg in form_ufl.arguments()]
        coefficients = {c: c for c in ufl.algorithms.extract_coefficients(form_ufl)}
        constants = {c: c for c in ufl.algorithms.analysis.extract_constants(form_ufl)}
        kernel = self._kernel(key, form_ufl)
        return dolfinx.fem.create_form(
            kernel, spaces, self.mesh,
            _pad_integral_ids(kernel, self._entity_set.integration_domains()),
            coefficients, constants, None)

    def has_compiled(self, key):
        return key in self._compiled

    def invalidate(self, keys=None, drop_kernels=False):
        """
        Drop cached bindings so the next request rebuilds them; with
        drop_kernels also the UFL expressions and their kernels (a change of
        the form itself).
        """
        if keys is None:
            self._compiled.clear()
            if drop_kernels:
                self._kernels.clear()
            return
        for key in ([keys] if isinstance(keys, (str, tuple)) else keys):
            self._compiled.pop(key, None)
            if drop_kernels:
                self._kernels.pop(key, None)

    # ------------------------------------------------------------------
    # Hyper-reduction: the restricted twin
    # ------------------------------------------------------------------
    def restrict(self, cells=None, interior_facets=None,
                 exterior_facets=None, entity_set=None):
        """
        A FormManager over a sampled entity set, sharing this one's UFL
        expressions and compiled kernels: the same forms, bound to fewer
        entities. Omitted entity arguments default to all of them. Moving the
        twin to another sample later is rebind(), which keeps the kernels.
        """
        if entity_set is None:
            entity_set = EntitySet(
                self.mesh, cells=cells, interior_facets=interior_facets,
                exterior_facets_by_tag=(
                    exterior_facets if exterior_facets is not None
                    else self._all_exterior_facets_by_tag()))
        return FormManager(self.mesh_obj, model=self.model,
                           measure_source=self._measures(),
                           entity_set=entity_set, kernels=self._kernels)

    def rebind(self, entity_set):
        """Point this manager at a different entity set, keeping its kernels."""
        self._entity_set = entity_set
        self._compiled.clear()

    @property
    def entity_set(self):
        return self._entity_set

    def entity_set_all(self, boundary_tags=None):
        """The full-order entity set; `boundary_tags` are the ds ids the
        model integrates over."""
        return EntitySet.all(self.mesh, facet_tags=self.facet_tags, tags=boundary_tags)

    def _all_exterior_facets_by_tag(self):
        """Every tagged boundary facet, grouped by tag; empty if untagged."""
        facet_tags = self.facet_tags
        if facet_tags is None:
            return {}
        return {int(tag): facet_tags.find(int(tag))
                for tag in np.unique(facet_tags.values) if int(tag) != 0}

    # ------------------------------------------------------------------
    # DOF <-> entity bridge (what turns a DEIM point set into an EntitySet)
    # ------------------------------------------------------------------
    def _state_space(self, space=None):
        if space is not None:
            return space
        if self.model is None or self.model.functionspaces is None:
            raise RuntimeError(
                "No function space available; pass one explicitly or call "
                "define_elements_functionspaces() first.")
        return self.model.functionspaces["V"]

    def _dof_to_cell(self, space=None):
        """Scalar dof index -> its cell. Well defined because the state space
        is DG: every dof belongs to exactly one cell."""
        V = self._state_space(space)
        tdim = self.mesh.topology.dim
        imap_cells = self.mesh.topology.index_map(tdim)
        n_cells = imap_cells.size_local + imap_cells.num_ghosts

        cell_dofs = V.dofmap.list[:n_cells]
        imap_dofs = V.dofmap.index_map
        n_dofs = imap_dofs.size_local + imap_dofs.num_ghosts

        dof_to_cell = np.full(n_dofs, -1, dtype=np.int32)
        dof_to_cell[cell_dofs.flatten()] = np.repeat(
            np.arange(n_cells, dtype=np.int32), cell_dofs.shape[1])
        return dof_to_cell

    def cells_for_dofs(self, dofs, space=None):
        """The cells whose element tensors write into the (blocked, local)
        dofs `dofs`: the rows a DEIM point set names, as mesh cells."""
        V = self._state_space(space)
        bs = V.dofmap.index_map_bs
        scalar_dofs = np.unique(np.asarray(dofs, dtype=np.int64) // bs)
        dof_to_cell = self._dof_to_cell(V)
        scalar_dofs = scalar_dofs[scalar_dofs < dof_to_cell.size]
        cells = dof_to_cell[scalar_dofs]
        return np.unique(cells[cells >= 0]).astype(np.int32)

    def dofs_for_cells(self, cells, space=None):
        """The blocked (local) dof indices of `cells`."""
        V = self._state_space(space)
        bs = V.dofmap.index_map_bs
        cells = np.asarray(cells, dtype=np.int32)
        if cells.size == 0:
            return np.zeros(0, dtype=np.int32)
        scalar = np.unique(V.dofmap.list[cells].flatten())
        blocked = (scalar[:, None] * bs + np.arange(bs)[None, :]).flatten()
        return np.unique(blocked).astype(np.int32)

    def _visible_marked_cells(self, cells):
        """The cells (owned and ghost) marked on any rank, from this rank's
        owned `cells`: a forward scatter of a cell marker. Collective."""
        tdim = self.mesh.topology.dim
        cell_map = self.mesh.topology.index_map(tdim)
        marker = dolfinx.la.vector(cell_map, 1, dtype=np.int32)
        marker.array[:] = 0
        cells = np.asarray(cells, dtype=np.int32)
        marker.array[cells[cells < cell_map.size_local]] = 1
        marker.scatter_forward()
        return np.nonzero(marker.array)[0].astype(np.int32)

    def cell_neighbours(self, cells):
        """
        The owned cells of `cells` and of their facet neighbours, on every
        rank: the closure one facet ring out, which a condensed (Green-Gauss)
        Jacobian needs for the gradient rows its sampled rows read.

        Collective. A neighbour can be a ghost on the rank that sampled the
        cell, so the marks are scattered forward (every rank sees which of its
        ghosts are sampled), extended by one ring, and summed back to the
        owners (reverse scatter).
        """
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim - 1)
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        c_to_f = self.mesh.topology.connectivity(tdim, tdim - 1)
        f_to_c = self.mesh.topology.connectivity(tdim - 1, tdim)
        cell_map = self.mesh.topology.index_map(tdim)

        visible = self._visible_marked_cells(cells)
        marker = dolfinx.la.vector(cell_map, 1, dtype=np.int32)
        marker.array[:] = 0
        marker.array[visible] = 1
        for c in visible:
            for f in c_to_f.links(int(c)):
                marker.array[f_to_c.links(int(f))] = 1
        marker.scatter_reverse(dolfinx.la.InsertMode.add)
        owned = marker.array[:cell_map.size_local]
        return np.nonzero(owned)[0].astype(np.int32)

    def entity_set_for_cells(self, cells):
        """
        The EntitySet that reproduces the residual (and Jacobian) rows of the
        owned `cells` exactly: their volume integrals and EVERY facet they
        touch, including facets shared with unsampled neighbours. Collective.

        Parallel: a facet is integrated by its owning rank only
        (compute_integration_domains drops the others), so a facet of a
        sampled cell owned by the neighbouring rank would be lost. Every rank
        therefore sees which of its ghost cells are sampled (a forward scatter)
        and contributes the facets it owns around any sampled cell; each facet
        is still integrated once.
        """
        cells = np.unique(np.asarray(cells, dtype=np.int32))
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim - 1)
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        c_to_f = self.mesh.topology.connectivity(tdim, tdim - 1)

        sampled_visible = self._visible_marked_cells(cells)
        facets = (np.unique(np.concatenate(
            [c_to_f.links(int(c)) for c in sampled_visible])).astype(np.int32)
            if sampled_visible.size else np.zeros(0, dtype=np.int32))

        f_to_c = self.mesh.topology.connectivity(tdim - 1, tdim)
        is_interior = np.array([len(f_to_c.links(int(f))) > 1 for f in facets], dtype=bool)
        interior_facets = facets[is_interior] if facets.size else facets
        exterior_facets = facets[~is_interior] if facets.size else facets

        exterior_by_tag = {}
        facet_tags = self.facet_tags
        if facet_tags is not None and exterior_facets.size:
            for tag in np.unique(facet_tags.values):
                if int(tag) == 0:
                    continue
                tagged = np.intersect1d(facet_tags.find(int(tag)), exterior_facets)
                if tagged.size:
                    exterior_by_tag[int(tag)] = tagged.astype(np.int32)

        return EntitySet(self.mesh, cells=cells, interior_facets=interior_facets,
                         exterior_facets_by_tag=exterior_by_tag)

    def entity_set_for_dofs(self, dofs, space=None):
        """The EntitySet reproducing the rows `dofs` (local, owned) exactly."""
        return self.entity_set_for_cells(self.cells_for_dofs(dofs, space=space))

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def assemble_vector_array(self, key, factory=None):
        """The assembled vector form as a numpy array (ghost-inclusive, ghost
        contributions accumulated)."""
        b = dolfinx.fem.assemble_vector(self.compiled(key, factory))
        b.scatter_reverse(dolfinx.la.InsertMode.add)
        return b.array

    def assemble_matrix_petsc(self, key, factory=None, result=None):
        """
        The assembled matrix form as a distributed PETSc Mat (aij). On a
        restricted twin it is full-size, with only the rows of the sampled
        cells exact. `result`: re-assemble in place into a Mat this method
        returned before (the restricted sparsity is fixed for an entity set);
        otherwise the caller owns the new Mat.
        """
        compiled = self.compiled(key, factory)
        A = dolfinx.fem.petsc.create_matrix(compiled) if result is None else result
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, compiled)
        A.assemble()
        return A

    def assemble_matrix_scipy(self, key, factory=None, csr=False):
        """
        The compiled form registered under `key`, assembled as a scipy sparse
        matrix.

        NOTE the format. MatrixCSR.to_scipy() returns BSR, not CSR, whenever
        the space is blocked -- which the Euler state space always is
        (block size 2 + dimensions). scipy's BSR does not implement row
        indexing at all (it raises NotImplementedError), so pass csr=True
        for anything that slices rows. The default keeps the block structure,
        which is the cheaper representation when you only need matvecs.
        """
        A = dolfinx.fem.assemble_matrix(self.compiled(key, factory))
        A.scatter_reverse()
        sparse = A.to_scipy()
        return sparse.tocsr() if csr else sparse
