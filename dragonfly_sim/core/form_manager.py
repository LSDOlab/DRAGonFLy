import numpy as np
import ufl
import dolfinx

# Subdomain IDs of the tagged integration measures
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

    The measures are built incrementally, because the two construction calls
    are independent and the existing call order varies: set_up_sim
    builds the boundary measure first (it needs the facet split to tag the
    wall before the mesh warper is constructed) and dx/dS later, while
    some tests build dx/dS and never build ds at all.
    """

    def __init__(self, mesh_obj, model=None, measure_source=None):
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

        # {key: ufl.Form} -- the UFL expressions registered on this manager.
        self._ufl_forms = {}

        # {key: dolfinx.fem.Form} -- the compiled bindings of those
        # expressions, cached so each form is compiled at most once.
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
        site applies its own tag: ds(WALL_TAG), ds(INLET_TAG), ...
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

    def ufl_form(self, key, factory=None):
        """The UFL expression registered under `key`, building it on demand."""
        if key not in self._ufl_forms:
            if factory is None:
                raise KeyError(
                    "No UFL form registered under {!r} and no factory given "
                    "to build one.".format(key))
            self._ufl_forms[key] = factory()
        return self._ufl_forms[key]

    def compiled(self, key, factory=None):
        """
        The compiled form (dolfinx.fem.form) of the expression registered
        under `key`, compiled on first request and cached.
        """
        if key not in self._compiled:
            self._compiled[key] = dolfinx.fem.form(self.ufl_form(key, factory))
        return self._compiled[key]

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
