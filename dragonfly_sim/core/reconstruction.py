"""
Green-Gauss gradient reconstruction for the p = 0 discretization.

At p = 0, grad U vanishes inside every cell, which leaves no viscous volume
term and no vorticity for the turbulence sources, and gives only
first-order face states. As a cell-centred finite-volume code does, a DG0
gradient G is reconstructed by Green-Gauss over each cell's facets, with
facet averages {U} inside and the registered boundary state U_b on the
boundary,

    |K| G_K = sum_f |f| U_f (x) n_f.

G is used in three places:

  * MUSCL: the interior-facet traces of the inviscid flux are reconstructed
    linearly, U_K + G_K . (x_f - x_K), optionally with a smooth van Albada
    limiter (see muscl_traces).
  * The viscous facet gradient, {G} with its component along the line
    between the two cell centres replaced by the direct difference (which
    keeps the two-point coupling that suppresses odd-even modes), or on a
    boundary facet with the normal component replaced by (U_b - U)/d_n.
  * The turbulence sources (vorticity, gradients of the transported scalars).

G is eliminated from the unknowns exactly (utils/gradient_condensation.py),
so Newton sees the true Jacobian. For adjoints and shape derivatives G and
the cell centres are DerivedFields (utils/derived_fields.py), with exact
derivatives with respect to U, alpha and the mesh coordinates.
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
import dolfinx.fem.petsc

from dragonfly_sim.utils.derived_fields import (DerivedField, diagonal_mass_inverse,
                                                mesh_coordinate_argument, destroy_operator)
from dragonfly_sim.utils.gradient_condensation import CondensedGradientProblem

_JIT = {"cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"]}


class CellCentres(DerivedField):
    """
    DG0 cell centres, each cell's reference midpoint mapped to the cell. For
    the degree-1 geometry used here that is the plain average of the cell's
    vertices, c = C x, so dc/dx is the constant averaging matrix C.
    """

    def __init__(self, mesh_obj):
        self.mesh_obj = mesh_obj
        # the reconstruction (set by its install()) decides whether dc/dx is built
        # vectorized and once -- it is a constant matrix -- or per call in a Python loop
        self.rec = None
        self._mesh_jac = None
        msh = mesh_obj.mesh
        gdim = msh.geometry.dim
        self.function = dolfinx.fem.Function(
            dolfinx.fem.functionspace(msh, ("DG", 0, (gdim,))), name="cell_centres")
        self.update_geometry()

    def update_geometry(self):
        gdim = self.mesh_obj.mesh.geometry.dim
        self.function.interpolate(lambda x: x[:gdim])
        self.function.x.scatter_forward()

    def mesh_jacobian(self):
        if self.rec is None or not self.rec.fast_cell_centre_jacobian:
            return self._mesh_jacobian_loop()
        if self._mesh_jac is None:
            self._mesh_jac = self._mesh_jacobian_vectorized()
        return self._mesh_jac

    def discard(self, obj):
        if obj is not self._mesh_jac:
            destroy_operator(obj)

    def release_cache(self):
        # the matrix is constant: kept until the reconstruction's cache is released
        if self._mesh_jac is not None:
            self._mesh_jac.destroy()
            self._mesh_jac = None

    def _mesh_jacobian_vectorized(self):
        """The same matrix as _mesh_jacobian_loop, as one CSR build: row (cell,
        component i) holds 1/nv at the cell's vertex dofs' component i."""
        msh = self.mesh_obj.mesh
        Vc = self.function.function_space
        Vx = self.mesh_obj.coordinate_space
        gdim = msh.geometry.dim
        n_own = msh.topology.index_map(msh.topology.dim).size_local
        rows_map, cols_map = Vc.dofmap.index_map, Vx.dofmap.index_map
        IT = PETSc.IntType
        # the rows of a CSR are in the order of the owned row dofs, not of the cells
        cell_dofs = np.asarray(Vc.dofmap.list[:n_own, 0])
        order = np.argsort(cell_dofs)
        assert np.array_equal(cell_dofs[order], np.arange(n_own)), "cell centre dofs are not the owned blocks"
        vertex_dofs = np.asarray(Vx.dofmap.list[:n_own])[order]
        nv = vertex_dofs.shape[1] if n_own else 1
        cols = cols_map.local_to_global(vertex_dofs.ravel()).reshape(n_own, nv).astype(np.int64)
        idx = cols[:, None, :] * gdim + np.arange(gdim)[None, :, None]       # (cell, component, vertex)
        idx = np.sort(idx.reshape(n_own * gdim, nv), axis=1)
        indptr = np.arange(n_own * gdim + 1, dtype=np.int64) * nv
        data = np.full(n_own * gdim * nv, 1.0 / nv)
        X = PETSc.Mat().createAIJ(
            ((rows_map.size_local * gdim, rows_map.size_global * gdim),
             (cols_map.size_local * gdim, cols_map.size_global * gdim)),
            csr=(indptr.astype(IT), idx.ravel().astype(IT), data), comm=msh.comm)
        X.assemble()
        return X

    def _mesh_jacobian_loop(self):
        msh = self.mesh_obj.mesh
        Vc = self.function.function_space
        Vx = self.mesh_obj.coordinate_space
        gdim = msh.geometry.dim
        n_own = msh.topology.index_map(msh.topology.dim).size_local
        rows_map, cols_map = Vc.dofmap.index_map, Vx.dofmap.index_map
        X = PETSc.Mat().createAIJ(
            ((rows_map.size_local * gdim, rows_map.size_global * gdim),
             (cols_map.size_local * gdim, cols_map.size_global * gdim)),
            comm=msh.comm)
        nv = Vx.dofmap.cell_dofs(0).size if n_own else 1
        X.setPreallocationNNZ((nv, nv))
        X.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        IT = PETSc.IntType
        for cell in range(n_own):
            r = int(rows_map.local_to_global(Vc.dofmap.cell_dofs(cell))[0])
            cols = cols_map.local_to_global(Vx.dofmap.cell_dofs(cell)).astype(IT)
            for i in range(gdim):
                X.setValues(np.array([r * gdim + i], dtype=IT), cols * gdim + i,
                            np.full(cols.size, 1.0 / cols.size))
        X.assemble()
        return X


class GreenGaussGradient(DerivedField):
    """
    G = M^-1 b(U, alpha, x), as a DerivedField (see the module docstring).

    Options (attributes of the reconstruction):
      cache_jacobians          keep dG/dU, dG/dalpha and dG/dx between calls
                               while the state, alpha and the mesh are unchanged
                               (several functionals and products of one totals
                               evaluation share them); freed by release_cache
      structured_state_jacobian  dG/dU as the operator W (x) I plus a small
                               boundary part (GreenGaussStateOperator) instead
                               of an assembled matrix
    """

    def __init__(self, reconstruction):
        self.rec = reconstruction
        self.function = reconstruction.G
        self.depends_on_state = True
        self._forms = {}
        self._cache = None

    def _form(self, key, factory):
        if key not in self._forms:
            self._forms[key] = dolfinx.fem.form(factory(), jit_options=_JIT)
        return self._forms[key]

    def update(self):
        self.rec.update_gradient()

    def update_geometry(self):
        self.rec.update_mass()

    # ---- content-keyed cache -------------------------------------------
    def _key(self):
        m = self.rec.model
        n = m.u_vec.x.petsc_vec.local_size
        return (np.array(m.u_vec.x.array[:n], copy=True), np.array(m.mesh_obj.mesh.geometry.x, copy=True),
                float(m.farfield_conditions['alpha']))

    def _same_key(self, key):
        a, b = self._cache["key"], key
        same = a[2] == b[2] and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        return bool(self.rec.mesh_obj.mesh.comm.allreduce(int(same), op=MPI.MIN))

    def _cached(self, name, build):
        """build() once per (state, alpha, mesh), or every call if caching is off."""
        if not self.rec.cache_jacobians:
            return build()
        key = self._key()
        if self._cache is None or not self._same_key(key):
            self.release_cache()
            self._cache = {"key": key}
        if name not in self._cache:
            self._cache[name] = build()
        return self._cache[name]

    def release_cache(self):
        cache, self._cache = self._cache, None
        if cache is not None:
            for name, obj in cache.items():
                if name != "key" and obj is not None:
                    destroy_operator(obj)

    def discard(self, obj):
        if self._cache is not None and any(obj is v for k, v in self._cache.items() if k != "key"):
            return
        destroy_operator(obj)

    # ---- the Jacobians -------------------------------------------------
    def state_jacobian(self):
        return self._cached("state", self._build_state_jacobian)

    def _build_state_jacobian(self):
        rec = self.rec
        if rec.structured_state_jacobian:
            return rec.state_operator()
        a = self._form("U", lambda: ufl.derivative(
            rec.gg_rhs, rec.model.u_vec, ufl.TrialFunction(rec.model.u_vec.function_space)))
        B = dolfinx.fem.petsc.assemble_matrix(a)
        B.assemble()
        B.diagonalScale(L=rec.inv_mass)
        return B

    def alpha_vector(self):
        return self._cached("alpha", self._build_alpha_vector)

    def _build_alpha_vector(self):
        rec = self.rec
        if getattr(rec.model, "alpha_var", None) is None:
            return None
        key = "alpha"
        if key not in self._forms:
            dform = ufl.algorithms.expand_derivatives(ufl.diff(rec.gg_rhs, rec.model.alpha_var))
            self._forms[key] = None if dform.empty() else dolfinx.fem.form(dform, jit_options=_JIT)
        if self._forms[key] is None:
            return None
        v = dolfinx.fem.petsc.assemble_vector(self._forms[key])
        v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        v.pointwiseMult(v, rec.inv_mass)
        return v

    def mesh_jacobian(self):
        return self._cached("mesh", self._build_mesh_jacobian)

    def _build_mesh_jacobian(self):
        # r_G(G, U, x) = int G : tau dx - b(U; tau) = 0, so dG/dx = -M^-1 dr_G/dx
        rec = self.rec
        G = rec.G
        tau = ufl.TestFunction(G.function_space)
        r_G = ufl.inner(G, tau) * rec.model.forms.dx - rec.gg_rhs
        Vx = rec.mesh_obj.coordinate_space
        a = self._form("x", lambda: ufl.derivative(
            r_G, ufl.SpatialCoordinate(rec.mesh_obj.mesh), mesh_coordinate_argument(r_G, Vx)))
        X = dolfinx.fem.petsc.assemble_matrix(a)
        X.assemble()
        X.diagonalScale(L=rec.inv_mass)
        X.scale(-1.0)
        return X


class GreenGaussStateOperator:
    """
    dG/dU = M_G^-1 db/dU as a python-context Mat (G layout x state layout)
    with mult and multTranspose, without assembling the 18-by-6 blocks.

    The interior-facet part of b is a facet average of U times the area
    vector, component by component: its derivative is W (x) I_n, with W the
    scalar operator (3 area-vector components per cell pair, assembled once
    per mesh) and I_n the identity over the n state components. Only the
    boundary-facet part, U_b(U) of the registered boundary states, couples
    components; it is assembled per state (boundary cells only).
    """

    def __init__(self, rec, W, B_bdry):
        self.rec, self.W, self.B_bdry = rec, W, B_bdry
        V = rec.model.u_vec.function_space
        self.n = V.dofmap.index_map.size_local
        self.ns, self.gdim = rec.model.n_state, rec.model.dimensions
        self._xk, self._vk = W.createVecRight(), W.createVecLeft()
        self._t = rec.inv_mass.duplicate()

    def mult(self, mat, x, y):
        n, ns, gdim = self.n, self.ns, self.gdim
        xa = x.getArray(readonly=True)[:n * ns].reshape(n, ns)
        ya = np.empty((n, ns, gdim))
        for k in range(ns):
            self._xk.setArray(np.ascontiguousarray(xa[:, k]))
            self.W.mult(self._xk, self._vk)
            ya[:, k, :] = self._vk.getArray(readonly=True).reshape(n, gdim)
        y.setArray(ya.reshape(-1))
        if self.B_bdry is not None:
            self.B_bdry.multAdd(x, y, y)
        y.pointwiseMult(y, self.rec.inv_mass)

    def multTranspose(self, mat, v, y):
        n, ns, gdim = self.n, self.ns, self.gdim
        t = self._t
        t.pointwiseMult(v, self.rec.inv_mass)
        ta = t.getArray(readonly=True)[:n * ns * gdim].reshape(n, ns, gdim)
        out = np.empty((n, ns))
        for k in range(ns):
            self._vk.setArray(np.ascontiguousarray(ta[:, k, :]).reshape(-1))
            self.W.multTranspose(self._vk, self._xk)
            out[:, k] = self._xk.getArray(readonly=True)
        y.setArray(out.reshape(-1))
        if self.B_bdry is not None:
            self.B_bdry.multTransposeAdd(t, y, y)

    def destroy(self, mat=None):
        # called by PETSc when the python Mat is destroyed; W belongs to the
        # reconstruction (one per mesh)
        for obj in (self.B_bdry, self._xk, self._vk, self._t):
            if obj is not None:
                obj.destroy()
        self.B_bdry = self._xk = self._vk = self._t = None


class GreenGaussReconstruction:
    """
    p = 0 gradient reconstruction attached to a CompressibleEulerModel host.

    install() plugs it in: the MUSCL face traces (if muscl), the derived
    fields (cell centres and G), and the condensed nonlinear problem.
    Call it after the function spaces exist and before the boundary
    conditions are defined; the Green-Gauss forms are built lazily, once
    every boundary condition has registered its state.

    limiter: None (unlimited MUSCL) or "vanalbada". limiter_scales() returns
    one magnitude per state component for the limiter's smoothing parameter.
    """

    def __init__(self, model, muscl=True, limiter=None, limiter_eps=1e-2, limiter_scales=None):
        if limiter not in (None, "vanalbada"):
            raise ValueError("limiter must be None or 'vanalbada'")
        self.model = model
        self.mesh_obj = model.mesh_obj
        self.muscl = muscl
        self.limiter = limiter
        self.limiter_eps = limiter_eps
        self.limiter_scales = limiter_scales
        msh = self.mesh_obj.mesh
        V_G = dolfinx.fem.functionspace(msh, ("DG", 0, (model.n_state, model.dimensions)))
        self.G = dolfinx.fem.Function(V_G, name="green_gauss_gradient")
        self.centres = CellCentres(self.mesh_obj)
        self.xc = self.centres.function
        self.gradient = GreenGaussGradient(self)
        # options of the adjoint's derived-field Jacobians (see GreenGaussGradient)
        self.fast_cell_centre_jacobian = False
        self.cache_jacobians = False
        self.structured_state_jacobian = False
        self._W = None
        self._W_key = None
        self.gg_rhs = None
        self._gg_form = None
        self.inv_mass = None

    def install(self):
        m = self.model
        m.reconstruction = self
        if self.muscl:
            m.face_traces = self.muscl_traces
        self.centres.rec = self
        m.derived_fields += [self.centres, self.gradient]
        m.problem_factory = self.make_problem

    # ------------------------------------------------------------------
    # Face states and gradients (UFL)
    # ------------------------------------------------------------------
    def muscl_traces(self):
        """
        Linear reconstruction of both interior-facet traces from G at the
        facet's quadrature points: U_K + G_K . (x_f - x_K) unlimited, or with
        limiter="vanalbada" the edge-based van Albada form (Blazek,
        Computational Fluid Dynamics, ch. 5) generalised to an off-centre
        facet. For the cell K and its neighbour N across the facet, with
        s = |x_f - x_K| / |x_N - x_K|,

            D+ = s (U_N - U_K)                  (central increment to the facet)
            D- = 2 G_K . (x_f - x_K) - D+       (upwind increment)
            U_f = U_K + Psi(D-, D+),
            Psi(a, b) = ((a^2 + e) b + (b^2 + e) a) / (a^2 + b^2 + 2e),

        componentwise. Psi(a, a) = a, so smooth regions keep the unlimited
        reconstruction; at a shock D- ~ 0 and the trace falls back to first
        order. Psi is smooth, so the exact Jacobian survives.
        e = (limiter_eps * S_k)^2 with S_k the component's scale.
        """
        U = self.model.u_vec
        G, xc = self.G, self.xc
        # x is continuous, so x('+') = x('-'); the explicit restriction is
        # what lets the form's mesh-coordinate derivative compile on dS
        x = ufl.SpatialCoordinate(self.mesh_obj.mesh)
        dp, dm = x('+') - xc('+'), x('-') - xc('-')
        if self.limiter is None:
            return U('+') + ufl.dot(G('+'), dp), U('-') + ufl.dot(G('-'), dm)
        r = xc('-') - xc('+')
        rr = ufl.sqrt(ufl.dot(r, r))
        eps = [(self.limiter_eps * s)**2 for s in self.limiter_scales()]
        n_state = self.model.n_state

        def trace(UK, UN, GK, d):
            s = ufl.sqrt(ufl.dot(d, d)) / rr
            Dp = s * (UN - UK)
            Dm = 2.0 * ufl.dot(GK, d) - Dp
            return ufl.as_vector([
                UK[k] + ((Dm[k]**2 + eps[k]) * Dp[k] + (Dp[k]**2 + eps[k]) * Dm[k])
                / (Dm[k]**2 + Dp[k]**2 + 2.0 * eps[k]) for k in range(n_state)])

        return trace(U('+'), U('-'), G('+'), dp), trace(U('-'), U('+'), G('-'), dm)

    def face_gradient_interior(self):
        U = self.model.u_vec
        G, xc = self.G, self.xc
        d = xc('-') - xc('+')
        dist = ufl.sqrt(ufl.dot(d, d))
        e = d / dist
        G_avg = 0.5 * (G('+') + G('-'))
        return G_avg + ufl.outer((U('-') - U('+')) / dist - ufl.dot(G_avg, e), e)

    def face_gradient_boundary(self, U_b, U=None):
        """The cell's G with its normal component replaced by (U_b - U)/d_n,
        d_n the cell centre's normal distance from the facet."""
        U = self.model.u_vec if U is None else U
        n = self.mesh_obj.n
        d_n = ufl.dot(ufl.SpatialCoordinate(self.mesh_obj.mesh) - self.xc, n)
        return self.G + ufl.outer((U_b - U) / d_n - ufl.dot(self.G, n), n)

    def green_gauss_rhs(self):
        """b(U; tau) with |K| G_K = b: facet averages inside, the registered
        boundary states on the boundary."""
        tau = ufl.TestFunction(self.G.function_space)
        U, n = self.model.u_vec, self.mesh_obj.n
        forms = self.model.forms
        b = ufl.inner(ufl.outer(0.5 * (U('+') + U('-')), n('+')), tau('+') - tau('-')) * forms.dS
        b_bdry = None
        for U_b, tag, _ in self.model.boundary_states:
            term = ufl.inner(ufl.outer(U_b, n), tau) * forms.ds(tag)
            b_bdry = term if b_bdry is None else b_bdry + term
        self.gg_rhs_boundary = b_bdry
        return b if b_bdry is None else b + b_bdry

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def set_up(self):
        if self._gg_form is not None:
            return
        msh = self.mesh_obj.mesh
        comm = msh.comm
        # Every boundary facet needs a facet state, or its cell's gradient
        # silently loses a term. Counted, not integrated: facet areas depend
        # on the quadrature on warped (bilinear) hex faces.
        tags = sorted({t for _, t, _ in self.model.boundary_states})
        fdim = msh.topology.dim - 1
        n_own = msh.topology.index_map(fdim).size_local
        mt = self.mesh_obj.meshtags
        tagged = np.unique(mt.indices[np.isin(mt.values, tags)])
        ext = dolfinx.mesh.exterior_facet_indices(msh.topology)
        missing = comm.allreduce(int(np.sum(~np.isin(ext[ext < n_own], tagged))), op=MPI.SUM)
        if missing:
            raise RuntimeError("Green-Gauss: {} boundary facets have no registered boundary "
                               "state (tags {})".format(missing, tags))
        self.gg_rhs = self.green_gauss_rhs()
        self._gg_form = dolfinx.fem.form(self.gg_rhs, jit_options=_JIT)
        self._gg_vec = dolfinx.fem.petsc.create_vector(self.G.function_space)
        self.inv_mass = diagonal_mass_inverse(self.G.function_space, self.model.forms.dx)

    # ------------------------------------------------------------------
    # dG/dU without assembling its 18-by-6 blocks (GreenGaussStateOperator)
    # ------------------------------------------------------------------
    def _weights(self):
        """W, the scalar operator of the interior-facet part of dG/dU
        before M_G^-1: (3 n_cells) x n_cells, assembled once per mesh."""
        msh = self.mesh_obj.mesh
        x = msh.geometry.x
        if self._W is not None:
            same = self._W_key.shape == x.shape and np.array_equal(self._W_key, x)
            if bool(msh.comm.allreduce(int(same), op=MPI.MIN)):
                return self._W
            self._W.destroy()
            self._W = None
        gdim = msh.geometry.dim
        V1 = dolfinx.fem.functionspace(msh, ("DG", 0))
        Vg = dolfinx.fem.functionspace(msh, ("DG", 0, (gdim,)))
        V = self.model.u_vec.function_space
        n = V.dofmap.index_map.size_local
        for W_ in (V1, Vg, self.G.function_space):
            if not np.array_equal(np.asarray(W_.dofmap.list[:n]), np.asarray(V.dofmap.list[:n])):
                raise RuntimeError("the DG0 dof numbering differs between the state and gradient spaces")
        u1, t3, nn = ufl.TrialFunction(V1), ufl.TestFunction(Vg), self.mesh_obj.n
        a = ufl.inner(0.5 * (u1('+') + u1('-')) * nn('+'), t3('+') - t3('-')) * self.model.forms.dS
        W = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a, jit_options=_JIT))
        W.assemble()
        self._W, self._W_key = W, np.array(x, copy=True)
        return W

    def state_operator(self):
        """dG/dU at the current state, as a python-context Mat."""
        self.set_up()
        W = self._weights()
        B_bdry = None
        if self.gg_rhs_boundary is not None:
            if getattr(self, "_a_bdry", None) is None:
                V = self.model.u_vec.function_space
                d = ufl.algorithms.expand_derivatives(
                    ufl.derivative(self.gg_rhs_boundary, self.model.u_vec, ufl.TrialFunction(V)))
                self._a_bdry = False if d.empty() else dolfinx.fem.form(d, jit_options=_JIT)
            if self._a_bdry is not False:
                B_bdry = dolfinx.fem.petsc.assemble_matrix(self._a_bdry)
                B_bdry.assemble()
        ctx = GreenGaussStateOperator(self, W, B_bdry)
        nG = self.G.function_space.dofmap.index_map
        nU = self.model.u_vec.function_space.dofmap.index_map
        bsG = self.G.function_space.dofmap.index_map_bs
        bsU = self.model.u_vec.function_space.dofmap.index_map_bs
        M = PETSc.Mat().createPython(
            ((nG.size_local * bsG, nG.size_global * bsG), (nU.size_local * bsU, nU.size_global * bsU)),
            comm=self.mesh_obj.mesh.comm)
        M.setPythonContext(ctx)
        M.setUp()
        return M

    def release_cache(self):
        self.gradient.release_cache()
        self.centres.release_cache()

    def update_mass(self):
        """Refresh M_G^-1 in place after mesh motion."""
        if self.inv_mass is None:
            return
        new = diagonal_mass_inverse(self.G.function_space, self.model.forms.dx)
        new.copy(self.inv_mass)
        new.destroy()

    def update_gradient(self):
        """G <- M_G^-1 b(U) at the current state and mesh. Collective."""
        self.set_up()
        b = self._gg_vec
        with b.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, self._gg_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.G.x.petsc_vec.pointwiseMult(b, self.inv_mass)
        self.G.x.scatter_forward()

    def make_problem(self, F_newton, J_newton):
        self.set_up()
        return CondensedGradientProblem(F_newton, J_newton, self.model.u_vec, self.G, self.gg_rhs,
                                        self.update_gradient, self.inv_mass, jit_options=_JIT)

    def compact_forms(self, F):
        """(a_UU, a_UG): the compiled forms of dR_U/dU and dR_U/dG, cached for F."""
        if getattr(self, "_compact_forms", None) is None or self._compact_forms[0] is not F:
            self.set_up()
            a_UU = dolfinx.fem.form(ufl.derivative(F, self.model.u_vec), jit_options=_JIT)
            a_UG = dolfinx.fem.form(ufl.derivative(F, self.G, ufl.TrialFunction(self.G.function_space)),
                                    jit_options=_JIT)
            self._compact_forms = (F, (a_UU, a_UG))
        return self._compact_forms[1]

    def condensed_jacobian(self, F):
        """The exact dR/dU of the residual form F, as a CondensedJacobian."""
        from dragonfly_sim.utils.gradient_condensation import CondensedJacobian
        self.set_up()
        return CondensedJacobian(F, self.model.u_vec, self.G, self.gg_rhs, self.inv_mass,
                                 jit_options=_JIT)
