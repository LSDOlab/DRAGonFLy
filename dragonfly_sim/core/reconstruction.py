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
                                                mesh_coordinate_argument)
from dragonfly_sim.utils.gradient_condensation import CondensedGradientProblem
from dragonfly_sim.utils.solver_profiling import PROFILER as _PROF

_JIT = {"cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"]}


class CellCentres(DerivedField):
    """
    DG0 cell centres, each cell's reference midpoint mapped to the cell. For
    the degree-1 geometry used here that is the plain average of the cell's
    vertices, c = C x, so dc/dx is the constant averaging matrix C.
    """

    def __init__(self, mesh_obj):
        self.mesh_obj = mesh_obj
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
    """G = M^-1 b(U, alpha, x), as a DerivedField (see the module docstring)."""

    def __init__(self, reconstruction):
        self.rec = reconstruction
        self.function = reconstruction.G
        self.depends_on_state = True
        self._forms = {}

    def _form(self, key, factory):
        if key not in self._forms:
            self._forms[key] = dolfinx.fem.form(factory(), jit_options=_JIT)
        return self._forms[key]

    def update(self):
        self.rec.update_gradient()

    def update_geometry(self):
        self.rec.update_mass()

    def state_jacobian(self):
        rec = self.rec
        a = self._form("U", lambda: ufl.derivative(
            rec.gg_rhs, rec.model.u_vec, ufl.TrialFunction(rec.model.u_vec.function_space)))
        B = dolfinx.fem.petsc.assemble_matrix(a)
        B.assemble()
        B.diagonalScale(L=rec.inv_mass)
        return B

    def alpha_vector(self):
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
        self.gg_rhs = None
        self._gg_form = None
        self.inv_mass = None

    def install(self):
        m = self.model
        m.reconstruction = self
        if self.muscl:
            m.face_traces = self.muscl_traces
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
        for U_b, tag, _ in self.model.boundary_states:
            b += ufl.inner(ufl.outer(U_b, n), tau) * forms.ds(tag)
        return b

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
        with _PROF.phase("assemble:gradient"):
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

    def condensed_jacobian(self, F):
        """The exact dR/dU of the residual form F, as a CondensedJacobian."""
        from dragonfly_sim.utils.gradient_condensation import CondensedJacobian
        self.set_up()
        return CondensedJacobian(F, self.model.u_vec, self.G, self.gg_rhs, self.inv_mass,
                                 jit_options=_JIT)
