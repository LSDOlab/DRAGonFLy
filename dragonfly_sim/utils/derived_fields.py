"""
Derived fields and the chain rule through them.

A derived field is a Function the residual (or a force functional) reads
that is not one of the unknowns, but is itself a function of the state U,
the angle of attack alpha and/or the mesh coordinates x:

    the Green-Gauss gradient    G = M^-1 b(U, alpha, x)   (core/reconstruction.py)
    the cell centres            c = C x                   (core/reconstruction.py)
    the wall distance           d = d(x)                  (core/wall_distance.py)

For R(U, f(U, alpha, x), alpha, x) every total derivative picks up one
chain term per field,

    dR/dU     = dR/dU|_f     + sum_f (dR/df) (df/dU)
    dR/dalpha = dR/dalpha|_f + sum_f (dR/df) (df/dalpha)
    dR/dx     = dR/dx|_f     + sum_f (dR/df) (df/dx)

and the same holds for a scalar functional J. A DerivedField supplies the
three right-hand factors; DerivedFieldChainRule assembles the partial
derivatives with UFL and combines them. With no derived fields nothing here
is used, and the model's derivative assemblers are the plain UFL ones.
"""
import numpy as np
import ufl
import dolfinx
import dolfinx.fem.petsc
from petsc4py import PETSc


class DerivedField:
    """
    Interface. `function` is the dolfinx Function the forms read.

    update()                 recompute the field at the current state and mesh
    update_geometry()        refresh whatever depends on the mesh only (called
                             after mesh motion, before update())
    state_jacobian()         PETSc Mat d(field)/dU (only if depends_on_state)
    alpha_vector()           PETSc Vec d(field)/dalpha, or None
    mesh_jacobian()          PETSc Mat d(field)/dx, columns in the mesh's
                             coordinate_space, or None
    """
    function = None
    # True if the field is a function of the state (then dR/dU is condensed)
    depends_on_state = False

    def update(self):
        pass

    def update_geometry(self):
        pass

    def state_jacobian(self):
        return None

    def alpha_vector(self):
        return None

    def mesh_jacobian(self):
        return None


def _nonempty(form):
    """The form with derivatives expanded, or None if it is identically zero."""
    form = ufl.algorithms.expand_derivatives(form)
    return None if form.empty() else form


def _assemble_vec(form):
    vec = dolfinx.fem.petsc.assemble_vector(form)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vec


def _assemble_mat(form):
    A = dolfinx.fem.petsc.assemble_matrix(form)
    A.assemble()
    return A


def mesh_coordinate_argument(form, coordinate_space):
    """The Argument a mesh-coordinate derivative of `form` is taken along."""
    args = form.arguments()
    number = max(a.number() for a in args) + 1 if args else 0
    return ufl.Argument(coordinate_space, number)


class DerivedFieldChainRule:
    """
    Total derivatives of a residual form F (rank 1) or a functional J
    (rank 0) through a list of DerivedFields. Compiled forms are cached per
    (form id, field) on this object, so reuse one instance per model.
    """

    def __init__(self, mesh_obj, u, fields, alpha_var=None, jit_options=None):
        self.mesh_obj = mesh_obj
        self.u = u
        self.fields = list(fields)
        self.alpha_var = alpha_var
        self.jit_options = jit_options or {}
        self._forms = {}

    # ----------------------------------------------------------------
    def _compiled(self, key, factory):
        if key not in self._forms:
            ufl_form = factory()
            self._forms[key] = None if ufl_form is None else dolfinx.fem.form(
                ufl_form, jit_options=self.jit_options)
        return self._forms[key]

    def _partial_field(self, form, key, field):
        """dform/dfield: a matrix form (rank-1 form) or a vector form (rank 0)."""
        V_f = field.function.function_space
        rank = len(form.arguments())
        arg = ufl.TrialFunction(V_f) if rank == 1 else ufl.TestFunction(V_f)
        return self._compiled((key, "field", id(field)),
                              lambda: _nonempty(ufl.derivative(form, field.function, arg)))

    def update(self):
        for f in self.fields:
            f.update()

    # ----------------------------------------------------------------
    # Residual (rank 1)
    # ----------------------------------------------------------------
    def residual_alpha(self, F, key, out):
        """out <- dR/dalpha (total); `out` is a Vec in the residual's layout."""
        dF = self._compiled((key, "alpha"), lambda: _nonempty(ufl.diff(F, self.alpha_var)))
        with out.localForm() as loc:
            loc.set(0.0)
        if dF is not None:
            dolfinx.fem.petsc.assemble_vector(out, dF)
            out.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for field in self.fields:
            a = field.alpha_vector()
            if a is None:
                continue
            dFdf = self._partial_field(F, key, field)
            if dFdf is None:
                a.destroy()
                continue
            A = _assemble_mat(dFdf)
            A.multAdd(a, out, out)
            A.destroy()
            a.destroy()
        return out

    def residual_mesh_operator(self, F, key):
        """dR/dx (total) as a PETSc python Mat with mult / multTranspose."""
        Vx = self.mesh_obj.coordinate_space
        spatial = ufl.SpatialCoordinate(self.mesh_obj.mesh)
        # coordinate derivatives are left to the form compiler (expanding
        # them early is not supported on interior-facet integrals)
        a_x = self._compiled((key, "mesh"), lambda: ufl.derivative(
            F, spatial, mesh_coordinate_argument(F, Vx)))
        A_x = _assemble_mat(a_x)
        chains = []
        for field in self.fields:
            X = field.mesh_jacobian()
            if X is None:
                continue
            dFdf = self._partial_field(F, key, field)
            if dFdf is None:
                X.destroy()
                continue
            chains.append((_assemble_mat(dFdf), X))
        return _shell_matrix(A_x, chains)

    # ----------------------------------------------------------------
    # Functionals (rank 0)
    # ----------------------------------------------------------------
    def functional_state(self, J, key):
        """dJ/dU (total) as a Vec in the state layout."""
        V = self.u.function_space
        dJ = self._compiled((key, "state"), lambda: _nonempty(
            ufl.derivative(J, self.u, ufl.TestFunction(V))))
        g = _assemble_vec(dJ)
        for field in self.fields:
            B = field.state_jacobian()
            if B is None:
                continue
            dJdf = self._partial_field(J, key, field)
            if dJdf is not None:
                v = _assemble_vec(dJdf)
                B.multTransposeAdd(v, g, g)
                v.destroy()
            # a fresh matrix per call: free it now, not at the next
            # PETSc.garbage_cleanup (several GB for a 3D Green-Gauss gradient)
            B.destroy()
        return g

    def functional_mesh(self, J, key):
        """dJ/dx (total) as a Vec on the mesh's coordinate_space."""
        Vx = self.mesh_obj.coordinate_space
        spatial = ufl.SpatialCoordinate(self.mesh_obj.mesh)
        dJ = self._compiled((key, "mesh"), lambda: ufl.derivative(
            J, spatial, ufl.TestFunction(Vx)))
        g = _assemble_vec(dJ)
        for field in self.fields:
            X = field.mesh_jacobian()
            if X is None:
                continue
            dJdf = self._partial_field(J, key, field)
            if dJdf is not None:
                v = _assemble_vec(dJdf)
                X.multTransposeAdd(v, g, g)
                v.destroy()
            X.destroy()
        return g


class _ChainedMeshOperator:
    """Python-Mat context: y = A_x v + sum_k A_k (X_k v), and its transpose."""

    def __init__(self, A_x, chains):
        self.A_x = A_x
        self.chains = chains
        self._work = [(A.createVecRight(), A.createVecLeft()) for A, _ in chains]

    def mult(self, mat, v, y):
        self.A_x.mult(v, y)
        for (A, X), (w_f, _) in zip(self.chains, self._work):
            X.mult(v, w_f)
            A.multAdd(w_f, y, y)

    def multTranspose(self, mat, w, y):
        self.A_x.multTranspose(w, y)
        for (A, X), (w_f, _) in zip(self.chains, self._work):
            A.multTranspose(w, w_f)
            X.multTransposeAdd(w_f, y, y)


def _shell_matrix(A_x, chains):
    ctx = _ChainedMeshOperator(A_x, chains)
    M = PETSc.Mat().createPython(A_x.getSizes(), comm=A_x.getComm())
    M.setPythonContext(ctx)
    M.setUp()
    return M


def destroy_chained_operator(M):
    """Free a residual_mesh_operator result and the matrices it holds now.
    Left to the garbage collector, a parallel petsc4py object is only queued,
    and freed at the next PETSc.garbage_cleanup; for a 3D RANS operator that
    is several GB held into the next adjoint."""
    ctx = M.getPythonContext()
    M.destroy()
    ctx.A_x.destroy()
    for (A, X), (w_f, w_r) in zip(ctx.chains, ctx._work):
        for obj in (A, X, w_f, w_r):
            obj.destroy()
    ctx.chains, ctx._work = [], []


def diagonal_mass_inverse(V, dx):
    """1/|K| per dof of a DG0 (blocked) space V: the inverse of its diagonal
    mass matrix, as a Vec in V's layout."""
    tau = ufl.TestFunction(V)
    shape = V.ufl_element().reference_value_shape
    ones = ufl.as_tensor(np.ones(shape).tolist()) if shape else 1.0
    m = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(ufl.inner(ones, tau) * dx))
    m.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    inv = m.duplicate()
    inv.setArray(1.0 / m.getArray(readonly=True))
    m.destroy()
    return inv
