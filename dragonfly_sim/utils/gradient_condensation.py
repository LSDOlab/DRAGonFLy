"""
Newton problems whose residual reads a gradient field that is eliminated
from the unknowns.

At p = 0 the RANS model reconstructs a DG0 gradient G of the state by
Green-Gauss (core/reconstruction.py),

    M_G G = b(U),    b(U; tau) = sum over facets of int u_f (x) n : tau ds,

with M_G the diagonal (cell-volume) DG0 mass matrix. G is linear in the
facet states, so it is eliminated exactly instead of being carried as an
unknown:

    R(U) = R_U(U, G(U)),    dR/dU = A_UU + A_UG M_G^-1 B_GU,

with A_UU = dR_U/dU at fixed G, A_UG = dR_U/dG and B_GU = db/dU. The product
widens the Jacobian's stencil to the neighbours' neighbours, which is the
stencil of the reconstructed scheme. Everything downstream (SNES, the
positivity limiter, the time integrator) still sees a single-field problem
in U.

jacobian_mode selects what Newton's linear solve uses (configure_snes wires
it into an existing SNES):

  exact       operator = preconditioning matrix = A_UU + A_UG M^-1 B_GU
  compact_pc  operator exact, preconditioning matrix A_UU: the compact,
              nearest-neighbour part (gradients frozen), as finite-volume
              codes precondition with their first-order Jacobian. The
              default: on the coarse SA NACA0012 C-grid it takes the same
              PTC path as MUMPS, 5x faster, and scales to the 181k-cell STW
              wing where every exact factorization runs out of memory.
  compact     operator = preconditioning matrix = A_UU: approximate Newton
              (defect correction); converges linearly
  pressure_pc operator exact, preconditioning matrix A_UU plus the
              reconstruction's terms (A_UG M^-1 B_GU) in the continuity rows
              and on the pressure column only: for incompressible (water)
              time steps, whose pressure-velocity coupling runs largely
              through the second-order terms the compact part drops, while
              the rest stays compact

The pressure is component 0 of every block of the (blocked) state.
"""
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.petsc import assemble_matrix, create_matrix

from dragonfly_sim.utils.Nonlinear_utils import NonlinearProblem_mod


class CondensedJacobian:
    """
    A = A_UU + A_UG M^-1 B_GU for a residual form F(U, G), with `b_G` the UFL
    Green-Gauss right-hand side (test function on G's space) and `inv_mass`
    M_G^-1 as a Vec in G's layout. The matrices are plain aij: MatMatMult
    does not take baij here. The product's sparsity is fixed by the
    operands', so one symbolic product sizes A for good.
    """

    def __init__(self, F, u, G, b_G, inv_mass, jit_options={}, A_UU=None, a_UU=None):
        self._inv_mass = inv_mass
        self._a = a_UU if a_UU is not None else _create_form(ufl.derivative(F, u), jit_options=jit_options)
        self._a_UG = _create_form(ufl.derivative(F, G, ufl.TrialFunction(G.function_space)),
                                  jit_options=jit_options)
        self._a_GU = _create_form(ufl.derivative(b_G, u, ufl.TrialFunction(u.function_space)),
                                  jit_options=jit_options)
        self.A_UU = A_UU if A_UU is not None else create_matrix(self._a)
        self.A_UG = create_matrix(self._a_UG)
        self.B_GU = create_matrix(self._a_GU)
        self._assemble_blocks()
        self._C = self.A_UG.matMult(self.B_GU)
        A = self.A_UU.duplicate(copy=True)
        A.axpy(1.0, self._C, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        A.zeroEntries()
        self.A = A

    def _assemble_blocks(self):
        for M, a in ((self.A_UU, self._a), (self.A_UG, self._a_UG), (self.B_GU, self._a_GU)):
            M.zeroEntries()
            assemble_matrix(M, a)
            M.assemble()
        self.B_GU.diagonalScale(L=self._inv_mass)

    def assemble(self):
        """Assemble A_UU, A_UG, M^-1 B_GU and A at the current state; G must
        be up to date."""
        self._assemble_blocks()
        self.A_UG.matMult(self.B_GU, result=self._C)
        self.A.zeroEntries()
        self.A.axpy(1.0, self.A_UU, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        self.A.axpy(1.0, self._C, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        self.A.assemble()
        return self.A


class TransposedCondensedOperator:
    """
    Python-Mat context of the transposed condensed Jacobian without forming it:

        A^T x = (A_UU + A_UG B)^T x = P x + B^T (A_UG^T x),    P = A_UU^T

    with B = M_G^-1 dG/dU (any Mat with mult / multTranspose, e.g. the
    structured GreenGaussStateOperator). Neither A nor the product A_UG B
    is ever assembled. It owns P (also the preconditioning matrix), A_UG and
    B (released through `release_B`, a callable) and frees them in destroy().
    """

    def __init__(self, P, A_UG, B, release_B):
        self.P, self.A_UG, self.B, self._release_B = P, A_UG, B, release_B
        self._w = A_UG.createVecRight()

    def mult(self, mat, x, y):
        self.A_UG.multTranspose(x, self._w)
        self.B.multTranspose(self._w, y)
        self.P.multAdd(x, y, y)

    def destroy(self, mat=None):
        # called by PETSc when the python Mat is destroyed
        if self.P is None:
            return
        self._release_B(self.B)
        for obj in (self._w, self.A_UG, self.P):
            obj.destroy()
        self.P = self.A_UG = self.B = None

    def create_mat(self, comm):
        M = PETSc.Mat().createPython(self.P.getSizes(), comm=comm)
        M.setPythonContext(self)
        M.setUp()
        return M


class CondensedGradientProblem(NonlinearProblem_mod):
    """
    NonlinearProblem_mod for R_U(U, G(U)).

    `G` is the gradient Function the residual reads, `b_G` the UFL form of
    the Green-Gauss right-hand side, `update_G()` recomputes G from the
    current u in place and `inv_mass` is M_G^-1 as a Vec in G's layout.
    """

    MODES = ("exact", "compact_pc", "compact", "pressure_pc")

    def __init__(self, F, J, u, G, b_G, update_G, inv_mass, jit_options={}):
        super().__init__(F, J, u, jit_options=jit_options, mat_type=None)
        self.G = G
        self.update_G = update_G
        self.update_G()
        self._condensed = CondensedJacobian(F, u, G, b_G, inv_mass, jit_options=jit_options,
                                            A_UU=self._A, a_UU=self._a)
        self._A_UU = self._condensed.A_UU
        self._A = self._condensed.A
        self.jacobian_mode = "exact"
        self.pc_log_stage = None      # a PETSc log stage timing the pressure_pc matrix, if set

    def configure_snes(self, snes, mode):
        """Re-wire `snes`'s Jacobian for jacobian_mode `mode` (see the module
        docstring). Call after the solver is built."""
        if mode not in self.MODES:
            raise ValueError("jacobian_mode must be one of {}".format(self.MODES))
        self.jacobian_mode = mode
        if mode == "exact":
            return

        def sync(x):
            x.copy(self.u.x.petsc_vec)
            self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                           mode=PETSc.ScatterMode.FORWARD)

        if mode == "compact_pc":
            def jac(snes_, x, A, P):
                sync(x)
                self.J(x, self._A)            # assembles A_UU on the way
            snes.setJacobian(jac, self._A, self._A_UU)
            return

        if mode == "pressure_pc":
            # P = A_UU + C on the pressure rows and column, set in one pass by
            # COO (A_UU's and C's values, C's off the pressure zeroed) into
            # the exact Jacobian's pattern, so its ILU(0) sees the same pattern
            bs = self._A.getBlockSize()
            self._P = None

            def build_P():
                C = self._condensed._C
                ijs, mask = [], None
                for M in (self._A_UU, C):
                    ia, ja, _ = M.getValuesCSR()
                    rows = M.getOwnershipRange()[0] + np.repeat(np.arange(len(ia) - 1), np.diff(ia))
                    ijs.append((rows, ja))
                    if M is C:
                        mask = ((rows % bs == 0) | (ja % bs == 0)).astype(float)
                P_ = PETSc.Mat().create(comm=self._A.getComm())
                P_.setSizes(self._A.getSizes(), bsize=bs)
                P_.setType(self._A.getType())
                P_.setPreallocationCOO(np.concatenate([i for i, _ in ijs]).astype(PETSc.IntType),
                                       np.concatenate([j for _, j in ijs]).astype(PETSc.IntType))
                self._P_mask = mask
                self._P_nnz = (len(ijs[0][1]), len(ijs[1][1]))
                return P_

            def jac(snes_, x, A, P):
                sync(x)
                self.J(x, self._A)            # assembles A_UU and C = A_UG M^-1 B_GU on the way
                if self.pc_log_stage is not None:
                    self.pc_log_stage.push()
                v_UU = self._A_UU.getValuesCSR()[2]
                v_C = self._condensed._C.getValuesCSR()[2]
                if (len(v_UU), len(v_C)) != self._P_nnz:
                    raise RuntimeError("pressure_pc: the Jacobian's sparsity changed")
                self._P.setValuesCOO(np.concatenate([v_UU, v_C * self._P_mask]),
                                     addv=PETSc.InsertMode.INSERT_VALUES)
                if self.pc_log_stage is not None:
                    self.pc_log_stage.pop()

            self._P = build_P()
            snes.setJacobian(jac, self._A, self._P)
            return

        def compact(snes_, x, A, P):
            sync(x)
            self.update_G()
            self._A_UU.zeroEntries()
            assemble_matrix(self._A_UU, self._a)
            self._A_UU.assemble()
        snes.setJacobian(compact, self._A_UU, self._A_UU)

    def F(self, x, b):
        self.update_G()
        super().F(x, b)

    def J(self, x, A):
        self.update_G()
        self._condensed.assemble()
        if A.handle != self._A.handle:
            A.assemble()
            self._A.copy(A)
