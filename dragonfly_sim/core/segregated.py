"""
Segregated Newton solve of a time step, for constant-density (water) RANS.

The monolithic Newton step couples all unknowns (p', u, T, nu_tilde). For a
constant-density fluid most of that coupling is absent or weak:

  * T enters neither the mass nor the momentum equations (rho and mu are
    constants), nor the turbulence model, and the energy equation is LINEAR
    in T at fixed (p', u, nu_tilde): h = c T + p'/rho, H and the low-Mach
    dissipation are linear in T, k is constant, MUSCL without a limiter is
    linear. So T is solved once per time step, after the flow has converged,
    by a single linear solve, which is exact.
  * nu_tilde couples to (p', u) through the eddy viscosity and the velocity
    gradients in its sources. By default (couple_nu) it stays in the Newton
    block with (p', u): splitting it off made its Newton steps stall at tight
    tolerances (Rizzi grid 3, 1e-8 ||V||: 7-26 Gauss-Seidel sweeps per step
    against 3-4 coupled Newton steps).

Each time step iterates Newton on the (p', u, nu_tilde) block, T frozen,
preconditioned as below with nu_tilde in the velocity field of the Schur
split, until the (p', u) and nu_tilde residuals both satisfy the criterion,
then solves T. With couple_nu=False it instead iterates block Gauss-Seidel
over

    (p', u)    one Newton step on its block of R*, nu_tilde and T frozen:
               FGMRES on the block of the exact Jacobian, preconditioned with
               the block of the model's preconditioning matrix (its
               jacobian_mode, e.g. pressure_pc) by point-block ILU(0) (blocks
               of the (p', u) unknowns of a cell, in ASM subdomains), or by
               the pressure Schur split with point-block ILU(0) on the
               velocity block and BoomerAMG on the pressure
    nu_tilde   one Newton step on its block, (p', u) at their new values;
               the update is scaled down as a whole so that no cell changes
               by more than nu_cap (|nu_tilde| + nu_tilde_inf): a bad step
               cannot wreck the eddy viscosity, and the step keeps the Newton
               direction (a per-cell clip does not, and stalled the line
               search where nu_tilde ~ 0)

with each block update backtracked (halved) until its block residual
decreases

until every block satisfies |R_b| <= max(atol, rtol |R_b,0|), or the combined
update has fallen below update_rtol times the first sweep's (the monolithic
incremental criterion), then solves T. The residual and the Jacobians are the model's own (the SNES
callbacks of model.solver, with the physical time term), so the converged
state is the monolithic one.

The work is timed in PETSc log stages (see -log_view): "seg: residual",
"seg: exact Jacobian", "seg: pressure_pc" (the preconditioning matrix built
from the exact Jacobian's parts), "seg: submatrices", "seg: PC setup" and
"seg: Krylov solve" of the flow block, and "seg: T solve" (its PC set-up
included); for a schur_pre other than "selfp" also "seg: Schur matrix" and
(when it uses the first-order Jacobian) "seg: first-order Jacobian".
"""
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc

from dragonfly_sim.core.time_integration import StepRecord


class SegregatedSolver:
    """
    pu_pc        the preconditioner of the (p', u[, nu_tilde]) Newton block:
                 "bilu0" (ASM with point-block ILU(0)), "schur" (pressure Schur
                 split, point-block ILU(0) on the velocity block, BoomerAMG on
                 the pressure) or "schur_ilu" (two-stage, CPR-like: the schur
                 preconditioner, then ASM/ILU(0) of the whole coupled block on
                 the residual it leaves, multiplicatively)
    couple_nu    solve nu_tilde in the Newton block with (p', u) (default), or
                 in a Gauss-Seidel sweep of its own, its update capped by nu_cap
    rtol, atol   the block residual criterion (|R_b| <= max(atol, rtol |R_b,0|))
    scaled_atol  if given, the criterion is instead |R_b| <= scaled_atol ||V||
                 for every block (V the whole state at the start of the step),
                 with no relative or incremental test
    update_rtol  ... or the incremental one: the combined (p', u, nu_tilde)
                 update of a sweep below update_rtol times the first sweep's
                 -- the monolithic solve's newton_criterion="incremental" (the
                 Rizzi grids' huge far-field cells put the residual's
                 round-off floor above useful absolute tolerances; a test per
                 block stalls on nu_tilde, whose own first update is that of
                 the step's start-up)
    max_outer    Gauss-Seidel sweeps per time step
    nu_cap       cap of a nu_tilde update in any cell, relative to |nu_tilde| +
                 its free-stream value, enforced by scaling the whole update
    krylov_rtol  relative tolerance of the (p', u) and nu_tilde linear solves;
                 the T solve uses t_rtol
    krylov_restart  FGMRES restart of the flow block (default max_krylov_it:
                 no restart)
    ksp_options  extra PETSc options of the flow block's KSP ({name: value}
                 without its prefix, e.g. {"pc_fieldsplit_schur_fact_type":
                 "full"}), set after (and so overriding) the defaults
    ew_rtol0     if given, the flow block's Krylov tolerance follows
                 Eisenstat-Walker (choice 2: gamma (|R_k| / |R_k-1|)^alpha,
                 gamma 0.9, alpha (1 + sqrt 5) / 2, safeguarded) from ew_rtol0
                 at the first sweep instead of krylov_rtol, floored at half
                 target / |R| (no over-solving of the last sweep), at most 0.9
    schur_pre    pu_pc "schur": the matrix the pressure AMG works on, S =
                 A11 - A10 diag(A00)^-1 A01 of
                   "selfp"             the preconditioning matrix (PETSc's
                                       selfp; under pressure_pc its pressure
                                       rows and column carry the MUSCL stencil)
                   "selfp_user"        the same, built here (a check of the
                                       user-matrix path)
                   "first_order"       the first-order Jacobian (no
                                       reconstruction: the gradient-frozen form
                                       at G = 0), nearest-neighbour coupling
                   "first_order_diag"  first_order with the diagonal of selfp's
                   "sparsified"        selfp's restricted to first_order's
                                       pattern, the dropped (second-neighbour)
                                       entries of each row added to its diagonal
                   "abs_lump"          selfp's with A00^-1 ~ diag(sum_j |A00_ij|)^-1
                   "neumann"           selfp's with A00^-1 ~ D^-1 + D^-1 (D - A00) D^-1
    """

    def __init__(self, model, pu_pc="bilu0", rtol=1e-3, atol=5e-6, max_outer=30, nu_cap=1.0,
                 krylov_rtol=1e-3, t_rtol=1e-10, max_krylov_it=300, max_backtracks=6, update_rtol=1e-4,
                 scaled_atol=None, couple_nu=True, schur_pre="selfp", krylov_restart=None, ew_rtol0=None,
                 ksp_options=None, verbose=True):
        if not model.material.incompressible:
            raise ValueError("the segregated solve assumes a constant-density fluid "
                             "(T decoupled from the flow and linear)")
        if model.n_aux:
            raise ValueError("the segregated solve has no block for auxiliary unknowns")
        self.model = model
        self.pu_pc, self.rtol, self.atol = pu_pc, float(rtol), float(atol)
        self.max_outer, self.nu_cap = int(max_outer), float(nu_cap)
        self.krylov_rtol, self.t_rtol, self.max_krylov_it = float(krylov_rtol), float(t_rtol), int(max_krylov_it)
        self.verbose, self.max_backtracks = verbose, int(max_backtracks)
        self.update_rtol = float(update_rtol)
        self.krylov_restart = self.max_krylov_it if krylov_restart is None else int(krylov_restart)
        self.ew_rtol0 = None if ew_rtol0 is None else float(ew_rtol0)
        self.ksp_options = dict(ksp_options or {})
        self.scaled_atol = None if scaled_atol is None else float(scaled_atol)
        ns, nm = model.n_state, model.n_mean
        self.couple_nu = bool(couple_nu)
        if schur_pre not in ("selfp", "selfp_user", "first_order", "first_order_diag", "sparsified",
                             "abs_lump", "neumann"):
            raise ValueError("unknown schur_pre {!r}".format(schur_pre))
        if schur_pre != "selfp" and pu_pc != "schur":
            raise ValueError("schur_pre {!r} needs pu_pc='schur'".format(schur_pre))
        self.schur_pre = schur_pre
        self.blocks = {"pu": list(range(nm - 1)), "nu": list(range(nm, nm + model.n_extra)), "T": [nm - 1]}
        # the Newton blocks of a sweep: (p', u, nu_tilde) together, or (p', u) then nu_tilde
        self.blocks["flow"] = self.blocks["pu"] + self.blocks["nu"]
        self.sweep_blocks = ["flow"] if self.couple_nu else ["pu", "nu"]
        A = model.solver.snes.getJacobian()[0]
        r0, r1 = A.getOwnershipRange()
        comp = np.arange(r0, r1) % ns
        comm = A.getComm()
        self.local = {b: np.nonzero(np.isin(comp, c))[0] for b, c in self.blocks.items()}
        self.iset = {b: PETSc.IS().createGeneral((r0 + i).astype(PETSc.IntType), comm=comm)
                     for b, i in self.local.items()}
        self._sub = {}
        self.ksp = {b: self._make_ksp(b) for b in self.sweep_blocks + ["T"]}
        self.nu_inf = max(abs(self._domain_mean(v)) for v in model._freestream_scalars())
        # the residual callback updates ghosts: a duplicate of the solver's own (ghosted) vector
        self.F = model.solver.b.duplicate()
        self.stages = {k: PETSc.Log.Stage("seg: " + k) for k in (
            "residual", "exact Jacobian", "pressure_pc", "first-order Jacobian", "Schur matrix", "submatrices",
            "PC setup", "Krylov solve", "T solve")}
        self.J1 = None
        if schur_pre in ("first_order", "first_order_diag", "sparsified"):
            self.J1 = model.problem._A_UU.duplicate()
        if schur_pre != "selfp":
            # p' and the rest within a Newton block's own numbering (p' its first component)
            self.split = {}
            for b in self.sweep_blocks:
                nb = len(self.blocks[b])
                n = len(self.local[b])
                b0 = model.mesh_obj.mesh.comm.exscan(n) or 0
                k = np.arange(n)
                self.split[b] = [PETSc.IS().createGeneral((b0 + k[sel]).astype(PETSc.IntType), comm=comm)
                                 for sel in (k % nb != 0, k % nb == 0)]
        # the pressure_pc Jacobian callback times its own part (utils/gradient_condensation.py)
        model.problem.pc_log_stage = self.stages["pressure_pc"]

    # -- set-up ------------------------------------------------------------
    def _make_ksp(self, block):
        comm = self.model.solver.snes.getComm()
        ksp = PETSc.KSP().create(comm=comm)
        prefix = "seg_{}_".format(block)
        ksp.setOptionsPrefix(prefix)
        o = PETSc.Options(prefix)
        o["ksp_diagonal_scale"] = ""
        o["ksp_diagonal_scale_fix"] = ""
        if block in ("pu", "flow"):
            o["ksp_type"] = "fgmres"
            o["ksp_gmres_restart"] = self.krylov_restart
            nb = len(self.blocks[block])   # p' first, then the velocity (and nu_tilde)
            if self.pu_pc in ("schur", "schur_ilu"):
                f = ""
                if self.pu_pc == "schur_ilu":
                    o["pc_type"] = "composite"
                    o["pc_composite_type"] = "multiplicative"
                    o["pc_composite_pcs"] = "fieldsplit,asm"
                    o["sub_1_pc_asm_overlap"] = 1
                    o["sub_1_sub_pc_type"] = "ilu"
                    o["sub_1_sub_pc_factor_levels"] = 0
                    f = "sub_0_"
                o[f + "pc_type"] = "fieldsplit"
                o[f + "pc_fieldsplit_block_size"] = nb
                o[f + "pc_fieldsplit_0_fields"] = ",".join(str(k) for k in range(1, nb))
                o[f + "pc_fieldsplit_1_fields"] = "0"
                o[f + "pc_fieldsplit_type"] = "schur"
                o[f + "pc_fieldsplit_schur_fact_type"] = "upper"
                o[f + "pc_fieldsplit_schur_precondition"] = "selfp"
                o[f + "fieldsplit_0_ksp_type"] = "preonly"
                o[f + "fieldsplit_0_pc_type"] = "asm"
                o[f + "fieldsplit_0_pc_asm_overlap"] = 1
                o[f + "fieldsplit_0_sub_pc_type"] = "ilu"
                o[f + "fieldsplit_0_sub_pc_factor_levels"] = 0
                o[f + "fieldsplit_1_ksp_type"] = "preonly"
                o[f + "fieldsplit_1_pc_type"] = "hypre"
                o[f + "fieldsplit_1_pc_hypre_type"] = "boomeramg"
            elif self.pu_pc == "bilu0":
                o["pc_type"] = "asm"
                o["pc_asm_overlap"] = 1
                o["sub_pc_type"] = "ilu"
                o["sub_pc_factor_levels"] = 0
            else:
                raise ValueError("pu_pc must be 'bilu0', 'schur' or 'schur_ilu'")
            for k, v in self.ksp_options.items():
                o[k] = v
            ksp.setTolerances(rtol=self.krylov_rtol, max_it=self.max_krylov_it)
        else:
            o["ksp_type"] = "gmres"
            o["ksp_gmres_restart"] = 100
            # the tolerance on the true residual (left preconditioning tests the
            # preconditioned one, which the scaling and ILU can make far smaller)
            o["ksp_pc_side"] = "right"
            o["pc_type"] = "asm"
            o["pc_asm_overlap"] = 1
            o["sub_pc_type"] = "ilu"
            o["sub_pc_factor_levels"] = 1
            ksp.setTolerances(rtol=self.t_rtol if block == "T" else self.krylov_rtol,
                              atol=1e-50, max_it=self.max_krylov_it)
        ksp.setFromOptions()
        return ksp

    def _domain_mean(self, expr):
        """The mean of a (constant) UFL expression over the domain."""
        dx = self.model.forms.dx
        comm = self.model.mesh_obj.mesh.comm
        val, vol = (comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(e * dx)), op=MPI.SUM)
                    for e in (expr, 1.0 + 0.0 * expr))
        return float(val / vol)

    @contextmanager
    def _stage(self, name):
        stage = self.stages[name]
        stage.push()
        try:
            yield
        finally:
            stage.pop()

    def _residual(self):
        with self._stage("residual"):
            self.model.solver.snes.computeFunction(self.model.u_vec.x.petsc_vec, self.F)

    def _print(self, *a):
        if self.verbose:
            PETSc.Sys.Print(*a)

    # -- one block Newton step ---------------------------------------------
    def _assemble(self):
        """R*, its Jacobian and preconditioning matrix at the current state."""
        snes = self.model.solver.snes
        self._residual()
        A, P = snes.getJacobian()[:2]
        with self._stage("exact Jacobian"):
            snes.computeJacobian(self.model.u_vec.x.petsc_vec, A, P)
        if self.J1 is not None:
            with self._stage("first-order Jacobian"):
                self._assemble_first_order()
        return A, P

    def _assemble_first_order(self):
        """J1: the first-order Jacobian of R* (the gradient-frozen form at G = 0,
        so face states are cell values), at the current state."""
        problem = self.model.problem
        G = problem.G.x.array
        saved = G.copy()
        G[:] = 0.0
        self.J1.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.J1, problem._a)
        self.J1.assemble()
        G[:] = saved

    def _selfp(self, M, block, ainv="diag"):
        """A11 - A10 W of M (a Newton block's matrix; 1 = p', 0 the rest), W ~
        A00^-1 A01 with A00^-1 approximated by
          "diag"      D^-1, D = diag(A00) (SIMPLE; PETSc's selfp)
          "abs_lump"  diag(sum_j |A00_ij|)^-1 (SIMPLEC-type lumping)
          "neumann"   D^-1 + D^-1 (D - A00) D^-1 (one Jacobi correction; one
                      ring wider)"""
        is0, is1 = self.split[block]
        A00 = M.createSubMatrix(is0, is0) if ainv != "diag" else None
        if ainv == "abs_lump":
            ia, ja, va = A00.getValuesCSR()
            w = A00.createVecLeft()
            w.setArray(1.0 / np.add.reduceat(np.abs(va), ia[:-1]))
        else:
            d = M.getDiagonal()
            sub = d.getSubVector(is0)
            w = sub.copy()
            d.restoreSubVector(is0, sub)
            d.destroy()
            w.reciprocal()
        A01 = M.createSubMatrix(is0, is1)
        A10 = M.createSubMatrix(is1, is0)
        S = M.createSubMatrix(is1, is1)
        A01.diagonalScale(L=w)                  # D^-1 A01
        if ainv == "neumann":
            T = A00.matMult(A01)                # A00 D^-1 A01
            T.diagonalScale(L=w)
            T.scale(-1.0)
            T.axpy(2.0, A01, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            A01.destroy()
            A01 = T                             # (2 D^-1 - D^-1 A00 D^-1) A01
        prod = A10.matMult(A01)
        S.axpy(-1.0, prod, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        for obj in (w, A01, A10, prod, A00):
            if obj is not None:
                obj.destroy()
        return S

    @staticmethod
    def _sparsify(S, pattern):
        """S restricted to `pattern`'s nonzeros (which include the diagonal),
        each row's dropped entries added to its diagonal."""
        ip, jp, _ = pattern.getValuesCSR()
        i, j, v = S.getValuesCSR()
        n, ncol = len(i) - 1, S.getSize()[1]
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(i))
        keep = np.isin(rows * ncol + j, np.repeat(np.arange(n, dtype=np.int64), np.diff(ip)) * ncol + jp)
        dropped = np.bincount(rows[~keep], v[~keep], minlength=n)
        v = np.where(keep, v, 0.0)
        r0 = S.getOwnershipRange()[0]
        v[(j == rows + r0) & keep] += dropped[rows[(j == rows + r0) & keep]]
        ik = np.concatenate([[0], np.cumsum(np.bincount(rows[keep], minlength=n))]).astype(PETSc.IntType)
        out = PETSc.Mat().createAIJ(S.getSizes(), csr=(ik, j[keep].astype(PETSc.IntType), v[keep]),
                                    comm=S.getComm())
        out.assemble()
        return out

    def _schur_matrix(self, block, P_b):
        """The pressure AMG's matrix for schur_pre, scaled as ksp_diagonal_scale
        scales the operators (diag(P_b)^-1/2 on either side)."""
        mode = self.schur_pre
        S = None
        if mode in ("abs_lump", "neumann"):
            S = self._selfp(P_b, block, ainv=mode)
        elif mode != "selfp_user":
            J = self.J1.createSubMatrix(self.iset[block], self.iset[block])
            S = self._selfp(J, block)
            J.destroy()
        if mode not in ("first_order", "abs_lump", "neumann"):
            S_full = self._selfp(P_b, block)
            if mode == "selfp_user":
                S = S_full
            elif mode == "first_order_diag":
                d = S_full.getDiagonal()
                S.setDiagonal(d)
                d.destroy()
                S_full.destroy()
            else:
                S_sp = self._sparsify(S_full, S)
                S.destroy(), S_full.destroy()
                S = S_sp
        is1 = self.split[block][1]
        dP = P_b.getDiagonal()
        sub = dP.getSubVector(is1)
        D = sub.copy()
        dP.restoreSubVector(is1, sub)
        D.sqrtabs()
        D.reciprocal()
        S.diagonalScale(L=D, R=D)
        dP.destroy(), D.destroy()
        return S

    def _block_norms(self):
        F = self.F.getArray(readonly=True)
        comm = self.model.mesh_obj.mesh.comm
        return {b: np.sqrt(comm.allreduce(float(np.sum(F[i] ** 2)))) for b, i in self.local.items()}

    def _sweep(self, A, P, r):
        """One Newton step on each of sweep_blocks in turn (reassembling after
        each). Returns (A, P, per-block (its, converged, theta, |update|))."""
        info = {}
        for b in self.sweep_blocks:
            d, its, ok = self._block_step(b, A, P)
            theta, _, upd = self._update(b, d, r[b])
            A, P = self._assemble()
            r = self._block_norms()
            info[b] = (its, ok, theta, upd)
        return A, P, info

    def _submatrices(self, block, A, P):
        """(A_bb, P_bb), freshly extracted (the assembled Jacobians' stored
        nonzero counts change between assemblies, so storage is not reused);
        the (p', u) preconditioning block in point-block (BAIJ) form."""
        for M in self._sub.pop(block, ()):
            M.destroy()
        iset = self.iset[block]
        A_b = A.createSubMatrix(iset, iset)
        P_b = A_b if P is A else P.createSubMatrix(iset, iset)
        own = [A_b] if P_b is A_b else [A_b, P_b]
        if block in ("pu", "flow"):
            nb = len(self.blocks[block])
            P_b.setBlockSizes(nb, nb)
            if self.pu_pc == "bilu0":
                P_b = P_b.convert("baij")
                own.append(P_b)
        self._sub[block] = own
        return A_b, P_b

    def _block_step(self, block, A, P):
        """Solve J_bb d = R_b; returns (d as a local array, Krylov iterations, converged)."""
        with self._stage("submatrices"):
            A_b, P_b = self._submatrices(block, A, P)
        ksp = self.ksp[block]
        ksp.setOperators(A_b, P_b)
        if self.schur_pre != "selfp" and block != "T":
            with self._stage("Schur matrix"):
                S = self._schur_matrix(block, P_b)
            ksp.getPC().setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)
            self._sub[block].append(S)
        rhs, d = A_b.createVecs()[::-1]
        rhs.setArray(self.F.getArray(readonly=True)[self.local[block]])
        with self._stage("T solve" if block == "T" else "PC setup"):
            ksp.setUp()
            ksp.setUpOnBlocks()
        with self._stage("T solve" if block == "T" else "Krylov solve"):
            ksp.solve(rhs, d)
        out = d.getArray().copy()
        its, ok = ksp.getIterationNumber(), ksp.getConvergedReason() > 0
        rhs.destroy(), d.destroy()
        return out, its, ok

    def _set(self, block, values):
        u = self.model.u_vec
        u.x.petsc_vec.getArray()[self.local[block]] = values
        u.x.scatter_forward()

    def _update(self, block, d, r_before):
        """x_b -= theta d (the Newton update; a separate nu_tilde block's capped), theta
        halved from 1 until the block's residual falls below r_before (at
        most max_backtracks times; the last trial is kept). Returns (theta,
        the block's new residual norm, the norm of the applied update)."""
        x0 = self.model.u_vec.x.petsc_vec.getArray()[self.local[block]].copy()
        theta = 1.0
        if block == "nu":
            # one scaling of the whole update (a per-cell clip would change
            # its direction, and the line search then stalls where nu_tilde
            # is ~0): no cell changes by more than nu_cap (|nu_tilde| + nu_inf)
            cap = self.nu_cap * (np.abs(x0) + self.nu_inf)
            ratio = np.max(np.abs(d) / cap, initial=0.0)
            ratio = self.model.mesh_obj.mesh.comm.allreduce(float(ratio), op=MPI.MAX)
            if ratio > 1.0:
                d = d / ratio
        for k in range(self.max_backtracks + 1):
            self._set(block, x0 - theta * d)
            self._residual()
            r = self._block_norms()[block]
            if r < r_before or k == self.max_backtracks:
                comm = self.model.mesh_obj.mesh.comm
                return theta, r, np.sqrt(comm.allreduce(float(np.sum((theta * d) ** 2))))
            theta *= 0.5

    def _forcing(self, r, target, eta, r_prev):
        """The Eisenstat-Walker Krylov tolerance of the next sweep (set on its
        KSPs); returns (it, the residual norm it was computed from)."""
        gamma, alpha = 0.9, 0.5 * (1.0 + np.sqrt(5.0))
        rn = float(np.sqrt(r["pu"] ** 2 + r["nu"] ** 2))
        if eta is None:
            new = self.ew_rtol0
        else:
            new = gamma * (rn / r_prev) ** alpha
            if gamma * eta ** alpha > 0.1:
                new = max(new, gamma * eta ** alpha)
        new = min(max(new, 0.5 * min(target.values()) / rn, 1e-8), 0.9)
        for b in self.sweep_blocks:
            self.ksp[b].setTolerances(rtol=new)
        return new, rn

    # -- the time step ---------------------------------------------------------
    def solve(self):
        """Converge the time step's R* by block Gauss-Seidel, then solve T.
        Returns (converged, StepRecord)."""
        A, P = self._assemble()
        r_full0 = self.F.norm()
        r0 = self._block_norms()
        if self.scaled_atol is not None:
            v_norm = self.model.u_vec.x.petsc_vec.norm()
            target = {b: self.scaled_atol * v_norm for b in ("pu", "nu")}
        else:
            target = {b: max(self.atol, self.rtol * r0[b]) for b in ("pu", "nu")}
        ksp_total, sweeps, converged = 0, 0, False
        r = r0
        upd, upd0 = None, None
        eta, r_prev = None, None

        def done():
            if all(r[b] <= target[b] for b in ("pu", "nu")):
                return True
            # the monolithic incremental criterion, on the combined (p', u, nu_tilde) update
            return self.scaled_atol is None and upd is not None and upd <= self.update_rtol * upd0

        while True:
            converged = done()
            if converged or sweeps >= self.max_outer or not np.isfinite(r["pu"]):
                break
            sweeps += 1
            if self.ew_rtol0 is not None:
                eta, r_prev = self._forcing(r, target, eta, r_prev)
            A, P, info = self._sweep(A, P, r)
            upd = float(np.sqrt(sum(v[3] ** 2 for v in info.values())))
            upd0 = max(upd, 1e-300) if upd0 is None else upd0
            ksp_total += sum(v[0] for v in info.values())
            r = self._block_norms()
            self._print("[seg] sweep {:2d}  |R_pu| {:.3e}  |R_nu| {:.3e}  |R_T| {:.3e}  |dx| {:.2e}  {}{}".format(
                sweeps, r["pu"], r["nu"], r["T"], upd, "" if eta is None else "rtol {:.1e}  ".format(eta), "  ".join(
                    "KSP {} {:3d}{} (theta {:.3g})".format(b, v[0], "" if v[1] else "*", v[2])
                    for b, v in info.items())))
        # T: linear and decoupled, one solve at the converged flow
        d, its_T, ok_T = self._block_step("T", A, P)
        self._set("T", self.model.u_vec.x.petsc_vec.getArray()[self.local["T"]] - d)
        ksp_total += its_T
        self._residual()
        r = self._block_norms()
        self._print("[seg] T solve: {} its{}, |R_T| {:.3e}".format(its_T, "" if ok_T else " (not converged)", r["T"]))
        rec = StepRecord(accepted=True, newton_its=sweeps, ksp_its=ksp_total,
                         res_initial=r_full0, res_final=self.F.norm())
        return bool(converged and np.isfinite(rec.res_final)), rec
