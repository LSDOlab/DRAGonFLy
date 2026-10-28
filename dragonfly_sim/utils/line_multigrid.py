"""
hp-multigrid with line smoothing for the Newton linear solves.

For stiff time steps on stretched near-wall cells (incompressible water in
particular) one-level preconditioners need a number of Krylov iterations that
grows like 1/h. The remedy of cell-centred and DG RANS solvers (Mavriplis;
Fidkowski, Oliver, Lu & Darmofal, JCP 2005) is line relaxation along the
strongly coupled (wall-normal) direction inside a multigrid cycle. One PCMG
V-cycle (FGMRES outside) runs through

  DG p -> DG p-1 -> ... -> DG 0       p-multigrid: DOLFINx interpolation
                                      (injection: the nodal basis is a
                                      partition of unity)
  DG 0 -> agglomerates -> ...         h-multigrid on agglomerated cells,
                                      each aggregate carrying a LINEAR
                                      function per unknown (value and slopes
                                      along its principal axes, scaled by
                                      its extent along each): the prolongation is piecewise
                                      linear, so a coarse correction puts no
                                      large jumps (of pressure, say) between
                                      the cells of an aggregate
  coarsest                            GMRES with the line smoother

Coarse operators are Galerkin, P^T A P, recomputed by PCMG at every set-up,
so every level inherits the fine discretization (numerical flux, interior
penalty, boundary terms, time terms). Everything below the finest level is
built from the matrices at the first set-up, rank-locally:

  strength    w_ij = |A_ij|_F / sqrt(|A_ii|_F |A_jj|_F) between neighbouring
              blocks i, j (one block: all unknowns of a cell or aggregate)
  lines       every block is on exactly one line: a line grows from the most
              strongly coupled free block along its strongest free neighbour,
              as long as the coupling is mutual (the current block is among
              the neighbour's two strongest); on thin wall cells these are the
              wall-normal grid lines. The p levels use the DG 0 level's lines
              (a DG cell's dofs are contiguous, in the DG 0 cell order).
  agglomerates  consecutive blocks of a line are paired (semi-coarsening the
              stiff direction), then aggregates are paired with their most
              strongly coupled neighbour, `rounds` times in all
  smoother    a few GMRES iterations preconditioned by the exact inverse of
              the block-tridiagonal part of the level's matrix along its
              lines (block Thomas algorithm, batched over lines with numpy):
              on the p levels the element-Jacobi smoother of DG p-multigrid,
              made implicit along the lines
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


# ----------------------------------------------------------------------
# Block structure of a matrix (rank-local)
# ----------------------------------------------------------------------
def _local_entries(A, bs):
    """(cell_i, cell_j, row_in_block, col_in_block, value) of the local rows'
    entries whose column cell is local too."""
    indptr, cols, vals = A.getValuesCSR()
    r0, r1 = A.getOwnershipRange()
    c0 = r0 // bs
    n = (r1 - r0) // bs
    rows = np.repeat(np.arange(r1 - r0), np.diff(indptr))
    cj = cols // bs - c0
    keep = (cj >= 0) & (cj < n)
    return (rows[keep] // bs, cj[keep], rows[keep] % bs, cols[keep] % bs, vals[keep]), n


def cell_graph(A, bs):
    """Neighbour lists and coupling strengths of the local cells."""
    (ci, cj, _, _, v), n = _local_entries(A, bs)
    key = ci * n + cj
    uk, inv = np.unique(key, return_inverse=True)
    norm = np.sqrt(np.bincount(inv, weights=v * v))
    bi, bj = uk // n, uk % n
    diag = np.zeros(n)
    d = bi == bj
    diag[bi[d]] = norm[d]
    off = ~d
    bi, bj, w = bi[off], bj[off], norm[off] / np.sqrt(np.maximum(diag[bi[off]] * diag[bj[off]], 1e-300))
    # symmetrize: strength of the pair is the larger of the two directions
    pairs = {}
    for i, j, x in zip(bi.tolist(), bj.tolist(), w.tolist()):
        k = (i, j) if i < j else (j, i)
        if x > pairs.get(k, 0.0):
            pairs[k] = x
    nbrs = [[] for _ in range(n)]
    for (i, j), x in pairs.items():
        nbrs[i].append((x, j))
        nbrs[j].append((x, i))
    for l in nbrs:
        l.sort(reverse=True)
    return nbrs


def strong_lines(nbrs, anisotropy=0.0, max_len=None):
    """Lines along mutual strongest couplings; every cell on exactly one. A
    line only grows across a coupling at least `anisotropy` times the
    strongest coupling transverse to it (the end cell's strongest coupling
    to a cell other than the two line neighbours), and to max_len cells."""
    n = len(nbrs)
    top2 = [set(j for _, j in l[:2]) for l in nbrs]
    free = np.ones(n, dtype=bool)
    order = sorted(range(n), key=lambda i: -(nbrs[i][0][0] if nbrs[i] else 0.0))
    max_len = max_len or n
    lines = []

    def transverse(i, a, b):
        return next((x for x, j in nbrs[i] if j != a and j != b), 0.0)

    def grow(line, end, prev, size):
        while size[0] < max_len:
            w, nxt = next(((x, j) for x, j in nbrs[end] if free[j]), (0.0, None))
            if nxt is None or end not in top2[nxt] or w < anisotropy * transverse(end, prev, nxt):
                return
            line.append(nxt)
            free[nxt] = False
            prev, end = end, nxt
            size[0] += 1

    for s in order:
        if not free[s]:
            continue
        free[s] = False
        size = [1]
        fwd = [s]
        grow(fwd, s, -1, size)
        bwd = []
        grow(bwd, s, fwd[1] if len(fwd) > 1 else -1, size)
        lines.append(np.array(bwd[::-1] + fwd, dtype=np.int64))
    return lines


def pair_along_lines(lines, n):
    """Aggregate id per block: consecutive blocks of each line paired."""
    agg = -np.ones(n, dtype=np.int64)
    k = 0
    for line in lines:
        for i in range(0, len(line), 2):
            agg[line[i:i + 2]] = k
            k += 1
    return agg, k


def pair_by_strength(nbrs, agg, n_agg):
    """One round of pairing aggregates with their most strongly coupled
    neighbour (summed block strengths); an aggregate left without a partner
    joins its most strongly coupled neighbour's pair. Returns the composed
    map."""
    link = {}
    for i, l in enumerate(nbrs):
        for x, j in l:
            a, b = agg[i], agg[j]
            if a != b:
                key = (a, b) if a < b else (b, a)
                link[key] = link.get(key, 0.0) + x
    group = -np.ones(n_agg, dtype=np.int64)
    k = 0
    ranked = sorted(link.items(), key=lambda kv: -kv[1])
    for (a, b), _ in ranked:
        if group[a] < 0 and group[b] < 0:
            group[a] = group[b] = k
            k += 1
    for (a, b), _ in ranked:          # leftovers join their strongest neighbour
        if group[a] < 0 and group[b] >= 0:
            group[a] = group[b]
        elif group[b] < 0 and group[a] >= 0:
            group[b] = group[a]
    for a in np.nonzero(group < 0)[0]:
        group[a] = k
        k += 1
    return group[agg], k


def agglomerate(nbrs, lines, rounds=2):
    """Aggregates of about 2^rounds blocks: pairs along the lines, then
    rounds - 1 rounds of strength pairing. Returns (agg, n_agg)."""
    agg, n_agg = pair_along_lines(lines, len(nbrs))
    for _ in range(rounds - 1):
        agg, n_agg = pair_by_strength(nbrs, agg, n_agg)
    return agg, n_agg


def aggregate_frames(agg, n_agg, x, E=None, s=None):
    """Centroid, principal axes (rows) and extents of each aggregate, from its
    members' centroids x and, for linear members, their own frames E, s."""
    n, d = x.shape
    cnt = np.bincount(agg, minlength=n_agg).astype(float)
    xc = np.stack([np.bincount(agg, weights=x[:, k], minlength=n_agg) for k in range(d)], 1) / cnt[:, None]
    dx = x - xc[agg]
    cov = dx[:, :, None] * dx[:, None, :]
    if E is not None:      # a member's own spread, E^T diag(s^2) E / 3
        cov = cov + np.einsum("nki,nk,nkj->nij", E, s ** 2, E) / 3.0
    C = np.zeros((n_agg, d, d))
    np.add.at(C, agg, cov)
    lam, V = np.linalg.eigh(C / cnt[:, None, None])
    Ec = np.transpose(V, (0, 2, 1))                    # rows: axes
    sc = 2.0 * np.sqrt(np.maximum(lam, 0.0))
    sc = np.maximum(sc, 1e-8 * sc.max(axis=1, keepdims=True) + 1e-300)
    return xc, Ec, sc


def linear_prolongation(comm, agg, n_agg, bs, x, E=None, s=None):
    """
    P from a linear function per aggregate -- block [value, slope along each
    principal axis (scaled by the extent along it)] x bs -- to the finer
    level: point values at the centroids x (E None: DG 0 cells) or again
    linear functions with frames (E, s). Returns (P, xc, Ec, sc) with the
    coarse frames.
    """
    n, d = x.shape
    nm = 1 + d
    xc, Ec, sc = aggregate_frames(agg, n_agg, x, E, s)
    Ea, sa = Ec[agg], sc[agg]
    xi = np.einsum("nij,nj->ni", Ea, x - xc[agg]) / sa
    fine_nm = 1 if E is None else nm
    coef = np.zeros((n, fine_nm, nm))
    coef[:, 0, 0] = 1.0
    coef[:, 0, 1:] = xi
    if E is not None:      # fine slope k = sum_j c_j s_k (e_k . e_j) / s_j
        coef[:, 1:, 1:] = s[:, :, None] * np.einsum("nki,nji->nkj", E, Ea) / sa[:, None, :]
    c_off = comm.exscan(n_agg * nm * bs) or 0
    comp = np.arange(bs)
    rows = (np.arange(n)[:, None, None, None] * fine_nm + np.arange(fine_nm)[None, :, None, None]) * bs \
        + comp[None, None, None, :]
    cols = (c_off + agg * nm * bs)[:, None, None, None] + np.arange(nm)[None, None, :, None] * bs \
        + comp[None, None, None, :]
    shape = (n, fine_nm, nm, bs)
    rows, cols = np.broadcast_to(rows, shape), np.broadcast_to(cols, shape)
    vals = np.broadcast_to(coef[:, :, :, None], shape)
    # rows are (n, fine_nm, ., bs): reorder to (n, fine_nm, bs, nm) so each row's columns are contiguous
    rows, cols, vals = (np.transpose(a, (0, 1, 3, 2)).ravel() for a in (rows, cols, vals))
    nrows = n * fine_nm * bs
    indptr = np.arange(0, nrows * nm + 1, nm, dtype=PETSc.IntType)
    P = PETSc.Mat().createAIJ(((nrows, None), (n_agg * nm * bs, None)),
                              csr=(indptr, cols.astype(PETSc.IntType), np.ascontiguousarray(vals)), comm=comm)
    return P, xc, Ec, sc


# ----------------------------------------------------------------------
# Line smoother (petsc4py Python-PC)
# ----------------------------------------------------------------------
class LineSmoother:
    """
    Exact inverse of the block-tridiagonal part of the PC's matrix along a set
    of lines (lists of local cells; cells on no line are lines of length 1):
    block Thomas, forward and backward, batched over lines. The lines are
    built from the matrix at the first set-up when not given.
    """

    def __init__(self, bs, lines=None, anisotropy=0.0, max_len=None):
        self.bs = bs
        self.lines = lines
        self.anisotropy, self.max_len = anisotropy, max_len
        self._ready = False

    def _layout(self, n):
        lines = sorted((np.asarray(l, dtype=np.int64) for l in self.lines), key=len, reverse=True)
        lens = np.array([len(l) for l in lines])
        self.L = int(lens.max())
        self.n_active = [int(np.sum(lens > k)) for k in range(self.L)]
        self.cells = [np.array([lines[j][k] for j in range(self.n_active[k])], dtype=np.int64)
                      for k in range(self.L)]
        self.n = n
        # the blocks the factorization needs: diagonal, previous and next cell
        pi, pj = [], []
        for k in range(self.L):
            c = self.cells[k]
            pi.append(c), pj.append(c)
            if k > 0:
                pi.append(c), pj.append(self.cells[k - 1][:len(c)])
            if k + 1 < self.L:
                nk = self.n_active[k + 1]
                pi.append(c[:nk]), pj.append(self.cells[k + 1])
        key = np.concatenate(pi) * n + np.concatenate(pj)
        self.keys = np.unique(key)
        self._ready = True

    def _blocks(self, A):
        """Dense bs x bs blocks of A for self.keys, read block by block (the
        fine DG levels' matrices are too large to unpack whole)."""
        bs = self.bs
        r0 = A.getOwnershipRange()[0]
        k = np.arange(bs, dtype=PETSc.IntType)
        B = np.empty((len(self.keys), bs, bs))
        for t, key in enumerate(self.keys.tolist()):
            i, j = divmod(key, self.n)
            B[t] = A.getValues(r0 + i * bs + k, r0 + j * bs + k)
        return B

    def _get(self, B, i, j):
        return B[np.searchsorted(self.keys, i * self.n + j)]

    def setUp(self, pc):
        A = pc.getOperators()[1]
        n = A.getLocalSize()[0] // self.bs
        if self.lines is None:
            self.lines = strong_lines(cell_graph(A, self.bs), self.anisotropy, self.max_len)
        if not self._ready:
            self._layout(n)
        B = self._blocks(A)
        self.Dinv, self.Lo, self.C = [], [], []
        for k in range(self.L):
            c = self.cells[k]
            D = self._get(B, c, c)
            Lo = None
            if k > 0:
                Lo = self._get(B, c, self.cells[k - 1][:len(c)])
                D = D - np.matmul(Lo, self.C[k - 1][:len(c)])
            try:
                Di = np.linalg.inv(D)
            except np.linalg.LinAlgError:      # a degenerate aggregate's null slope
                Di = np.linalg.pinv(D)
            nxt = self.n_active[k + 1] if k + 1 < self.L else 0
            Up = self._get(B, c[:nxt], self.cells[k + 1]) if nxt else np.zeros((0, self.bs, self.bs))
            self.Dinv.append(Di)
            self.Lo.append(Lo)
            self.C.append(np.matmul(Di[:nxt], Up))

    def apply(self, pc, x, y):
        b = x.getArray(readonly=True).reshape(self.n, self.bs)
        z = [None] * self.L
        for k in range(self.L):
            r = b[self.cells[k]]
            if k > 0:
                r = r - np.matmul(self.Lo[k], z[k - 1][:len(r), :, None])[..., 0]
            z[k] = np.matmul(self.Dinv[k], r[..., None])[..., 0]
        out = np.empty_like(b)
        xn = None
        for k in range(self.L - 1, -1, -1):
            xk = z[k]
            if xn is not None and len(self.C[k]):
                xk = xk.copy()
                xk[:len(self.C[k])] -= np.matmul(self.C[k], xn[..., None])[..., 0]
            out[self.cells[k]] = xk
            xn = xk
        y.setArray(out.reshape(-1))


# ----------------------------------------------------------------------
# The V-cycle (petsc4py Python-PC around PCMG)
# ----------------------------------------------------------------------
class HPMultigridPC:
    """
    Python-PC context: the hp-multigrid V-cycle of the module docstring,
    built from the preconditioning matrix at the first set-up, then handed
    each new matrix (PCMG recomputes the Galerkin operators and the
    smoothers their line factorizations).

    p_interpolations  DG(q-1) -> DG(q) interpolation matrices, q = p .. 1
                      (finest first; empty for p = 0)
    n_state           unknowns per node (the blocked state's block size)
    cell_nodes        nodes per cell of each p level, finest first, ending
                      with DG 0's 1
    centroids         (local cells, dim) cell midpoints, in the DG 0 order
    rounds            pairings per agglomeration level (~2^rounds cells)
    coarse_size       stop agglomerating below this many aggregates (global)
    smooth_its        GMRES iterations per smoothing
    anisotropy, max_len  line options (strong_lines)
    cycle             "v" or "w"
    """

    def __init__(self, p_interpolations, n_state, cell_nodes, centroids, rounds=2, max_levels=8,
                 coarse_size=500, smooth_its=2, coarse_its=30, anisotropy=0.0, max_len=None,
                 first_rounds=None, cycle="v"):
        self.p_interp = list(p_interpolations)
        self.n_state, self.cell_nodes = n_state, list(cell_nodes)
        self.x = np.asarray(centroids, dtype=float)
        self.rounds, self.first_rounds = rounds, first_rounds or rounds + 1
        self.max_levels, self.coarse_size = max_levels, coarse_size
        self.smooth_its, self.coarse_its = smooth_its, coarse_its
        self.anisotropy, self.max_len = anisotropy, max_len
        self.cycle = cycle
        self.mg = None

    def _build(self, pc, A):
        comm = A.getComm().tompi4py()
        ns = self.n_state
        # p levels down to DG 0 (Galerkin, only to build the hierarchy below)
        mats, prolongs, blocks = [A], list(self.p_interp), [n * ns for n in self.cell_nodes]
        for P in self.p_interp:
            mats.append(mats[-1].ptap(P))
        nbrs = cell_graph(mats[-1], ns)
        dg0_lines = strong_lines(nbrs, self.anisotropy, self.max_len)
        lines = [dg0_lines] * len(mats)
        sizes = [comm.allreduce(len(nbrs), op=MPI.SUM)] * len(mats)
        # agglomeration levels: linear per aggregate
        x, E, sx, rounds = self.x, None, None, self.first_rounds
        while len(mats) < self.max_levels and sizes[-1] > self.coarse_size:
            agg, n_agg = agglomerate(nbrs, lines[-1], rounds)
            P, x, E, sx = linear_prolongation(comm, agg, n_agg, ns, x, E, sx)
            prolongs.append(P)
            mats.append(mats[-1].ptap(P))
            blocks.append((1 + x.shape[1]) * ns)
            nbrs = cell_graph(mats[-1], blocks[-1])
            lines.append(strong_lines(nbrs, self.anisotropy, self.max_len))
            sizes.append(comm.allreduce(n_agg, op=MPI.SUM))
            rounds = self.rounds
        for M in mats[1:]:
            M.destroy()
        nlev = len(mats)
        mg = PETSc.PC().create(comm=pc.getComm())
        mg.setOptionsPrefix((pc.getOptionsPrefix() or "") + "hpmg_")
        mg.setType(PETSc.PC.Type.MG)
        mg.setMGLevels(nlev)
        mg.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
        mg.setMGCycleType(PETSc.PC.MGCycleType.W if self.cycle == "w" else PETSc.PC.MGCycleType.V)
        PETSc.Options()[mg.getOptionsPrefix() + "pc_mg_galerkin"] = "both"
        # PETSc's level 0 is the coarsest; our index 0 the finest
        for l in range(1, nlev):
            mg.setMGInterpolation(l, prolongs[nlev - 1 - l])
        for l in range(1, nlev):
            k = mg.getMGSmoother(l)
            k.setType(PETSc.KSP.Type.GMRES)
            k.setTolerances(max_it=self.smooth_its)
            k.setNormType(PETSc.KSP.NormType.NONE)
            k.getPC().setType(PETSc.PC.Type.PYTHON)
            k.getPC().setPythonContext(LineSmoother(blocks[nlev - 1 - l], lines[nlev - 1 - l]))
        k = mg.getMGCoarseSolve()
        k.setType(PETSc.KSP.Type.GMRES)
        k.setTolerances(rtol=1e-2, max_it=self.coarse_its)
        k.getPC().setType(PETSc.PC.Type.PYTHON)
        k.getPC().setPythonContext(LineSmoother(blocks[-1], lines[-1]))
        mg.setFromOptions()
        PETSc.Sys.Print("hp-multigrid: {} levels, blocks per level {} (block sizes {})".format(
            nlev, sizes, blocks))
        return mg

    def setUp(self, pc):
        _, P = pc.getOperators()
        if self.mg is None:
            self.mg = self._build(pc, P)
        self.mg.setOperators(P, P)
        self.mg.setUp()

    def apply(self, pc, x, y):
        self.mg.apply(x, y)
