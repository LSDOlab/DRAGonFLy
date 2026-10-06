"""
Non-reflecting far-field refinements on top of the characteristic far field
(CompressibleEulerModel.define_farfield_bc).

TransverseFarfield ('riemann2') integrates the incoming Riemann invariant's
transverse terms along the boundary, which turns the plain characteristic
condition (exact only at normal incidence) into a second-order absorbing one
(Engquist-Majda / Higdon) for oblique waves. Unsteady runs only.
"""
import numpy as np
from mpi4py import MPI
import ufl
import basix
import dolfinx
import dolfinx.fem.petsc

from dragonfly_sim.utils.Euler_utils import pressure


class TransverseFarfield:
    """
    Riemann far-field with transverse terms in the incoming characteristic.

    At the far field the incoming invariant R- = u.n - 2c/(g-1) obeys
    (straight boundary, tangent tau = (-n_y, n_x))

        dR-/dt + (u.n - c) dR-/dn + u_t dR-/dtau - c du_t/dtau = 0.

    Dropping only the normal (incoming-wave) term and keeping beta times the
    transverse ones, plus a weak relaxation to the freestream, gives

        dR-/dt = -beta (u_t dR-/dtau - c du_t/dtau) - K (R- - R-_inf).

    beta = 0 with R-(0) = R-_inf is exactly riemann_farfield_state. For
    linear acoustics on a boundary with tangential mean flow the plane-wave
    reflection coefficient is

        r(theta) = [a(s-1) + beta r^2] / [a(s+1) - beta r^2],
        s = cos(theta), r = sin(theta), a = 1 + (1-beta) M r,

    so beta = 0 gives (s-1)/(s+1), beta = 1 gives (1-s)/(1+s) (no better),
    and beta = 1/2 gives -((1-s)/(1+s))^2 for any tangential Mach number:
    the second-order Engquist-Majda/Higdon absorbing condition.
    K = sigma c_inf (1 - M^2) / L (Poinsot & Lele) only stops long-term drift.

    The equation is integrated with the solver's own BDF coefficients for the
    deviation delta = R- - R-_inf (a0 + a1 + a2 = 0 cancels R-_inf), with
    the transverse terms extrapolated from the two previous levels
    (IMEX/SBDF2; on a BDF1 step T* = T(U^n)):

        delta^{n+1} = (-a1 delta^n - a2 delta^{n-1} - dt beta T*) / (a0 + K dt),
        T* = 2 T(U^n) - T(U^{n-1}).

    Explicit is safe here: the far-field cells are coarse (h ~ 1.35), so the
    transverse transport has CFL ~ U dt/h ~ c dt/h ~ 0.04. Keeping T implicit
    put gradients of the nonlinear primitives inside the Newton Jacobian,
    whose form took longer to compile than dolfinx's JIT wait on the other
    ranks; this way the residual only sees delta^{n+1} as a stored field and
    its Jacobian is the plain Riemann one.

    T is taken from the interior trace. The levels live in DG fields of the
    solution degree whose traces on the far-field facets are the boundary
    data: delta^{n+1} is evaluated at 3 Gauss points per facet and
    L2-projected (cell-wise least squares; a corner cell's two facets share
    its corner dof, so there the trace is a compromise). Storing delta rather
    than R- keeps the freestream exact: delta stays identically zero for
    beta = 0.

    Usage: construct it, pass it to model.define_farfield_bc(tag,
    transverse=...), and register it with the model's TimeIntegrator
    (attach(integrator)), which then calls prepare() before every physical
    step's Newton solve and commit() after the step is accepted. Its two
    history levels are in `functions`, for checkpoints.

    K defaults to sigma c_inf (1 - M^2) / L with sigma = 0.25 and L the
    far-field distance `relax_length`.

    Tangential derivatives at p = 0: the state is constant in each cell, so
    ufl.grad of it vanishes and T would be identically zero (riemann2 would
    reduce to riemann). The derivatives are therefore taken from a
    Green-Gauss gradient of the state at the two history levels
    (core/reconstruction.py): the model's own reconstruction when it has one
    (RANS), otherwise a standalone one that is not part of the residual.
    """

    def __init__(self, model, farfield_tag, beta=0.5, K=None, relax_length=50.0, relax_sigma=0.25):
        if model.dimensions != 2:
            raise NotImplementedError("the transverse far field (riemann2) is 2D only")
        mesh_obj = model.mesh_obj
        self.mesh = mesh_obj.mesh
        self.gamma = gamma = model.gamma
        self.n = mesh_obj.n
        self.farfield_tag = farfield_tag
        inlet_bc = model.boundary_conditions['inlet']
        V_s = model.functionspaces["V_scalar"]
        self.delta_n = dolfinx.fem.Function(V_s, name="ff_delta_n")
        self.delta_nm1 = dolfinx.fem.Function(V_s, name="ff_delta_nm1")
        self.delta_next = dolfinx.fem.Function(V_s, name="ff_delta_next")
        self.functions = [self.delta_n, self.delta_nm1]
        if K is None:
            K = relax_sigma * (gamma * inlet_bc['p'] / inlet_bc['rho']) ** 0.5 \
                * (1.0 - inlet_bc['M'] ** 2) / relax_length
        self.beta, self.K = float(beta), float(K)
        # this boundary equation's own copies of the step size and BDF
        # coefficients, set in prepare()
        self.dt = dolfinx.fem.Constant(self.mesh, 1.0)
        self.a = tuple(dolfinx.fem.Constant(self.mesh, c) for c in (1.0, -1.0, 0.0))
        self._coef = (self.dt, self.a, self.beta, self.K)
        # extrapolation weight dt_n / dt_{n-1}
        self.ext_w = dolfinx.fem.Constant(self.mesh, 1.0)
        # p = 0: Green-Gauss gradients of U^n and U^{n-1} (set up in attach)
        self.model = model
        self._rec = None
        self.G_n = self.G_nm1 = None
        if model.poly_order == 0:
            from dragonfly_sim.core.reconstruction import GreenGaussReconstruction
            self._rec = getattr(model, "reconstruction", None) or \
                GreenGaussReconstruction(model, muscl=False)
            V_G = self._rec.G.function_space
            self.G_n = dolfinx.fem.Function(V_G, name="ff_grad_n")
            self.G_nm1 = dolfinx.fem.Function(V_G, name="ff_grad_nm1")
            self.functions += [self.G_n, self.G_nm1]

        c_inf = (gamma * inlet_bc['p'] / inlet_bc['rho']) ** 0.5
        self.R_in = ufl.dot(inlet_bc['u'], self.n) - 2.0 * c_inf / (gamma - 1.0) \
            + self.delta_next
        self._exprs = None
        self._setup_projection(mesh_obj, V_s)

    def transverse_term(self, U, G=None):
        """T = u_t dR-/dtau - c du_t/dtau of state U on the far field (UFL),
        with grad U from the field G (n_state x 2) when given, else ufl.grad."""
        gamma, n = self.gamma, self.n
        n_state = U.ufl_shape[0]
        Um = ufl.as_vector([U[i] for i in range(4)]) if n_state > 4 else U  # mean flow
        rho = Um[0]
        vel = ufl.as_vector([Um[1] / rho, Um[2] / rho])
        c = ufl.sqrt(gamma * pressure(Um, gamma) / rho)
        tau = ufl.as_vector([-n[1], n[0]])
        if G is None:
            grad_vel, grad_c = ufl.grad(vel), ufl.grad(c)
        else:
            from dragonfly_sim.utils.NS_utils import primitive_gradients
            grad_vel, grad_T, _, _ = primitive_gradients(U, G, gamma, 4)   # grad u, grad(p/rho)
            grad_c = gamma / (2.0 * c) * grad_T
        dvel_dtau = ufl.dot(grad_vel, tau)   # grad(vel)[i, j] = d u_i / d x_j
        # R-_int = u.n - 2c/(g-1); n is constant along the straight far field
        dR_dtau = ufl.dot(dvel_dtau, n) - 2.0 / (gamma - 1.0) * ufl.dot(grad_c, tau)
        return ufl.dot(vel, tau) * dR_dtau - c * ufl.dot(dvel_dtau, tau)

    def bind_history(self, U_n, U_nm1):
        # Separate BDF1 and BDF2 expressions rather than one with an
        # extrapolation weight: 0 * T(U^{n-1}) would still be evaluated and
        # turn NaN if U^{n-1} were not a valid state.
        dt, (a0, a1, a2), beta, K = self._coef
        T_n = self.transverse_term(U_n, self.G_n)
        # linear extrapolation to t^{n+1}; w = dt_n / dt_{n-1} (1 at constant dt)
        w = self.ext_w
        T_star = {1: T_n, 2: (1.0 + w) * T_n - w * self.transverse_term(U_nm1, self.G_nm1)}
        self.delta_new = {k: (-a1 * self.delta_n - a2 * self.delta_nm1 - dt * beta * T)
                          / (a0 + K * dt) for k, T in T_star.items()}
        self._exprs = {k: dolfinx.fem.Expression(e, self._s[:, None])
                       for k, e in self.delta_new.items()}

    def _setup_projection(self, mesh_obj, V_s):
        mesh = self.mesh
        tdim = mesh.topology.dim
        n_owned = mesh.topology.index_map(tdim).size_local
        mesh.topology.create_connectivity(tdim - 1, tdim)
        mesh.topology.create_connectivity(tdim, tdim - 1)
        f2c = mesh.topology.connectivity(tdim - 1, tdim)
        c2f = mesh.topology.connectivity(tdim, tdim - 1)

        # 3-point Gauss-Legendre on the reference facet [0, 1]
        g = np.sqrt(3.0 / 5.0)
        s = np.array([0.5 - 0.5 * g, 0.5, 0.5 + 0.5 * g])
        w = np.array([5.0, 8.0, 5.0]) / 18.0
        self._s = s

        element = V_s.element.basix_element
        topo = basix.topology(element.cell_type)[tdim - 1]
        ref = basix.geometry(element.cell_type)
        tab = [element.tabulate(0, ref[f[0]] + s[:, None] * (ref[f[1]] - ref[f[0]]))[0, :, :, 0]
               for f in topo]

        per_cell = {}
        for f in mesh_obj.meshtags.find(self.farfield_tag):
            cell = f2c.links(f)[0]
            if cell >= n_owned:
                continue
            per_cell.setdefault(cell, []).append(int(np.where(c2f.links(cell) == f)[0][0]))

        x_nodes = mesh.geometry.x[:, :tdim]
        gdofs = mesh.geometry.dofmaps[0]
        cmap = mesh.geometry.cmaps[0]
        # Expression.eval on facets applies dolfinx's facet reflection, which
        # makes both neighbours see a facet's points in the same order. On a
        # reflected facet the point at s therefore lies at reference
        # va + (1-s)(vb-va). Detect the orientation per facet from the
        # physical coordinates instead of relying on the permutation bits.
        ent_all = np.array([(c, lf) for c, lfs in per_cell.items() for lf in lfs],
                           dtype=np.int32).reshape(-1, 2)
        # Building an Expression is a collective JIT call: every rank must
        # build it, including ranks without far-field facets.
        x_expr = dolfinx.fem.Expression(ufl.SpatialCoordinate(mesh), s[:, None])
        x_eval = x_expr.eval(mesh, ent_all) if len(ent_all) else np.zeros((0, len(s), tdim))
        entities, ops, cell_dofs = [], [], []
        k_ent = 0
        for cell, lfs in per_cell.items():
            rows, weights = [], []
            nodes = mesh.geometry.x[gdofs[cell]]
            for lf in lfs:
                va, vb = (x_nodes[gdofs[cell][topo[lf][k]]] for k in (0, 1))
                ra, rb = ref[topo[lf][0]], ref[topo[lf][1]]
                fwd = cmap.push_forward(ra + s[:, None] * (rb - ra), nodes)[:, :tdim]
                got = x_eval[k_ent][:, :tdim]
                k_ent += 1
                scale = np.linalg.norm(vb - va)
                if np.allclose(got, fwd, rtol=0, atol=1e-10 * scale):
                    rows.append(tab[lf])
                elif np.allclose(got, fwd[::-1], rtol=0, atol=1e-10 * scale):
                    rows.append(tab[lf][::-1])
                else:
                    raise RuntimeError("cannot match facet quadrature points of cell "
                                       "{} facet {}".format(cell, lf))
                weights.append(w * scale)
                entities.append((cell, lf))
            B = np.vstack(rows)
            sw = np.sqrt(np.concatenate(weights))
            # weighted least squares, minimum norm for dofs no facet sees
            ops.append(np.linalg.pinv(sw[:, None] * B) * sw[None, :])
            cell_dofs.append(V_s.dofmap.cell_dofs(cell))
        self._entities = np.array(entities, dtype=np.int32).reshape(-1, 2)
        self._ops, self._cell_dofs = ops, cell_dofs
        self._facets_per_cell = [len(l) for l in per_cell.values()]
        self.n_facets = mesh.comm.allreduce(len(self._entities), MPI.SUM)

    def attach(self, integrator):
        """Register with a TimeIntegrator: bind the solution history and
        receive prepare()/commit() calls from it."""
        self.bind_history(integrator.term.U_n, integrator.term.U_nm1)
        integrator.boundary_updaters.append(self)
        if self._rec is not None:
            # both history gradients from the current state (U^n = U^{n-1} at
            # the start; a restart overwrites them from the checkpoint)
            self._gradient_into(self.G_n)
            self.G_nm1.x.array[:] = self.G_n.x.array

    def _gradient_into(self, target):
        self._rec.update_gradient()
        target.x.array[:] = self._rec.G.x.array
        target.x.scatter_forward()

    def prepare(self, order, ratio=None, dt=None):
        """Before a step's Newton solve: delta^{n+1} into delta_next, for a
        BDF `order` step of size dt with step ratio dt_n/dt_{n-1}. Collective.
        Returns max |delta^{n+1}| over the far-field quadrature points."""
        from dragonfly_sim.core.time_integration import bdf_coefficients
        for c, val in zip(self.a, bdf_coefficients(order, ratio)):
            c.value = val
        if dt is not None:
            self.dt.value = dt
        self.ext_w.value = 1.0 if ratio is None else ratio
        expr = self._exprs[2 if order == 2 else 1]
        vals = expr.eval(self.mesh, self._entities) if len(self._entities) \
            else np.zeros((0, 3))
        new = np.zeros_like(self.delta_next.x.array)
        i = 0
        for op, dofs, nf in zip(self._ops, self._cell_dofs, self._facets_per_cell):
            new[dofs] = op @ vals[i:i + nf].reshape(-1)
            i += nf
        self.delta_next.x.array[:] = new
        self.delta_next.x.scatter_forward()
        return self.mesh.comm.allreduce(float(np.abs(vals).max(initial=0.0)), MPI.MAX)

    def commit(self):
        """After the step is accepted (u_vec = U^{n+1}): shift the boundary
        history."""
        self.delta_nm1.x.array[:] = self.delta_n.x.array
        self.delta_n.x.array[:] = self.delta_next.x.array
        if self._rec is not None:
            self.G_nm1.x.array[:] = self.G_n.x.array
            self._gradient_into(self.G_n)
