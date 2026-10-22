"""
Barter-Darmofal artificial viscosity for the p = 0 (cell-centred) models.

This is the p = 0 form of the PDE-based artificial viscosity of the
project-sandbox CompressibleEuler/2D code (core/coupled_bd_stabilization.py,
after Barter & Darmofal, JCP 2010): a shock sensor drives the source of a
screened-Poisson equation whose solution sigma is the viscosity field, and
sigma is solved monolithically with the flow, as an auxiliary unknown of the
model (CompressibleEulerModel(shock_capturing=...)), so Newton keeps its
exact Jacobian and the adjoint and shape derivatives stay exact. A lagged
viscosity field oscillated and stalled Newton at ~1e-6 in the sandbox.

At p = 0 the sandbox's interior-penalty viscosity reduces to its facet
penalty term, and with its anisotropic metric the penalty coefficient is
~ lambda in every direction: the artificial viscosity is a smooth,
sensor-driven blend towards first-order dissipation,

    R_U += int_dS C lambda avg(sigma) [[U(V)]] . [[v]],

on the cell values (not the MUSCL traces), lambda the facet's largest wave
speed (preconditioned, if the model is). sigma in [0, 1] solves, per cell K,

    sigma_K |K| + sum_f L^2 d_f |f| (sigma_K - sigma_N) = int_K D g,
    g = tanh(s / eta)  averaged over the cell's facets,
    s = |p'_+ - p'_-| (1/c^2) / rho  (= |dp| / (gamma p) for an ideal gas),

the sandbox's pressure-jump sensor (normalised by the facet's rho c^2
instead of p, which is fluid independent and vanishes identically for an
incompressible fluid), taken between the MUSCL-reconstructed facet traces
when the model has them, else between the cell values. At p = 0 the jump
of the cell values is O(h) in smooth flow, so on a coarse grid a suction
peak trips the sensor as readily as a shock; the jump the reconstruction
leaves is O(h^2) there and O(dp) at a shock -- the p = 0 counterpart of the
sandbox's p > 0 sensors, which measure what the polynomial cannot resolve. d_f = avg(|K|) / |f| is the cell extent across the
facet, so the smoothing spreads sigma about L cells in every direction,
including the wall-normal direction of a stretched boundary-layer cell (the
sandbox's isotropic L h_vol would spread it across ~sqrt(aspect ratio)
cells there). D is the Ducros dilatation switch (div u)^2 / ((div u)^2 +
|omega|^2 + eps) from the reconstructed gradient, used whenever the model
has one: it is ~1 at a shock and ~0 throughout a boundary layer, so the
viscosity cannot leak into the layer at a shock foot. `directional` further
weights the facet dissipation with (n . grad p)^2 / |grad p|^2, so only the
facets crossing a shock get it.

Everything is a smooth function of the state (tanh, smooth absolute
values), so the residual is differentiable everywhere.
"""
import ufl
import dolfinx

from dragonfly_sim.utils.NS_utils import gradient_rows, velocity_gradient


class BarterDarmofalAV:
    """
    The artificial viscosity of a CompressibleEulerModel; created by the model
    (shock_capturing=True or a dict of these options), which then carries
    sigma as its auxiliary unknown.

    C               dissipation coefficient: C sigma = 1/2 adds one Rusanov
                    dissipation on top of the scheme's own
    eta             sensor scale of the normalised pressure jump (tanh(s/eta))
    jump_eps        smoothing of |s| at zero jump
    smooth_length   L, the smoothing length in cells
    ducros          use the Ducros switch (None: whenever the model has a
                    reconstructed gradient)
    directional     weight the dissipation with the facet's share of grad p
    """

    def __init__(self, model, C=0.5, eta=0.1, jump_eps=1e-3, smooth_length=1.0, ducros=None,
                 directional=False):
        self.model = model
        self.C, self.eta, self.jump_eps = float(C), float(eta), float(jump_eps)
        self.smooth_length = float(smooth_length)
        self.ducros, self.directional = ducros, bool(directional)
        model.residual_terms.append(self.residual)

    @property
    def index(self):
        """sigma's component in the state."""
        m = self.model
        return m.n_mean + m.n_extra

    def sigma(self, V=None):
        V = self.model.u_vec if V is None else V
        return V[self.index]

    def _gradient(self):
        rec = getattr(self.model, "reconstruction", None)
        return None if rec is None else rec.G

    def facet_sensor(self, V_p, V_m):
        """tanh(s/eta) of the normalised pressure jump across a facet."""
        m = self.model
        th_p, th_m = m.thermo(V_p), m.thermo(V_m)
        x = (V_p[0] - V_m[0]) * 0.5 * (th_p.inv_c2 / th_p.rho + th_m.inv_c2 / th_m.rho)
        s = x * x / ufl.sqrt(x * x + self.jump_eps ** 2)
        return ufl.tanh(s / self.eta)

    def ducros_switch(self, G):
        n_mean = self.model.n_mean
        grad_u = velocity_gradient(G, n_mean)
        div2 = ufl.tr(grad_u) ** 2
        W = grad_u - grad_u.T
        return div2 / (div2 + 0.5 * ufl.inner(W, W) + 1e-12)

    def residual(self):
        m = self.model
        U, v, psi = m.u_vec, m.v_vec, m.v_test[self.index]
        dS, dx = m.forms.dS, m.forms.dx
        msh = m.mesh_obj.mesh
        # pointwise from the Jacobians (ufl.CellVolume / FacetArea are not
        # lowered for quads and hexes); exact on straight-sided cells
        vol = m.mesh_obj.cell_volume
        area = (abs(ufl.classes.FacetJacobianDeterminant(msh))
                * ufl.classes.ReferenceFacetVolume(msh))('+')
        n_facets = dolfinx.cpp.mesh.cell_num_entities(msh.topology.cell_type, msh.topology.dim - 1)
        d_f = 0.5 * (vol('+') + vol('-')) / area
        sigma = self.sigma()

        # the sigma equation: screened Poisson with the facet-averaged sensor as source
        traces = m.face_traces() if m.face_traces is not None else (U('+'), U('-'))
        g = self.facet_sensor(*traces)
        G = self._gradient()
        use_ducros = (G is not None) if self.ducros is None else self.ducros
        if use_ducros:
            if G is None:
                raise RuntimeError("the Ducros switch needs a reconstructed gradient (MUSCL or RANS)")
            D = self.ducros_switch(G)
            D_p, D_m = D('+'), D('-')
        else:
            D_p = D_m = 1.0
        L2 = self.smooth_length ** 2
        R = (sigma * psi * dx
             + L2 * d_f * ufl.jump(sigma) * ufl.jump(psi) * dS
             - g / (n_facets * area) * (vol('+') * D_p * psi('+') + vol('-') * D_m * psi('-')) * dS)

        # the viscosity: blend towards first-order dissipation on the cell values
        h = m.mesh_obj.h_min
        lam = 0.5 * (m._lambda_max_expr(U('+'), h('+')) + m._lambda_max_expr(U('-'), h('-')))
        coeff = self.C * lam * ufl.avg(sigma)
        if self.directional:
            if G is None:
                raise RuntimeError("directional artificial viscosity needs a reconstructed gradient")
            grad_p = 0.5 * (gradient_rows(G, 0)('+') + gradient_rows(G, 0)('-'))
            n = m.mesh_obj.n('+')
            coeff = coeff * ufl.dot(grad_p, n) ** 2 / (ufl.dot(grad_p, grad_p) + 1e-12)
        R += coeff * ufl.inner(m.conserved(U('+')) - m.conserved(U('-')), v('+') - v('-')) * dS
        return R
