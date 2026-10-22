"""
Compressible RANS (and laminar Navier-Stokes) at p = 0, built on the Euler
model by composition.

CompressibleRANSModel owns a CompressibleEulerModel (`self.flow`) whose state
carries the turbulence model's transported scalars after the primitive mean
flow (nu_tilde for Spalart-Allmaras), and plugs the viscous physics into it.
Everything else -- spaces, boundary conditions, the Newton solver and
positivity limiter, pseudo-transient continuation, time stepping, adjoint
and shape-derivative assembly -- is the Euler model's own code. Attributes
and methods the RANS model does not define itself are forwarded to
`self.flow` (reads and writes), so DG_windtunnel_model and DG_postprocessor
use it exactly like the Euler model.

Discretization (cell-centred finite volume, as DG0)
---------------------------------------------------
Inviscid: Euler_utils.hll_flux (the same fan upwinding the scalars), with
MUSCL-reconstructed interior-facet traces of the primitive variables.
Without MUSCL the first-order HLL flux puts acoustic-speed dissipation on
the tangential momentum at wall-parallel facets, a numerical viscosity that
exceeds the turbulent one through the boundary layer (SA NACA0012 at M 0.7:
friction drag ~10x too high).

Viscous: through each facet, F_v(V_f, grad V_f) . n with the reconstructed
facet gradient of core/reconstruction.py; no penalty term. Viscosity and
conductivity are the material's (core/materials.py). Turbulence
sources use the Green-Gauss gradient G. G is condensed out of Newton
exactly, and is a derived field for adjoints and shape derivatives, as are
the cell centres and the wall distance.

Nondimensionalization: the material's (rho_inf = |u_inf| = T_inf = 1, the
mesh unit), so mu_inf = 1/Re (Re held fixed under set_mach).

Boundary conditions (kind: extra-variable values)
-------------------------------------------------
farfield  the characteristic state; scalars from the free stream where it
          flows in, from the interior where it flows out
wall      adiabatic no-slip (define_wall_bc): u = 0, interior pressure and
          temperature, the turbulence model's wall values; zero heat flux
slip      (symmetry) the Euler slip state, interior scalars, no viscous flux

Solver defaults
---------------
Pseudo-transient continuation from the free stream (use_ptc, CFL0 5), one
Newton step per pseudo step; FGMRES on the exact condensed Jacobian,
preconditioned with ASM(1)/ILU(2) of the compact gradient-frozen part
(jacobian_mode "compact_pc", utils/gradient_condensation.py). Set
direct_linear_solver for MUMPS on the exact Jacobian instead.
"""
import ufl

from dragonfly_sim.core.Euler_model import CompressibleEulerModel
from dragonfly_sim.core.reconstruction import GreenGaussReconstruction
from dragonfly_sim.core.turbulence_models import SpalartAllmarasNeg
from dragonfly_sim.core.wall_distance import WallDistance
from dragonfly_sim.core.materials import IdealGas
from dragonfly_sim.utils.Euler_utils import hll_flux
from dragonfly_sim.utils.NS_utils import (extra_components, concat_state, viscous_flux,
                                          viscous_stress, velocity_gradient)


class CompressibleRANSModel:

    # Attributes that live on the RANS object itself; every other attribute
    # read or written goes to self.flow.
    _OWN = frozenset((
        "flow", "turbulence_model", "Re", "Pr_t", "muscl", "limiter", "limiter_eps",
        "jacobian_mode", "reconstruction", "wall_distance", "wall_tags", "mu_inf_const"))

    def __init__(self, mesh, Re, turbulence_model=None, gamma=1.4, Pr=0.72, Pr_t=0.9,
                 viscosity_law="sutherland", T_inf_dim=300.0, sutherland_S_dim=110.4,
                 muscl=True, limiter=None, limiter_eps=1e-2, jacobian_mode="compact_pc",
                 material=None, preconditioning=None, preconditioning_floor=1.0, shock_capturing=None):
        """`material` (core/materials.py) is the fluid; by default an IdealGas
        from gamma, Pr, viscosity_law and the Sutherland constants.
        preconditioning, preconditioning_floor, shock_capturing: the Euler
        host's low-Mach preconditioning and artificial viscosity
        (CompressibleEulerModel)."""
        tm = SpalartAllmarasNeg() if turbulence_model is None else turbulence_model
        if material is None:
            material = IdealGas(gamma, Pr=Pr, viscosity_law=viscosity_law,
                                S_ratio=float(sutherland_S_dim) / float(T_inf_dim))
        self.turbulence_model = tm
        self.Re = float(Re)
        self.Pr_t = float(Pr_t)
        self.muscl = muscl
        self.limiter = limiter
        self.limiter_eps = limiter_eps
        self.jacobian_mode = jacobian_mode
        self.reconstruction = None
        self.wall_distance = None
        self.wall_tags = []
        self.mu_inf_const = None

        self.flow = flow = CompressibleEulerModel(mesh, poly_order=0, flux_function=hll_flux,
                                                  extra_variables=tm.names, material=material,
                                                  preconditioning=preconditioning,
                                                  preconditioning_floor=preconditioning_floor,
                                                  shock_capturing=shock_capturing)
        # solver defaults (see the module docstring)
        flow.use_ptc = True
        flow.ptc_settings = {"cfl0": 5.0}
        flow.fom_newton_inner_max_it = 1
        flow.krylov_type = "fgmres"
        flow.asm_overlap = 1
        flow.ilu_levels = 2
        # physics plug-ins
        flow.freestream_scalars = self._freestream_extras
        flow.boundary_scalars = self._boundary_scalars
        flow.residual_terms.append(self._interior_terms)
        flow.boundary_terms.append(self._viscous_boundary_terms)
        flow.wall_bc = self.define_noslip_wall_bc
        flow.wall_traction = self._wall_traction
        flow.time_scale_terms.append(self._viscous_time_scale)
        flow.preconditioning_speeds.append(self._viscous_cutoff_speed)
        flow.extra_scales = self._extra_scales

    # ------------------------------------------------------------------
    # Forwarding to the Euler host
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # only called for attributes not found on the RANS object
        if name.startswith("__") or name in CompressibleRANSModel._OWN:
            raise AttributeError(name)
        return getattr(self.__dict__["flow"], name)

    def __setattr__(self, name, value):
        if name in CompressibleRANSModel._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.flow, name, value)

    # ------------------------------------------------------------------
    # Set-up (overrides of the host's steps)
    # ------------------------------------------------------------------
    def define_elements_functionspaces(self):
        self.flow.define_elements_functionspaces()
        if self.reconstruction is None:
            self.reconstruction = GreenGaussReconstruction(
                self.flow, muscl=self.muscl, limiter=self.limiter, limiter_eps=self.limiter_eps,
                limiter_scales=self._limiter_scales)
            self.reconstruction.install()

    def define_farfield_conditions(self, bc_dict):
        self.flow.define_farfield_conditions(bc_dict)
        ff = self.flow.farfield_conditions
        # the material's viscosity scale: mu_inf = rho_inf U_inf / Re = 1/Re
        self.mu_inf_const = self.flow.material.mu_ref
        self.mu_inf_const.value = ff['rho'] * ff['speed'] / self.Re

    def set_up_solver(self):
        flow = self.flow
        was_set_up = flow.solver_is_set_up
        flow.set_up_solver()
        if was_set_up:
            return
        mode = "exact" if flow.direct_linear_solver else self.jacobian_mode
        if mode != "exact":
            flow.problem.configure_snes(flow.solver.snes, mode)
            flow.solver.krylov_solver.setFromOptions()

    # ------------------------------------------------------------------
    # Material laws and free stream
    # ------------------------------------------------------------------
    def laminar_viscosity(self, U):
        return self.flow.material.mu(U[self.flow.n_mean - 1])

    def eddy_viscosity(self, U):
        return self.turbulence_model.eddy_viscosity(U, self.flow.thermo(U), self.laminar_viscosity(U),
                                                    self.flow.n_mean)

    def viscous_flux(self, U, G, heat_flux=True):
        flow = self.flow
        th = flow.thermo(U)
        mu = self.laminar_viscosity(U)
        tm = self.turbulence_model
        return viscous_flux(th, G, flow.n_mean, mu, tm.eddy_viscosity(U, th, mu, flow.n_mean),
                            flow.material.k(th.T), self.Pr_t,
                            extra_fluxes=tm.diffusive_fluxes(U, th, mu, flow.n_mean),
                            heat_flux=heat_flux)

    def freestream(self):
        """The free-stream dict TurbulenceModel methods take (solver units)."""
        ff = self.flow.farfield_conditions
        return {"rho": ff['rho'], "speed": ff['speed'], "mu": self.mu_inf_const}

    def _freestream_extras(self):
        return self.turbulence_model.freestream_values(self.freestream())

    def _extra_scales(self):
        return list(self.turbulence_model.limiter_scales(self.freestream()))

    def _limiter_scales(self):
        return self.flow.state_scale_expressions()

    def _boundary_scalars(self, kind, U, inflow):
        nm, ne = self.flow.n_mean, self.flow.n_extra
        scalars = extra_components(U, nm)[:ne]
        if kind == "slip":
            return scalars
        if kind == "farfield":
            return [ufl.conditional(inflow, s_inf, s)
                    for s_inf, s in zip(self._freestream_extras(), scalars)]
        if kind == "wall":
            return self.turbulence_model.wall_values(U, nm)
        raise ValueError("unknown boundary kind {!r}".format(kind))

    # ------------------------------------------------------------------
    # Residual terms (plugged into the host)
    # ------------------------------------------------------------------
    def _interior_terms(self):
        flow, rec = self.flow, self.reconstruction
        U, v, n = flow.u_vec, flow.v_vec, flow.mesh_obj.n
        nm = flow.n_mean
        jump_v = ufl.outer(v('+') - v('-'), n('+'))
        F = - ufl.inner(flow.embed(self.viscous_flux(0.5 * (U('+') + U('-')), rec.face_gradient_interior())),
                        jump_v) * flow.forms.dS
        if flow.n_extra:
            # the WallDistance field, or a UFL expression standing in for it
            # (manufactured-solution tests)
            d = self.wall_distance
            d = 1.0 if d is None else getattr(d, "function", d)
            mu = self.laminar_viscosity(U)
            S = self.turbulence_model.sources(U, flow.thermo(U), rec.G, mu, d, nm)
            F -= sum(S[k] * v[nm + k] for k in range(flow.n_extra)) * flow.forms.dx
        return F

    def _viscous_boundary_terms(self, U_b, meshtag, kind):
        if kind == "slip":
            return None   # symmetry: zero shear stress and heat flux
        flow = self.flow
        Fv = self.viscous_flux(U_b, self.reconstruction.face_gradient_boundary(U_b),
                               heat_flux=(kind != "wall"))
        return - ufl.inner(flow.embed(ufl.dot(Fv, flow.mesh_obj.n)), flow.v_vec) * flow.forms.ds(meshtag)

    def _viscous_cutoff_speed(self, U, h):
        """(nu/h)^2: low-Mach preconditioning stops at the viscous speed, so
        viscous-dominated near-wall cells are not over-preconditioned."""
        nu = self.laminar_viscosity(U) / ufl.max_value(self.flow.thermo(U).rho, 1e-8)
        return (nu / h) ** 2

    def _viscous_time_scale(self, U):
        rho = ufl.max_value(self.flow.thermo(U).rho, 1e-8)
        nu_eff = self.flow.material.viscous_time_factor() * (
            self.laminar_viscosity(U) + ufl.max_value(self.eddy_viscosity(U), 0.0)) / rho
        h = self.flow.mesh_obj.h_min
        return 2.0 * nu_eff / h**2

    # ------------------------------------------------------------------
    # No-slip wall
    # ------------------------------------------------------------------
    def noslip_wall_state(self, U):
        d = self.flow.dimensions
        return self.flow.with_aux(concat_state(ufl.as_vector([U[0]] + [0.0] * d + [U[1 + d]]),
                                               self._boundary_scalars("wall", U, None)), U)

    def define_noslip_wall_bc(self, meshtag):
        """
        Adiabatic no-slip wall. The inviscid flux is the exact flux of the
        wall state, (0, p' n, 0, 0): no mass, energy or scalar flux through
        the wall, the interior pressure on it. Conduction is dropped from the
        viscous flux, which is the q.n = 0 condition.
        """
        flow = self.flow
        flow.add_boundary_condition(meshtag, self.noslip_wall_state(flow.u_vec), "wall",
                                    inviscid_flux="exact")
        self.wall_tags.append(meshtag)
        if self.turbulence_model.needs_wall_distance:
            if self.wall_distance is None:
                self.wall_distance = WallDistance(flow.mesh_obj, self.wall_tags)
                flow.derived_fields.append(self.wall_distance)
            else:
                self.wall_distance.wall_tags = list(self.wall_tags)
                self.wall_distance.update_geometry()

    def define_wall_bc(self, meshtag):
        self.define_noslip_wall_bc(meshtag)

    def _wall_stress_arguments(self, U):
        """(state, gradient) the wall shear is evaluated with: the wall state
        and the reconstructed facet gradient, exactly what the residual's wall
        flux uses."""
        U_w = self.noslip_wall_state(U)
        return U_w, self.reconstruction.face_gradient_boundary(U_w, U)

    def _wall_traction(self, U, n):
        U_s, G_s = self._wall_stress_arguments(U)
        grad_u = velocity_gradient(G_s, self.flow.n_mean)
        tau = viscous_stress(grad_u, self.laminar_viscosity(U_s) + self.eddy_viscosity(U_s))
        return - ufl.dot(tau, n)

    def skin_friction_vector(self, U, n, q_inf):
        """tau_w / q_inf: the wall shear (tangential traction) coefficient
        vector. Call update_derived_fields() before assembling it."""
        U_s, G_s = self._wall_stress_arguments(U)
        grad_u = velocity_gradient(G_s, self.flow.n_mean)
        t = ufl.dot(viscous_stress(grad_u, self.laminar_viscosity(U_s)), n)
        return (t - ufl.dot(t, n) * n) / q_inf

    def eddy_viscosity_ratio(self):
        """mu_t / mu of the current state, as a UFL expression."""
        U = self.flow.u_vec
        return self.eddy_viscosity(U) / self.laminar_viscosity(U)
