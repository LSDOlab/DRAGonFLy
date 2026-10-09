"""
Euler equations at p = 0 with MUSCL reconstruction: second-order inviscid
fluxes.

The plain p = 0 CompressibleEulerModel evaluates the HLL flux on the cell
averages, a first-order scheme whose numerical dissipation dominates the drag
of a wing (it scales with the wetted area). MUSCLEulerModel installs the same
Green-Gauss reconstruction the RANS model uses (core/reconstruction.py) on the
Euler model: the interior-facet traces become U_K + G_K . (x_f - x_K), the
gradient G is condensed out of Newton exactly, and the adjoint and shape
derivatives include the chain terms through G and the cell centres.

The solver defaults are the RANS model's: pseudo-transient continuation from
the free stream (CFL0 5), one Newton step per pseudo step, FGMRES on the exact
condensed Jacobian, preconditioned with ASM(1)/ILU(2) of its compact,
gradient-frozen part (jacobian_mode "compact_pc").

Use it through DG_windtunnel_model(model_class=MUSCLEulerModel,
model_kwargs={...}).
"""
from dragonfly_sim.core.Euler_model import CompressibleEulerModel
from dragonfly_sim.core.reconstruction import GreenGaussReconstruction
from dragonfly_sim.utils.Euler_utils import hll_flux


class MUSCLEulerModel(CompressibleEulerModel):
    """CompressibleEulerModel with a MUSCL (Green-Gauss) reconstruction.

    limiter: None (unlimited MUSCL, the RANS default) or "vanalbada" (smooth,
        so the exact Jacobian survives; see GreenGaussReconstruction).
    limiter_eps: the van Albada smoothing parameter, relative to the
        free-stream scale of each state component.
    jacobian_mode: what Newton's linear solve uses (utils/gradient_condensation.py).
    use_ptc, cfl0: pseudo-transient continuation from the free stream.
    """

    def __init__(self, mesh, poly_order=0, gamma=1.4, limiter=None, limiter_eps=1e-2,
                 jacobian_mode="compact_pc", use_ptc=True, cfl0=5.0):
        super().__init__(mesh, poly_order=poly_order, gamma=gamma, flux_function=hll_flux)
        self.limiter = limiter
        self.limiter_eps = limiter_eps
        self.jacobian_mode = jacobian_mode
        self.reconstruction = None
        # solver defaults (see the module docstring)
        self.use_ptc = use_ptc
        self.ptc_settings = {"cfl0": cfl0}
        if use_ptc:
            self.fom_newton_inner_max_it = 1
        self.krylov_type = "fgmres"
        self.asm_overlap = 1
        self.ilu_levels = 2

    def define_elements_functionspaces(self):
        super().define_elements_functionspaces()
        if self.reconstruction is None:
            GreenGaussReconstruction(self, muscl=True, limiter=self.limiter,
                                     limiter_eps=self.limiter_eps,
                                     limiter_scales=self._limiter_scales).install()

    def _limiter_scales(self):
        """Free-stream magnitude of each state component (van Albada limiter)."""
        inlet = self.boundary_conditions['inlet']
        rho, p = inlet['rho'], inlet['p']
        speed = inlet['M'] * inlet['c']
        return ([rho] + [rho * speed] * self.dimensions
                + [p / (self.gamma - 1.0) + 0.5 * rho * speed**2])

    def set_up_solver(self):
        was_set_up = self.solver_is_set_up
        super().set_up_solver()
        if was_set_up:
            return
        mode = "exact" if self.direct_linear_solver else self.jacobian_mode
        if mode != "exact":
            self.problem.configure_snes(self.solver.snes, mode)
            self.solver.krylov_solver.setFromOptions()
