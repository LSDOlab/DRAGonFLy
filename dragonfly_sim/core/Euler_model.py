import ctypes
import gc
import contextlib

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import ufl.classes
import dolfinx
import dolfinx.fem.petsc

from dragonfly_sim.utils.Nonlinear_utils import NonlinearProblem_mod, SNESNewtonSolver
from dragonfly_sim.utils.petsc_utils import set_petsc_vec_array
from dragonfly_sim.utils.solver_utils import (locate_positivity_limiting_nodes, ksp_converged_reason_name,
                                              ksp_norm_type_name, ksp_pc_side_name)
from dragonfly_sim.core.form_manager import FormManager
from dragonfly_sim.utils.Euler_utils import (slip_wall_state, hll_flux, boundary_flux, spectral_radius,
                                             smooth_max)
from dragonfly_sim.utils.NS_utils import mean_flow, concat_state
from dragonfly_sim.core.materials import IdealGas, thermo, state_vector
from dragonfly_sim.utils.filewriter import FileWriter


class CompressibleEulerModel():
    """
    The p = 0 DG (cell-centred finite-volume) compressible Euler model, and
    the host every other flow model in the package is built on.

    State: the primitive mean flow V = (p', u, T) -- gauge pressure,
    velocity, temperature, n_mean = 2 + dim components per cell -- optionally
    followed by `extra_variables` transported scalars phi_k (n_state
    components in all). The residual is the conservative one in U(V) =
    (rho, rho u, rho E', rho phi) (core/materials.py): every flux is
    evaluated on the ThermoState of its traces, so shocks keep their
    conservative jump conditions. The scalars are convected with the mean
    flow and are not positivity-limited.

    The fluid is the `material` (core/materials.py; an IdealGas with
    `gamma` by default), the only fluid-specific code. All quantities are
    scaled with the far-field state (see core/materials.py); the energy
    equation is weighted by 1/(h_T T_ref), which keeps the residual rows
    of one size at every Mach number.

    Plug-in points. CompressibleRANSModel fills these; for the plain Euler
    model they are all empty or None, and the residual, Jacobian and
    derivative forms are the same expressions as without them.

        face_traces         () -> (U+, U-) interior-facet traces of the
                            inviscid flux (MUSCL reconstruction)
        residual_terms      callables () -> UFL form added to the residual
        boundary_terms      callables (U_b, tag, kind) -> UFL form or None,
                            added to every boundary condition
        boundary_scalars    (kind, U, inflow) -> list of the extra variables'
                            boundary values (kind: slip, farfield,
                            wall); required when n_extra > 0
        freestream_scalars  () -> list of the extra variables' free-stream values
        wall_bc             (tag) -> None, replaces the slip wall in define_wall_bc
        wall_traction       (U, n) -> traction added to p n in wall_force_density
        time_scale_terms    callables (U) -> added to the PTC inverse time scale
        preconditioning_speeds
                            callables (U, h) -> a squared speed the low-Mach
                            reference velocity U_r may not drop below (e.g.
                            the viscous (nu/h)^2)
        derived_fields      DerivedField objects (utils/derived_fields.py):
                            fields the residual reads that are not unknowns
                            (a reconstructed gradient, cell centres, the wall
                            distance). Every derivative assembler adds their
                            chain-rule terms.
        problem_factory     (F, J) -> nonlinear problem, replacing
                            NonlinearProblem_mod in set_up_solver
        extra_scales        () -> list of the extra variables' typical
                            magnitudes (numbers or UFL expressions of
                            Constants); required by state_scales when
                            n_extra > 0

    Every boundary condition records (U_b, tag, kind) in boundary_states.

    Time integration (core/time_integration.py): use_ptc makes solve_system
    a pseudo-transient continuation solve; enable_time_stepping() adds a
    BDF physical time term for unsteady runs. Both live in newton_forms()
    only, so self.F stays the steady residual that the adjoint
    differentiates.
    """
    def __init__(self, mesh, poly_order=0, gamma=1.4, flux_function=hll_flux, extra_variables=(),
                 material=None, preconditioning=None, preconditioning_floor=1.0, shock_capturing=None):
        """
        preconditioning: local low-Mach (Weiss-Smith) preconditioning of the
            flux dissipation and the pseudo-time term (see
            low_mach_preconditioning); None means on exactly when the
            material needs it (an incompressible fluid).
        preconditioning_floor: the reference velocity's floor, as a fraction
            of the free-stream speed. 1 (U_r >= u_inf, as Turkel recommends):
            smaller floors slowed the transonic RANS start from the free
            stream twofold (0.1, 0.3 on Rizzi grid 1) or stalled it (0.1 on
            grid 4); U_r = O(u_inf) keeps the low-Mach scaling either way.
        shock_capturing: None, or True or a dict of options (possibly empty) for the
            Barter-Darmofal artificial viscosity (core/shock_capturing.py),
            which adds its viscosity field as an auxiliary unknown.
        """
        if poly_order > 0 and shock_capturing not in (None, False):
            raise ValueError("the artificial viscosity is the p = 0 form (core/shock_capturing.py)")

        self.mesh_obj = mesh
        self.poly_order = poly_order
        self.material = IdealGas(gamma) if material is None else material
        if preconditioning is None:
            preconditioning = self.material.incompressible
        if self.material.incompressible and not preconditioning:
            raise ValueError("an incompressible fluid needs the low-Mach preconditioning")
        self.preconditioning = bool(preconditioning)
        self.preconditioning_floor = float(preconditioning_floor)
        # kept for callers that read it; None for a fluid without one
        self.gamma = getattr(self.material, "gamma", None)
        self.dimensions = self.mesh_obj.mesh._ufl_domain._geometric_dimension

        self.flux_function = flux_function

        # State layout: mean flow, then the transported scalars, then the
        # auxiliary (not conserved, not transported) unknowns
        self.extra_variables = tuple(extra_variables)
        # True: the artificial viscosity with its default options; False/None: none
        shock_capturing = {True: {}, False: None}.get(shock_capturing, shock_capturing) \
            if isinstance(shock_capturing, bool) else shock_capturing
        self.aux_variables = ("sigma",) if shock_capturing is not None else ()
        self.n_mean = 2 + self.dimensions
        self.n_extra = len(self.extra_variables)
        self.n_aux = len(self.aux_variables)
        self.n_state = self.n_mean + self.n_extra + self.n_aux

        # Plug-in points (see the class docstring); all inert by default
        self.boundary_states = []
        self.face_traces = None
        self.residual_terms = []
        self.boundary_terms = []
        self.boundary_scalars = None
        self.freestream_scalars = None
        self.wall_bc = None
        self.wall_traction = None
        self.time_scale_terms = []
        self.preconditioning_speeds = []
        self.shock_capturing = None
        if shock_capturing is not None:
            from dragonfly_sim.core.shock_capturing import BarterDarmofalAV
            if self.material.incompressible:
                raise ValueError("shock capturing for an incompressible fluid: there are no shocks")
            self.shock_capturing = BarterDarmofalAV(self, **shock_capturing)
        self.derived_fields = []
        self.problem_factory = None
        self.extra_scales = None

        # Time integration (core/time_integration.py). time_integrator is
        # created by set_up_solver when use_ptc is set or by
        # enable_time_stepping(); ptc_settings are CFLController keywords.
        self.use_ptc = False
        self.ptc_settings = {}
        # a steady PTC solve that starts from the end state of the previous
        # converged steady solve continues at that solve's final CFL instead
        # of ramping up from cfl0 (TimeIntegrator.solve_steady)
        self.ptc_warm_start = True
        self.time_integrator = None
        self._time_stepping = None
        # Smallest positivity step length applied since the caller last
        # reset it; the PTC CFL law reads it.
        self.last_step_theta = 1.0
        # Largest relative change of the absolute pressure and the
        # temperature per Newton step (positivity_step_length): an update of
        # the primitive unknowns misses the quadratic kinetic-energy term
        # an update of rho E carries, which overheats cells in the large
        # steps of an impulsive start. None: positivity only.
        self.max_relative_change = 0.5

        # Krylov solver of the Newton steps (see apply_krylov_solver_settings);
        # direct_linear_solver switches every linear solve to MUMPS.
        self.krylov_type = "gmres"
        self.krylov_diagonal_scale = True
        self.direct_linear_solver = False
        # relative tolerance of that Krylov solve, set before set_up_solver
        self.krylov_rtol = 1e-3

        # Instantiate the FormManager
        self.forms = FormManager(mesh, model=self,
                                 measure_source=mesh.form_manager)

        self.functionspaces = None
        self.bcs = []
        self.farfield_conditions = None
        self.initial_conditions = None
        self.F = None
        self.J = None
        self.solver_is_set_up = False
        self.solver_converged = False



        # Newton steps taken per pass of solve_system's loop
        self.fom_newton_inner_max_it = 5

        # Report which nodes bound the Newton step whenever the positivity
        # limiter cuts theta below the step-length cap -- see
        # _report_positivity_limiting_node. Costs three allreduces and one
        # small allgather, and ONLY on iterations where theta was already cut,
        # so it is on by default: a collapsed theta is the failure this most
        # often needs explaining, and the limiter itself reports only the
        # surviving step length. Set False once a case is solving cleanly and
        # the per-iteration lines are just noise.
        self.report_positivity_limiter = True

        # ASM overlap and ILU fill for the preconditioner 
        self.asm_overlap = 2
        self.ilu_levels = 2
        # ASM subdomain solver: "ilu" (ILU(ilu_levels)) or "lu" (exact
        # subdomain factorizations: more memory, robust where ILU stalls,
        # e.g. the zero pressure diagonal of an incompressible time step)
        self.subdomain_solver = "ilu"
        # Preconditioner of the Newton steps' Krylov solve:
        #   "asm"    additive Schwarz with subdomain ILU/LU (the default; the
        #            compressible flows' setting)
        #   "schur"  a Schur-complement field split of the pressure p' from
        #            the other unknowns: ASM/ILU on the velocity-scalar block,
        #            BoomerAMG on the assembled Schur approximation (selfp).
        #            For incompressible (water) and low-Mach time steps, whose
        #            pressure couples globally (a Poisson-like Schur
        #            complement) where local ILU stalls. Needs an AIJ
        #            Jacobian (the condensed MUSCL/RANS ones are).
        #   "mg"     hp-multigrid (utils/line_multigrid.py): p-multigrid DG p
        #            -> ... -> DG 0, then agglomeration levels with a linear
        #            function per aggregate, Galerkin coarse operators and line
        #            smoothing along the strongly coupled (wall-normal) lines on
        #            every level; with FGMRES. For stiff time steps on stretched
        #            near-wall cells, where the one-level preconditioners'
        #            iterations grow like 1/h. multigrid_options: its keywords
        #            (HPMultigridPC).
        self.linear_preconditioner = "asm"
        # extra PETSc options of the Newton steps' Krylov solve, {name without
        # the prefix: value}, applied after the model's own (experiments with
        # preconditioner variants, e.g. the Schur complement approximation)
        self.newton_ksp_options = {}
        self.multigrid_options = {}

        self.fom_solution_writer = None
        self._output_fields = []
        # False until define_sol_export_files actually opens the writers, which
        # keeps write_solution_output inert for a model that never opened them
        self.export_solutions = False


    def output_field_expressions(self):
        """The fields of the solution file, as (name, value shape, UFL
        expression of u_vec), in the solver's units: the unknowns as separate
        fields -- the gauge pressure p' (pressure; C_p = 2 p'), the velocity
        u (a vector), the temperature T and each transported scalar under its
        own name (e.g. nu_tilde) -- then the density and, for a compressible
        fluid, the Mach number."""
        dim = self.dimensions
        th = self.thermo(self.u_vec)
        fields = [("pressure", (), th.p), ("velocity", (dim,), th.u), ("temperature", (), th.T)]
        fields += [(name, (), self.u_vec[self.n_mean + k])
                   for k, name in enumerate(self.extra_variables + self.aux_variables)]
        fields += [("density", (), th.rho)]
        if not self.material.incompressible:
            fields += [("mach", (), ufl.sqrt(ufl.dot(th.u, th.u) * th.inv_c2))]
        return fields

    def define_sol_export_files(self, func_space=None, file_name_addendum=None, mesh_deformation=False,
                                time_dependent=False, file_name=None):
        """Open FOM_solution_<suffix>.bp, one file holding every output field
        (output_field_expressions) as a separate, named field; the mesh is
        stored once per frame for all of them. Collective.

        It is the ALIGNED series: exactly one frame per design evaluation,
        written by write_solution_output() at eval_idx, so step k here is step
        k in mesh_deformation_<suffix>.bp. It carries the accepted solution --
        see DG_windtunnel_model.solve_residual_equations. mesh_deformation=True
        writes each frame on the deformed mesh it was solved on. func_space is
        unused (the fields define their own spaces); it is kept for callers.

        time_dependent: frames are labelled with the physical time
        (write_solution_output(..., time=t)) instead of the evaluation index,
        for unsteady runs; file_name replaces FOM_solution_<suffix>.
        """
        spaces = {}
        self._output_fields = []
        for name, shape, expr in self.output_field_expressions():
            if shape not in spaces:
                spaces[shape] = (self.functionspaces["V_scalar"] if shape == () else
                                 dolfinx.fem.functionspace(self.mesh_obj.mesh,
                                                           ("DG", self.poly_order, shape)))
            V = spaces[shape]
            # compiled once here (collective), interpolated at every write
            self._output_fields.append(
                (name, dolfinx.fem.Function(V, name=name),
                 dolfinx.fem.Expression(expr, V.element.interpolation_points)))
        self.fom_solution_writer = FileWriter(
            file_name or "FOM_solution_{}".format(file_name_addendum), self.mesh_obj.mesh.comm,
            {name: f.function_space for name, f, _ in self._output_fields},
            mesh_deformation=mesh_deformation, time_dependent=time_dependent)
        self.export_solutions = True

    def _write_output_fields(self, write_counter, time=None):
        """Interpolate the output fields off the current u_vec and write one
        frame. u_vec must already hold the state being exported. Collective."""
        for _, f, expr in self._output_fields:
            f.interpolate(expr)
        label = dict(time=time) if self.fom_solution_writer.time_dependent else dict(write_counter=write_counter)
        self.fom_solution_writer.interpolate_and_write(
            {name: f for name, f, _ in self._output_fields}, **label)

    def write_solution_output(self, write_counter, solution_array=None, time=None):
        """Write one aligned output frame at index `write_counter`.

        Called once per design evaluation from
        DG_windtunnel_model.solve_residual_equations with write_counter =
        eval_idx, so FOM_solution_<suffix>.bp steps in lockstep with
        mesh_deformation_<suffix>.bp.

        `solution_array` is the accepted solution of the evaluation, passed
        explicitly rather than read off self.u_vec. Passing None writes
        whatever u_vec currently holds. `time` labels the frame of a
        time-dependent writer (define_sol_export_files(time_dependent=True)).
        """
        if not self.export_solutions:
            return

        if solution_array is not None:
            self.interpolate_solution_vector(solution_array)

        self._write_output_fields(write_counter, time)

    def define_elements_functionspaces(self):
        # Throughout this class we need access to various function spaces; this is where we define them
        self.functionspaces = {
            "V": dolfinx.fem.functionspace(self.mesh_obj.mesh, ("DG", self.poly_order, (self.n_state,))),
            "V_scalar": dolfinx.fem.functionspace(self.mesh_obj.mesh, ("DG", self.poly_order)),
        }

    def define_trial_testfunctions(self):
        # Define the trial and test functions for the weak form. Every form
        # tests with v_vec, the test function with the energy row weighted by
        # 1/(h_T T_ref): rho u H carries the enthalpy h ~ c_p T, which is
        # 1/((gamma - 1) M^2) times the kinetic terms, and would otherwise
        # make the energy rows dominate every residual norm at low Mach.
        self.u_vec = dolfinx.fem.Function(self.functionspaces["V"], name="u")
        self.v_test = ufl.TestFunction(self.functionspaces["V"])
        w_energy = 1.0 / self.material.energy_scale()
        e = self.n_mean - 1
        self.v_vec = ufl.as_vector([w_energy * self.v_test[i] if i == e else self.v_test[i]
                                    for i in range(self.n_state)])

    def define_farfield_conditions(self, farfield):
        """
        Set the free-stream state the far-field boundary condition relaxes to.

        `farfield` is the material's far-field dict, for an ideal gas
        {'rho', 'p', 'M', 'alpha'}: density, pressure, Mach number and angle
        of attack (rad). This is all the user states; the boundary treatment
        (inflow, outflow) follows from it per facet in define_farfield_bc,
        through the characteristic state. The material scales every quantity
        with it (core/materials.py), so farfield_conditions holds the free
        stream in the solver's units: rho = 1, speed |u| = 1, the absolute
        pressure p (1/(gamma M^2) for an ideal gas) and the sound speed c.

        NOTE: the ideal gas's far field is the subsonic characteristic one,
        so M < 1. A supersonic free stream needs a different boundary state
        (all variables from the free stream at inflow, all from the interior
        at outflow); IdealGas.farfield_state is the one place that would
        change.
        """
        ref = self.material.bind(self.mesh_obj.mesh, farfield)
        alpha = float(farfield['alpha'])

        # Since angle of attack is a design parameter we define it as a UFL variable;
        # this allows us to use Dolfin's automatic differentiation for it
        self.alpha_const = dolfinx.fem.Constant(self.mesh_obj.mesh, alpha)
        self.alpha_var = ufl.variable(self.alpha_const)

        inflow_dir = self._inflow_direction_ufl()

        # unit vector along the free stream (the drag direction)
        flow_dir = dolfinx.fem.Constant(
            self.mesh_obj.mesh, self._inflow_direction_numpy(alpha))

        # the free-stream speed, the velocity unit
        self.u_ref_const = dolfinx.fem.Constant(self.mesh_obj.mesh, ref['speed'])
        u_inf = self.u_ref_const * inflow_dir

        self.farfield_conditions = dict(ref)
        self.farfield_conditions.update({'alpha': alpha,
                                         'inflow_angle': self._inflow_direction_numpy(alpha),
                                         'flow_dir': flow_dir,
                                         'u': u_inf,
                                         'u_ref': self.u_ref_const})

    def _inflow_direction_ufl(self):
        # unit freestream direction as a UFL expression
        if self.dimensions == 2:
            # x = chordwise, y = vertical
            return ufl.as_vector([ufl.cos(self.alpha_var), ufl.sin(self.alpha_var)])
        elif self.dimensions == 3:
            # x = chordwise, y = spanwise, z = vertical
            return ufl.as_vector([ufl.cos(self.alpha_var), 0., ufl.sin(self.alpha_var)])
        else:
            raise ValueError("Unsupported dimension {}".format(self.dimensions))

    def _inflow_direction_numpy(self, alpha):
        # unit freestream direction as a numpy array
        if self.dimensions == 2:
            return np.array([np.cos(alpha), np.sin(alpha)], dtype=np.double)
        elif self.dimensions == 3:
            return np.array([np.cos(alpha), 0., np.sin(alpha)], dtype=np.double)
        else:
            raise ValueError("Unsupported dimension {}".format(self.dimensions))

    def set_angle_of_attack(self, alpha):
        # propagate the new angle of attack `alpha` to all model dependencies
        alpha = float(alpha)

        self.alpha_const.value = alpha

        inflow_angle = self._inflow_direction_numpy(alpha)
        self.farfield_conditions['alpha'] = alpha
        self.farfield_conditions['inflow_angle'] = inflow_angle
        self.farfield_conditions['flow_dir'].value = inflow_angle

    def set_mach(self, M):
        """Propagate a new Mach number to the material's constants. The
        velocity unit stays the free-stream speed, so the far-field state in
        the solver's units is unchanged; the absolute pressure p_ref =
        1/(gamma M^2) and the sound speed move."""
        self.farfield_conditions.update(self.material.set_mach(M))

    def freestream_state(self):
        """The free stream V_inf = (0, u_inf, 1, phi_inf) as a UFL vector."""
        ff = self.farfield_conditions
        extras = []
        if self.n_extra:
            # plain numbers become Constants on this mesh, so the state stays
            # interpolable
            extras = [v if isinstance(v, ufl.core.expr.Expr)
                      else dolfinx.fem.Constant(self.mesh_obj.mesh, float(v))
                      for v in self._freestream_scalars()]
        msh = self.mesh_obj.mesh
        zero = dolfinx.fem.Constant(msh, 0.0)
        return state_vector(zero, ff['u'], dolfinx.fem.Constant(msh, 1.0),
                            extras + [zero] * self.n_aux)

    def compute_initial_conditions_from_farfield(self):
        # The initial guess used in the Newton solver: the free stream.
        self.initial_conditions = self.freestream_state()

    def interpolate_solution_vector(self, interpolant=None):
        if interpolant is None:
            interpolant = self.initial_conditions

        if isinstance(interpolant, (np.ndarray, PETSc.Vec)):
            set_petsc_vec_array(self.u_vec.x.petsc_vec, interpolant)
        else:
            for j in range(interpolant.ufl_shape[0]):
                self.u_vec.sub(j).interpolate(
                    dolfinx.fem.Expression(interpolant[j], self.functionspaces['V'].sub(j).element.interpolation_points,
                                           comm=self.mesh_obj.mesh.comm))

        self.u_vec.x.scatter_forward()

    # ------------------------------------------------------------------
    # State layout
    # ------------------------------------------------------------------
    def mean_flow(self, U):
        """The mean-flow slice (p', u, T) of a state expression."""
        return mean_flow(U, self.n_mean)

    def thermo(self, V):
        """The ThermoState (core/materials.py) of a state expression V: the
        conserved state, the physical flux and the material derivatives. V
        may be the mean flow alone."""
        return thermo(self.material, V, self.n_mean, min(self.n_extra, V.ufl_shape[0] - self.n_mean))

    def embed(self, x):
        """A vector (or the rows of a tensor) over the conserved components,
        padded with zero rows for the auxiliary unknowns, which have no
        flux and no time derivative. The identity without them."""
        if not self.n_aux:
            return x
        if len(x.ufl_shape) == 1:
            return ufl.as_vector([x[i] for i in range(x.ufl_shape[0])] + [0.0] * self.n_aux)
        zero = [0.0] * x.ufl_shape[1]
        rows = [[x[i, j] for j in range(x.ufl_shape[1])] for i in range(x.ufl_shape[0])]
        return ufl.as_tensor(rows + [zero] * self.n_aux)

    def with_aux(self, V_b, V=None):
        """A boundary state V_b given without the auxiliary unknowns, completed
        with V's (the interior's: zero normal gradient)."""
        if not self.n_aux or V_b.ufl_shape[0] == self.n_state:
            return V_b
        V = self.u_vec if V is None else V
        n_c = self.n_mean + self.n_extra
        return ufl.as_vector([V_b[i] for i in range(n_c)] + [V[i] for i in range(n_c, self.n_state)])

    def conserved(self, V):
        """The conserved state U(V) the residual and the physical time term
        are written in (embedded: zero for the auxiliary unknowns)."""
        return self.embed(self.thermo(V).U)

    def pseudo_time_increment(self, V_old, dV):
        """
        The pseudo-time term's increment Gamma dV at V_old: dU/dV dV, plus
        with low-Mach preconditioning the Weiss-Smith correction of the
        pressure column, (1/U_r^2 - 1/c^2) dp' (1, u, H, phi). The physical
        time derivative is never preconditioned.
        """
        Vv = ufl.variable(V_old)
        inc = ufl.dot(ufl.diff(self.conserved(Vv), Vv), dV)
        if not self.preconditioning:
            return inc
        th = self.low_mach_preconditioning(self.thermo(V_old), V_old, self.mesh_obj.h_min)
        d = self.dimensions
        w = ufl.as_vector([1.0] + [th.u[i] for i in range(d)] + [th.H] + list(th.phi))
        return inc + (th.inv_Ur2 - th.inv_c2) * dV[0] * self.embed(w)

    def low_mach_preconditioning(self, th, V, h, pressure_jump=None):
        """
        th with its preconditioning set: 1/U_r^2 (ThermoState.inv_Ur2), the
        local reference velocity of the Weiss-Smith preconditioner,

            1/U_r^2 = max(1/c^2, 1/max(|u|, eps u_inf, sqrt(|dp'|/rho), cutoffs)^2),

        with smooth maxima (kinks stall Newton). The 1/c^2 floor caps U_r at
        c, so the preconditioning switches itself off wherever the flow is
        sonic or faster; for an incompressible fluid (1/c^2 = 0) it is
        always active. The pressure-jump term (faces only, pressure_jump =
        p'_+ - p'_-) keeps stagnation points robust; the
        preconditioning_speeds plug-ins add e.g. the viscous cutoff nu/h.
        Unchanged (inv_Ur2 None) without preconditioning.
        """
        if not self.preconditioning:
            return th
        Ur2 = smooth_max(ufl.dot(th.u, th.u), (self.preconditioning_floor * self.u_ref_const) ** 2)
        if pressure_jump is not None:
            Ur2 = smooth_max(Ur2, ufl.sqrt(pressure_jump ** 2 + 1e-24) / th.rho)
        for term in self.preconditioning_speeds:
            Ur2 = smooth_max(Ur2, term(V, h))
        th.inv_Ur2 = smooth_max(th.inv_c2, 1.0 / Ur2)
        return th

    def _flux_states(self, V_a, V_b, h):
        """ThermoStates of a facet's two states, preconditioned with the
        facet's pressure jump."""
        th_a, th_b = self.thermo(V_a), self.thermo(V_b)
        dp = V_a[0] - V_b[0]
        return (self.low_mach_preconditioning(th_a, V_a, h, dp),
                self.low_mach_preconditioning(th_b, V_b, h, dp))

    def _freestream_scalars(self):
        if self.freestream_scalars is None:
            raise RuntimeError("{} extra variables but no freestream_scalars plug-in".format(self.n_extra))
        return list(self.freestream_scalars())

    def _boundary_state(self, U_b_mean, kind, inflow=None):
        """The full boundary state: the mean-flow state and, with extra
        variables, their values from the boundary_scalars plug-in."""
        if not self.n_extra:
            return U_b_mean
        if self.boundary_scalars is None:
            raise RuntimeError("{} extra variables but no boundary_scalars plug-in".format(self.n_extra))
        return concat_state(U_b_mean, list(self.boundary_scalars(kind, self.u_vec, inflow)))

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    def add_boundary_condition(self, meshtag, U_b, kind, inviscid_flux="llf"):
        """
        Weakly impose the boundary state U_b (a state V_b in the unknowns'
        layout) on ds(meshtag) and register it.

        inviscid_flux="llf" is the one-sided Lax-Friedrichs boundary flux
        against U_b (Euler_utils.boundary_flux); "numerical" is the interior
        numerical flux with U_b as the ghost state (the upwinding picks the
        incoming characteristics from it); "exact" is the physical flux of
        U_b itself, for states that already carry the upwinding (the
        characteristic far field) or that must have no mass flux (no-slip
        wall). The boundary_terms plug-ins add their terms (e.g. viscous
        fluxes) to the same integral.
        """
        n, h = self.mesh_obj.n, self.mesh_obj.h_min
        U_b = self.with_aux(U_b)
        if inviscid_flux == "llf":
            f = boundary_flux(*self._flux_states(self.u_vec, U_b, h), n)
        elif inviscid_flux == "numerical":
            f = self.flux_function(*self._flux_states(self.u_vec, U_b, h), n)
        elif inviscid_flux == "exact":
            f = self.thermo(U_b).normal_flux(n)
        else:
            raise ValueError("inviscid_flux must be 'llf', 'numerical' or 'exact'")
        F = ufl.inner(self.embed(f), self.v_vec) * self.forms.ds(meshtag)
        for term in self.boundary_terms:
            extra = term(U_b, meshtag, kind)
            if extra is not None:
                F = F + extra
        self.bcs += [F]
        self.boundary_states.append((U_b, meshtag, kind))

    def define_slipwall_bc(self, meshtag):
        # define the weak form term for natural enforcement of the slip wall boundary condition
        u_wall = slip_wall_state(self.mean_flow(self.u_vec), self.mesh_obj.n, self.n_mean)
        self.add_boundary_condition(meshtag, self._boundary_state(u_wall, "slip"), "slip")

    def define_wall_bc(self, meshtag):
        """The wall of the body: a slip wall, unless a wall_bc plug-in
        (e.g. the RANS model's no-slip wall) replaces it."""
        if self.wall_bc is not None:
            self.wall_bc(meshtag)
        else:
            self.define_slipwall_bc(meshtag)

    def define_farfield_bc(self, meshtag, transverse=None):
        """
        Characteristic (Riemann-invariant) far field on ds(meshtag), for the
        whole outer boundary: no inflow/outflow division is made, each facet
        takes its character from the sign of the characteristic speeds, so the
        angle of attack can change freely. Outgoing acoustic waves leave the domain (exactly at normal incidence)
        instead of reflecting off a fixed state. The inviscid flux is the
        physical flux of the characteristic state, which already carries the
        upwinding; extra variables take their free-stream values where the
        flow enters and the interior values where it leaves.

        `transverse` (core/farfield.py:TransverseFarfield) adds the
        transverse correction to the incoming invariant: the 'riemann2' far
        field of unsteady runs.

        With low-Mach preconditioning (or a material without Riemann
        invariants) the far field is instead the free stream as a ghost
        state through the preconditioned numerical flux. The Riemann
        invariants are those of the unpreconditioned system: their boundary
        pressure follows the normal velocity through the acoustic impedance
        rho c, 1/M times the dynamic pressure scale, which leaves the
        pressure level of a preconditioned low-Mach solution all but
        undetermined (it drifts, and PTC stalls, at M = 0.001). The ghost
        state pins it through the same pressure-jump dissipation dp'/U_r
        that couples pressure and velocity inside the domain, and the
        upwinding still takes each facet's character from the (now
        preconditioned) wave speeds, so no inflow/outflow division is made
        either; the transported scalars are upwinded with it.

        The state depends on alpha through the free-stream velocity, so
        dR/dalpha follows automatically.
        """
        if self.preconditioning or not hasattr(self.material, "farfield_state"):
            if transverse is not None:
                raise NotImplementedError("the transverse far field (riemann2) corrects the "
                                          "unpreconditioned characteristic far field")
            self.add_boundary_condition(meshtag, self.freestream_state(), "farfield",
                                        inviscid_flux="numerical")
            return
        R_in = None if transverse is None else transverse.R_in
        U_b, inflow = self.material.farfield_state(self.thermo(self.mean_flow(self.u_vec)),
                                                   self.mesh_obj.n, self.farfield_conditions['u'],
                                                   R_in=R_in)
        self.add_boundary_condition(meshtag, self._boundary_state(U_b, "farfield", inflow),
                                    "farfield", inviscid_flux="exact")
        if transverse is not None:
            self.transverse_farfield = transverse

    def compute_interior_integral_terms(self):
        U, v, n = self.u_vec, self.v_vec, self.mesh_obj.n
        # the volume term vanishes at p = 0 (grad v = 0); kept for the form's generality
        F = -ufl.inner(ufl.grad(v), self.embed(self.thermo(U).flux())) * self.forms.dx
        U_p, U_m = self.face_traces() if self.face_traces is not None else (U('+'), U('-'))
        h = self.mesh_obj.h_min
        F_num = self.flux_function(*self._flux_states(U_p, U_m, 0.5 * (h('+') + h('-'))), n('+'))
        F += ufl.inner(self.embed(F_num), v('+') - v('-')) * self.forms.dS
        for term in self.residual_terms:
            F += term()
        return F

    def _lambda_max_expr(self, U, h=None):
        """The largest wave speed |u| + c of the state U, |u'| + c' with
        low-Mach preconditioning (Euler_utils.spectral_radius); h is the
        length of the preconditioning's cutoffs (h_min by default)."""
        h = self.mesh_obj.h_min if h is None else h
        th = self.low_mach_preconditioning(self.thermo(self.mean_flow(U)), U, h)
        return spectral_radius(th)

    def wall_force_density(self, U, n):
        """Force per unit wall area on the body: the gauge pressure p' (the
        reference pressure integrates to zero over a closed body), plus the
        wall_traction plug-in's viscous traction when there is one."""
        f = U[0] * n
        if self.wall_traction is not None:
            f = f + self.wall_traction(U, n)
        return f

    def assemble_dRdalpha_vec(self, u_vec_entries=None):
        """
        The derivative dR/d(alpha) of the converged residual, assembled as a Vec.

        A VECTOR, not a matrix, because alpha is a single global scalar: the
        derivative of the residual with respect to it is one column, and
        ufl.diff against the stored alpha_var yields a rank-1 form that
        assembles straight into it. Nothing has to be contracted afterwards.

        Note this is ufl.diff, not the ufl.derivative used by every other
        assembler here. ufl.derivative only accepts Coefficients and
        SpatialCoordinate and rejects a Constant outright; ufl.diff against a
        ufl.variable-wrapped Constant is the route that works. See
        define_farfield_conditions for why alpha_var must be the single
        stored instance.

        Sparse in practice: alpha only enters the residual through the
        free-stream velocity in the far-field boundary state, so the result is
        supported on far-field-adjacent dofs alone. The interior terms and the
        slip wall (which uses the facet normal) carry no alpha dependence.

        Caches its own form and Vec.
        """
        if u_vec_entries is not None:
            self.set_u_vec(u_vec_entries)

        if self.derived_fields:
            self.update_derived_fields()
            if not hasattr(self, 'dRdalpha_vec'):
                self.dRdalpha_vec = dolfinx.fem.petsc.create_vector(
                    dolfinx.fem.extract_function_spaces(self.F_physical_form))
            return self._chain.residual_alpha(self.F, "F", self.dRdalpha_vec)

        if not hasattr(self, 'dRdalpha_form'):
            self.dRdalpha_form = dolfinx.fem.form(
                ufl.diff(self.F, self.alpha_var))
            self.dRdalpha_vec = dolfinx.fem.petsc.create_vector(
                dolfinx.fem.extract_function_spaces(self.dRdalpha_form))

        with self.dRdalpha_vec.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.dRdalpha_vec, self.dRdalpha_form)
        self.dRdalpha_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                      mode=PETSc.ScatterMode.REVERSE)

        return self.dRdalpha_vec

    def compute_weakform(self):
        if self.derived_fields and not getattr(self, "_geometry_listener_added", False):
            # derived fields follow the mesh through every node motion
            self.mesh_obj.add_geometry_listener(self.update_geometry_fields)
            self._geometry_listener_added = True
        if self.F is not None and self.J is not None:
            return

        self.F_physical = self.forms.build_residual()

        self.F_physical_form = self.forms.compiled(
            "F_physical", lambda: self.F_physical)

        self.F = self.F_physical

        self.J = self.compute_dRdu_mat()

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def enable_time_stepping(self, dual_time=False):
        """
        Add the BDF physical time term to the Newton forms (and, with
        dual_time, the pseudo-time term as well). Call before set_up_solver.
        Returns the model's TimeIntegrator.
        """
        if self.solver_is_set_up:
            raise RuntimeError("enable_time_stepping must be called before set_up_solver")
        self._time_stepping = {"dual_time": bool(dual_time)}
        return self._make_time_integrator()

    def _make_time_integrator(self):
        from dragonfly_sim.core.time_integration import TimeTerm, TimeIntegrator
        if self.time_integrator is None:
            physical = self._time_stepping is not None
            pseudo = self.use_ptc or (physical and self._time_stepping["dual_time"])
            if physical or pseudo:
                self.time_integrator = TimeIntegrator(self, TimeTerm(self, physical=physical, pseudo=pseudo))
        return self.time_integrator

    def newton_forms(self):
        """
        The residual/Jacobian pair Newton drives: model.F with the time
        terms the model uses (core/time_integration.py). Without PTC or time
        stepping this is (self.F, self.J), the plain steady solve.
        """
        integrator = self._make_time_integrator()
        if integrator is None:
            return self.F, self.J
        F = integrator.term.augment(self.F)
        return F, ufl.derivative(F, self.u_vec)

    def set_up_solver(self):
        if not self.solver_is_set_up:
            # PCFIELDSPLIT does not split BAIJ matrices by component, and
            # PCMG's Galerkin products (MatPtAP) do not take them either
            mat_type = "aij" if self.linear_preconditioner in ("schur", "mg") else "baij"
            F_newton, J_newton = self.newton_forms()
            if self.problem_factory is not None:
                self.problem = self.problem_factory(F_newton, J_newton)
            else:
                self.problem = NonlinearProblem_mod(F_newton, J_newton, self.u_vec,
                                                    jit_options={"cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"]},
                                                    mat_type=mat_type)
            self.solver = SNESNewtonSolver(self.mesh_obj.mesh.comm, self.problem)
            self.solver.convergence_criterion = "incremental"  # convergence is determined through the norm of the iterative solution update
            self.solver.rtol = 1e-6
            # Decoupled from rtol: with ||x|| ~ O(1e4), the old atol=1e-8
            # floor demanded ~1e-12 RELATIVE precision on ||dx|| -- far
            # tighter than rtol=1e-6 -- so Newton was burning its full
            # max_it budget chasing an unreachable absolute target on every
            # solve instead of stopping once rtol was satisfied. Setting
            # atol to a floor well below anything reachable makes rtol the
            # sole criterion carrying the Eisenstat-Walker forcing, as
            # intended.
            self.solver.atol = 1e-11
            self.solver.max_it = int(self.fom_newton_inner_max_it)

            self.solver.error_on_nonconvergence = False

            def limiter(solver, x, dx):
                ksp = solver.krylov_solver

                # Print solver summary (number of iterations, current residual norm, reason)
                PETSc.Sys.Print("nls_solve_ solve: {} iterations, residual norm {} [{}]".format(
                    ksp.getIterationNumber(), ksp.getResidualNorm(),
                    ksp_converged_reason_name(ksp.getConvergedReason())))
                dx_norm = dx.norm()
                PETSc.Sys.Print("x pre norm: {}".format(x.norm()))

                theta, initial_theta = self.positivity_step_length(x, dx)
                self.last_step_theta = min(self.last_step_theta, theta)

                # Print (density and pressure) positivity-preserving under-relaxation summary
                PETSc.Sys.Print("dx norm: {}; Initial theta: {}, theta applied: {}".format(dx_norm, initial_theta, theta))
                self._report_positivity_limiting_node(x, dx, theta, initial_theta)

                # The solver lands on x - dx, so the limiter's job is to
                # rescale dx in place; x itself must not be touched.
                dx.scale(theta)

            self.solver.set_step_limiter(limiter)

            ksp = self.solver.krylov_solver
            self.apply_krylov_solver_settings(ksp, monitor_convergence=False, max_it=300,
                                              rtol=self.krylov_rtol)
            if self.newton_ksp_options:
                opts = PETSc.Options()
                for key, value in self.newton_ksp_options.items():
                    opts[ksp.getOptionsPrefix() + key] = value
                ksp.setFromOptions()
            self.solver_is_set_up = True

            # Print Jacobian matrix block size
            PETSc.Sys.Print("Euler Jacobian matrix type: {}, block size: {}".format(
                self.solver.A.getType(), self.solver.A.getBlockSize()))

            # Print summary of various solver parameters
            rtol, _, _, ksp_max_it = ksp.getTolerances()
            PETSc.Sys.Print(
                "Euler forward KSP: type={}, pc={}, norm={}, pc_side={}, "
                "rtol={:.1e}, max_it={}".format(
                    ksp.getType(), ksp.getPC().getType(),
                    ksp_norm_type_name(ksp.getNormType()),
                    ksp_pc_side_name(ksp.getPCSide()), rtol, ksp_max_it))

    def positivity_step_length(self, x, dx, tau=1.0):
        """
        Step length for the update x - theta dx: (theta, cap), with the
        step-length cap = min((tau / (||dx|| / max(||x||, 1)))^0.5, 1) and theta
        <= cap the largest step keeping the material's positivity constraints
        (absolute pressure and temperature) in every cell, and changing them
        by at most max_relative_change; transported extra variables are not
        limited. Used by the Newton step limiter and by the
        reduced-order solve (core/reduced_order_model.py). Collective.
        """
        dx_norm = dx.norm()
        relative_dx = dx_norm / max(x.norm(), 1.)
        initial_theta = min((tau / relative_dx)**0.5, 1.0) if dx_norm > 1e-16 else 0.

        from dragonfly_sim.utils.solver_utils import compute_positivity_preserving_theta
        theta = compute_positivity_preserving_theta(
            self.mesh_obj.mesh.comm, x.array, dx.array, self.positivity_constraints(),
            initial_theta=initial_theta, block_size=self.n_state, max_change=self.max_relative_change)
        return theta, initial_theta

    def positivity_constraints(self):
        """[(state component, lower bound)] the Newton step limiter holds."""
        return self.material.positivity_constraints(self.n_mean)

    @contextlib.contextmanager
    def steady_residual(self):
        """
        Within this block the Newton problem (self.problem) assembles the
        steady residual and its Jacobian: the pseudo-time and physical time
        terms of core/time_integration.py are switched off (their Constants
        set to zero) and restored on exit. A no-op without time terms.
        """
        term = None if self.time_integrator is None else self.time_integrator.term
        if term is None:
            yield
            return
        saved = (float(term.inv_cfl.value), float(term.inv_dt.value))
        term.inv_cfl.value, term.inv_dt.value = 0.0, 0.0
        try:
            yield
        finally:
            term.inv_cfl.value, term.inv_dt.value = saved

    def state_scale_expressions(self):
        """
        Typical magnitudes of the state components: the free-stream dynamic
        pressure rho U^2, the speed U (per direction) and the temperature
        of the mean flow -- all 1 in the solver's units -- then the
        extra_scales plug-in's values for the extra variables (numbers or UFL
        expressions of Constants).
        """
        scales = [1.0] * self.n_mean
        if self.n_extra:
            if self.extra_scales is None:
                raise RuntimeError("{} extra variables but no extra_scales plug-in".format(self.n_extra))
            scales += list(self.extra_scales())
        # the auxiliary unknowns (the artificial viscosity sigma) are O(1)
        return scales + [1.0] * self.n_aux

    def state_scales(self):
        """state_scale_expressions evaluated to floats, one per state
        component. Collective (UFL expressions are evaluated by assembly)."""
        msh = self.mesh_obj.mesh
        dx = ufl.dx(domain=msh)
        volume = None
        values = []
        for s in self.state_scale_expressions():
            if isinstance(s, ufl.core.expr.Expr):
                if volume is None:
                    volume = msh.comm.allreduce(
                        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * dx)), op=MPI.SUM)
                s = msh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(s * dx)),
                                       op=MPI.SUM) / volume
            values.append(float(s))
        return values

    def apply_krylov_solver_settings(self, ksp, max_it=100, gmres_restart=None,
                                     monitor_convergence=True,
                                     rtol=1e-3, atol=None):
        """
        Configure `ksp` through the PETSc options DB.

        rtol/atol are parameters rather than literals because this method is
        shared by three very different solves. The DEFAULTS are the forward
        Euler Newton inner KSP's settings -- a loose rtol=1e-3 carrying the
        Eisenstat-Walker forcing (see set_up_solver's atol comment), and no
        atol at all, i.e. PETSc's 1e-50 floor. The adjoint and tangent solves
        pass the model's standardized linear-solve pair in instead.

        atol=None means "leave the key unset" and NOT "set zero": writing 0
        would be a real change from PETSc's 1e-50 default.

        Both of those call sites previously called ksp.setTolerances() AFTER
        this method, silently overriding the rtol written here -- so this
        method's rtol only ever governed the forward solve, which was
        readable from neither end. Passing tolerances in keeps one source of
        truth per KSP.
        """
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()

        if monitor_convergence:
            opts[f"{option_prefix}ksp_converged_reason"] = ""
            opts[f"{option_prefix}ksp_monitor"] = ""

        if self.direct_linear_solver:
            opts[f"{option_prefix}ksp_type"] = "preonly"
            opts[f"{option_prefix}pc_type"] = "lu"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            ksp.setFromOptions()
            return

        # Plain gmres, not fgmres: fgmres only earns its double vector
        # storage when the preconditioner itself varies iteration to
        # iteration (e.g. an inner Krylov solve run to a loose, changing
        # tolerance). The PC configured below (ASM+ILU) is fixed, so
        # fgmres was paying for flexibility nothing here uses.
        opts[f"{option_prefix}ksp_type"] = self.krylov_type
        opts[f"{option_prefix}ksp_max_it"] = "{}".format(max_it)
        opts[f"{option_prefix}ksp_rtol"] = rtol
        if atol is not None:
            opts[f"{option_prefix}ksp_atol"] = atol
        # Default restart length to max_it: with a fixed (non-flexible)
        # preconditioner there is no reason to discard the Krylov subspace
        # before the iteration budget itself is exhausted -- a restart
        # shorter than max_it just forces a mid-solve subspace discard for
        # no memory savings that matter at this scale. Callers that pass
        # gmres_restart explicitly (e.g. the adjoint solve, which pairs a
        # much larger max_it with a deliberately smaller restart to bound
        # memory) are unaffected.
        if gmres_restart is None:
            gmres_restart = max_it
        opts[f"{option_prefix}ksp_gmres_restart"] = "{}".format(gmres_restart)
        if self.linear_preconditioner == "schur":
            self._schur_fieldsplit_options(opts, option_prefix)
            sub = option_prefix + "fieldsplit_0_"         # ASM/ILU below acts on the velocity block
        elif self.linear_preconditioner == "asm":
            opts[f"{option_prefix}pc_type"] = "asm"
            sub = option_prefix
        elif self.linear_preconditioner == "mg":
            opts[f"{option_prefix}ksp_type"] = "fgmres"
            opts[f"{option_prefix}pc_type"] = "python"
            sub = None
        else:
            raise ValueError("linear_preconditioner must be 'asm', 'schur' or 'mg'")
        option_prefix_pc, option_prefix = option_prefix, sub or option_prefix
        opts[f"{option_prefix}pc_asm_overlap"] = "{}".format(self.asm_overlap)

        opts[f"{option_prefix}sub_pc_type"] = self.subdomain_solver
        # opts[f"{option_prefix}sub_pc_factor_levels"] = "{}".format(int((self.poly_order + 1) ** self.dimensions))
        opts[f"{option_prefix}sub_pc_factor_levels"] = "{}".format(self.ilu_levels)
        # ILU(k)'s fill pattern is computed purely from the local
        # block's sparsity GRAPH (level-of-fill is symbolic, not
        # value-dependent), so reordering the unknowns before
        # factoring can reduce the fill actually generated at a given
        # k without changing accuracy -- a free lever, unlike k
        # itself which is now pinned between "stalls" (k=3, see
        # terminal_output_3Dtest_ILU_fillin=3.txt) and "OOM" (k=
        # dimensions+2) in 3D. Was previously scaffolded but unused
        # ("natural", i.e. no reordering, per that log's ksp_view).
        # (nested dissection for exact factorizations, which RCM would fill)
        opts[f"{option_prefix}sub_pc_factor_mat_ordering_type"] = "rcm" if self.subdomain_solver == "ilu" else "nd"
        option_prefix = option_prefix_pc

        # the options database is global and the prefix is shared by every KSP of that name:
        # clear a setting an earlier KSP made (diagonal scaling cannot work on a matrix-free operator)
        for name in ("ksp_diagonal_scale", "ksp_diagonal_scale_fix"):
            if opts.hasName(option_prefix + name):
                opts.delValue(option_prefix + name)
        if self.krylov_diagonal_scale:
            opts[f"{option_prefix}ksp_diagonal_scale"] = ""
            opts[f"{option_prefix}ksp_diagonal_scale_fix"] = ""

        ksp.setFromOptions()
        if self.linear_preconditioner == "mg" and not self.direct_linear_solver:
            ksp.getPC().setPythonContext(self._hp_multigrid())

    def _hp_multigrid(self):
        """The 'mg' preconditioner's context (see linear_preconditioner)."""
        from dragonfly_sim.utils.line_multigrid import HPMultigridPC
        msh = self.mesh_obj.mesh
        tdim = msh.topology.dim
        spaces = [self.functionspaces["V"]] + [
            dolfinx.fem.functionspace(msh, ("DG", q, (self.n_state,))) for q in range(self.poly_order - 1, -1, -1)]
        interp = []
        for fine, coarse in zip(spaces[:-1], spaces[1:]):
            P = dolfinx.fem.petsc.interpolation_matrix(coarse, fine)
            P.assemble()
            interp.append(P)
        n_cells = msh.topology.index_map(tdim).size_local
        centroids = dolfinx.mesh.compute_midpoints(msh, tdim, np.arange(n_cells, dtype=np.int32))[:, :tdim]
        return HPMultigridPC(interp, self.n_state, [V.dofmap.dof_layout.num_dofs for V in spaces], centroids,
                             **self.multigrid_options)

    def _schur_fieldsplit_options(self, opts, prefix):
        """The 'schur' preconditioner (see linear_preconditioner): fields by
        component of the blocked state, p' (component 0) against the rest."""
        others = ",".join(str(k) for k in range(1, self.n_state))
        for key, value in {
                "pc_type": "fieldsplit",
                "pc_fieldsplit_block_size": self.n_state,
                "pc_fieldsplit_0_fields": others,
                "pc_fieldsplit_1_fields": "0",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "upper",
                "pc_fieldsplit_schur_precondition": "selfp",
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "asm",
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "hypre",
                "fieldsplit_1_pc_hypre_type": "boomeramg"}.items():
            opts[prefix + key] = value

    def _assemble_physical_residual(self):
        """
        Re-assemble F_physical into self.residual_vec_physical and return its
        norm. Derived fields (e.g. a reconstructed gradient) are brought up
        to date with the current state first.
        """
        self.update_derived_fields()
        with self.residual_vec_physical.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.residual_vec_physical, self.F_physical_form)
        self.residual_vec_physical.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return self.residual_vec_physical.norm()

    def relative_residual_norm(self, full_norm):
        """
        eta = ||r|| / ||u||, the dimensionless residual measure.

        Absolute residual norms are not comparable across problems: they
        scale with sqrt(n_dofs) and with the magnitude of the conserved
        variables, so a threshold calibrated on one mesh is meaningless on
        the next. Dividing by the current ||u|| removes both, which is what
        makes a single tolerance transferable between the 2D and 3D cases.

        Collective (Vec.norm). Returns nan rather than raising when u_vec
        has blown up, so a diverged solve still prints a number.
        """
        u_norm = self.u_vec.x.petsc_vec.norm()
        if not np.isfinite(u_norm) or u_norm == 0.0:
            return np.nan
        return full_norm / u_norm

    def linearized_solver(self, transpose=False, prefix="lin_", rtol=1e-10,
                          atol=1e-11, max_it=1000, gmres_restart=200, release_operands=False,
                          matrix_free=False, ksp_options=None):
        """
        The KSP of dR/dU x = rhs (or its transpose, for the adjoint) at the
        current state, for models whose dR/dU is a condensed product
        (has_condensed_jacobian). FGMRES on the exact operator, right
        preconditioned with ASM/ILU of the compact A_UU part -- the forward
        solve's compact_pc pairing -- or MUMPS with direct_linear_solver.
        The transposes are formed explicitly. With release_operands and
        transpose, the condensed dR/dU (and its blocks) is freed once the
        transposes exist, before the preconditioner is set up: only the
        transposes are used from then on (release_solver). The KSP holds its
        operators and, after its first solve, the preconditioner's factors, so
        it solves any number of right-hand sides at this state; the caller
        destroys it. Collective.

        matrix_free (with transpose): the operator is the python Mat of
        TransposedCondensedOperator, A_UU^T x + B^T (A_UG^T x), which never
        forms A or the product A_UG B; its context owns the matrices and is
        destroyed with destroy_adjoint_operator(ksp).
        """
        if transpose and matrix_free:
            return self._matrix_free_adjoint_solver(prefix, rtol, atol, max_it, gmres_restart, ksp_options)
        A = self.assemble_dRdu_mat()
        P = self._dRdu_condensed.A_UU
        if transpose:
            # out of place: Mat.transpose() without an argument transposes the
            # cached matrices themselves
            A, P = A.transpose(PETSc.Mat()), P.transpose(PETSc.Mat())
            if release_operands:
                self._release_condensed_jacobian()
                self._return_freed_memory()
        ksp = PETSc.KSP().create(self.mesh_obj.mesh.comm)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(A, A if self.direct_linear_solver else P)
        saved = (self.krylov_type, self.krylov_diagonal_scale)
        self.krylov_type, self.krylov_diagonal_scale = "fgmres", False
        try:
            self.apply_krylov_solver_settings(ksp, max_it=max_it, gmres_restart=gmres_restart,
                                              monitor_convergence=False, rtol=rtol, atol=atol)
        finally:
            self.krylov_type, self.krylov_diagonal_scale = saved
        self._apply_ksp_options(ksp, ksp_options)
        return ksp

    # PETSc option keys written by _apply_ksp_options: the options database is global, so they are
    # removed again when the next KSP is configured instead of leaking into it
    _applied_ksp_option_keys = set()

    @classmethod
    def _apply_ksp_options(cls, ksp, ksp_options):
        """Override KSP settings through the PETSc options DB, after the model's own:
        a dict of option names without the prefix, e.g. {"ksp_type": "dgmres",
        "ksp_gmres_restart": 400}. The overrides last until the next KSP is configured here
        (the preconditioner reads its options at the first solve), then are deleted. An option
        the PETSc build does not know (a KSP type it was not built with) raises a PETSc error."""
        opts = PETSc.Options()
        for key in cls._applied_ksp_option_keys:
            opts.delValue(key)
        cls._applied_ksp_option_keys = set()
        if not ksp_options:
            return
        prefix = ksp.getOptionsPrefix()
        for key, value in ksp_options.items():
            opts[prefix + key] = value
            cls._applied_ksp_option_keys.add(prefix + key)
        ksp.setFromOptions()

    def _matrix_free_adjoint_solver(self, prefix, rtol, atol, max_it, gmres_restart, ksp_options=None):
        from dragonfly_sim.utils.gradient_condensation import TransposedCondensedOperator
        self.update_derived_fields()
        rec = self.reconstruction
        forms = rec.compact_forms(self.F)
        A_UU = dolfinx.fem.petsc.create_matrix(forms[0])
        A_UG = dolfinx.fem.petsc.create_matrix(forms[1])
        for M, a in ((A_UU, forms[0]), (A_UG, forms[1])):
            M.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(M, a)
            M.assemble()
        P = A_UU.transpose(PETSc.Mat())
        A_UU.destroy()
        gradient = rec.gradient
        B = gradient.state_jacobian()
        ctx = TransposedCondensedOperator(P, A_UG, B, gradient.discard)
        A = ctx.create_mat(self.mesh_obj.mesh.comm)
        self._return_freed_memory()
        ksp = PETSc.KSP().create(self.mesh_obj.mesh.comm)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(A, A if self.direct_linear_solver else P)
        saved = (self.krylov_type, self.krylov_diagonal_scale)
        self.krylov_type, self.krylov_diagonal_scale = "fgmres", False
        try:
            self.apply_krylov_solver_settings(ksp, max_it=max_it, gmres_restart=gmres_restart,
                                              monitor_convergence=False, rtol=rtol, atol=atol)
        finally:
            self.krylov_type, self.krylov_diagonal_scale = saved
        self._apply_ksp_options(ksp, ksp_options)
        return ksp

    @staticmethod
    def destroy_adjoint_operator(ksp):
        """Destroy a KSP of linearized_solver and, for a matrix-free one, what
        its operator's context owns."""
        A = ksp.getOperators()[0]
        ksp.destroy()
        if A.getType() == PETSc.Mat.Type.PYTHON:
            A.destroy()                  # PETSc calls the context's destroy(mat)

    def release_derived_caches(self):
        """Free what the derived fields cache between calls (Jacobians kept
        while the state, alpha and mesh are unchanged). Collective."""
        for f in self.derived_fields:
            f.release_cache()

    def solve_linearized(self, rhs, out, **kwargs):
        """Solve dR/dU x = rhs once (see linearized_solver for the arguments):
        returns the KSP (iterations, residual norm and reason are read off it)."""
        ksp = self.linearized_solver(**kwargs)
        ksp.solve(rhs, out)
        return ksp

    def _report_max_residual_location(self):
        """
        Print the magnitude, solution component and physical coordinate of
        the largest entry in the current residual_vec_physical.

        Collective: every rank participates in the allgather, and the
        lookup that decides which rank owns the max dof is derived from a
        collective Vec.max(), so all ranks agree on the answer by
        construction.
        """
        PETSc.Sys.Print("max physical residual entry: {}".format(
            self.residual_vec_physical.max()[1]))
        PETSc.Sys.Print("max physical residual index: {}".format(
            self.residual_vec_physical.max()[0]))

        max_residual_global_idx, max_residual_val = self.residual_vec_physical.max()

        bs = self.functionspaces["V"].dofmap.index_map_bs
        block_global_idx = max_residual_global_idx // bs
        component = max_residual_global_idx % bs

        block_local_idx = self.functionspaces["V"].dofmap.index_map.global_to_local(
            np.array([block_global_idx], dtype=np.int32))[0]

        comm = self.mesh_obj.mesh.comm
        if block_local_idx >= 0:
            dof_coords = self.functionspaces["V"].tabulate_dof_coordinates()
            payload = (comm.rank, dof_coords[block_local_idx, :].tolist(),
                       int(component), max_residual_val)
        else:
            payload = None

        owner = next(p for p in comm.allgather(payload) if p is not None)
        if comm.rank == 0:
            rank, coord, component, val = owner
            PETSc.Sys.Print(
                f"max residual {val:.6e} on rank {rank}, component {component}, coord {coord}")

    def _report_positivity_limiting_node(self, x, dx, theta, initial_theta,
                                         min_theta=1e-6):
        """
        Print which node bounded the Newton step length.
        """
        if not self.report_positivity_limiter or theta >= initial_theta:
            return

        comm = self.mesh_obj.mesh.comm

        if theta > 0.0:
            theta_probe = 2.0*theta
        else:
            # theta == 0 means the bisection fell BELOW min_theta and gave up.
            # The smallest step it actually tested is the last halving still
            # >= min_theta, which is strictly larger than min_theta itself
            # (1.907e-6 for the defaults, not 1e-6). Probing at min_theta
            # would evaluate a step the limiter never rejected, and then
            # truthfully report finding no violator -- which reads as a bug in
            # the diagnostic rather than as the deadlock it actually is.
            theta_probe = initial_theta
            while theta_probe * 0.5 >= min_theta:
                theta_probe *= 0.5

        constraints = self.positivity_constraints()
        names = ["pressure" if k == 0 else "temperature" if k == self.n_mean - 1
                 else "x[{}]".format(k) for k, _ in constraints]
        local = locate_positivity_limiting_nodes(
            x.array, dx.array, theta_probe, constraints, block_size=self.n_state, names=names)
        counts = {name: comm.allreduce(c, op=MPI.SUM) for name, c in local["counts"].items()}

        # Same allgather-and-pick idiom as _report_max_residual_location. The
        # payload is a handful of floats, mpi4py returns it in rank order, and
        # `min` is stable, so every rank selects the same node.
        payload = None
        if local["block"] is not None:
            dof_coords = self.functionspaces["V"].tabulate_dof_coordinates()
            payload = (local["severity"], comm.rank,
                       dof_coords[local["block"], :].tolist(), local)
        gathered = [p for p in comm.allgather(payload) if p is not None]

        PETSc.Sys.Print(
            "[theta-limit] theta {:.6e} (step-length cap {:.6e}); {} node(s) block "
            "the step at theta = {:.6e} -- {}".format(
                theta, initial_theta, sum(counts.values()), theta_probe,
                ", ".join("{} {}".format(c, name) for name, c in counts.items())))

        if not gathered:
            # theta was cut, yet nothing violates at twice the accepted value.
            # Not reachable from the bisection alone, so it means x or dx moved
            # between the limiter call and this one.
            PETSc.Sys.Print(
                "[theta-limit]   no violating node found at the probe step -- "
                "state or update changed since theta was computed")
            return

        _, rank, coord, info = min(gathered, key=lambda p: p[0])
        PETSc.Sys.Print(
            "[theta-limit]   worst: rank {}, local block {}, coord {}, binding "
            "constraint: {}".format(rank, info["block"], coord, info["constraint"]))
        PETSc.Sys.Print(
            "[theta-limit]   value {:.6e} -> trial {:.6e} (update {:.6e})".format(
                info["value"], info["trial"], info["update"]))

    def incoming_state_converged(self):
        """
        True, with solver_converged set, if the current u_vec already
        satisfies residual_conv_limit. Needs only the residual form, not the
        solver: the caller can skip set_up_solver and solve_system. The
        incoming state routinely is converged, since every evaluation that
        re-solves at an unchanged design arrives here converged, and a full
        Newton step would walk it backwards. Leaves the residual in
        residual_vec_physical either way. Collective.
        """
        self._ensure_residual_vector()
        residual_norm = self._assemble_physical_residual()
        self.solver_converged = bool(residual_norm < self.residual_conv_limit)
        if self.solver_converged:
            PETSc.Sys.Print(
                "Incoming state already converged ({:.6e} < {:.1e}); "
                "skipping the solve.".format(residual_norm, self.residual_conv_limit))
            self.u_vec.x.scatter_forward()
        return self.solver_converged

    def solve_system(self):
        """
        Newton solve of the Euler system, starting from the current u_vec.

        SNESNewtonSolver runs in passes of fom_newton_inner_max_it steps, and
        the physical residual is tested after each pass, until it drops below
        residual_conv_limit or the max_newton_iterations budget is spent.
        With use_ptc it is instead a pseudo-transient continuation solve
        (TimeIntegrator.solve_steady), max_newton_iterations pseudo steps of
        one Newton step each.
        Returns the physical residual vector (residual_vec_physical).
        """
        if self.use_ptc:
            return self.time_integrator.solve_steady()

        if self.incoming_state_converged():
            return self.residual_vec_physical

        solver = self.solver

        conv_limit = self.residual_conv_limit
        # FOM: max_newton_iterations is a TOTAL Newton-step budget, and each
        # outer pass now runs fom_newton_inner_max_it steps, so divide to hold
        # the product fixed -- otherwise raising the inner count silently
        # multiplies the budget and any gain is confounded with 5x the work.
        max_outer = (max(1, int(self.max_newton_iterations)
                              // max(1, int(self.fom_newton_inner_max_it))))

        residual_norm = self.residual_vec_physical.norm()
        self.solver_converged = False

        for _ in range(max_outer):
            solver.solve(self.u_vec)
            self.u_vec.x.scatter_forward()
            residual_norm = self._assemble_physical_residual()
            self._report_max_residual_location()
            if not np.isfinite(residual_norm):
                PETSc.Sys.Print("Non-finite residual; stopping.")
                break
            if residual_norm < conv_limit:
                self.solver_converged = True
                break
        else:
            PETSc.Sys.Print(
                "WARNING: {} Newton iterations exhausted ({} passes x {}) "
                "at {:.6e}".format(
                    max_outer * (int(self.fom_newton_inner_max_it)),
                    max_outer,
                    int(self.fom_newton_inner_max_it),
                    residual_norm))

        self.u_vec.x.scatter_forward()
        return self.residual_vec_physical

    def set_u_vec(self, u_vec_entries):
        # set `self.u_vec` to the input linearization point

        set_petsc_vec_array(self.u_vec.x.petsc_vec, u_vec_entries)
        self.u_vec.x.petsc_vec.ghostUpdate()

    def _ensure_residual_vector(self):
        if not hasattr(self, 'residual_vec_physical'):
            self.residual_vec_physical = dolfinx.fem.petsc.create_vector(
                dolfinx.fem.extract_function_spaces(self.F_physical_form))

    # ------------------------------------------------------------------
    # Derived fields (utils/derived_fields.py)
    # ------------------------------------------------------------------
    def update_derived_fields(self):
        """Bring every derived field up to date with the current state.
        Collective; a no-op without derived fields."""
        for f in self.derived_fields:
            f.update()

    def update_geometry_fields(self):
        """Refresh what depends on the mesh coordinates alone (cell centres,
        wall distance, the reconstruction's cell volumes). Runs automatically
        after every Mesh.apply_node_motions / reset_nodes once the model has
        derived fields (see compute_weakform). A no-op for the plain Euler
        model."""
        for f in self.derived_fields:
            f.update_geometry()

    @property
    def _chain(self):
        if getattr(self, "_chain_rule", None) is None:
            from dragonfly_sim.utils.derived_fields import DerivedFieldChainRule
            self._chain_rule = DerivedFieldChainRule(
                self.mesh_obj, self.u_vec, self.derived_fields,
                alpha_var=getattr(self, "alpha_var", None),
                jit_options={"cffi_extra_compile_args": ["-O3", "-march=native", "-ffast-math"]})
        return self._chain_rule

    def release_solver(self):
        """
        Free the forward solver (SNES, KSP and its preconditioner), the
        Newton problem's matrices, and the cached condensed dR/dU of
        assemble_dRdu_mat. set_up_solver and assemble_dRdu_mat rebuild them
        when next needed. For memory-bound runs, e.g. a 3D RANS shape
        optimization: the forward solve's matrices and the adjoint's then
        never coexist. Collective.
        """
        if self.solver_is_set_up:
            self.solver.destroy()
            condensed = getattr(self.problem, "_condensed", None)
            if condensed is not None:
                self._destroy_condensed(condensed)
            self.problem._A.destroy()
            self.problem._b.destroy()
            self.problem = None
            self.solver = None
            self.solver_is_set_up = False
        self._release_condensed_jacobian()
        self._return_freed_memory()

    def _return_freed_memory(self):
        """Collect garbage, flush petsc4py's queue of dropped parallel objects
        and hand freed heap back to the OS (glibc malloc_trim). Rebuilding the
        multi-GB 3D RANS matrices on every solve otherwise fragments the heap,
        and the resident size creeps up ~2 GB per optimization iteration."""
        gc.collect()
        PETSc.garbage_cleanup(self.mesh_obj.mesh.comm)
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass                                  # not glibc

    @staticmethod
    def _destroy_condensed(condensed):
        # the matrices a CondensedJacobian owns; _inv_mass is the reconstruction's
        for name in ("A_UU", "A_UG", "B_GU", "_C", "A"):
            getattr(condensed, name).destroy()

    def _release_condensed_jacobian(self):
        """Free assemble_dRdu_mat's cached condensed dR/dU; rebuilt when next needed."""
        if hasattr(self, "_dRdu_condensed"):
            self._destroy_condensed(self._dRdu_condensed)
            del self._dRdu_condensed

    @property
    def has_condensed_jacobian(self):
        """True when dR/dU is a condensed product (a derived field depends on
        U), which the default ASM/ILU-on-dR/dU solves cannot precondition;
        the linearized solves then go through solve_linearized."""
        return any(f.depends_on_state for f in self.derived_fields)

    @staticmethod
    def _form_key(form):
        """Cache key of a form: its signature AND the identity of every
        coefficient and constant in it. The signature alone is the same for
        two forms that differ only in which Constant they hold (e.g. the lift
        and drag directions), and a compiled form is bound to its objects."""
        return (form.signature(), tuple(id(c) for c in form.coefficients()),
                tuple(id(c) for c in form.constants()))

    def state_derivative(self, form):
        """dJ/dU of a scalar functional J (rank-0 UFL form), through the
        derived fields: a Vec in the state layout."""
        self.update_derived_fields()
        return self._chain.functional_state(form, self._form_key(form))

    def alpha_derivative(self, form):
        """dJ/dalpha of a scalar functional J at fixed U: its explicit alpha
        dependence plus that of the derived fields (e.g. a reconstructed
        gradient reading the alpha-dependent far-field states). A number.
        Collective."""
        self.update_derived_fields()
        return self._chain.functional_alpha(form, self._form_key(form))

    def mesh_derivative(self, form):
        """dJ/dx of a scalar functional J through the derived fields: a Vec on
        the mesh's coordinate_space."""
        self.update_derived_fields()
        return self._chain.functional_mesh(form, self._form_key(form))

    def compute_dRdu_mat(self, u_vec_entries=None):
        """
        The forward Newton Jacobian, dR/du, as a UFL expression.
        """
        if u_vec_entries is not None:
            self.set_u_vec(u_vec_entries)
        return ufl.derivative(self.F, self.u_vec)

    def assemble_dRdu_mat(self, u_vec_entries=None):
        """
        The forward Newton Jacobian, assembled.

        Caches its own form/matrix, mirroring compute_dRdxmesh_mat's
        convention. With a reconstructed gradient this is the exact
        condensed dR/dU of the steady residual (an aij Mat).
        """
        if u_vec_entries is not None:
            self.set_u_vec(u_vec_entries)

        if self.has_condensed_jacobian:
            self.update_derived_fields()
            if not hasattr(self, "_dRdu_condensed"):
                self._dRdu_condensed = self.reconstruction.condensed_jacobian(self.F)
            return self._dRdu_condensed.assemble()

        if not hasattr(self, "dRdu_form"):
            self.dRdu_form = dolfinx.fem.form(self.compute_dRdu_mat())
            self.dRdu_mat = dolfinx.fem.petsc.create_matrix(self.dRdu_form)

        self.dRdu_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.dRdu_mat, self.dRdu_form)
        self.dRdu_mat.assemble()

        return self.dRdu_mat

    def compute_dRdxmesh_mat(self, u_vec_entries, V):
        self.set_u_vec(u_vec_entries)

        if self.derived_fields:
            # dR/dx through the derived fields, as a matrix-free operator
            # (mult / multTranspose); V must be the mesh's coordinate_space
            self.update_derived_fields()
            return self._chain.residual_mesh_operator(self.F, "F")

        if not hasattr(self, 'dRdxmesh_form'):
            # test out derivative calculation
            spatial_coordinate = ufl.SpatialCoordinate(self.mesh_obj.mesh)

            args = self.F.arguments()
            # # UFL arguments need unique indices within a form
            n = max(a.number() for a in args) if args else -1
            dX = ufl.Argument(V, n + 1)

            a_deriv = ufl.derivative(self.F, spatial_coordinate, dX)
            self.dRdxmesh_form = dolfinx.fem.form(a_deriv)
            self.dRdxmesh_mat = dolfinx.fem.petsc.create_matrix(self.dRdxmesh_form)

        self.dRdxmesh_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.dRdxmesh_mat, self.dRdxmesh_form)
        self.dRdxmesh_mat.assemble()

        return self.dRdxmesh_mat
