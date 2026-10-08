import ctypes
import gc

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
from dragonfly_sim.utils.Euler_utils import primitives, pressure, energy_density, flux, subsonic_inflow_state, subsonic_outflow_state, slip_wall_state, hll_flux, boundary_flux
from dragonfly_sim.utils.NS_utils import (mean_flow, concat_state, extended_flux, extended_normal_flux,
                                          extended_hll_flux, extended_boundary_flux)
from dragonfly_sim.utils.farfield_utils import riemann_farfield_state
from dragonfly_sim.utils.filewriter import FileWriter


class CompressibleEulerModel():
    """
    The p = 0 DG (cell-centred finite-volume) compressible Euler model, and
    the host every other flow model in the package is built on.

    State: the conservative mean flow (rho, rho u, rho E), n_mean = 2 + dim
    components per cell, optionally followed by `extra_variables`
    transported scalars (n_state components in all). The scalars are
    convected with the mean flow (NS_utils.extended_flux) and are not
    positivity-limited. With no extra variables every form below is the
    plain Euler one.

    Plug-in points. CompressibleRANSModel fills these; for the plain Euler
    model they are all empty or None, and the residual, Jacobian and
    derivative forms are the same expressions as without them.

        face_traces         () -> (U+, U-) interior-facet traces of the
                            inviscid flux (MUSCL reconstruction)
        residual_terms      callables () -> UFL form added to the residual
        boundary_terms      callables (U_b, tag, kind) -> UFL form or None,
                            added to every boundary condition
        boundary_scalars    (kind, U, inflow) -> list of the extra variables'
                            boundary values (kind: inflow, outflow, slip,
                            farfield, wall); required when n_extra > 0
        freestream_scalars  () -> list of the extra variables' free-stream values
        wall_bc             (tag) -> None, replaces the slip wall in define_wall_bc
        wall_traction       (U, n) -> traction added to p n in wall_force_density
        time_scale_terms    callables (U) -> added to the PTC inverse time scale
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
    def __init__(self, mesh, poly_order=0, gamma=1.4, flux_function=hll_flux, extra_variables=()):  # llf_flux  # hll_flux
        if poly_order > 0:
            raise ValueError("This release only support 0th-order DG bases for now")

        self.mesh_obj = mesh
        self.poly_order = poly_order
        self.gamma = gamma
        self.dimensions = self.mesh_obj.mesh._ufl_domain._geometric_dimension

        self.flux_function = flux_function

        # State layout: mean flow, then the transported scalars
        self.extra_variables = tuple(extra_variables)
        self.n_mean = 2 + self.dimensions
        self.n_extra = len(self.extra_variables)
        self.n_state = self.n_mean + self.n_extra
        if self.n_extra and flux_function is not hll_flux:
            raise NotImplementedError(
                "Extra transported variables need the HLL flux (NS_utils.extended_hll_flux); "
                "the other Riemann fluxes have no extended-state version.")

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
        self.derived_fields = []
        self.problem_factory = None
        self.extra_scales = None

        # Time integration (core/time_integration.py). time_integrator is
        # created by set_up_solver when use_ptc is set or by
        # enable_time_stepping(); ptc_settings are CFLController keywords.
        self.use_ptc = False
        self.ptc_settings = {}
        self.time_integrator = None
        self._time_stepping = None
        # Smallest positivity step length applied since the caller last
        # reset it; the PTC CFL law reads it.
        self.last_step_theta = 1.0

        # Krylov solver of the Newton steps (see apply_krylov_solver_settings);
        # direct_linear_solver switches every linear solve to MUMPS.
        self.krylov_type = "gmres"
        self.krylov_diagonal_scale = True
        self.direct_linear_solver = False

        # Instantiate the FormManager
        self.forms = FormManager(mesh, model=self,
                                 measure_source=mesh.form_manager)

        self.functionspaces = None
        self.bcs = []
        self.boundary_conditions = None
        self.initial_conditions = None
        self.F = None
        self.J = None
        self.solver_is_set_up = False
        self.solver_converged = False


        # Initial angle of attack at which the boundary conditions are constructed;
        # most notable the inlet/outlet split. This is important because changing 
        # the boundary conditions would impose a discontinuity in optimization
        # problems. Therefore we have to keep track of large angle of attack changes 
        self.alpha_tagging = None
        self.alpha_tagging_tol = np.radians(1.0)


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

        self.fom_solution_writer = None
        self._output_fields = []
        # False until define_sol_export_files actually opens the writers, which
        # keeps write_solution_output inert for a model that never opened them
        self.export_solutions = False


    def output_field_expressions(self):
        """The fields of the solution file, as (name, value shape, UFL
        expression of u_vec): the conservative state as separate fields --
        density rho, momentum rho*u (a vector), energy rho*E, and each
        transported scalar under its own name (e.g. rho_nu_tilde) -- then the
        velocity u (a vector) and the pressure p."""
        dim = self.dimensions
        U = self.mean_flow(self.u_vec)
        momentum = ufl.as_vector([U[1 + i] for i in range(dim)])
        fields = [("density", (), U[0]),
                  ("momentum", (dim,), momentum),
                  ("energy", (), U[1 + dim])]
        fields += [(name, (), self.u_vec[self.n_mean + k])
                   for k, name in enumerate(self.extra_variables)]
        fields += [("velocity", (dim,), momentum / U[0]),
                   ("pressure", (), pressure(U, self.gamma))]
        return fields

    def define_sol_export_files(self, func_space=None, file_name_addendum=None, mesh_deformation=False):
        """Open FOM_solution_<suffix>.bp, one file holding every output field
        (output_field_expressions) as a separate, named field; the mesh is
        stored once per frame for all of them. Collective.

        It is the ALIGNED series: exactly one frame per design evaluation,
        written by write_solution_output() at eval_idx, so step k here is step
        k in mesh_deformation_<suffix>.bp. It carries the accepted solution --
        see DG_windtunnel_model.solve_residual_equations. mesh_deformation=True
        writes each frame on the deformed mesh it was solved on. func_space is
        unused (the fields define their own spaces); it is kept for callers.
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
            "FOM_solution_{}".format(file_name_addendum), self.mesh_obj.mesh.comm,
            {name: f.function_space for name, f, _ in self._output_fields},
            mesh_deformation=mesh_deformation)
        self.export_solutions = True

    def _write_output_fields(self, write_counter):
        """Interpolate the output fields off the current u_vec and write one
        frame. u_vec must already hold the state being exported. Collective."""
        for _, f, expr in self._output_fields:
            f.interpolate(expr)
        self.fom_solution_writer.interpolate_and_write(
            {name: f for name, f, _ in self._output_fields}, write_counter=write_counter)

    def write_solution_output(self, write_counter, solution_array=None):
        """Write one aligned output frame at index `write_counter`.

        Called once per design evaluation from
        DG_windtunnel_model.solve_residual_equations with write_counter =
        eval_idx, so FOM_solution_<suffix>.bp steps in lockstep with
        mesh_deformation_<suffix>.bp.

        `solution_array` is the accepted solution of the evaluation, passed
        explicitly rather than read off self.u_vec. Passing None writes
        whatever u_vec currently holds.
        """
        if not self.export_solutions:
            return

        if solution_array is not None:
            self.interpolate_solution_vector(solution_array)

        self._write_output_fields(write_counter)

    def define_elements_functionspaces(self):
        # Throughout this class we need access to various function spaces; this is where we define them
        self.functionspaces = {
            "V": dolfinx.fem.functionspace(self.mesh_obj.mesh, ("DG", self.poly_order, (self.n_state,))),
            "V_scalar": dolfinx.fem.functionspace(self.mesh_obj.mesh, ("DG", self.poly_order)),
        }

    def define_trial_testfunctions(self):
        # Define the trial and test functions for the weak form
        self.u_vec = dolfinx.fem.Function(self.functionspaces["V"], name="u")
        self.v_vec = ufl.TestFunction(self.functionspaces["V"])

    def define_inlet_outlet_conditions(self, bc_dict):
        # Define the inlet and outlet flow conditions.
        # NOTE: This explicitly encodes subsonic boundary conditions; 
        #       supersonic in- and outflows are currently not supported, but can be
        #       implemented with a few small changes
        rho = bc_dict['inlet']['rho']  # inlet mass density
        p = bc_dict['inlet']['p']  # inlet pressure
        M = bc_dict['inlet']['M']  # inlet Mach number
        alpha = bc_dict['inlet']['alpha']  # initial inlet angle of attack

        c = abs(self.gamma*p/rho)**0.5  # speed of sound

        # Since angle of attack is a design parameter we define it as a UFL variable;
        # this allows us to use Dolfin's automatic differentiation for it
        self.alpha_const = dolfinx.fem.Constant(self.mesh_obj.mesh, float(alpha))
        self.alpha_var = ufl.variable(self.alpha_const)

        inflow_dir = self._inflow_direction_ufl()

        # inflow and outflow directions
        n_in = dolfinx.fem.Constant(
            self.mesh_obj.mesh, -self._inflow_direction_numpy(alpha))
        n_out = dolfinx.fem.Constant(
            self.mesh_obj.mesh, self._inflow_direction_numpy(alpha))

        # defining the reference and inlet velocity so 
        # we can update it later if Mach number gets modified
        self.u_ref_const = dolfinx.fem.Constant(self.mesh_obj.mesh, M*c)
        u_in = self.u_ref_const * inflow_dir

        # define a dictionary with all of the inlet and outlet conditions
        self.boundary_conditions = {'inlet': {'rho': rho,
                                            'p': p,
                                            'M': M,
                                            'alpha': float(alpha),
                                            'inflow_angle': self._inflow_direction_numpy(alpha),
                                            'c': c,
                                            'n': n_in,  # outward-pointing normal vector
                                            'u': u_in,
                                            'u_ref': self.u_ref_const
                                            },
                                    'outlet': {'p': p,
                                               'n': n_out  # outward-pointing normal vector
                                               }
                                 }

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

        if getattr(self, 'alpha_tagging', None) is not None:
            # we verify whether the angle of attack is changed by a large enough increment
            # to change the inflow/outflow boundary division
            if abs(alpha - self.alpha_tagging) > self.alpha_tagging_tol:
                raise ValueError(
                    "alpha = {:.6f} rad ({:.3f} deg) is more than {:.6f} rad from the "
                    "alpha the inlet/outlet facet split was tagged at ({:.6f} rad, "
                    "{:.3f} deg). The split is frozen at tagging time, so this would "
                    "silently mis-classify far-field facets. Re-tag the mesh or tighten "
                    "the design-variable bounds.".format(
                        alpha, np.degrees(alpha), self.alpha_tagging_tol,
                        self.alpha_tagging, np.degrees(self.alpha_tagging)))

        self.alpha_const.value = alpha

        inflow_angle = self._inflow_direction_numpy(alpha)
        self.boundary_conditions['inlet']['alpha'] = alpha
        self.boundary_conditions['inlet']['inflow_angle'] = inflow_angle
        self.boundary_conditions['inlet']['n'].value = -inflow_angle
        self.boundary_conditions['outlet']['n'].value = inflow_angle

    def set_mach(self, M):
        # propagate the new Mach number `M` to all model dependencies
        M = float(M)
        c = self.boundary_conditions['inlet']['c']
        self.u_ref_const.value = M * c
        self.boundary_conditions['inlet']['M'] = M

    def compute_initial_conditions_from_inlet(self):
        # The initial guess used in the Newton solver. Here we use the inlet flow.
        inlet_conditions = self.boundary_conditions['inlet']
        rho_in = dolfinx.fem.Constant(self.mesh_obj.mesh, inlet_conditions['rho'])
        rhoE_in_guess = energy_density(inlet_conditions['p'], rho_in, inlet_conditions['u'], self.gamma)        
        self.initial_conditions = ufl.as_vector((rho_in, *[rho_in*inlet_conditions['u'][i] for i in range(self.dimensions)], rhoE_in_guess))
        if self.n_extra:
            # plain numbers become Constants on this mesh, so the initial
            # condition stays interpolable
            extras = [v if isinstance(v, ufl.core.expr.Expr)
                      else dolfinx.fem.Constant(self.mesh_obj.mesh, float(v))
                      for v in self._freestream_scalars()]
            self.initial_conditions = concat_state(self.initial_conditions, extras)

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
        """The mean-flow slice (rho, rho u, rho E) of a state expression."""
        return mean_flow(U, self.n_mean)

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
        Weakly impose the boundary state U_b on ds(meshtag) and register it.

        inviscid_flux="llf" is the one-sided Lax-Friedrichs boundary flux
        against U_b (Euler_utils.boundary_flux); "exact" is the physical flux
        of U_b itself, for states that already carry the upwinding (the
        characteristic far field) or that must have no mass flux (no-slip
        wall). The boundary_terms plug-ins add their terms (e.g. viscous
        fluxes) to the same integral.
        """
        U, n = self.u_vec, self.mesh_obj.n
        if inviscid_flux == "llf":
            if self.n_extra:
                f = extended_boundary_flux(U, U_b, n, self.gamma, self.n_mean)
            else:
                f = boundary_flux(U, U_b, n, self.gamma)
        elif inviscid_flux == "exact":
            f = extended_normal_flux(U_b, n, self.gamma, self.n_mean)
        else:
            raise ValueError("inviscid_flux must be 'llf' or 'exact'")
        F = ufl.inner(f, self.v_vec) * self.forms.ds(meshtag)
        for term in self.boundary_terms:
            extra = term(U_b, meshtag, kind)
            if extra is not None:
                F = F + extra
        self.bcs += [F]
        self.boundary_states.append((U_b, meshtag, kind))

    def define_subsonic_inflow_bc(self, meshtag):
        # define the weak form term for natural enforcement of the inflow boundary condition 
        inlet_conditions = self.boundary_conditions['inlet']
        inflow = subsonic_inflow_state(self.mean_flow(self.u_vec), inlet_conditions['rho'], inlet_conditions['u'], self.gamma)
        self.add_boundary_condition(meshtag, self._boundary_state(inflow, "inflow"), "inflow")

    def define_subsonic_outflow_bc(self, meshtag):
        # define the weak form term for natural enforcement of the outflow boundary condition
        outlet_conditions = self.boundary_conditions['outlet']
        outflow = subsonic_outflow_state(self.mean_flow(self.u_vec), outlet_conditions['p'], self.gamma)
        self.add_boundary_condition(meshtag, self._boundary_state(outflow, "outflow"), "outflow")

    def define_slipwall_bc(self, meshtag):
        # define the weak form term for natural enforcement of the slip wall boundary condition
        u_wall = slip_wall_state(self.mean_flow(self.u_vec), self.mesh_obj.n)
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
        Characteristic (Riemann-invariant) far field on ds(meshtag), meant
        for the whole outer boundary in place of the inflow/outflow split.
        Outgoing acoustic waves leave the domain (exactly at normal incidence)
        instead of reflecting off a fixed state. The inviscid flux is the
        physical flux of the characteristic state, which already carries the
        upwinding; extra variables take their free-stream values where the
        flow enters and the interior values where it leaves.

        `transverse` (core/farfield.py:TransverseFarfield) adds the
        transverse correction to the incoming invariant: the 'riemann2' far
        field of unsteady runs.

        The state depends on alpha through the free-stream velocity, so
        dR/dalpha follows automatically.
        """
        inlet = self.boundary_conditions['inlet']
        R_in = None if transverse is None else transverse.R_in
        U_b, inflow = riemann_farfield_state(self.mean_flow(self.u_vec), self.mesh_obj.n,
                                             inlet['rho'], inlet['u'], inlet['p'], self.gamma,
                                             R_in=R_in)
        self.add_boundary_condition(meshtag, self._boundary_state(U_b, "farfield", inflow),
                                    "farfield", inviscid_flux="exact")
        if transverse is not None:
            self.transverse_farfield = transverse

    def compute_interior_integral_terms(self):
        if not self.n_extra and self.face_traces is None and not self.residual_terms:
            # the plain Euler residual, written exactly as it always was
            F_vol = ufl.inner(ufl.grad(self.v_vec), flux(self.u_vec, self.gamma)) * self.forms.dx

            F_int = ufl.inner(
                self.flux_function(self.u_vec('+'), self.u_vec('-'), self.mesh_obj.n('+'), self.gamma),
                self.v_vec('+') - self.v_vec('-')) * self.forms.dS
            return -F_vol + F_int

        U, v, n = self.u_vec, self.v_vec, self.mesh_obj.n
        F = -ufl.inner(ufl.grad(v), extended_flux(U, self.gamma, self.n_mean)) * self.forms.dx
        U_p, U_m = self.face_traces() if self.face_traces is not None else (U('+'), U('-'))
        if self.n_extra:
            F_num = extended_hll_flux(U_p, U_m, n('+'), self.gamma, self.n_mean)
        else:
            F_num = self.flux_function(U_p, U_m, n('+'), self.gamma)
        F += ufl.inner(F_num, v('+') - v('-')) * self.forms.dS
        for term in self.residual_terms:
            F += term()
        return F

    def _lambda_max_expr(self, U):
        """The largest wave speed |u| + c of the state U (mean flow), with rho
        and p floored as in Euler_utils.signed_wave_speeds."""
        U = self.mean_flow(U)
        rho, u_vel, _ = primitives(U)
        p = pressure(U, self.gamma)
        rho_safe = ufl.max_value(rho, 1e-8)
        p_safe = ufl.max_value(p, 1e-8)
        c = ufl.sqrt(self.gamma * p_safe / rho_safe)
        return ufl.sqrt(ufl.dot(u_vel, u_vel)) + c

    def wall_force_density(self, U, n):
        """Force per unit wall area on the body: pressure, plus the
        wall_traction plug-in's viscous traction when there is one."""
        f = pressure(self.mean_flow(U), self.gamma) * n
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
        define_inlet_outlet_conditions for why alpha_var must be the single
        stored instance.

        Sparse in practice: alpha only enters the residual through u_in in the
        subsonic inflow BC, so the result is supported on inlet-adjacent dofs
        alone. The interior terms, the outflow BC (which prescribes p) and the
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
            mat_type = "baij"
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
            self.apply_krylov_solver_settings(ksp, monitor_convergence=False, max_it=300)
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
        <= cap the largest step keeping density and pressure positive in every
        cell (transported extra variables are not limited). Used by the Newton
        step limiter and by the reduced-order solve
        (core/reduced_order_model.py). Collective.
        """
        dx_norm = dx.norm()
        relative_dx = dx_norm / max(x.norm(), 1.)
        initial_theta = min((tau / relative_dx)**0.5, 1.0) if dx_norm > 1e-16 else 0.

        from dragonfly_sim.utils.solver_utils import compute_positivity_preserving_theta
        theta = compute_positivity_preserving_theta(
            self.mesh_obj.mesh.comm, x.array, dx.array, gamma=self.gamma, initial_theta=initial_theta,
            block_size=self.n_state, n_mean=self.n_mean if self.n_extra else None)
        return theta, initial_theta

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
        Typical magnitudes of the state components: the free-stream
        [rho, rho U (per direction), p/(gamma - 1) + rho U^2/2] of the mean
        flow, then the extra_scales plug-in's values for the extra variables
        (numbers or UFL expressions of Constants).
        """
        inlet = self.boundary_conditions['inlet']
        rho, p = inlet['rho'], inlet['p']
        speed = inlet['M'] * inlet['c']
        scales = ([rho] + [rho * speed] * self.dimensions
                  + [p / (self.gamma - 1.0) + 0.5 * rho * speed**2])
        if self.n_extra:
            if self.extra_scales is None:
                raise RuntimeError("{} extra variables but no extra_scales plug-in".format(self.n_extra))
            scales += list(self.extra_scales())
        return scales

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
        opts[f"{option_prefix}pc_type"] = "asm"
        opts[f"{option_prefix}pc_asm_overlap"] = "{}".format(self.asm_overlap)

        opts[f"{option_prefix}sub_pc_type"] = "ilu"
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
        opts[f"{option_prefix}sub_pc_factor_mat_ordering_type"] = "rcm"

        if self.krylov_diagonal_scale:
            opts[f"{option_prefix}ksp_diagonal_scale"] = ""
            opts[f"{option_prefix}ksp_diagonal_scale_fix"] = ""

        ksp.setFromOptions()

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

    def solve_linearized(self, rhs, out, transpose=False, prefix="lin_", rtol=1e-10,
                         atol=1e-11, max_it=1000, gmres_restart=200, release_operands=False):
        """
        Solve dR/dU x = rhs (or its transpose, for the adjoint) at the
        current state, for models whose dR/dU is a condensed product
        (has_condensed_jacobian). FGMRES on the exact operator, right
        preconditioned with ASM/ILU of the compact A_UU part -- the forward
        solve's compact_pc pairing -- or MUMPS with direct_linear_solver.
        The transposes are formed explicitly. With release_operands and
        transpose, the condensed dR/dU (and its blocks) is freed once the
        transposes exist, before the preconditioner is set up: only the
        transposes are used from then on (release_solver). Returns the KSP
        (iterations, residual norm and reason are read off it).
        """
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

        local = locate_positivity_limiting_nodes(
            x.array, dx.array, theta_probe, gamma=self.gamma,
            block_size=self.n_state, n_mean=self.n_mean if self.n_extra else None)

        n_density = comm.allreduce(local["n_density"], op=MPI.SUM)
        n_pressure = comm.allreduce(local["n_pressure"], op=MPI.SUM)
        n_nonfinite = comm.allreduce(local["n_nonfinite"], op=MPI.SUM)

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
            "the step at theta = {:.6e} -- {} density, {} pressure, {} non-finite".format(
                theta, initial_theta, n_density + n_pressure + n_nonfinite,
                theta_probe, n_density, n_pressure, n_nonfinite))

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
            "[theta-limit]   state rho = {:.6e}, p = {:.6e}  ->  trial rho = {:.6e}, "
            "p = {:.6e}".format(
                info["rho"], info["p"], info["rho_trial"], info["p_trial"]))
        PETSc.Sys.Print(
            "[theta-limit]   update d_rho = {:.6e}, |d_rhou| = {:.6e}, "
            "d_rhoE = {:.6e}".format(
                info["d_rho"], info["d_rhou_norm"], info["d_rhoE"]))

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

        self._ensure_residual_vector()

        solver = self.solver

        conv_limit = (self.residual_conv_limit)
        # FOM: max_newton_iterations is a TOTAL Newton-step budget, and each
        # outer pass now runs fom_newton_inner_max_it steps, so divide to hold
        # the product fixed -- otherwise raising the inner count silently
        # multiplies the budget and any gain is confounded with 5x the work.
        max_outer = (max(1, int(self.max_newton_iterations)
                              // max(1, int(self.fom_newton_inner_max_it))))

        residual_norm = self._assemble_physical_residual()
        self.solver_converged = False

        # The incoming state may ALREADY satisfy conv_limit, and routinely
        # does: every evaluation that re-solves at an unchanged design arrives
        # here converged. Without this test the loop still runs a full Newton
        # step and can walk a converged state backwards.
        if residual_norm < conv_limit:
            self.solver_converged = True
            PETSc.Sys.Print(
                "Incoming state already converged ({:.6e} < {:.1e}); "
                "skipping the solve.".format(residual_norm, conv_limit))
            self.u_vec.x.scatter_forward()
            return self.residual_vec_physical

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
