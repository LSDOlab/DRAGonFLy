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
from dragonfly_sim.utils.Euler_utils import pressure, energy_density, flux, subsonic_inflow_state, subsonic_outflow_state, slip_wall_state, hll_flux, boundary_flux
from dragonfly_sim.utils.filewriter import FileWriter


class CompressibleEulerModel():
    """

    """
    def __init__(self, mesh, poly_order=0, gamma=1.4, flux_function=hll_flux):  # llf_flux  # hll_flux
        if poly_order > 0:
            raise ValueError("This release only support 0th-order DG bases for now")

        self.mesh_obj = mesh
        self.poly_order = poly_order
        self.gamma = gamma
        self.dimensions = self.mesh_obj.mesh._ufl_domain._geometric_dimension

        self.flux_function = flux_function

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

        self.fom_pressure_writer = None
        self.pressure_func = None
        # False until define_sol_export_files actually opens the writers, which
        # keeps write_solution_output inert for a model that never opened them
        self.export_solutions = False


    def define_sol_export_files(self, func_space, file_name_addendum=None):
        # FOM_solution_<suffix>.bp is the ALIGNED series: exactly one frame per
        # design evaluation, written by write_solution_output() at eval_idx, so
        # step k here is step k in mesh_deformation_<suffix>.bp. It carries the
        # accepted solution -- see DG_windtunnel_model.solve_residual_equations.
        self.fom_solution_writer = FileWriter("FOM_solution_{}".format(file_name_addendum), self.mesh_obj.mesh.comm, func_space)
        self.fom_pressure_writer = FileWriter("FOM_pressure_{}".format(file_name_addendum), self.mesh_obj.mesh.comm, self.functionspaces["V_scalar"])
        self.pressure_func = dolfinx.fem.Function(self.functionspaces["V_scalar"], name="pressure")
        self.export_solutions = True

    def _write_derived_fields(self, solution_writer, pressure_writer, write_counter):
        """Interpolate the derived fields off the current u_vec and write them.

        u_vec must already hold the state being exported.
        """
        p_expr = dolfinx.fem.Expression(pressure(self.u_vec, self.gamma), self.functionspaces["V_scalar"].element.interpolation_points)
        self.pressure_func.interpolate(p_expr)

        solution_writer.interpolate_and_write(self.u_vec, write_counter=write_counter)
        pressure_writer.interpolate_and_write(self.pressure_func, write_counter=write_counter)

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

        self._write_derived_fields(
            self.fom_solution_writer, self.fom_pressure_writer, write_counter)

    def define_elements_functionspaces(self):
        # Throughout this class we need access to various function spaces; this is where we define them
        self.functionspaces = {
            "V": dolfinx.fem.functionspace(self.mesh_obj.mesh, ("DG", self.poly_order, (2 + self.dimensions,))),
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

    def interpolate_solution_vector(self, interpolant=None):
        if interpolant is None:
            interpolant = self.initial_conditions

        if isinstance(interpolant, (np.ndarray, PETSc.Vec)):
            set_petsc_vec_array(self.u_vec.x.petsc_vec, interpolant)
        else:
            for j in range(interpolant.ufl_shape[0]):
                self.u_vec.sub(j).interpolate(
                    dolfinx.fem.Expression(interpolant[j], self.functionspaces['V'].sub(j).element.interpolation_points))

        self.u_vec.x.scatter_forward()

    def define_subsonic_inflow_bc(self, meshtag):
        # define the weak form term for natural enforcement of the inflow boundary condition 
        inlet_conditions = self.boundary_conditions['inlet']
        inflow = subsonic_inflow_state(self.u_vec, inlet_conditions['rho'], inlet_conditions['u'], self.gamma)

        flux_in = boundary_flux(self.u_vec, inflow, self.mesh_obj.n, self.gamma)
        F_in = ufl.inner(flux_in, self.v_vec) * self.forms.ds(meshtag)

        self.bcs += [F_in]

    def define_subsonic_outflow_bc(self, meshtag):
        # define the weak form term for natural enforcement of the outflow boundary condition
        outlet_conditions = self.boundary_conditions['outlet']
        # outflow = dolfin_dg.aero.subsonic_outflow(outlet_conditions['p'], self.u_vec, self.gamma)
        outflow = subsonic_outflow_state(self.u_vec, outlet_conditions['p'], self.gamma)

        flux_out = boundary_flux(self.u_vec, outflow, self.mesh_obj.n, self.gamma)
        F_out = ufl.inner(flux_out, self.v_vec) * self.forms.ds(meshtag)

        self.bcs += [F_out]

    def define_slipwall_bc(self, meshtag):
        # define the weak form term for natural enforcement of the slip wall boundary condition
        u_wall = slip_wall_state(self.u_vec, self.mesh_obj.n)

        flux_wall = boundary_flux(self.u_vec, u_wall, self.mesh_obj.n, self.gamma)
        F_wall = ufl.inner(flux_wall, self.v_vec) * self.forms.ds(meshtag)
        self.bcs += [F_wall]

    def compute_interior_integral_terms(self):
        F_vol = ufl.inner(ufl.grad(self.v_vec), flux(self.u_vec, self.gamma)) * self.forms.dx

        F_int = ufl.inner(
            self.flux_function(self.u_vec('+'), self.u_vec('-'), self.mesh_obj.n('+'), self.gamma),
            self.v_vec('+') - self.v_vec('-')) * self.forms.dS
        return -F_vol + F_int

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
        if self.F is not None and self.J is not None:
            return

        self.F_physical = self.forms.build_residual()

        self.F_physical_form = self.forms.compiled(
            "F_physical", lambda: self.F_physical)

        self.F = self.F_physical

        self.J = self.compute_dRdu_mat()

    def set_up_solver(self):
        if not self.solver_is_set_up:
            mat_type = "baij"
            self.problem = NonlinearProblem_mod(self.F, self.J, self.u_vec,
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
                tau = 1.0
                eps = 1e-16
                ksp = solver.krylov_solver

                # Print solver summary (number of iterations, current residual norm, reason)
                PETSc.Sys.Print("nls_solve_ solve: {} iterations, residual norm {} [{}]".format(
                    ksp.getIterationNumber(), ksp.getResidualNorm(),
                    ksp_converged_reason_name(ksp.getConvergedReason())))
                dx_norm = dx.norm()
                x_norm = x.norm()
                PETSc.Sys.Print("x pre norm: {}".format(x_norm))

                relative_dx = dx_norm / max(x_norm, 1.)
                if dx_norm > eps:
                    initial_theta = min((tau/relative_dx)**0.5, 1.0)
                else:
                    initial_theta = 0.

                from dragonfly_sim.utils.solver_utils import compute_positivity_preserving_theta
                bs = 2 + self.dimensions

                theta = compute_positivity_preserving_theta(self.mesh_obj.mesh.comm, x.array, dx.array, gamma=self.gamma, initial_theta=initial_theta, block_size=bs)

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

        # Plain gmres, not fgmres: fgmres only earns its double vector
        # storage when the preconditioner itself varies iteration to
        # iteration (e.g. an inner Krylov solve run to a loose, changing
        # tolerance). The PC configured below (ASM+ILU) is fixed, so
        # fgmres was paying for flexibility nothing here uses.
        opts[f"{option_prefix}ksp_type"] = "gmres"
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

        opts[f"{option_prefix}ksp_diagonal_scale"] = ""
        opts[f"{option_prefix}ksp_diagonal_scale_fix"] = ""

        ksp.setFromOptions()

    def _assemble_physical_residual(self):
        """
        Re-assemble F_physical into self.residual_vec_physical and return its
        norm.
        """
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
            block_size=2 + self.dimensions)

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
        Returns the physical residual vector (residual_vec_physical).
        """
        if not hasattr(self, 'residual_vec_physical'):
            self.residual_vec_physical = dolfinx.fem.petsc.create_vector(
                dolfinx.fem.extract_function_spaces(self.F_physical_form))

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
        convention.
        """
        if u_vec_entries is not None:
            self.set_u_vec(u_vec_entries)

        if not hasattr(self, "dRdu_form"):
            self.dRdu_form = dolfinx.fem.form(self.compute_dRdu_mat())
            self.dRdu_mat = dolfinx.fem.petsc.create_matrix(self.dRdu_form)

        self.dRdu_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.dRdu_mat, self.dRdu_form)
        self.dRdu_mat.assemble()

        return self.dRdu_mat

    def compute_dRdxmesh_mat(self, u_vec_entries, V):
        self.set_u_vec(u_vec_entries)

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
