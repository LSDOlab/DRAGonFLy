import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx

import csdl_alpha as csdl

from dragonfly_sim.utils.petsc_utils import set_petsc_vec_array
from dragonfly_sim.utils.postprocessor_utils import moment_integrand
from dragonfly_sim.utils.solver_utils import ksp_converged_reason_name
from dragonfly_sim.utils.Euler_utils import pressure


class DG_postprocessor(csdl.CustomExplicitOperation):
    ordered_callbacks = True    # as DG_windtunnel_model: collectives and the shared mesh

    def __init__(self, Mesh_obj, sim_model, wall_meshtag_val, aero_center=None,
                 ref_area_dim=None, ref_chord_dim=None, testing_mode=False,
                 linear_solver_rtol=1e-16, linear_solver_atol=1e-11,
                 linear_solver_max_it=1000, linear_residual_accept_limit=1e-10,
                 p_inf_dim=None, L_ref_dim=1.0):
        super().__init__()

        self.testing_mode = testing_mode

        self.mesh_obj = Mesh_obj
        self.sim_model = sim_model
        self.linear_solver_rtol = linear_solver_rtol
        self.linear_solver_atol = linear_solver_atol
        self.linear_solver_max_it = linear_solver_max_it
        self.linear_residual_accept_limit = linear_residual_accept_limit
        # c_m_alpha is MONITORED ONLY -- no objective or constraint uses it,
        # and nothing supplies its derivative. It nevertheless costs a tangent
        # solve per evaluation and RAISES when that solve degrades, which is
        # enough to abort a long sweep over a quantity nobody consumed. Set
        # False to skip it and report nan.
        self.log_cm_alpha = True
        self.wall_meshtag_val = wall_meshtag_val
        if aero_center is None:
            # quarter chord of a unit chord at the origin
            aero_center = np.array([0.25] + [0.0] * (sim_model.dimensions - 1))
        self.aero_center = dolfinx.fem.Constant(self.mesh_obj.mesh, np.asarray(aero_center, dtype=np.double))

        # Reference quantities
        self.ref_area_dim = ref_area_dim  # reference area (if None, set to 1 in 2D and computed planform area in 3D)
        self.ref_chord_dim = ref_chord_dim  # reference chord length (if None, set to 1)
        self.p_inf_dim = p_inf_dim  # dimensionalized freestream pressure, used for computing physical forces
        self.L_ref_dim = L_ref_dim  # dimensionalized length scale, used for computing physical forces

        # Compute quantities that are constant over the course of an optimization
        self.compute_initial_and_constant_quantities()


    def compute_initial_and_constant_quantities(self):
        # This function computes quantities that are unchanged during an optimization,
        # as well as some quantities that need to be initialized here and are modified later
        n = self.sim_model.dimensions
        if self.p_inf_dim is not None and self.L_ref_dim is not None:
            self.force_scale = float(self.p_inf_dim) * float(self.L_ref_dim) ** (n - 1)
            self.moment_scale = float(self.p_inf_dim) * float(self.L_ref_dim) ** n
        else:
            self.force_scale = 1.0
            self.moment_scale = 1.0

        rho_in = dolfinx.fem.Constant(self.mesh_obj.mesh, self.sim_model.boundary_conditions['inlet']['rho'])
        u_ref = self.sim_model.boundary_conditions['inlet']['u_ref']

        self._S_ref_const = dolfinx.fem.Constant(self.mesh_obj.mesh, self._compute_S_ref())
        self._chord_ref_const = dolfinx.fem.Constant(self.mesh_obj.mesh, self._compute_chord_ref(self._S_ref_const.value))

        self.C_infty = 0.5*rho_in*(u_ref**2)*self._S_ref_const
        self.C_infty_moment = self.C_infty * self._chord_ref_const

        # Drag and lift vector directions
        self.psi_drag = self.sim_model.boundary_conditions['outlet']['n']
        self.psi_lift = dolfinx.fem.Constant(
            self.mesh_obj.mesh, np.zeros((self.sim_model.dimensions,), dtype=np.double))

        self.set_angle_of_attack(self.sim_model.boundary_conditions['inlet']['alpha'])

    def _compute_S_ref(self):
        # Compute the reference area or chord length for force, moment coefficient calculation
        if self.ref_area_dim is not None:
            # Skip the calculation, use the user-defined value instead
            return self.ref_area_dim
        if self.sim_model.dimensions == 2:
            return 1.0
        elif self.sim_model.dimensions == 3:
            # S_ref is equal to 0.5 times the wing planform area, calculated as
            # the wing surface projected onto the z=0 plane
            S_ref = 0.5 * self.mesh_obj.mesh.comm.allreduce(
                dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                    abs(self.mesh_obj.n[2]) * self.mesh_obj.form_manager.ds(self.wall_meshtag_val))), MPI.SUM)
            PETSc.Sys.Print("Reference (planform) area: {}".format(S_ref))
            return S_ref

    def _compute_chord_ref(self, S_ref_value):
        # Compute the reference chord length for moment coefficient calculation
        if self.ref_chord_dim is not None:
            # Skip the calculation, use the user-defined value instead
            return self.ref_chord_dim
        if self.sim_model.dimensions == 2:
            return 1.0
        elif self.sim_model.dimensions == 3:
            # Calculate the mean geometric chord of the current geometry
            bbox = self.mesh_obj.compute_wall_bounding_box(self.wall_meshtag_val)
            span = bbox[1][1] - bbox[1][0]
            return S_ref_value / span

    def refresh_geometry_dependent_reference_quantities(self):
        # Recompute the reference area and chord length after a geometry update
        self._S_ref_const.value = self._compute_S_ref()
        self._chord_ref_const.value = self._compute_chord_ref(self._S_ref_const.value)

    def set_angle_of_attack(self, alpha):
        # update the angle of attack, and propagate the update to the lift vector direction
        # NOTE: The drag vector direction is handled differently, and is propagated automatically
        self.sim_model.set_angle_of_attack(alpha)

        inflow_vec = self.sim_model.boundary_conditions['inlet']['inflow_angle']
        if self.sim_model.dimensions == 2:
            lift_unit_vec = np.array([-inflow_vec[1], inflow_vec[0]], dtype=np.double)
        elif self.sim_model.dimensions == 3:
            # wing pitches about the spanwise (y) axis, so the lift direction
            # is the cross product of the inflow and y-direction
            zero_direction_vec = np.zeros((3,))
            zero_direction_vec[1] = 1.
            lift_unit_vec = np.cross(inflow_vec, zero_direction_vec)

        self.psi_lift.value = lift_unit_vec

    def _pressure_force_vector(self, u_dof_vec):
        """The force density on the wall: pressure, plus the viscous traction
        for a viscous model (sim_model.wall_force_density)."""
        self.sim_model.set_u_vec(u_dof_vec)
        if self.sim_model.derived_fields:
            # the viscous traction reads the reconstructed gradient
            self.sim_model.update_derived_fields()
            return self.sim_model.wall_force_density(self.sim_model.u_vec, self.mesh_obj.n)
        p = pressure(self.sim_model.u_vec, self.sim_model.gamma)
        return p*self.mesh_obj.n

    def _assemble_value_and_coefficient(self, form, normalizer):
        raw_val = self.mesh_obj.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), MPI.SUM)
        coeff_val = self.mesh_obj.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form((1.0/normalizer)*form)), MPI.SUM)
        return raw_val, coeff_val

    def _assemble_state_derivative(self, form, label):
        if self.sim_model.derived_fields:
            # total derivative through the reconstructed gradient
            vec = self.sim_model.state_derivative(form)
            PETSc.Sys.Print("{} norm: {}".format(label, vec.norm()))
            out = vec.getArray().copy()
            vec.destroy()
            return out
        d_duvec = ufl.derivative(form, self.sim_model.u_vec)
        d_duvec_vec = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(d_duvec))
        d_duvec_vec.assemble()

        PETSc.Sys.Print("{} norm: {}".format(label, d_duvec_vec.norm()))

        output_vec = d_duvec_vec.getArray().copy()
        d_duvec_vec.destroy()
        return output_vec

    def _assemble_mesh_derivative(self, form, label):
        if self.sim_model.derived_fields:
            # total derivative through the derived fields (gradient, cell
            # centres, wall distance)
            dK = self.sim_model.mesh_derivative(form)
            PETSc.Sys.Print("{} norm: {}".format(label, dK.norm()))
            d_dmesh = dK.getArray().copy()
            dK.destroy()
            return (self.mesh_obj.dof_to_node_map.T @ d_dmesh.reshape(-1, self.mesh_obj.mesh.geometry.dim)).flatten()

        spatial_coordinate = ufl.SpatialCoordinate(self.mesh_obj.mesh)

        args = form.arguments()
        n = max(a.number() for a in args) if args else -1
        dX = ufl.Argument(self.mesh_obj.coordinate_space, n + 1)

        a_deriv_form = dolfinx.fem.form(ufl.derivative(form, spatial_coordinate, dX))

        dK = dolfinx.fem.petsc.assemble_vector(a_deriv_form)
        dK.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        PETSc.Sys.Print("{} norm: {}".format(label, dK.norm()))

        d_dmesh = dK.getArray().copy()
        dK.destroy()

        d_dmesh_shape_1_dim = self.mesh_obj.dof_to_node_map.T@d_dmesh.reshape(-1, self.mesh_obj.mesh.geometry.dim)
        return d_dmesh_shape_1_dim.flatten()

    def compute_force_in_direction(self, u_dof_vec, direction_vec):
        force_density = self._pressure_force_vector(u_dof_vec)
        return ufl.dot(direction_vec, force_density)*self.mesh_obj.form_manager.ds(self.wall_meshtag_val)

    def compute_cl(self, u_dof_vec):
        lift = self.compute_force_in_direction(u_dof_vec, self.psi_lift)
        return self._assemble_value_and_coefficient(lift, self.C_infty)

    def compute_L_deriv(self, u_dof_vec):
        lift = self.compute_force_in_direction(u_dof_vec, self.psi_lift)
        return self._assemble_state_derivative(lift, "dL_duvec")

    def compute_L_meshderiv(self, u_dof_vec):
        lift = self.compute_force_in_direction(u_dof_vec, self.psi_lift)
        return self._assemble_mesh_derivative(lift, "dL_dmesh")

    def compute_cd(self, u_dof_vec):
        drag = self.compute_force_in_direction(u_dof_vec, self.psi_drag)
        return self._assemble_value_and_coefficient(drag, self.C_infty)

    def compute_D_deriv(self, u_dof_vec):
        drag = self.compute_force_in_direction(u_dof_vec, self.psi_drag)
        return self._assemble_state_derivative(drag, "dD_duvec")

    def compute_D_meshderiv(self, u_dof_vec):
        drag = self.compute_force_in_direction(u_dof_vec, self.psi_drag)
        return self._assemble_mesh_derivative(drag, "dD_dmesh")

    def compute_moment_form(self, u_dof_vec):
        force_density = self._pressure_force_vector(u_dof_vec)
        coords = ufl.SpatialCoordinate(self.mesh_obj.mesh)
        return moment_integrand(force_density, coords - self.aero_center,
                                self.sim_model.dimensions) * self.mesh_obj.form_manager.ds(self.wall_meshtag_val)

    def compute_cm(self, u_dof_vec):
        moment = self.compute_moment_form(u_dof_vec)
        return self._assemble_value_and_coefficient(moment, self.C_infty_moment)

    def compute_cm_deriv(self, u_dof_vec):
        moment_coeff = (1.0/self.C_infty_moment)*self.compute_moment_form(u_dof_vec)
        return self._assemble_state_derivative(moment_coeff, "dCm_duvec")

    def compute_M_deriv(self, u_dof_vec):
        moment = self.compute_moment_form(u_dof_vec)
        return self._assemble_state_derivative(moment, "dM_duvec")

    def compute_cm_alpha(self, u_dof_vec):
        # Compute d cm / d alpha by solving a linear system;
        # NOTE: this quantity is not used in the optimization, and is merely present as a logging output
        local_size = self.sim_model.u_vec.x.petsc_vec.local_size

        if self.sim_model.has_condensed_jacobian:
            # condensed dR/dU: the model's own linearized solve
            self.sim_model.set_u_vec(u_dof_vec)
            dr_dalpha = self.sim_model.assemble_dRdalpha_vec()
            A = self.sim_model.assemble_dRdu_mat()
            rhs, sol = A.createVecRight(), A.createVecRight()
            set_petsc_vec_array(rhs, -dr_dalpha.getArray()[:local_size])
            ksp = self.sim_model.solve_linearized(
                rhs, sol, prefix="tangent_", rtol=self.linear_solver_rtol,
                atol=self.linear_solver_atol, max_it=self.linear_solver_max_it)
            PETSc.Sys.Print("tangent_ solve: {} iterations, residual norm {} [{}]".format(
                ksp.getIterationNumber(), ksp.getResidualNorm(),
                ksp_converged_reason_name(ksp.getConvergedReason())))
            ksp.destroy()
            dcm_du = self.compute_cm_deriv(u_dof_vec)
            local_contribution = float(np.dot(dcm_du, sol.getArray()[:local_size]))
            return self.mesh_obj.mesh.comm.allreduce(local_contribution, op=MPI.SUM)

        drdu_mat = self.sim_model.assemble_dRdu_mat(u_dof_vec)
        dr_dalpha = self.sim_model.assemble_dRdalpha_vec(u_dof_vec)

        if not hasattr(self, '_tangent_solver'):
            # build a tangent solver if it doesn't exist already
            self._tangent_solver = PETSc.KSP().create(self.mesh_obj.mesh.comm)
            self._tangent_solver.setOperators(drdu_mat)
            self._tangent_solver.setOptionsPrefix("tangent_")

            self.sim_model.apply_krylov_solver_settings(
                self._tangent_solver, max_it=self.linear_solver_max_it,
                gmres_restart=100,
                monitor_convergence=False,
                rtol=self.linear_solver_rtol,
                atol=self.linear_solver_atol)
            self._tangent_rhs = drdu_mat.createVecRight()
            self._tangent_sol = drdu_mat.createVecRight()

        set_petsc_vec_array(self._tangent_rhs, -dr_dalpha.getArray()[:local_size])

        self._tangent_solver.solve(self._tangent_rhs, self._tangent_sol)

        rnorm = self._tangent_solver.getResidualNorm()
        PETSc.Sys.Print("tangent_ solve: {} iterations, residual norm {} [{}]".format(
            self._tangent_solver.getIterationNumber(), rnorm,
            ksp_converged_reason_name(self._tangent_solver.getConvergedReason())))
        if rnorm > self.linear_residual_accept_limit:
            raise Warning(
                "compute_cm_alpha tangent solve residual norm is too high: "
                "{:.6e} > linear_residual_accept_limit {:.1e}".format(
                    rnorm, self.linear_residual_accept_limit))

        dcm_du = self.compute_cm_deriv(u_dof_vec)
        local_contribution = float(np.dot(dcm_du, self._tangent_sol.getArray()[:local_size]))
        return self.mesh_obj.mesh.comm.allreduce(local_contribution, op=MPI.SUM)

    def forces_and_coefficients(self, u_dof_vec=None, alpha=None):
        """
        Lift, drag and pitching moment and their coefficients of a flow state,
        outside any CSDL graph: no FFD block, mesh motion or recorder needed.

        u_dof_vec: the owned state entries (default: the model's current
        u_vec); alpha: angle of attack in radians (default: the model's). For a
        viscous model, D_friction / c_d_friction is the part of the drag from
        the viscous wall traction. Collective. Returns a dict.
        """
        if u_dof_vec is None:
            u_dof_vec = self.sim_model.u_vec.x.petsc_vec.getArray().copy()
        if alpha is not None:
            self.set_angle_of_attack(alpha)
        self.refresh_geometry_dependent_reference_quantities()
        L, c_l = self.compute_cl(u_dof_vec)
        D, c_d = self.compute_cd(u_dof_vec)
        M, c_m = self.compute_cm(u_dof_vec)
        out = dict(L=L, D=D, M=M, c_l=c_l, c_d=c_d, c_m=c_m)
        if getattr(self.sim_model, "wall_traction", None) is not None:
            # drag of the viscous traction alone
            self.sim_model.set_u_vec(u_dof_vec)
            self.sim_model.update_derived_fields()
            t = self.sim_model.wall_traction(self.sim_model.u_vec, self.mesh_obj.n)
            form = ufl.dot(self.psi_drag, t) * self.mesh_obj.form_manager.ds(self.wall_meshtag_val)
            out["D_friction"], out["c_d_friction"] = self._assemble_value_and_coefficient(form, self.C_infty)
        return out

    def compute_M_meshderiv(self, u_dof_vec):
        moment = self.compute_moment_form(u_dof_vec)
        return self._assemble_mesh_derivative(moment, "dM_dmesh")

    def evaluate(self, u_vec, mesh_deformation=None, alpha=None, shape_param_inputs=None):
        """Forces and coefficients of the state u_vec. mesh_deformation is the
        volume mesh motion of a shape optimization; leave it out (None) for a
        forward analysis on the fixed mesh. alpha is required."""
        if alpha is None:
            raise ValueError("evaluate needs alpha")
        # declare input quantities
        self.declare_input('u_vec', u_vec)
        if mesh_deformation is not None:
            self.declare_input('mesh_deformation', mesh_deformation)
        self.declare_input('alpha', alpha)
        if shape_param_inputs is not None:
            self.declare_input('shape_param_inputs', shape_param_inputs)

        # declare output variables
        c_l = self.create_output('c_l', (1,))
        c_d = self.create_output('c_d', (1,))
        c_m = self.create_output('c_m', (1,))
        c_m_alpha = self.create_output('c_m_alpha', (1,))
        L = self.create_output('L', (1,))
        D = self.create_output('D', (1,))
        M = self.create_output('M', (1,))

        # Force and moment coefficients (and derivatives) are only used as monitoring outputs;
        # to avoid CSDL from computing its derivatives we set these to zero
        for _input_name in ('u_vec', 'alpha') + (('mesh_deformation',) if mesh_deformation is not None else ()):
            for _output_name in ('c_l', 'c_d', 'c_m', 'c_m_alpha'):
                self.declare_derivative_parameters(of=_output_name, wrt=_input_name, dependent=False)
        if shape_param_inputs is not None:
            for _output_name in ('c_l', 'c_d', 'c_m', 'c_m_alpha', 'L', 'D', 'M'):
                self.declare_derivative_parameters(of=_output_name, wrt='shape_param_inputs',
                                                   dependent=False)

        # construct output of the model
        output = csdl.VariableGroup()
        output.c_l = c_l
        output.c_d = c_d
        output.c_m = c_m
        output.c_m_alpha = c_m_alpha
        output.L = L
        output.D = D
        output.M = M

        return output
    
    def compute(self, input_vals, output_vals):
        PETSc.Sys.Print("starting compute in DG_postprocessor...")

        u_vec = input_vals['u_vec']
        alpha = input_vals['alpha'][0]
        deform_array = input_vals['mesh_deformation'] if 'mesh_deformation' in input_vals else None

        # Update the angle of attack and downstream quantities
        self.set_angle_of_attack(alpha)

        # Apply the mesh deformations to the mesh (none in a forward analysis)
        if deform_array is not None:
            self.mesh_obj.apply_node_motions(deform_array)

        # check whether the mesh and solution on it are valid
        if (self.mesh_obj.is_valid and self.sim_model.last_solve_converged and not np.isnan(np.linalg.norm(u_vec))) or self.testing_mode:
            # update geometry-dependent quantities, under the influence of the deformed mesh
            self.refresh_geometry_dependent_reference_quantities()

            # compute all output quantities and print them
            lift_val, c_l = self.compute_cl(u_vec)
            output_vals['L'] = lift_val
            output_vals['c_l'] = c_l
            drag_val, c_d_initial = self.compute_cd(u_vec)
            output_vals['D'] = drag_val
            output_vals['c_d'] = c_d_initial #+ max(0., 10.*(0.285 - c_l))
            moment_val, output_vals['c_m'] = self.compute_cm(u_vec)
            output_vals['M'] = moment_val
            output_vals['c_m_alpha'] = (self.compute_cm_alpha(u_vec)
                                        if self.log_cm_alpha else np.nan)

            PETSc.Sys.Print("c_d: {}, c_l: {}, c_m: {}, c_m_alpha: {}".format(np.round(c_d_initial, 6), np.round(output_vals['c_l'], 6), np.round(output_vals['c_m'], 6), np.round(output_vals['c_m_alpha'], 6)))
            PETSc.Sys.Print("D: {}, L: {}, M: {} (nondimensional)".format(np.round(output_vals['D'], 6), np.round(output_vals['L'], 6), np.round(output_vals['M'], 6)))
            
            # print dimensional forces and moments, just for monitoring purposes
            PETSc.Sys.Print("D: {}, L: {}, M: {} (dimensional)".format(
                np.round(output_vals['D'] * self.force_scale, 6),
                np.round(output_vals['L'] * self.force_scale, 6),
                np.round(output_vals['M'] * self.moment_scale, 6)))
        else:
            PETSc.Sys.Print("Mesh or solution is not valid, setting outputs to high drag.")
            output_vals['L'], output_vals['c_l'] = 0.0, 0.0
            output_vals['D'], output_vals['c_d'] = 10., 1.0
            output_vals['M'], output_vals['c_m'] = 0.0, 0.0
            output_vals['c_m_alpha'] = 0.0

        # reset the mesh to its undeformed state
        self.mesh_obj.reset_nodes()

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        # Compute the derivatives of the output quantities w.r.t. the inputs
        PETSc.Sys.Print("starting compute_derivatives in DG_postprocessor...")

        u_vec = input_vals['u_vec']
        alpha = input_vals['alpha'][0]
        deform_array = input_vals['mesh_deformation'] if 'mesh_deformation' in input_vals else None

        # Update the angle of attack and downstream quantities
        self.set_angle_of_attack(alpha)

        # Apply the mesh deformations to the mesh and update geometry-dependent quantities
        if deform_array is not None:
            self.mesh_obj.apply_node_motions(deform_array)
        self.refresh_geometry_dependent_reference_quantities()

        # Compute all force and moment derivatives w.r.t. the nondimensional state and mesh vertex movement
        derivatives['L', 'u_vec'] = self.compute_L_deriv(u_vec)[None, :]
        derivatives['D', 'u_vec'] = self.compute_D_deriv(u_vec)[None, :]
        derivatives['M', 'u_vec'] = self.compute_M_deriv(u_vec)[None, :]

        if deform_array is not None:
            derivatives['L', 'mesh_deformation'] = self.compute_L_meshderiv(u_vec)[None, :]
            derivatives['D', 'mesh_deformation'] = self.compute_D_meshderiv(u_vec)[None, :]
            derivatives['M', 'mesh_deformation'] = self.compute_M_meshderiv(u_vec)[None, :]

        # Compute the force derivatives w.r.t. angle of attack changes; the arithmetic for this looks odd but the math checks out
        L_val, _ = self.compute_cl(u_vec)
        D_val, _ = self.compute_cd(u_vec)
        derivatives['L', 'alpha'] = np.array([[-D_val]], dtype=np.float64)
        derivatives['D', 'alpha'] = np.array([[ L_val]], dtype=np.float64)
        derivatives['M', 'alpha'] = np.array([[0.0]], dtype=np.float64)

        self.mesh_obj.reset_nodes()
