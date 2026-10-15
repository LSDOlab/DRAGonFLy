import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx

import csdl_alpha as csdl

from dragonfly_sim.utils.postprocessor_utils import moment_about
from dragonfly_sim.utils.Euler_utils import pressure


class MomentSpec:
    """One moment of the wall force: about `point`, along unit `direction`
    (a 3-vector in 2D too, where only the z entry is read), normalised in the
    coefficient by q*S_ref*ref_length. The values are dolfinx Constants, so the
    forms built on them stay valid when they change."""

    def __init__(self, mesh, name, point, direction, ref_length=None):
        self.name = name
        self.point = dolfinx.fem.Constant(mesh, point)
        self.direction = dolfinx.fem.Constant(mesh, direction)
        # None: the postprocessor's reference chord
        self.ref_length = None if ref_length is None else dolfinx.fem.Constant(mesh, float(ref_length))
        self.point_array = point
        self.direction_array = direction


class DG_postprocessor(csdl.CustomExplicitOperation):
    """
    Forces, and moments about user-given points and directions, of the wall
    force (pressure, plus the viscous traction for a viscous model), with their
    derivatives for CSDL.

    moments: sequence of (name, point, direction[, ref_length]) tuples, each
    giving the moment M = d . (r x F), r = X - point, of the wall force F along
    the direction d (right-hand rule); CSDL outputs M_<name> and c_m_<name>.
    point has one entry per dimension; direction is a nonzero vector,
    normalised to unit length, or the string "pitch": the nose-up-positive
    pitching moment, (0, 1, 0) in 3D (y spanwise) and (0, 0, -1) in 2D. In 2D
    only the out-of-plane direction (0, 0, +-1) exists. c_m_<name> is
    M / (q S_ref ref_length), the reference chord by default. The directions
    are fixed in the geometry axes and the points fixed in space, also while
    a shape design variable moves the wall. Only M_<name> is differentiated:
    S_ref and the reference chord change with the 3D geometry, so constrain
    M_<name>, not c_m_<name>. For a half-wing on a symmetry plane y = 0, the
    root bending moment is ("root_bending", (x0, 0, 0), (1, 0, 0)).

    Each moment costs its own value, state-derivative and mesh-derivative
    assembly (a chain-rule evaluation per form for derived-field models).
    Since M_k = d_k . int(X x f) - (d_k x x0_k) . int(f), all moments are
    linear combinations of 6 base functionals (3 in 2D), so only 6 derivative
    evaluations would suffice regardless of the number of moments; this is not
    yet exploited.
    """
    ordered_callbacks = True    # as DG_windtunnel_model: collectives and the shared mesh

    def __init__(self, Mesh_obj, sim_model, wall_meshtag_val, moments=None,
                 ref_area_dim=None, ref_chord_dim=None, testing_mode=False,
                 p_inf_dim=None, L_ref_dim=1.0):
        super().__init__()

        self.testing_mode = testing_mode

        self.mesh_obj = Mesh_obj
        self.sim_model = sim_model
        self.wall_meshtag_val = wall_meshtag_val
        self.moment_specs = self._parse_moments(moments)

        # Reference quantities
        self.ref_area_dim = ref_area_dim  # reference area (if None, set to 1 in 2D and computed planform area in 3D)
        self.ref_chord_dim = ref_chord_dim  # reference chord length (if None, set to 1)
        self.p_inf_dim = p_inf_dim  # dimensionalized freestream pressure, used for computing physical forces
        self.L_ref_dim = L_ref_dim  # dimensionalized length scale, used for computing physical forces

        # Compute quantities that are constant over the course of an optimization
        self.compute_initial_and_constant_quantities()


    def _parse_moments(self, moments):
        # Validate the user's (name, point, direction[, ref_length]) tuples and
        # turn them into MomentSpecs
        dim = self.sim_model.dimensions
        specs = []
        for entry in (moments if moments is not None else ()):
            if not isinstance(entry, (tuple, list)) or len(entry) not in (3, 4):
                raise ValueError("a moment is a (name, point, direction[, ref_length]) tuple, got {!r}".format(entry))
            name, point, direction = entry[:3]
            ref_length = entry[3] if len(entry) == 4 else None
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("moment name {!r} is not a valid identifier".format(name))
            if name in [spec.name for spec in specs]:
                raise ValueError("duplicate moment name {!r}".format(name))
            point = np.asarray(point, dtype=np.double)
            if point.shape != (dim,):
                raise ValueError("moment {!r}: point needs {} entries, got {}".format(name, dim, point.shape))
            if isinstance(direction, str):
                if direction != "pitch":
                    raise ValueError("moment {!r}: unknown direction preset {!r}".format(name, direction))
                direction = [0., 1., 0.] if dim == 3 else [0., 0., -1.]
            elif np.ndim(direction) == 0 and dim == 2:
                direction = [0., 0., direction]     # scalar shorthand for the out-of-plane axis
            direction = np.asarray(direction, dtype=np.double)
            if direction.shape != (3,):
                raise ValueError("moment {!r}: direction needs 3 entries, got {}".format(name, direction.shape))
            norm = np.linalg.norm(direction)
            if not norm > 0.0:
                raise ValueError("moment {!r}: direction is zero".format(name))
            direction = direction / norm
            if dim == 2 and (direction[0] != 0.0 or direction[1] != 0.0):
                raise ValueError("moment {!r}: in 2D only the out-of-plane direction (0, 0, +-1) exists".format(name))
            if ref_length is not None and not float(ref_length) > 0.0:
                raise ValueError("moment {!r}: ref_length must be positive".format(name))
            specs.append(MomentSpec(self.mesh_obj.mesh, name, point, direction, ref_length))
        return specs

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

        rho_in = dolfinx.fem.Constant(self.mesh_obj.mesh, self.sim_model.farfield_conditions['rho'])
        u_ref = self.sim_model.farfield_conditions['u_ref']

        self._S_ref_const = dolfinx.fem.Constant(self.mesh_obj.mesh, self._compute_S_ref())
        self._chord_ref_const = dolfinx.fem.Constant(self.mesh_obj.mesh, self._compute_chord_ref(self._S_ref_const.value))

        self.C_infty = 0.5*rho_in*(u_ref**2)*self._S_ref_const

        # Drag and lift vector directions
        self.psi_drag = self.sim_model.farfield_conditions['flow_dir']
        self.psi_lift = dolfinx.fem.Constant(
            self.mesh_obj.mesh, np.zeros((self.sim_model.dimensions,), dtype=np.double))

        self.set_angle_of_attack(self.sim_model.farfield_conditions['alpha'])

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

        inflow_vec = self.sim_model.farfield_conditions['inflow_angle']
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

    def _assemble_alpha_derivative(self, form):
        """d(form)/dalpha at fixed U through the derived fields, e.g. a
        reconstructed gradient whose boundary facets carry the alpha-dependent
        far-field state; zero without derived fields. The force directions are
        Constants (set_angle_of_attack), so their rotation is not part of it."""
        if not self.sim_model.derived_fields:
            return 0.0
        return self.sim_model.alpha_derivative(form)

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

    def _moment_normalizer(self, spec):
        ref_length = self._chord_ref_const if spec.ref_length is None else spec.ref_length
        return self.C_infty * ref_length

    def compute_moment_form(self, u_dof_vec, spec):
        """The moment M = d . (r x F) of the wall force about spec.point along
        spec.direction, as a scalar form."""
        force_density = self._pressure_force_vector(u_dof_vec)
        coords = ufl.SpatialCoordinate(self.mesh_obj.mesh)
        return moment_about(force_density, coords - spec.point, spec.direction,
                            self.sim_model.dimensions) * self.mesh_obj.form_manager.ds(self.wall_meshtag_val)

    def compute_moment(self, u_dof_vec, spec):
        """(M, c_m) of one moment."""
        moment = self.compute_moment_form(u_dof_vec, spec)
        return self._assemble_value_and_coefficient(moment, self._moment_normalizer(spec))

    def compute_moment_deriv(self, u_dof_vec, spec):
        moment = self.compute_moment_form(u_dof_vec, spec)
        return self._assemble_state_derivative(moment, "dM_{}_duvec".format(spec.name))

    def compute_moment_meshderiv(self, u_dof_vec, spec):
        moment = self.compute_moment_form(u_dof_vec, spec)
        return self._assemble_mesh_derivative(moment, "dM_{}_dmesh".format(spec.name))

    def compute_moment_alphaderiv(self, u_dof_vec, spec):
        # explicit alpha dependence at fixed state: the direction is fixed in the geometry axes,
        # so only the derived fields carry any
        return self._assemble_alpha_derivative(self.compute_moment_form(u_dof_vec, spec))

    def forces_and_coefficients(self, u_dof_vec=None, alpha=None):
        """
        Lift and drag and their coefficients of a flow state, and the moments
        and their coefficients, outside any CSDL graph: no FFD block, mesh
        motion or recorder needed.

        u_dof_vec: the owned state entries (default: the model's current
        u_vec); alpha: angle of attack in radians (default: the model's). For a
        viscous model, D_friction / c_d_friction is the part of the drag from
        the viscous wall traction. Collective. Returns a dict; "moments" maps
        each moment name to dict(M=..., c_m=...).
        """
        if u_dof_vec is None:
            u_dof_vec = self.sim_model.u_vec.x.petsc_vec.getArray().copy()
        if alpha is not None:
            self.set_angle_of_attack(alpha)
        self.refresh_geometry_dependent_reference_quantities()
        L, c_l = self.compute_cl(u_dof_vec)
        D, c_d = self.compute_cd(u_dof_vec)
        out = dict(L=L, D=D, c_l=c_l, c_d=c_d)
        out["moments"] = {}
        for spec in self.moment_specs:
            M, c_m = self.compute_moment(u_dof_vec, spec)
            out["moments"][spec.name] = dict(M=M, c_m=c_m)
        if getattr(self.sim_model, "wall_traction", None) is not None:
            # drag of the viscous traction alone
            self.sim_model.set_u_vec(u_dof_vec)
            self.sim_model.update_derived_fields()
            t = self.sim_model.wall_traction(self.sim_model.u_vec, self.mesh_obj.n)
            form = ufl.dot(self.psi_drag, t) * self.mesh_obj.form_manager.ds(self.wall_meshtag_val)
            out["D_friction"], out["c_d_friction"] = self._assemble_value_and_coefficient(form, self.C_infty)
        return out

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
        L = self.create_output('L', (1,))
        D = self.create_output('D', (1,))
        M = {spec.name: self.create_output('M_' + spec.name, (1,)) for spec in self.moment_specs}
        c_m = {spec.name: self.create_output('c_m_' + spec.name, (1,)) for spec in self.moment_specs}

        # Force and moment coefficients are only used as monitoring outputs;
        # to avoid CSDL from computing its derivatives we set these to zero
        input_names = ('u_vec', 'alpha') + (('mesh_deformation',) if mesh_deformation is not None else ())
        coefficient_names = ('c_l', 'c_d') + tuple('c_m_' + spec.name for spec in self.moment_specs)
        for _input_name in input_names:
            for _output_name in coefficient_names:
                self.declare_derivative_parameters(of=_output_name, wrt=_input_name, dependent=False)
        if shape_param_inputs is not None:
            for _output_name in coefficient_names + ('L', 'D') + tuple('M_' + spec.name for spec in self.moment_specs):
                self.declare_derivative_parameters(of=_output_name, wrt='shape_param_inputs',
                                                   dependent=False)

        # construct output of the model
        output = csdl.VariableGroup()
        output.c_l = c_l
        output.c_d = c_d
        output.L = L
        output.D = D
        for spec in self.moment_specs:
            setattr(output, 'M_' + spec.name, M[spec.name])
            setattr(output, 'c_m_' + spec.name, c_m[spec.name])

        return output

    def compute(self, input_vals, output_vals):
        PETSc.Sys.Print("starting compute in DG_postprocessor...")

        u_vec = input_vals['u_vec']
        alpha = input_vals['alpha'][0]
        deform_array = input_vals['mesh_deformation'] if 'mesh_deformation' in input_vals else None
        shape_parameters = input_vals['shape_param_inputs'] if 'shape_param_inputs' in input_vals else None

        # Update the angle of attack and downstream quantities
        self.set_angle_of_attack(alpha)

        # Apply the mesh deformations to the mesh (none in a forward analysis)
        if deform_array is not None:
            self.mesh_obj.apply_node_motions(deform_array, design=shape_parameters)

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

            PETSc.Sys.Print("c_d: {}, c_l: {}".format(np.round(c_d_initial, 6), np.round(output_vals['c_l'], 6)))
            PETSc.Sys.Print("D: {}, L: {} (nondimensional)".format(np.round(output_vals['D'], 6), np.round(output_vals['L'], 6)))

            # print dimensional forces, just for monitoring purposes
            PETSc.Sys.Print("D: {}, L: {} (dimensional)".format(
                np.round(output_vals['D'] * self.force_scale, 6),
                np.round(output_vals['L'] * self.force_scale, 6)))

            for spec in self.moment_specs:
                moment_val, output_vals['c_m_' + spec.name] = self.compute_moment(u_vec, spec)
                output_vals['M_' + spec.name] = moment_val
                PETSc.Sys.Print("moment {} (point {}, direction {}): M: {} (nondimensional), c_m: {}, M: {} (dimensional)".format(
                    spec.name, np.round(spec.point_array, 6), np.round(spec.direction_array, 6),
                    np.round(moment_val, 6), np.round(output_vals['c_m_' + spec.name], 6),
                    np.round(moment_val * self.moment_scale, 6)))
        else:
            PETSc.Sys.Print("Mesh or solution is not valid, setting outputs to high drag.")
            output_vals['L'], output_vals['c_l'] = 0.0, 0.0
            output_vals['D'], output_vals['c_d'] = 10., 1.0
            for spec in self.moment_specs:
                output_vals['M_' + spec.name], output_vals['c_m_' + spec.name] = 0.0, 0.0

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        # Compute the derivatives of the output quantities w.r.t. the inputs
        PETSc.Sys.Print("starting compute_derivatives in DG_postprocessor...")

        u_vec = input_vals['u_vec']
        alpha = input_vals['alpha'][0]
        deform_array = input_vals['mesh_deformation'] if 'mesh_deformation' in input_vals else None
        shape_parameters = input_vals['shape_param_inputs'] if 'shape_param_inputs' in input_vals else None

        # Update the angle of attack and downstream quantities
        self.set_angle_of_attack(alpha)

        # Apply the mesh deformations to the mesh and update geometry-dependent quantities
        if deform_array is not None:
            self.mesh_obj.apply_node_motions(deform_array, design=shape_parameters)
        self.refresh_geometry_dependent_reference_quantities()

        # Compute all force and moment derivatives w.r.t. the nondimensional state and mesh vertex movement
        derivatives['L', 'u_vec'] = self.compute_L_deriv(u_vec)[None, :]
        derivatives['D', 'u_vec'] = self.compute_D_deriv(u_vec)[None, :]
        for spec in self.moment_specs:
            derivatives['M_' + spec.name, 'u_vec'] = self.compute_moment_deriv(u_vec, spec)[None, :]

        if deform_array is not None:
            derivatives['L', 'mesh_deformation'] = self.compute_L_meshderiv(u_vec)[None, :]
            derivatives['D', 'mesh_deformation'] = self.compute_D_meshderiv(u_vec)[None, :]
            for spec in self.moment_specs:
                derivatives['M_' + spec.name, 'mesh_deformation'] = self.compute_moment_meshderiv(u_vec, spec)[None, :]

        # Compute the force derivatives w.r.t. angle of attack changes at fixed u_vec: the
        # rotation of the lift and drag directions (dL = -D dalpha, dD = L dalpha), plus
        # the forces' dependence on alpha through the derived fields
        L_val, _ = self.compute_cl(u_vec)
        D_val, _ = self.compute_cd(u_vec)
        dL = self._assemble_alpha_derivative(self.compute_force_in_direction(u_vec, self.psi_lift))
        dD = self._assemble_alpha_derivative(self.compute_force_in_direction(u_vec, self.psi_drag))
        derivatives['L', 'alpha'] = np.array([[-D_val + dL]], dtype=np.float64)
        derivatives['D', 'alpha'] = np.array([[ L_val + dD]], dtype=np.float64)
        for spec in self.moment_specs:
            derivatives['M_' + spec.name, 'alpha'] = np.array(
                [[self.compute_moment_alphaderiv(u_vec, spec)]], dtype=np.float64)
