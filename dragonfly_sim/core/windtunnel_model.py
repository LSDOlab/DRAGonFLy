import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

import datetime
import resource
import os
import gc

import csdl_alpha as csdl

from time import perf_counter

from dragonfly_sim.core.mesh_manager import Mesh
from dragonfly_sim.core.Euler_model import CompressibleEulerModel
from dragonfly_sim.core.meshwarping import IDWarp_jax
from dragonfly_sim.core.shape_parameterization import WallFFD, WallGeometry
from dragonfly_sim.utils.meshwarping_utils import inner_bdry_function_from_corner_list
from dragonfly_sim.utils.data_handler import DataStore

from dragonfly_sim.utils.petsc_utils import set_petsc_vec_array
from dragonfly_sim.utils.solver_utils import ksp_converged_reason_name
from dragonfly_sim.core.postprocessor import DG_postprocessor


# TODO: Look into using entropy variables instead of conservative variables;
#       these would potentially allow us to do simulations from low Mach all
#       the way to supersonic
class DG_windtunnel_model(csdl.experimental.CustomImplicitOperation):
    # The solve/adjoint callbacks contain MPI collectives (PETSc, mpi4py) and deform
    # the shared mesh, so they must never overlap -- also on a single process.
    ordered_callbacks = True

    def __init__(self, mesh, boundary_dict, ffd_shape, aero_center,
                 mesh_inner_bdry_function=None, ffd_block_corner_list=None, cp_coord_opt_idxs=None,
                 poly_order=1, gamma=1.4, ffd_degree=2,
                 filename_suffix="test",
                 asm_overlap=None, ilu_levels=None):
        """
        The shape of the object being optimized is defined either through `mesh_inner_bdry_function`
        or through ffd_block_corner_list, which defines the box the inner boundary
        facets are contained in and can therefore supply the same signal. At least
        one of the two is required; when both are given the explicit function
        wins, since it can encode exclusions the FFD block cannot.
        """
        super().__init__()

        self.asm_overlap = asm_overlap
        self.ilu_levels = ilu_levels

        # --- Linear-solve convergence criteria ----------------------------
        # We use one set of convergence criteria for all linear solves.
        # This does not cover the Newton solver for the Euler model itself,
        # but all other things (adjoint, Helmholtz, elasticity-based 
        # mesh warping, etc.) when used.
        self.linear_solver_rtol = 1e-20
        self.linear_solver_atol = 1e-11
        self.linear_solver_max_it = 1000

        # If linear solves converge above this tolerance, we raise an error
        self.linear_residual_accept_limit = 1e-10

        # --- Nonlinear (Euler) convergence criteria -----------------------
        self.euler_residual_conv_limit = 1e-9
        # Total Newton-step budget for the FOM solve
        self.max_newton_iterations = 100
        # KDTree coincidence radius for the dof <-> mesh-node permutation.
        # These two orderings are 1:1 by construction and differ only in
        # index order, so this tolerance accounts for round-off errors
        self.mesh_node_match_tol = 1e-10
        # Newton tolerance for projecting mesh boundary nodes into the FFD
        # block's parametric space.
        self.ffd_projection_newton_tol = 1e-12
        # Containment test used to pick the inner-boundary facets out of the
        # FFD block when no explicit mesh_inner_bdry_function is given.
        self.inner_bdry_rel_tol = 1e-6
        self.inner_bdry_overset_margin = 1e-4

        # Resolved AFTER the tolerance block above, which is what it reads.
        if mesh_inner_bdry_function is None:
            if ffd_block_corner_list is None:
                raise ValueError("Provide either mesh_inner_bdry_function or "
                                 "ffd_block_corner_list to locate the inner boundary facets.")
            mesh_inner_bdry_function = inner_bdry_function_from_corner_list(
                ffd_block_corner_list,
                overset_margin=self.inner_bdry_overset_margin,
                rel_tol=self.inner_bdry_rel_tol)

        # Initial instantiation of mesh and simulation model
        self.mesh = Mesh(mesh, mesh_inner_bdry_function,
                         node_match_tol=self.mesh_node_match_tol)
        _euler_kwargs = dict(poly_order=poly_order, gamma=gamma)
        self.sim_model = CompressibleEulerModel(self.mesh, **_euler_kwargs)
        # TODO: Perhaps split up CompressibleEulerModel into an Euler model class and a solver class that facilitates solver interactions; 
        #       the solver has become so complicated recently that spinning it off into its own class would likely help with code readability

        if asm_overlap is not None:
            self.sim_model.asm_overlap = asm_overlap
        if ilu_levels is not None:
            self.sim_model.ilu_levels = ilu_levels

        # Set by solve_residual_equations, once, as the LAST write to this
        # name for the evaluation: whether the solve that produced the
        # accepted u_vec actually converged. 
        # Mainly used for testing purposes.
        self.sim_model.last_solve_converged = False

        self.ffd_shape = ffd_shape
        self.ffd_block_corner_list = ffd_block_corner_list
        self.ffd_degree = ffd_degree
        self.cp_coord_opt_idxs = cp_coord_opt_idxs

        # Initialize some mesh deformation properties, 
        # defined in set_up_sim()
        self.mesh_warper = None
        self.ffd = None
        self._wall_geometry = None

        # Define data storage object for external post-processing
        self.data_store = DataStore()

        self.previous_solution = None

        # Naming suffix for exported files
        self.filename_suffix = filename_suffix

        # Include dc_m/d(alpha) in printed coefficient outputs 
        # (costs one extra linear solve per print)
        self.log_cm_alpha = True


        # Discretization and physics-related parameters
        self.poly_order:int = poly_order
        self.gamma:float = gamma
        self.boundary_condition_dict = boundary_dict
        self.aero_center = aero_center

        # Dolfinx mesh tags for the various boundaries
        self.INLET_TAG = 3  # flow inlet
        self.OUTLET_TAG = 4  # flow outlet
        self.WALL_TAG = 2  # walls (including wing/airfoil boundaries)
        self.SYMMETRY_TAG = 5  # symmetry, only used in 3D thus far

        # global evaluation index of solve_residual_equations
        self.eval_idx = 0

    def set_up_sim(self):
        # pass the various tolerances and limits to the Euler sim model
        self.sim_model.residual_conv_limit = self.euler_residual_conv_limit
        self.sim_model.max_newton_iterations = self.max_newton_iterations

        # use the boundary condition dictionary to define inlet and outlet BCs
        self.sim_model.define_inlet_outlet_conditions(self.boundary_condition_dict)

        # compute boundary facet boolean arrays
        inner_facets = self.mesh.inner_bdry_facet_mask
        
        # Find the symmetry face facets (only relevant in 3D at the moment)
        # NOTE: The symmetry plane is currently hardcoded at y=0
        if self.sim_model.dimensions == 3:
            symmetry_facets = self.mesh.bdry_midpoints[:, 1] <= 1e-6
        else:
            symmetry_facets = np.array([False]*inner_facets.shape[0])

        # Next we split the outer boundaries into inlet and outlet; 
        # the dividing line is perpendicular to the inflow angle due to angle of attack.
        # NOTE: This is fixed under angle of attack changes, 
        # so could be a source of error for large angle of attack changes
        mesh_x_max, mesh_x_min = self.mesh.mesh.comm.allreduce(self.mesh.mesh.geometry.x[:, 0].max(), MPI.MAX), self.mesh.mesh.comm.allreduce(self.mesh.mesh.geometry.x[:, 0].min(), MPI.MIN)

        alpha_tagging = self.boundary_condition_dict['inlet']['alpha']
        inlet_outlet_bdry = -np.tan(alpha_tagging)*self.mesh.bdry_midpoints[:, 2] + (mesh_x_max + mesh_x_min) / 2

        inlet_facets = (self.mesh.bdry_midpoints[:, 0] <= inlet_outlet_bdry) & ~np.asarray(inner_facets, dtype=bool) & ~np.asarray(symmetry_facets, dtype=bool)
        outlet_facets = (self.mesh.bdry_midpoints[:, 0] > inlet_outlet_bdry) & ~np.asarray(inner_facets, dtype=bool) & ~np.asarray(symmetry_facets, dtype=bool)

        # Print the local number of facets of each type for each MPI rank
        PETSc.Sys.syncPrint("Rank {}, number of wing boundary facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(inner_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of symmetry facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(symmetry_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of inlet facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(inlet_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of outlet facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(outlet_facets)))
        PETSc.Sys.syncFlush()

        # Pair the boundary facet sets with the various mesh tags
        bdry_meshtags_and_ds_list = [(inlet_facets, self.INLET_TAG), (outlet_facets, self.OUTLET_TAG), (inner_facets, self.WALL_TAG)]
        if self.sim_model.dimensions == 3:
            bdry_meshtags_and_ds_list += [(symmetry_facets, self.SYMMETRY_TAG)]

        # Build the boundary facet tags and define the corresponding quadrature measure
        self.mesh.tag_boundary_facets(bdry_meshtags_and_ds_list)
        self.mesh.form_manager.build_boundary_measure(
            measure_metadata={'quadrature_degree': 3*self.poly_order + 1})

        # The inlet-outlet split above is now frozen in the meshtags; arm the guard in
        # CompressibleEulerModel.set_angle_of_attack against it.
        self.sim_model.alpha_tagging = alpha_tagging

        self.mesh.configure_output(self.filename_suffix, deformation=True)

        # The JAX-based IDWarp implementation as mesh warper
        self.mesh_warper = IDWarp_jax(self.mesh, wall_meshtag_val=self.WALL_TAG,
                                      anchor_meshtag_vals=(self.INLET_TAG, self.OUTLET_TAG))

        PETSc.Sys.Print("Initialized mesh warping object")

        # The lsdo_geo FFD block around the wall nodes. Shape layers
        # (shape_parameterization.SectionalShape, WallFFD.apply_cp_motions) turn
        # the design variables into its coefficients; wall_displacement() is the
        # replicated (M_global, dim) array mesh_warper.evaluate takes, in the
        # numbering of the mesh's wall node set.
        wall_nodes = self.mesh.boundary_node_set((self.WALL_TAG,))
        self.ffd = WallFFD(self.mesh.mesh.comm, wall_nodes.coords,
                           wall_nodes.global_idx, self.ffd_shape,
                           degree=self.ffd_degree,
                           ffd_block_corner_list=self.ffd_block_corner_list,
                           projection_newton_tol=self.ffd_projection_newton_tol)

        # define the integration measure for the mesh cells and non-boundary facets
        self.mesh.form_manager.build_cell_and_interior_facet_measures(
            measure_metadata={'quadrature_degree': 3*self.poly_order + 1})

        # Define the required Dolfinx objects for the simulation, including the initial conditions
        self.sim_model.define_elements_functionspaces()
        self.sim_model.define_trial_testfunctions()
        self.sim_model.compute_initial_conditions_from_inlet()
        self.sim_model.interpolate_solution_vector()

        # We output the facet mesh tags to an xdmf file, so we can inspect the BCs in paraview
        self.mesh.mesh.topology.create_connectivity(self.sim_model.dimensions-1, self.sim_model.dimensions)
        with dolfinx.io.XDMFFile(self.mesh.mesh.comm, "output_meshtags.xdmf", "w") as xdmf:
            # Write the Mesh object to the file
            xdmf.write_mesh(self.mesh.mesh)
            # Write the MeshTags object to the file
            xdmf.write_meshtags(self.mesh.meshtags, self.mesh.mesh.geometry)

        # Define the boundary condition terms of the Euler weak form
        self.sim_model.define_subsonic_inflow_bc(self.INLET_TAG)
        self.sim_model.define_subsonic_outflow_bc(self.OUTLET_TAG)
        self.sim_model.define_slipwall_bc(self.WALL_TAG)
        if self.sim_model.dimensions == 3:
            self.sim_model.define_slipwall_bc(self.SYMMETRY_TAG)

        # Construct the Euler weak form for forward evaluation
        self.sim_model.compute_weakform()

        # Define the file names and objects for solution writes to external files for Paraview
        self.sim_model.define_sol_export_files(self.sim_model.functionspaces["V"], 
                                               file_name_addendum=self.filename_suffix)

        # We instantiate a separate DG_postprocessor object here so we can print 
        # forces and moments and coefficients after each forward solve
        self.postprocessor = DG_postprocessor(
            self.mesh, self.sim_model, self.WALL_TAG,
            aero_center=self.aero_center,
            linear_solver_rtol=self.linear_solver_rtol,
            linear_solver_atol=self.linear_solver_atol,
            linear_solver_max_it=self.linear_solver_max_it,
            linear_residual_accept_limit=self.linear_residual_accept_limit, 
            p_inf_dim=101325.)

    def _log_memory(self, stage):
        # We write a log file for each MPI rank that documents the peak local memory usage
        # at various points during an optimization run.
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rank = MPI.COMM_WORLD.Get_rank()
        fname = 'memory_logs/r{}_{}.log'.format(rank, self.filename_suffix)
        os.makedirs('memory_logs', exist_ok=True)
        with open(fname, 'a') as f:
            f.write('{}   {}  {}  {}  {}\n'.format(
                self.eval_idx, datetime.datetime.now(), os.getpid(), mem, stage))

    def wall_geometry(self):
        """Geometric constraint quantities of the FFD-deformed wall (area/volume,
        thickness, planform); see shape_parameterization.WallGeometry.

        Built on first use. Collective: call it on every rank.
        """
        if self._wall_geometry is None:
            self._wall_geometry = WallGeometry(self.ffd, self.mesh.wall_facets(self.WALL_TAG))
        return self._wall_geometry

    def evaluate(self, mesh_node_motions:csdl.Variable, cp_motion_inputs:csdl.Variable,
                 alpha:csdl.Variable):

        # set inputs using self.declare_input
        self.declare_input('mesh_node_motions', mesh_node_motions)
        self.declare_input('cp_motion_inputs', cp_motion_inputs)
        self.declare_input('alpha', alpha)

        # The output on this rank is only the local portion of the solution vector
        u_vec = self.create_output('u_vec', (self.sim_model.u_vec.x.petsc_vec.local_size,))

        # We have declared cp_motion_inputs as an input here, just for logging and computation purposes,
        # not because we actually need it; mesh_node_motions carries the effect of the cp_motion_inputs.
        self.declare_derivative_parameters(of='u_vec', wrt='cp_motion_inputs', dependent=False)

        return u_vec

    def solve_residual_equations(self, input_vals, output_vals):
        # Define inputs
        mesh_node_motions = input_vals['mesh_node_motions']
        cp_motion_inputs = input_vals['cp_motion_inputs']
        alpha = input_vals['alpha'][0]

        # Update the angle of attack to its current value
        self.sim_model.set_angle_of_attack(alpha)

        PETSc.Sys.Print("angle of attack: {} deg".format(np.degrees(alpha)))
        PETSc.Sys.Print("cp motion inputs: {}".format(cp_motion_inputs))
        PETSc.Sys.Print("solve_residual_equations, stepping into mesh.apply_node_motions")
        PETSc.Sys.Print("mesh_node_motions 2-norm: {}".format(np.linalg.norm(mesh_node_motions)))

        # Apply the mesh node motions to the mesh
        self.mesh.apply_node_motions(mesh_node_motions)

        # If the mesh is not valid (e.g. has inverted cells), set the output to zero and don't run any simulations
        if not self.mesh.is_valid:
            PETSc.Sys.Print("Mesh is not valid, skipping simulations.")
            output_vals['u_vec'] = np.zeros((self.sim_model.u_vec.x.petsc_vec.local_size,))
            self.mesh.reset_nodes()
            return

        # re-define the current weak form if necessary
        self.sim_model.compute_weakform()

        # --- FOM solve ----------------------------------------------------
        solution_output, fom_converged = self._run_fom_solve()

        PETSc.Sys.Print("Accepting FOM solution")

        self._log_memory("after FOM solve, before setting u_vec output")

        self.sim_model.last_solve_converged = fom_converged

        PETSc.Sys.Print("setting u_vec output...")
        output_vals['u_vec'] = solution_output

        # Warm start: keep the FOM solution unless it has blown up
        if np.isfinite(np.linalg.norm(solution_output)):
            PETSc.Sys.Print("Updating warm start solution")
            self.previous_solution = solution_output
        else:
            PETSc.Sys.Print("Not updating warm start solution, using previous value")

        PETSc.Sys.Print("Writing aligned output frame at eval_idx {} (FOM solution)".format(
            self.eval_idx))
        # we write the deformation and the solution outputs to the same time step indices
        self.mesh.write_deformation_output(mesh_node_motions, self.eval_idx)
        self.sim_model.write_solution_output(self.eval_idx,
                                             solution_array=solution_output)

        self.mesh.reset_nodes()

        # Explicitly collect garbage and cleanup PETSc objects
        gc.collect()
        PETSc.garbage_cleanup(self.mesh.mesh.comm)

        self._log_memory("after Euler PDE solve")

        self.eval_idx += 1

    def _cm_alpha_for_logging(self, solution):
        # Compute d c_m / d alpha: The total derivative of the moment 
        # coefficient w.r.t. the angle of attack. This step seems 
        # somewhat unstable and does not come for free, since it requires
        # solving a linear system (it's a derivative evaluation after all)
        # TODO: Look into why this solution step is seemingly so unstable,
        #       even at well-converged states
        if not self.log_cm_alpha:
            return np.nan
        if solution is None or not np.isfinite(np.linalg.norm(solution)):
            return np.nan
        return self.postprocessor.compute_cm_alpha(solution)

    def _run_fom_solve(self):
        # Run the FOM model and output its solution

        # store pre-FOM time stamp for performance profiling
        fom_pre_time = perf_counter()

        # use the most recent converged solution as warm start
        initial_condition = self.previous_solution
        self.sim_model.interpolate_solution_vector(initial_condition)

        # solve the FOM
        self.sim_model.set_up_solver()
        residual_norm = self.sim_model.solve_system().norm()
        fom_converged = self.sim_model.solver_converged
        eta = self.sim_model.relative_residual_norm(residual_norm)
        solution_fom = self.sim_model.u_vec.x.petsc_vec.getArray().copy()

        self._log_memory("after FOM solve, before doing anything else")

        # store post-FOM time stamp for performance profiling
        fom_post_time = perf_counter()
        PETSc.Sys.Print("FOM wall time: {}".format(fom_post_time - fom_pre_time))

        L_fom, cl_fom = self.postprocessor.compute_cl(solution_fom)
        D_fom, cd_fom = self.postprocessor.compute_cd(solution_fom)
        M_fom, cm_fom = self.postprocessor.compute_cm(solution_fom)
        cma_fom = self._cm_alpha_for_logging(solution_fom)

        if self.mesh.mesh.comm.Get_rank() == 0:
            self.data_store.FOM_walltime += [(self.eval_idx, fom_post_time - fom_pre_time)]
            self.data_store.fom_relative_residuals += [(self.eval_idx, eta)]
            self.data_store.FOM_force_coefficients += [(self.eval_idx, cd_fom, cl_fom, cm_fom, cma_fom)]
        PETSc.Sys.Print("FOM coefficients: c_d: {}, c_l: {}, c_m: {}, c_m_alpha: {}".format(cd_fom, cl_fom, cm_fom, cma_fom))
        PETSc.Sys.Print("FOM relative residual norm eta: {} (||r||: {})".format(
            eta, residual_norm))

        return solution_fom, fom_converged

    def apply_inverse_jacobian(self, input_vals, outputs, d_outputs, d_residuals, mode):
        pre_apply_inverse_jacobian_time = perf_counter()
        # Here we solve the adjoint equation when calculating derivatives for optimization
        print("starting apply_inverse_jacobian in DG_windtunnel_model...")

        mesh_node_motions = input_vals['mesh_node_motions'] # numpy array
        alpha = input_vals['alpha'][0]
        u_vec = outputs['u_vec'] # numpy array

        if mode == 'rev':
            # compute d_residual = (dr_du^-1)*d_outputs

            # apply the input mesh node motions to the mesh of the sim model
            # and set the angle of attack before constructing the weak form
            # and the corresponding solution that we are linearizing around
            PETSc.Sys.Print("apply_inverse_jacobian, stepping into mesh.apply_node_motions")
            self.mesh.apply_node_motions(mesh_node_motions)
            self.sim_model.set_angle_of_attack(alpha)
            self.sim_model.compute_weakform()
            self.sim_model.set_u_vec(u_vec)

            self._drdu_mat = self.sim_model.assemble_dRdu_mat()
            if not hasattr(self, '_d_output_petsc'):
                self._d_output_petsc = self._drdu_mat.createVecLeft()
                self._d_residual_petsc = self._drdu_mat.createVecLeft()
                self._F_form = dolfinx.fem.form(self.sim_model.F)
                self._residual_vec_petsc = dolfinx.fem.petsc.create_vector(
                    dolfinx.fem.extract_function_spaces(self._F_form))

            with self._residual_vec_petsc.localForm() as loc:
                loc.set(0.0)
            dolfinx.fem.petsc.assemble_vector(self._residual_vec_petsc, self._F_form)
            self._residual_vec_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            PETSc.Sys.Print("max residual entry: {}".format(self._residual_vec_petsc.max()[1]))
            residual_norm = self._residual_vec_petsc.norm()
            PETSc.Sys.Print("forward eval residual norm: {}".format(residual_norm))

            # (no assembly here any more: assemble_dRdu_mat above already
            # returns the matrix assembled. It hands back the same cached
            # Mat object every call, so the KSP's operator reference below
            # stays valid.)
            if not hasattr(self, '_adj_solver'):
                self._adj_solver = PETSc.KSP().create(self.mesh.mesh.comm)
                self._adj_solver.setOperators(self._drdu_mat)
                self._adj_solver.setOptionsPrefix("adj_")

                # define the adjoint solver settings
                self.sim_model.apply_krylov_solver_settings(
                    self._adj_solver, max_it=self.linear_solver_max_it,
                    gmres_restart=200,
                    monitor_convergence=False,
                    rtol=self.linear_solver_rtol,
                    atol=self.linear_solver_atol)

            set_petsc_vec_array(self._d_output_petsc, d_outputs['u_vec'])

            # solve the transposed linear system and output the solution
            self._adj_solver.solveTranspose(self._d_output_petsc, self._d_residual_petsc)

            rnorm = self._adj_solver.getResidualNorm()
            PETSc.Sys.Print("adj_ solve: {} iterations, residual norm {} [{}]".format(
                self._adj_solver.getIterationNumber(), rnorm,
                ksp_converged_reason_name(self._adj_solver.getConvergedReason())))

            if rnorm > self.linear_residual_accept_limit:
                raise ValueError(
                    "apply_inverse_jacobian residual norm is too high: {:.6e} > "
                    "linear_residual_accept_limit {:.1e}".format(
                        rnorm, self.linear_residual_accept_limit))

            PETSc.Sys.Print("d_residual norm: {}".format(self._d_residual_petsc.norm()))

            PETSc.Sys.Print("mesh_node_motions min, max: {}, {}".format(mesh_node_motions.min(), mesh_node_motions.max()))
            PETSc.Sys.Print("alpha: {}".format(np.degrees(alpha)))
            PETSc.Sys.Print("cp motions: {}".format(input_vals['cp_motion_inputs']))

            d_residuals['u_vec'] = self._d_residual_petsc.getArray()

            PETSc.garbage_cleanup()

            self.mesh.reset_nodes()

            post_apply_inverse_jacobian_time = perf_counter()
            total_apply_inverse_jacobian_time = post_apply_inverse_jacobian_time - pre_apply_inverse_jacobian_time

            PETSc.Sys.Print("apply_inverse_jacobian total time: {}".format(total_apply_inverse_jacobian_time))
        else:
            raise ValueError("Reverse mode required")

    def compute_jacvec_product(self, input_vals, outputs, d_inputs, d_outputs, d_residuals, mode):
        # Here we compute the product of the adjoint solution with d_residual
        pre_compute_jacvec_product_time = perf_counter()
        print("starting compute_jacvec_product in DG_windtunnel_model...")

        mesh_node_motions = input_vals['mesh_node_motions'] # numpy array
        alpha = input_vals['alpha'][0]
        u_vec = outputs['u_vec'] # numpy array

        if mode == 'rev':
            # compute d_input = (dr_dinput^T)*d_residual

            # apply the input mesh node motions to the mesh of the sim model,
            # apply the angle of attack and compute the weak form
            PETSc.Sys.Print("compute_jacvec_product, stepping into mesh.apply_node_motions")
            self.mesh.apply_node_motions(mesh_node_motions)
            self.sim_model.set_angle_of_attack(alpha)
            self.sim_model.compute_weakform()

            if 'alpha' in d_inputs:
                # alpha is a single global scalar, so dR/d(alpha) is one column
                # and this reverse product is a plain dot product between vectors
                dr_dalpha = self.sim_model.assemble_dRdalpha_vec(u_vec)

                # we first have to assert that the MPI objects are compatible
                local_size = self.sim_model.u_vec.x.petsc_vec.local_size
                dr_dalpha_local = dr_dalpha.getArray()[:local_size]
                assert dr_dalpha_local.shape == d_residuals['u_vec'].shape, (
                    "dR/dalpha local slice {} does not match d_residuals['u_vec'] {}".format(
                        dr_dalpha_local.shape, d_residuals['u_vec'].shape))

                # we do the multiplication of the local objects through Numpy,
                # after which we sum the contributions on every MPI rank
                local_contribution = float(np.dot(dr_dalpha_local, d_residuals['u_vec']))
                d_inputs['alpha'] = np.array(
                    [self.mesh.mesh.comm.allreduce(local_contribution, op=MPI.SUM)],
                    dtype=np.float64)

                PETSc.Sys.Print("d_inputs['alpha']: {}".format(d_inputs['alpha']))

            if 'mesh_node_motions' in d_inputs:
                dr_dxmesh = self.sim_model.compute_dRdxmesh_mat(u_vec, self.mesh.coordinate_space)

                if not hasattr(self, '_d_residual_jvp_petsc'):
                    self._d_residual_jvp_petsc = dr_dxmesh.createVecLeft()
                    self._d_input_jvp_petsc = dr_dxmesh.createVecRight()

                set_petsc_vec_array(self._d_residual_jvp_petsc, d_residuals['u_vec'])

                # compute d_input
                self._d_input_jvp_petsc.set(0.0)
                dr_dxmesh.multTranspose(self._d_residual_jvp_petsc, self._d_input_jvp_petsc)
                d_input = self._d_input_jvp_petsc

                self.mesh.reset_nodes()

                PETSc.Sys.Print("d_input norm: {}".format(d_input.norm()))

                d_input_cg1 = d_input.getArray().reshape(-1, self.mesh.mesh.geometry.dim)   # (N_CG1_dofs, gdim)
                d_input_geom = self.mesh.dof_to_node_map.T @ d_input_cg1  # (N_geom_nodes, gdim)
                d_inputs['mesh_node_motions'] = d_input_geom

                PETSc.garbage_cleanup()
            else:
                # The mesh branch above owns the reset; without it, an
                # alpha-only reverse pass would leave the mesh deformed.
                self.mesh.reset_nodes()
        else:
            raise ValueError("Reverse mode required")
        
        PETSc.Sys.Print("DG_windtunnel_model compute_jacvec_product done")

        post_compute_jacvec_product_time = perf_counter()
        total_compute_jacvec_product_time = post_compute_jacvec_product_time - pre_compute_jacvec_product_time

        PETSc.Sys.Print("compute_jacvec_product total time: {}".format(total_compute_jacvec_product_time))
