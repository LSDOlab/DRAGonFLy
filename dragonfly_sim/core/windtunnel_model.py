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

    def __init__(self, mesh, boundary_dict, shape_parameterization=None, aero_center=None,
                 mesh_inner_bdry_function=None,
                 poly_order=1, gamma=1.4,
                 filename_suffix="test",
                 asm_overlap=None, ilu_levels=None,
                 model_class=CompressibleEulerModel, model_kwargs=None, farfield="split",
                 reduced_order_model=None):
        """
        The shape of the object being optimized is defined either through `mesh_inner_bdry_function`
        or through the shape parameterization's FFD block corners
        (shape_parameterization.inner_boundary_function()), which bound the box
        the inner boundary facets are contained in and can therefore supply the
        same signal. At least one of the two is required; when both are given
        the explicit function wins, since it can encode exclusions the FFD block
        cannot.

        shape_parameterization: the wall-shape parameterization, e.g. a
        shape_design.FFDShapeParameterization -- the FFD block and the layers
        that define the shape design variables and their bounds. The model does
        not need to know what those variables are: set_up_sim builds the IDWarp
        mesh warper (self.mesh_warper) and hands the wall to
        shape_parameterization.setup(); deform_mesh() turns its wall
        displacement into mesh node motions; evaluate() takes those and the
        flat shape parameter vector: shape optimization. Without it (None) no
        warper is built, the mesh stays fixed, and evaluate() takes alpha only:
        a forward flow analysis, with derivatives with respect to alpha.
        solve_forward() runs one flow solve without a CSDL graph at all.

        aero_center: moment reference point; defaults to (0.25, 0[, 0]).

        model_class / model_kwargs select the flow model: CompressibleEulerModel
        (the default) or e.g. core.RANS_model.CompressibleRANSModel with
        model_kwargs={'Re': 6e6}. The body's wall is the model's
        define_wall_bc (slip for Euler, adiabatic no-slip for RANS).

        farfield: "split" divides the outer boundary into a subsonic inflow
        and outflow perpendicular to the free stream (the default); "riemann"
        puts the characteristic far field (define_farfield_bc) on all of it,
        which needs no split and so no guard on large alpha changes.

        reduced_order_model: a core.reduced_order_model.ReducedOrderModel to
        try before every flow solve (see _solve), or None for full-order
        solves only.
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

        if shape_parameterization is not None and not hasattr(shape_parameterization, 'setup'):
            raise TypeError("shape_parameterization must be a shape parameterization object "
                            "(e.g. shape_design.FFDShapeParameterization), got {!r}. The FFD "
                            "block arguments (ffd_shape, ffd_degree, ffd_block_corner_list, "
                            "cp_coord_opt_idxs) moved there.".format(shape_parameterization))
        if mesh_inner_bdry_function is None and shape_parameterization is not None:
            mesh_inner_bdry_function = shape_parameterization.inner_boundary_function()
        if mesh_inner_bdry_function is None:
            raise ValueError("Provide either mesh_inner_bdry_function or a shape "
                             "parameterization with FFD block corners to locate the inner "
                             "boundary facets.")

        # Initial instantiation of mesh and simulation model
        self.mesh = Mesh(mesh, mesh_inner_bdry_function,
                         node_match_tol=self.mesh_node_match_tol)
        if farfield not in ("split", "riemann"):
            raise ValueError("farfield must be 'split' or 'riemann'")
        self.farfield = farfield
        if model_class is CompressibleEulerModel:
            _euler_kwargs = dict(poly_order=poly_order, gamma=gamma)
            self.sim_model = CompressibleEulerModel(self.mesh, **_euler_kwargs)
        else:
            self.sim_model = model_class(self.mesh, gamma=gamma, **(model_kwargs or {}))
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

        # Mesh warping only with a shape parameterization (see __init__)
        self.shape_parameterization = shape_parameterization
        self.shape_parameterized = shape_parameterization is not None

        # The mesh warper, built in set_up_sim()
        self.mesh_warper = None

        # Define data storage object for external post-processing
        self.data_store = DataStore()

        self.previous_solution = None

        # POD reduced-order model (None: full-order solves only)
        self.reduced_order_model = reduced_order_model

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
        self.FARFIELD_TAG = 6  # characteristic far field (farfield="riemann")

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
        if self.farfield == "riemann":
            farfield_facets = inlet_facets | outlet_facets

        # Print the local number of facets of each type for each MPI rank
        PETSc.Sys.syncPrint("Rank {}, number of wing boundary facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(inner_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of symmetry facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(symmetry_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of inlet facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(inlet_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of outlet facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(outlet_facets)))
        PETSc.Sys.syncFlush()

        # Pair the boundary facet sets with the various mesh tags
        if self.farfield == "riemann":
            bdry_meshtags_and_ds_list = [(farfield_facets, self.FARFIELD_TAG), (inner_facets, self.WALL_TAG)]
        else:
            bdry_meshtags_and_ds_list = [(inlet_facets, self.INLET_TAG), (outlet_facets, self.OUTLET_TAG), (inner_facets, self.WALL_TAG)]
        if self.sim_model.dimensions == 3:
            bdry_meshtags_and_ds_list += [(symmetry_facets, self.SYMMETRY_TAG)]

        # Build the boundary facet tags and define the corresponding quadrature measure
        self.mesh.tag_boundary_facets(bdry_meshtags_and_ds_list)
        self.mesh.form_manager.build_boundary_measure(
            measure_metadata={'quadrature_degree': 3*self.poly_order + 1})

        # The inlet-outlet split above is now frozen in the meshtags; arm the guard in
        # CompressibleEulerModel.set_angle_of_attack against it. The
        # characteristic far field has no split to guard.
        self.sim_model.alpha_tagging = alpha_tagging if self.farfield == "split" else None

        self.mesh.configure_output(self.filename_suffix, deformation=self.shape_parameterized)

        if self.shape_parameterized:
            # The JAX-based IDWarp implementation as mesh warper
            anchors = ((self.FARFIELD_TAG,) if self.farfield == "riemann"
                       else (self.INLET_TAG, self.OUTLET_TAG))
            self.mesh_warper = IDWarp_jax(self.mesh, wall_meshtag_val=self.WALL_TAG,
                                          anchor_meshtag_vals=anchors)

            PETSc.Sys.Print("Initialized mesh warping object")

            # The shape parameterization gets the wall nodes (replicated, in the
            # numbering of the mesh's wall node set) and facets; its
            # wall_displacement() is the (M_global, dim) array mesh_warper.evaluate
            # takes.
            wall_nodes = self.mesh.boundary_node_set((self.WALL_TAG,))
            self.shape_parameterization.setup(self.mesh.mesh.comm, wall_nodes.coords,
                                              wall_nodes.global_idx,
                                              self.mesh.wall_facets(self.WALL_TAG))
        else:
            PETSc.Sys.Print("No shape parameterization given: fixed mesh, forward flow analysis")

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
        if self.farfield == "riemann":
            self.sim_model.define_farfield_bc(self.FARFIELD_TAG)
        else:
            self.sim_model.define_subsonic_inflow_bc(self.INLET_TAG)
            self.sim_model.define_subsonic_outflow_bc(self.OUTLET_TAG)
        self.sim_model.define_wall_bc(self.WALL_TAG)
        if self.sim_model.dimensions == 3:
            self.sim_model.define_slipwall_bc(self.SYMMETRY_TAG)

        # Construct the Euler weak form for forward evaluation
        self.sim_model.compute_weakform()

        # Define the file names and objects for solution writes to external files for Paraview
        self.sim_model.define_sol_export_files(self.sim_model.functionspaces["V"],
                                               file_name_addendum=self.filename_suffix,
                                               mesh_deformation=self.shape_parameterized)

        if self.aero_center is None:
            self.aero_center = np.array([0.25] + [0.0] * (self.sim_model.dimensions - 1))

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

        if self.reduced_order_model is not None:
            self.reduced_order_model.set_up(self.sim_model)

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

    @property
    def ffd(self):
        """The shape parameterization's FFD block (None without one, or before
        set_up_sim)."""
        if self.shape_parameterization is None:
            return None
        return getattr(self.shape_parameterization, 'ffd', None)

    def wall_geometry(self):
        """Geometric constraint quantities of the FFD-deformed wall (area/volume,
        thickness, planform); the shape parameterization's
        shape_parameterization.WallGeometry.

        Built on first use. Collective: call it on every rank.
        """
        if self.ffd is None:
            raise RuntimeError("wall_geometry needs a shape parameterization (an FFD block), "
                               "set up by set_up_sim")
        return self.shape_parameterization.geometry

    def deform_mesh(self):
        """Mesh node motions (CSDL) for the shape parameterization's current
        wall displacement: the first input of evaluate(). Call after the
        shape design variables are declared."""
        if not self.shape_parameterized:
            raise RuntimeError("deform_mesh needs a shape parameterization")
        shape = self.shape_parameterization
        return self.mesh_warper.evaluate(shape.wall_displacement(), shape.parameter_vector())

    def evaluate(self, mesh_node_motions:csdl.Variable = None, shape_parameters:csdl.Variable = None,
                 alpha:csdl.Variable = None):
        """
        The flow state u_vec as a CSDL output.

        Shape optimization (a shape parameterization was given):
        evaluate(mesh_node_motions, shape_parameters, alpha), with
        mesh_node_motions from deform_mesh() and shape_parameters the flat
        shape parameter vector. Forward analysis on the fixed mesh:
        evaluate(alpha=alpha). alpha is always required.
        """
        if alpha is None:
            raise ValueError("evaluate needs alpha")
        if (mesh_node_motions is None) != (shape_parameters is None):
            raise ValueError("pass mesh_node_motions and shape_parameters together, or neither")
        if mesh_node_motions is not None and not self.shape_parameterized:
            raise ValueError("mesh_node_motions given, but there is no shape parameterization / "
                             "mesh warper (pass shape_parameterization to DG_windtunnel_model)")

        # set inputs using self.declare_input
        if mesh_node_motions is not None:
            self.declare_input('mesh_node_motions', mesh_node_motions)
            self.declare_input('shape_parameters', shape_parameters)
        self.declare_input('alpha', alpha)

        # The output on this rank is only the local portion of the solution vector
        u_vec = self.create_output('u_vec', (self.sim_model.u_vec.x.petsc_vec.local_size,))

        if mesh_node_motions is not None:
            # We have declared shape_parameters as an input here, just for logging and computation purposes,
            # not because we actually need it; mesh_node_motions carries the effect of the shape_parameters.
            self.declare_derivative_parameters(of='u_vec', wrt='shape_parameters', dependent=False)

        return u_vec

    def solve_forward(self, alpha=None):
        """
        One flow solve on the current (undeformed) mesh, without CSDL: set
        alpha (radians; default: the boundary dictionary's), solve, print the
        coefficients. Warm-starts from the previous solve. Call set_up_sim()
        first. Returns (u_vec array, converged, forces), forces being
        DG_postprocessor.forces_and_coefficients (L, D, M, c_l, c_d, c_m, and
        the friction drag for a viscous model).
        """
        if alpha is not None:
            self.sim_model.set_angle_of_attack(alpha)
            self.postprocessor.set_angle_of_attack(alpha)
        self.sim_model.compute_weakform()
        solution, converged = self._solve(
            self._parameter_vector(None, self.sim_model.boundary_conditions['inlet']['alpha']))
        self.sim_model.last_solve_converged = converged
        if np.isfinite(np.linalg.norm(solution)):
            self.previous_solution = solution
        return solution, converged, self.postprocessor.forces_and_coefficients(solution)

    def solve_residual_equations(self, input_vals, output_vals):
        # Define inputs
        mesh_node_motions = input_vals['mesh_node_motions'] if 'mesh_node_motions' in input_vals else None
        shape_parameters = input_vals['shape_parameters'] if 'shape_parameters' in input_vals else None
        alpha = input_vals['alpha'][0]

        # Update the angle of attack to its current value
        self.sim_model.set_angle_of_attack(alpha)

        PETSc.Sys.Print("angle of attack: {} deg".format(np.degrees(alpha)))
        PETSc.Sys.Print("shape parameters: {}".format(shape_parameters))
        PETSc.Sys.Print("solve_residual_equations, stepping into mesh.apply_node_motions")
        if mesh_node_motions is not None:
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

        # --- flow solve (ROM, else FOM) -----------------------------------
        solution_output, solution_converged = self._solve(self._parameter_vector(shape_parameters, alpha))

        self._log_memory("after flow solve, before setting u_vec output")

        self.sim_model.last_solve_converged = solution_converged

        PETSc.Sys.Print("setting u_vec output...")
        output_vals['u_vec'] = solution_output

        # Warm start: keep the accepted solution unless it has blown up
        if np.isfinite(np.linalg.norm(solution_output)):
            PETSc.Sys.Print("Updating warm start solution")
            self.previous_solution = solution_output
        else:
            PETSc.Sys.Print("Not updating warm start solution, using previous value")

        PETSc.Sys.Print("Writing aligned output frame at eval_idx {}".format(self.eval_idx))
        # we write the deformation and the solution outputs to the same time step indices
        if mesh_node_motions is not None:
            self.mesh.write_deformation_output(mesh_node_motions, self.eval_idx)
        self.sim_model.write_solution_output(self.eval_idx,
                                             solution_array=solution_output)

        self.mesh.reset_nodes()

        # Explicitly collect garbage and cleanup PETSc objects
        gc.collect()
        PETSc.garbage_cleanup(self.mesh.mesh.comm)

        self._log_memory("after Euler PDE solve")

        self.eval_idx += 1

    @staticmethod
    def _parameter_vector(shape_parameters, alpha):
        """The ROM snapshot parameter vector of an evaluation: the flat shape
        parameters (if any), then alpha."""
        parts = [] if shape_parameters is None else [np.asarray(shape_parameters, dtype=float).ravel()]
        return np.concatenate(parts + [np.array([float(alpha)])])

    def _solve(self, parameter_vector):
        """
        The accepted flow solution of one evaluation: (owned solution entries,
        converged).

        Without a reduced-order model this is the full-order (FOM) solve from
        the warm start. With one, the ROM is tried first (once it holds enough
        snapshots), from the warm start projected onto the basis, and its
        solution is accepted when eta = ||R||/||u|| < eta_threshold. Otherwise
        the FOM solves from the warm start exactly as without a ROM, and its
        solution, if converged and finite, becomes a snapshot at
        `parameter_vector`.
        """
        rom = self.reduced_order_model
        if rom is None:
            return self._run_fom_solve()

        self.sim_model.interpolate_solution_vector(self.previous_solution)
        rom_result = rom.solve(self.sim_model, parameter_vector)
        rom_forces = None
        if rom_result is not None:
            rom_forces = self._log_rom_solve(rom_result)
        accept_rom = rom_result is not None and rom_result.accepted
        if accept_rom and not rom.run_fom_every_evaluation:
            PETSc.Sys.Print("Accepting ROM solution")
            return rom_result.solution, True

        solution_fom, fom_converged = self._run_fom_solve()
        if rom_forces is not None and fom_converged:
            self._log_rom_error(rom_forces)
        if accept_rom:
            PETSc.Sys.Print("Accepting ROM solution (FOM run for comparison)")
            return rom_result.solution, True

        PETSc.Sys.Print("Accepting FOM solution")
        finite = self.mesh.mesh.comm.allreduce(bool(np.all(np.isfinite(solution_fom))), op=MPI.LAND)
        if fom_converged and finite:
            self.sim_model.set_u_vec(solution_fom)
            rom.add_snapshot(self.sim_model, parameter_vector)
            sigma = rom.snapshot_singular_values()
            if self.mesh.mesh.comm.Get_rank() == 0 and sigma is not None:
                self.data_store.global_snapshot_sing_vals_per_iteration += [(self.eval_idx, sigma)]
        else:
            PETSc.Sys.Print("Not adding a ROM snapshot: the FOM solve did not converge")
        return solution_fom, fom_converged

    def _log_rom_solve(self, result):
        # coefficients of the ROM solution, logged like the FOM's
        _, cl = self.postprocessor.compute_cl(result.solution)
        _, cd = self.postprocessor.compute_cd(result.solution)
        _, cm = self.postprocessor.compute_cm(result.solution)
        if self.mesh.mesh.comm.Get_rank() == 0:
            self.data_store.ROM_walltime += [(self.eval_idx, result.walltime)]
            self.data_store.rom_relative_residuals += [(self.eval_idx, result.eta)]
            self.data_store.ROM_force_coefficients += [(self.eval_idx, cd, cl, cm, np.nan)]
            self.data_store.ROM_meets_threshold += [(self.eval_idx, result.accepted)]
        PETSc.Sys.Print("ROM coefficients: c_d: {}, c_l: {}, c_m: {}".format(cd, cl, cm))
        return cd, cl, cm

    def _log_rom_error(self, rom_forces):
        # ROM-vs-FOM coefficient errors against the FOM solve just run
        errors = tuple(float(r - f) for r, f in zip(rom_forces, self._last_fom_coefficients))
        if self.mesh.mesh.comm.Get_rank() == 0:
            self.data_store.rom_coefficient_errors += [(self.eval_idx,) + errors]
        PETSc.Sys.Print("ROM - FOM coefficient errors: c_d: {:.3e}, c_l: {:.3e}, c_m: {:.3e}".format(*errors))

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
        self._last_fom_coefficients = (cd_fom, cl_fom, cm_fom)

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

        mesh_node_motions = input_vals['mesh_node_motions'] if 'mesh_node_motions' in input_vals else None
        alpha = input_vals['alpha'][0]
        u_vec = outputs['u_vec'] # numpy array

        if mode == 'rev':
            # compute d_residual = (dr_du^-1)*d_outputs

            # apply the input mesh node motions to the mesh of the sim model
            # and set the angle of attack before constructing the weak form
            # and the corresponding solution that we are linearizing around
            if mesh_node_motions is not None:
                PETSc.Sys.Print("apply_inverse_jacobian, stepping into mesh.apply_node_motions")
                self.mesh.apply_node_motions(mesh_node_motions)
            self.sim_model.set_angle_of_attack(alpha)
            self.sim_model.compute_weakform()
            self.sim_model.set_u_vec(u_vec)

            if self.sim_model.has_condensed_jacobian:
                self._condensed_adjoint(d_outputs, d_residuals)
                self.mesh.reset_nodes()
                PETSc.garbage_cleanup()
                PETSc.Sys.Print("apply_inverse_jacobian total time: {}".format(
                    perf_counter() - pre_apply_inverse_jacobian_time))
                return

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

            if mesh_node_motions is not None:
                PETSc.Sys.Print("mesh_node_motions min, max: {}, {}".format(mesh_node_motions.min(), mesh_node_motions.max()))
                PETSc.Sys.Print("shape parameters: {}".format(input_vals['shape_parameters']))
            PETSc.Sys.Print("alpha: {}".format(np.degrees(alpha)))

            d_residuals['u_vec'] = self._d_residual_petsc.getArray()

            PETSc.garbage_cleanup()

            self.mesh.reset_nodes()

            post_apply_inverse_jacobian_time = perf_counter()
            total_apply_inverse_jacobian_time = post_apply_inverse_jacobian_time - pre_apply_inverse_jacobian_time

            PETSc.Sys.Print("apply_inverse_jacobian total time: {}".format(total_apply_inverse_jacobian_time))
        else:
            raise ValueError("Reverse mode required")

    def _condensed_adjoint(self, d_outputs, d_residuals):
        """The adjoint solve for a model whose dR/dU is a condensed product
        (a reconstructed gradient): the model's own transposed linearized
        solve (CompressibleEulerModel.solve_linearized)."""
        A = self.sim_model.assemble_dRdu_mat()
        rhs, sol = A.createVecLeft(), A.createVecLeft()
        set_petsc_vec_array(rhs, d_outputs['u_vec'])
        ksp = self.sim_model.solve_linearized(
            rhs, sol, transpose=True, prefix="adj_", rtol=self.linear_solver_rtol,
            atol=self.linear_solver_atol, max_it=self.linear_solver_max_it)
        rnorm = ksp.getResidualNorm()
        PETSc.Sys.Print("adj_ solve: {} iterations, residual norm {} [{}]".format(
            ksp.getIterationNumber(), rnorm, ksp_converged_reason_name(ksp.getConvergedReason())))
        ksp.destroy()
        if rnorm > self.linear_residual_accept_limit:
            raise ValueError(
                "apply_inverse_jacobian residual norm is too high: {:.6e} > "
                "linear_residual_accept_limit {:.1e}".format(rnorm, self.linear_residual_accept_limit))
        d_residuals['u_vec'] = sol.getArray().copy()

    def compute_jacvec_product(self, input_vals, outputs, d_inputs, d_outputs, d_residuals, mode):
        # Here we compute the product of the adjoint solution with d_residual
        pre_compute_jacvec_product_time = perf_counter()
        print("starting compute_jacvec_product in DG_windtunnel_model...")

        mesh_node_motions = input_vals['mesh_node_motions'] if 'mesh_node_motions' in input_vals else None
        alpha = input_vals['alpha'][0]
        u_vec = outputs['u_vec'] # numpy array

        if mode == 'rev':
            # compute d_input = (dr_dinput^T)*d_residual

            # apply the input mesh node motions to the mesh of the sim model,
            # apply the angle of attack and compute the weak form
            if mesh_node_motions is not None:
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
