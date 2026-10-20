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
from dragonfly_sim.utils.derived_fields import destroy_chained_operator
from dragonfly_sim.utils.solver_utils import ksp_converged_reason_name
from dragonfly_sim.core.postprocessor import DG_postprocessor


# TODO: Look into using entropy variables instead of conservative variables;
#       these would potentially allow us to do simulations from low Mach all
#       the way to supersonic
class DG_windtunnel_model(csdl.experimental.CustomImplicitOperation):
    # The solve/adjoint callbacks contain MPI collectives (PETSc, mpi4py) and deform
    # the shared mesh, so they must never overlap -- also on a single process.
    ordered_callbacks = True

    def __init__(self, mesh, boundary_dict, shape_parameterization=None, moments=None,
                 mesh_inner_bdry_function=None, wall_tags=None, symmetry_tags=None,
                 poly_order=1, gamma=1.4,
                 filename_suffix="test",
                 asm_overlap=None, ilu_levels=None,
                 model_class=CompressibleEulerModel, model_kwargs=None,
                 reduced_order_model=None, compute_all_adjoint_errors=False):
        """
        boundary_dict: the free-stream state {'rho', 'p', 'M', 'alpha'} (density,
        pressure, Mach number, angle of attack in rad). The whole outer boundary
        is a characteristic far field built from it, so the angle of attack can
        change by any amount during an optimization; there is no inflow/outflow
        division to choose or to keep valid.

        The shape of the object being optimized (the wall) is defined in one of
        three ways, in this order of precedence:
        1. `mesh_inner_bdry_function`, a function of the boundary facet midpoints
           (shape (dim, n)) returning a boolean wall mask. It can encode
           exclusions the FFD block cannot. If the mesh also carries boundary
           facet tags (utils/mesh_io_utils.py:load_dolfinx_mesh stores them on
           the mesh), a warning lists them: they could be used instead.
        2. `wall_tags`: the mesh's boundary facet tags (values or names, one or
           several) that are the wall, and `symmetry_tags` those that are
           symmetry planes (slip walls). Every other boundary is far field.
           In 3D without tags, the symmetry plane is y = 0.
        3. The shape parameterization's FFD block corners
           (shape_parameterization.inner_boundary_function()), which bound the
           box the inner boundary facets are contained in.
        At least one is required.

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

        moments: the moments of the wall force to compute, print and differentiate,
        a sequence of (name, point, direction[, ref_length]) tuples (see
        DG_postprocessor); default none. E.g. the pitching moment about the quarter
        chord of a wing, ("pitch", (0.25, 0., 0.), "pitch"), and its root
        bending moment, ("root_bending", (0.25, 0., 0.), (1., 0., 0.)).

        model_class / model_kwargs select the flow model: CompressibleEulerModel
        (the default) or e.g. core.RANS_model.CompressibleRANSModel with
        model_kwargs={'Re': 6e6}. The body's wall is the model's
        define_wall_bc (slip for Euler, adiabatic no-slip for RANS).

        reduced_order_model: a core.reduced_order_model.ReducedOrderModel to
        try before every flow solve (see _solve), or None for full-order
        solves only. Where its solution is accepted, the derivatives use the
        adjoint of its projected equations (see apply_inverse_jacobian).

        compute_all_adjoint_errors: diagnostic, default False. When a derivative
        uses the ROM adjoint, also solve the full-order adjoint at the same state
        and right-hand side and record the ROM adjoint's error against it in
        data_store.adjoint_errors. The adjoint handed to the optimizer is the
        ROM's either way; the extra full-order solve costs about as much as the
        adjoint the ROM saves, and is not part of data_store.adjoint_solves.
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
        if mesh_inner_bdry_function is None and wall_tags is None and shape_parameterization is not None:
            mesh_inner_bdry_function = shape_parameterization.inner_boundary_function()
        if mesh_inner_bdry_function is None and wall_tags is None:
            raise ValueError("Provide either mesh_inner_bdry_function, wall_tags (the mesh's own "
                             "boundary tags) or a shape parameterization with FFD block corners "
                             "to locate the inner boundary facets.")

        # Initial instantiation of mesh and simulation model
        self.mesh = Mesh(mesh, mesh_inner_bdry_function,
                         node_match_tol=self.mesh_node_match_tol,
                         wall_tags=wall_tags, symmetry_tags=symmetry_tags)
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
        self.compute_all_adjoint_errors = compute_all_adjoint_errors
        # (eval_idx, number of adjoint solves so far at it): the output row of
        # the derivative evaluation that CSDL is on
        self._adjoint_row = (None, 0)
        # (parameter vector, converged) of the last evaluation if the FOM
        # solved it, else None: a re-evaluation of that design skips the ROM
        # (see _solve)
        self._fom_design = None

        # Naming suffix for exported files
        self.filename_suffix = filename_suffix

        # Free the flow solver's matrices after every forward solve and the
        # condensed dR/dU after every condensed adjoint
        # (sim_model.release_solver), so the two never coexist; they are
        # rebuilt when next needed. For memory-bound runs (3D RANS).
        self.release_solver_memory = False

        # Adjoint reuse. The transposed dR/dU and the preconditioned KSP of the
        # state last linearized (condensed Jacobian), kept for the adjoint
        # solves of every output of that design: {'key': state key, 'ksp': KSP};
        # freed when the next flow solve starts or a different state is
        # linearized.
        self._adjoint_cache = None

        # Options of the adjoint of a model with a condensed Jacobian (default off;
        # set_adjoint_options sets them together with those of the reconstruction):
        #   adjoint_matrix_free   the adjoint operator A_UU^T x + B^T (A_UG^T x), without
        #                         forming A or the product A_UG B
        #   cache_jacvec_operators  the dR/dx operator and dR/dalpha of the
        #                         compute_jacvec_product calls of all outputs of one
        #                         design are built once (freed with the adjoint cache)
        self.adjoint_matrix_free = False
        self.cache_jacvec_operators = False
        # PETSc options (names without the "adj_" prefix) overriding the adjoint KSP's settings of the
        # model, e.g. {"ksp_type": "hpddm", "ksp_hpddm_type": "gcrodr", "ksp_hpddm_recycle": 20}; the KSP
        # is kept for all outputs of a design, so a recycling method carries its subspace between them
        self.adjoint_ksp_options = None
        self._dRdx_cache = None
        self._dRdalpha_cache = None

        # Start each condensed adjoint solve from a first-order solution: the
        # solution of the same transposed system with the compact (nearest
        # neighbour, gradients frozen) part A_UU of the Jacobian alone -- the
        # first-order Jacobian the preconditioner is built from -- solved to a
        # relative tolerance first_order_adjoint_rtol. It reuses the
        # preconditioner's factors, so costs only Krylov iterations. Applies to
        # models with a condensed Jacobian (MUSCL / RANS); the guess is always
        # used, and its effect is logged (self._last_first_order_guess).
        self.adjoint_first_order_guess = False
        self.first_order_adjoint_rtol = 1e-2
        self._last_first_order_guess = None

        # Write mesh_deformation_<suffix>.bp (the node motions of every design
        # evaluation). The solution files are written either way.
        self.write_mesh_deformation = True

        # Write the output frame (solution and mesh deformation) only for the
        # first evaluation of each design: an optimizer that evaluates the
        # same design again (CSDL re-runs the model for the derivatives) then
        # does not duplicate the frame. Frames keep the eval_idx of that first
        # evaluation, so their labels have gaps.
        self.write_once_per_design = False
        self._last_written_design = None


        # Discretization and physics-related parameters
        self.poly_order:int = poly_order
        self.gamma:float = gamma
        self.boundary_condition_dict = boundary_dict
        self.moments = moments

        # Dolfinx mesh tags for the various boundaries
        self.WALL_TAG = 2  # walls (including wing/airfoil boundaries)
        self.SYMMETRY_TAG = 5  # symmetry planes (3D: y = 0, or the mesh's symmetry_tags)
        self.FARFIELD_TAG = 6  # characteristic far field: the whole outer boundary

        # global evaluation index of solve_residual_equations
        self.eval_idx = 0

    def set_up_sim(self):
        # pass the various tolerances and limits to the Euler sim model
        self.sim_model.residual_conv_limit = self.euler_residual_conv_limit
        self.sim_model.max_newton_iterations = self.max_newton_iterations

        # use the free-stream dictionary to define the far-field conditions
        self.sim_model.define_farfield_conditions(self.boundary_condition_dict)

        # compute boundary facet boolean arrays
        inner_facets = self.mesh.inner_bdry_facet_mask
        
        # Find the symmetry face facets (only relevant in 3D at the moment)
        # NOTE: The symmetry plane is currently hardcoded at y=0
        has_symmetry = self.mesh.symmetry_facet_mask is not None or self.sim_model.dimensions == 3
        if self.mesh.symmetry_facet_mask is not None:
            symmetry_facets = self.mesh.symmetry_facet_mask  # from the mesh's boundary tags
        elif self.sim_model.dimensions == 3:
            symmetry_facets = self.mesh.bdry_midpoints[:, 1] <= 1e-6
        else:
            symmetry_facets = np.array([False]*inner_facets.shape[0])

        # Everything that is neither wall nor symmetry plane is far field. The
        # characteristic far-field state decides facet by facet whether the flow
        # enters or leaves, so no inflow/outflow division (and nothing to
        # re-tag when the angle of attack changes) is needed.
        farfield_facets = ~np.asarray(inner_facets, dtype=bool) & ~np.asarray(symmetry_facets, dtype=bool)

        # Print the local number of facets of each type for each MPI rank
        PETSc.Sys.syncPrint("Rank {}, number of wing boundary facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(inner_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of symmetry facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(symmetry_facets)))
        PETSc.Sys.syncPrint("Rank {}, number of far-field facets: {}".format(self.mesh.mesh.comm.Get_rank(), np.count_nonzero(farfield_facets)))
        PETSc.Sys.syncFlush()

        # Pair the boundary facet sets with the various mesh tags
        bdry_meshtags_and_ds_list = [(farfield_facets, self.FARFIELD_TAG), (inner_facets, self.WALL_TAG)]
        if has_symmetry:
            bdry_meshtags_and_ds_list += [(symmetry_facets, self.SYMMETRY_TAG)]

        # Build the boundary facet tags and define the corresponding quadrature measure
        self.mesh.tag_boundary_facets(bdry_meshtags_and_ds_list)
        self.mesh.form_manager.build_boundary_measure(
            measure_metadata={'quadrature_degree': 3*self.poly_order + 1})

        self.mesh.configure_output(self.filename_suffix,
                                   deformation=self.shape_parameterized and self.write_mesh_deformation)

        if self.shape_parameterized:
            # The JAX-based IDWarp implementation as mesh warper
            self.mesh_warper = IDWarp_jax(self.mesh, wall_meshtag_val=self.WALL_TAG,
                                          anchor_meshtag_vals=(self.FARFIELD_TAG,))

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
        self.sim_model.compute_initial_conditions_from_farfield()
        self.sim_model.interpolate_solution_vector()

        # We output the facet mesh tags to an xdmf file, so we can inspect the BCs in paraview
        self.mesh.mesh.topology.create_connectivity(self.sim_model.dimensions-1, self.sim_model.dimensions)
        with dolfinx.io.XDMFFile(self.mesh.mesh.comm, "output_meshtags.xdmf", "w") as xdmf:
            # Write the Mesh object to the file
            xdmf.write_mesh(self.mesh.mesh)
            # Write the MeshTags object to the file
            xdmf.write_meshtags(self.mesh.meshtags, self.mesh.mesh.geometry)

        # Define the boundary condition terms of the Euler weak form
        self.sim_model.define_farfield_bc(self.FARFIELD_TAG)
        self.sim_model.define_wall_bc(self.WALL_TAG)
        if has_symmetry:
            self.sim_model.define_slipwall_bc(self.SYMMETRY_TAG)

        # Construct the Euler weak form for forward evaluation
        self.sim_model.compute_weakform()

        # Define the file names and objects for solution writes to external files for Paraview
        self.sim_model.define_sol_export_files(self.sim_model.functionspaces["V"],
                                               file_name_addendum=self.filename_suffix,
                                               mesh_deformation=self.shape_parameterized)

        # We instantiate a separate DG_postprocessor object here so we can print 
        # forces and moments and coefficients after each forward solve
        self.postprocessor = DG_postprocessor(
            self.mesh, self.sim_model, self.WALL_TAG,
            moments=self.moments,
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

    def solve_forward(self, alpha=None, snapshot=True):
        """
        One flow solve on the current (undeformed) mesh, without CSDL: set
        alpha (radians; default: the boundary dictionary's), solve, print the
        coefficients. Warm-starts from the previous solve. Call set_up_sim()
        first. Returns (u_vec array, converged, forces), forces being
        DG_postprocessor.forces_and_coefficients (L, D, c_l, c_d, the moments, and
        the friction drag for a viscous model).

        The ROM (if any) stores the solve as a snapshot at the parameter vector
        (alpha). With shape parameters that vector is longer, so a solve for a
        baseline that is not a design of the optimization passes snapshot=False.
        """
        if alpha is not None:
            self.sim_model.set_angle_of_attack(alpha)
            self.postprocessor.set_angle_of_attack(alpha)
        self.sim_model.compute_weakform()
        rom, self.reduced_order_model = self.reduced_order_model, (self.reduced_order_model if snapshot else None)
        try:
            solution, converged = self._solve(
                self._parameter_vector(None, self.sim_model.farfield_conditions['alpha']))
        finally:
            self.reduced_order_model = rom
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
            self.mesh.apply_node_motions(mesh_node_motions, design=shape_parameters)

        # If the mesh is not valid (e.g. has inverted cells), set the output to zero and don't run any simulations
        if not self.mesh.is_valid:
            PETSc.Sys.Print("Mesh is not valid, skipping simulations.")
            output_vals['u_vec'] = np.zeros((self.sim_model.u_vec.x.petsc_vec.local_size,))
            self._fom_design = None
            if self.reduced_order_model is not None:
                self.reduced_order_model.discard_adjoint()
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

        design = np.concatenate([[alpha], np.ravel(shape_parameters)
                                 if shape_parameters is not None else []])
        if (self.write_once_per_design and self._last_written_design is not None
                and np.array_equal(design, self._last_written_design)):
            PETSc.Sys.Print("Same design as the last written frame: not writing eval_idx {}".format(
                self.eval_idx))
        else:
            PETSc.Sys.Print("Writing aligned output frame at eval_idx {} (FOM solution)".format(
                self.eval_idx))
            # we write the deformation and the solution outputs to the same time step indices
            if mesh_node_motions is not None:
                self.mesh.write_deformation_output(mesh_node_motions, self.eval_idx)
            self.sim_model.write_solution_output(self.eval_idx,
                                                 solution_array=solution_output)
            self._last_written_design = design

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
        solution is accepted when eta = ||R||/||u|| < eta_threshold; the ROM
        then holds the adjoint of that solution (rom.adjoint), which
        apply_inverse_jacobian uses. Otherwise the FOM solves from the warm
        start exactly as without a ROM, and its solution, if converged and
        finite, becomes a snapshot at `parameter_vector`.

        A re-evaluation of the design the FOM solved last skips the ROM: the
        FOM restarts from its own solution (no Newton steps if that
        converged), so the design keeps the FOM solution and the full-order
        adjoint. CSDL re-runs the model before every derivative evaluation,
        and by then the ROM would reproduce that solution, now a snapshot, and
        accept it.
        """
        # the forward solve needs the memory of the last design's adjoint operators
        self._release_adjoint_cache()
        rom = self.reduced_order_model
        if rom is None:
            return self._run_fom_solve()
        rom.discard_adjoint()

        repeat = self._fom_design is not None and np.array_equal(parameter_vector, self._fom_design[0])
        rom_result = rom_forces = None
        if repeat:
            PETSc.Sys.Print("Re-evaluating the design the FOM solved last: skipping the ROM")
        else:
            self.sim_model.interpolate_solution_vector(self.previous_solution)
            rom_result = rom.solve(self.sim_model, parameter_vector)
            if rom_result is not None:
                rom_forces = self._log_rom_solve(rom_result)
        accept_rom = rom_result is not None and rom_result.accepted
        if accept_rom and not rom.run_fom_every_evaluation:
            PETSc.Sys.Print("Accepting ROM solution")
            self._fom_design = None
            return rom_result.solution, True

        solution_fom, fom_converged = self._run_fom_solve()
        if rom_forces is not None and fom_converged:
            self._log_rom_error(rom_forces)
            self._log_rom_state_error(rom_result.solution, solution_fom)
        if accept_rom:
            PETSc.Sys.Print("Accepting ROM solution (FOM run for comparison)")
            self._fom_design = None
            return rom_result.solution, True

        PETSc.Sys.Print("Accepting FOM solution")
        finite = self.mesh.mesh.comm.allreduce(bool(np.all(np.isfinite(solution_fom))), op=MPI.LAND)
        converged = fom_converged and finite
        if converged and repeat and self._fom_design[1]:
            PETSc.Sys.Print("The snapshot of this design is already stored")
        elif converged:
            self.sim_model.set_u_vec(solution_fom)
            rom.add_snapshot(self.sim_model, parameter_vector)
            sigma = rom.snapshot_singular_values()
            if self.mesh.mesh.comm.Get_rank() == 0 and sigma is not None:
                self.data_store.global_snapshot_sing_vals_per_iteration += [(self.eval_idx, sigma)]
        else:
            PETSc.Sys.Print("Not adding a ROM snapshot: the FOM solve did not converge")
        self._fom_design = (np.array(parameter_vector, dtype=float), converged)
        return solution_fom, fom_converged

    def _log_rom_solve(self, result):
        # coefficients of the ROM solution, logged like the FOM's
        outputs, (cd, cl, *cms) = self._evaluate_outputs(result.solution)
        cms = tuple(cms)
        if self.mesh.mesh.comm.Get_rank() == 0:
            self.data_store.record_outputs("ROM", self.eval_idx, outputs)
            self.data_store.ROM_walltime += [(self.eval_idx, result.walltime)]
            self.data_store.rom_relative_residuals += [(self.eval_idx, result.eta)]
            self.data_store.ROM_force_coefficients += [(self.eval_idx, cd, cl) + cms]
            self.data_store.ROM_meets_threshold += [(self.eval_idx, result.accepted)]
        PETSc.Sys.Print("ROM coefficients: c_d: {}, c_l: {}".format(cd, cl) + self._moment_log_string(cms))
        return (cd, cl) + cms

    def _log_rom_error(self, rom_forces):
        # ROM-vs-FOM coefficient errors against the FOM solve just run
        errors = tuple(float(r - f) for r, f in zip(rom_forces, self._last_fom_coefficients))
        if self.mesh.mesh.comm.Get_rank() == 0:
            self.data_store.rom_coefficient_errors += [(self.eval_idx,) + errors]
        PETSc.Sys.Print("ROM - FOM coefficient errors: c_d: {:.3e}, c_l: {:.3e}".format(*errors[:2])
                        + "".join(", c_m_{}: {:.3e}".format(spec.name, e)
                                  for spec, e in zip(self.postprocessor.moment_specs, errors[2:])))

    def _evaluate_outputs(self, solution):
        """Every output of the postprocessor for a flow state: ({name: value},
        (c_d, c_l, c_m_1, ...)). The names are DG_postprocessor.evaluate's (L, D,
        c_l, c_d, M_<name>, c_m_<name>); the tuple is the layout of the
        *_force_coefficients records, one moment per moment spec, in order."""
        L, c_l = self.postprocessor.compute_cl(solution)
        D, c_d = self.postprocessor.compute_cd(solution)
        outputs = {"L": L, "D": D, "c_l": c_l, "c_d": c_d}
        cms = []
        for spec in self.postprocessor.moment_specs:
            M, c_m = self.postprocessor.compute_moment(solution, spec)
            outputs["M_" + spec.name], outputs["c_m_" + spec.name] = M, c_m
            cms.append(c_m)
        return outputs, (c_d, c_l, *cms)

    def _log_rom_state_error(self, solution_rom, solution_fom):
        # absolute and relative L2 errors of the ROM state against the FOM state just
        # run, over the domain and over the wall. Both are this rank's owned entries.
        if not self.mesh.mesh.comm.allreduce(bool(np.all(np.isfinite(solution_rom))), op=MPI.LAND):
            return
        diff = np.subtract(solution_fom, solution_rom)
        l2, l2_rel = self.sim_model.compute_L2_norm(diff, normalization_vec=solution_fom)
        l2_wall, l2_wall_rel = self.sim_model.compute_L2_norm(
            diff, normalization_vec=solution_fom, wall_tag=self.WALL_TAG)
        if self.mesh.mesh.comm.Get_rank() == 0:
            ds = self.data_store
            ds.absolute_l2_errors_per_iteration += [(self.eval_idx, l2)]
            ds.relative_l2_errors_per_iteration += [(self.eval_idx, l2_rel)]
            ds.absolute_l2_bdry_errors_per_iteration += [(self.eval_idx, l2_wall)]
            ds.relative_l2_bdry_errors_per_iteration += [(self.eval_idx, l2_wall_rel)]
        PETSc.Sys.Print("ROM L2 error relative to the FOM: domain {:.3e}, wall {:.3e}".format(l2_rel, l2_wall_rel))

    def _moment_log_string(self, coefficients):
        return "".join(", c_m_{}: {}".format(spec.name, c)
                       for spec, c in zip(self.postprocessor.moment_specs, coefficients))

    def _run_fom_solve(self):
        # Run the FOM model and output its solution

        # store pre-FOM time stamp for performance profiling
        fom_pre_time = perf_counter()

        # use the most recent converged solution as warm start
        initial_condition = self.previous_solution
        self.sim_model.interpolate_solution_vector(initial_condition)

        # solve the FOM
        # an incoming state that already meets the convergence criterion
        # needs no solver (SNES, KSP/PC, matrices): skip building it
        if self.sim_model.incoming_state_converged():
            residual_norm = self.sim_model.residual_vec_physical.norm()
        else:
            self.sim_model.set_up_solver()
            residual_norm = self.sim_model.solve_system().norm()
        fom_converged = self.sim_model.solver_converged
        eta = self.sim_model.relative_residual_norm(residual_norm)
        solution_fom = self.sim_model.u_vec.x.petsc_vec.getArray().copy()
        if self.release_solver_memory:
            self.sim_model.release_solver()

        self._log_memory("after FOM solve, before doing anything else")

        # store post-FOM time stamp for performance profiling
        fom_post_time = perf_counter()
        PETSc.Sys.Print("FOM wall time: {}".format(fom_post_time - fom_pre_time))

        outputs_fom, (cd_fom, cl_fom, *cms_fom) = self._evaluate_outputs(solution_fom)
        cms_fom = tuple(cms_fom)
        self._last_fom_coefficients = (cd_fom, cl_fom) + cms_fom

        if self.mesh.mesh.comm.Get_rank() == 0:
            self.data_store.record_outputs("FOM", self.eval_idx, outputs_fom)
            self.data_store.FOM_walltime += [(self.eval_idx, fom_post_time - fom_pre_time)]
            self.data_store.fom_relative_residuals += [(self.eval_idx, eta)]
            self.data_store.FOM_force_coefficients += [(self.eval_idx, cd_fom, cl_fom) + cms_fom]
        PETSc.Sys.Print("FOM coefficients: c_d: {}, c_l: {}".format(cd_fom, cl_fom) + self._moment_log_string(cms_fom))
        PETSc.Sys.Print("FOM relative residual norm eta: {} (||r||: {})".format(
            eta, residual_norm))

        return solution_fom, fom_converged

    def apply_inverse_jacobian(self, input_vals, outputs, d_outputs, d_residuals, mode):
        pre_apply_inverse_jacobian_time = perf_counter()
        # Here we solve the adjoint equation when calculating derivatives for optimization
        print("starting apply_inverse_jacobian in DG_windtunnel_model...")

        mesh_node_motions = input_vals['mesh_node_motions'] if 'mesh_node_motions' in input_vals else None
        shape_parameters = input_vals['shape_parameters'] if 'shape_parameters' in input_vals else None
        alpha = input_vals['alpha'][0]
        u_vec = outputs['u_vec'] # numpy array

        if mode == 'rev':
            # compute d_residual = (dr_du^-1)*d_outputs

            # apply the input mesh node motions to the mesh of the sim model
            # and set the angle of attack before constructing the weak form
            # and the corresponding solution that we are linearizing around
            if mesh_node_motions is not None:
                PETSc.Sys.Print("apply_inverse_jacobian, stepping into mesh.apply_node_motions")
                self.mesh.apply_node_motions(mesh_node_motions, design=shape_parameters)
            self.sim_model.set_angle_of_attack(alpha)
            self.sim_model.compute_weakform()
            self.sim_model.set_u_vec(u_vec)

            rom = self.reduced_order_model
            if (rom is not None and rom.adjoint is not None
                    and rom.adjoint.matches(self._parameter_vector(shape_parameters, alpha), u_vec)):
                # The accepted solution is the ROM's: the adjoint of its
                # projected (LSPG) equations, lifted to the full state, which
                # compute_jacvec_product then uses like the full-order one
                d_residuals['u_vec'] = rom.adjoint.solve(self.sim_model, d_outputs['u_vec'])
                self._log_adjoint("rom", pre_apply_inverse_jacobian_time)
                if self.compute_all_adjoint_errors:
                    # after the ROM adjoint's own timing; d_residuals is not touched
                    self._log_adjoint_error(input_vals, u_vec, alpha, mesh_node_motions,
                                            d_outputs, d_residuals['u_vec'])
                return

            self._fom_adjoint(input_vals, u_vec, alpha, mesh_node_motions, d_outputs, d_residuals,
                              pre_apply_inverse_jacobian_time)
        else:
            raise ValueError("Reverse mode required")

    def _fom_adjoint(self, input_vals, u_vec, alpha, mesh_node_motions, d_outputs, d_residuals,
                     start_time, log=True):
        """The full-order adjoint psi = (dR/dU)^-T d_outputs['u_vec'] at the model's current
        state, written to d_residuals['u_vec']. log=False leaves it out of data_store.adjoint_solves
        (the reference solve of compute_all_adjoint_errors)."""
        if self.sim_model.has_condensed_jacobian:
            self._condensed_adjoint(d_outputs, d_residuals,
                                    self._adjoint_state_key(u_vec, alpha, mesh_node_motions))
            if self.release_solver_memory:
                self.sim_model.release_solver()
            PETSc.garbage_cleanup()
            if log:
                self._log_adjoint("fom", start_time)
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

        if log:
            self._log_adjoint("fom", start_time)

    def _log_adjoint_error(self, input_vals, u_vec, alpha, mesh_node_motions, d_outputs, psi_rom):
        """Record the error of the ROM adjoint psi_rom against the full-order adjoint at the
        same state, design and right-hand side: (eval_idx, output row, absolute 2-norm error,
        relative error, ||psi_fom||) in data_store.adjoint_errors. The row is the order of the
        adjoint solves within the evaluation, i.e. of CSDL's derivative rows. Collective."""
        scratch = {}
        try:
            self._fom_adjoint(input_vals, u_vec, alpha, mesh_node_motions, d_outputs, scratch,
                              perf_counter(), log=False)
        except ValueError as exc:
            # a diagnostic must not stop the optimization
            PETSc.Sys.Print("adjoint error: the full-order reference adjoint failed: {}".format(exc))
            return
        # copy: the plain path returns a view of a PETSc vector reused by the next solve
        psi_fom = np.array(scratch['u_vec'], dtype=float)
        comm = self.mesh.mesh.comm
        diff = comm.allreduce(float(np.sum((np.asarray(psi_rom) - psi_fom)**2)), op=MPI.SUM)
        ref = comm.allreduce(float(psi_fom @ psi_fom), op=MPI.SUM)
        abs_err, fom_norm = np.sqrt(diff), np.sqrt(ref)
        rel_err = abs_err / fom_norm if fom_norm > 0.0 else np.nan

        eval_idx = self.eval_idx - 1
        row = self._adjoint_row[1] if self._adjoint_row[0] == eval_idx else 0
        self._adjoint_row = (eval_idx, row + 1)
        if comm.Get_rank() == 0:
            self.data_store.adjoint_errors += [(eval_idx, row, float(abs_err), float(rel_err), float(fom_norm))]
        PETSc.Sys.Print("ROM adjoint error (output row {}): |psi_rom - psi_fom| = {:.3e}, relative {:.3e}".format(
            row, abs_err, rel_err))

    def _log_adjoint(self, kind, start_time):
        # which adjoint ('rom' or 'fom') the derivative of the last evaluation
        # used, and its wall time
        walltime = perf_counter() - start_time
        if self.mesh.mesh.comm.Get_rank() == 0:
            self.data_store.adjoint_solves += [(self.eval_idx - 1, kind, walltime)]
        PETSc.Sys.Print("apply_inverse_jacobian total time ({} adjoint): {}".format(kind, walltime))

    def _condensed_adjoint(self, d_outputs, d_residuals, state_key):
        """The adjoint solve for a model whose dR/dU is a condensed product
        (a reconstructed gradient): the model's own transposed linearized
        solve (CompressibleEulerModel.linearized_solver). The transposed
        matrix and the preconditioned KSP are built once per state and reused
        by the adjoint solves of all outputs at it."""
        if self._adjoint_cache_matches(state_key):
            PETSc.Sys.Print("adj_: reusing the transposed Jacobian and preconditioner of this state")
        else:
            self._release_adjoint_cache()
            if self.release_solver_memory:
                # free queued PETSc objects before the large assembly
                gc.collect()
                PETSc.garbage_cleanup(self.mesh.mesh.comm)
            # the condensed dR/dU is freed once its transposes exist: only they are used from then on
            ksp = self.sim_model.linearized_solver(
                transpose=True, prefix="adj_", rtol=self.linear_solver_rtol,
                atol=self.linear_solver_atol, max_it=self.linear_solver_max_it,
                release_operands=True, matrix_free=self.adjoint_matrix_free,
                ksp_options=self.adjoint_ksp_options)
            self._adjoint_cache = {'key': state_key, 'ksp': ksp}
        ksp = self._adjoint_cache['ksp']
        A = ksp.getOperators()[0]
        rhs, sol = A.createVecLeft(), A.createVecLeft()
        set_petsc_vec_array(rhs, d_outputs['u_vec'])
        guess_used = self.adjoint_first_order_guess and self._first_order_adjoint_guess(ksp, A, rhs, sol)
        ksp.setInitialGuessNonzero(bool(guess_used))
        ksp.solve(rhs, sol)
        rnorm = ksp.getResidualNorm()
        PETSc.Sys.Print("adj_ solve: {} iterations, residual norm {} [{}]".format(
            ksp.getIterationNumber(), rnorm, ksp_converged_reason_name(ksp.getConvergedReason())))
        if rnorm > self.linear_residual_accept_limit:
            raise ValueError(
                "apply_inverse_jacobian residual norm is too high: {:.6e} > "
                "linear_residual_accept_limit {:.1e}".format(rnorm, self.linear_residual_accept_limit))
        d_residuals['u_vec'] = sol.getArray().copy()
        rhs.destroy()
        sol.destroy()

    # ---- adjoint reuse: cached operators -----------------------------------
    def _first_order_adjoint_guess(self, ksp, A, rhs, sol):
        """
        Fill `sol` with the first-order adjoint solution of `rhs`: the solution
        of P x = rhs, P the transposed compact part A_UU of the Jacobian (the
        KSP's preconditioning matrix), to a relative tolerance
        first_order_adjoint_rtol. The KSP is switched to operator P for the
        solve, which keeps its preconditioner (P's factors), and back to
        (A, P) afterwards, with its tolerances. Returns False, leaving `sol`
        zero, if the first-order solve did not converge. The residual of the
        exact system at the guess, |rhs - A guess| / |rhs|, the iterations and
        the time are logged and kept in self._last_first_order_guess.
        Collective.
        """
        P = ksp.getOperators()[1]
        if P.handle == A.handle:
            return False                   # direct solver: A is P, nothing to precondition with
        pre_time = perf_counter()
        rtol, atol, divtol, max_it = ksp.getTolerances()
        ksp.setOperators(P, P)
        ksp.setTolerances(rtol=self.first_order_adjoint_rtol, atol=0.0, max_it=self.linear_solver_max_it)
        ksp.setInitialGuessNonzero(False)
        try:
            ksp.solve(rhs, sol)
            iterations, reason = ksp.getIterationNumber(), ksp.getConvergedReason()
        finally:
            ksp.setOperators(A, P)
            ksp.setTolerances(rtol=rtol, atol=atol, divtol=divtol, max_it=max_it)
        if reason <= 0:
            PETSc.Sys.Print("adj_ first-order guess: no convergence after {} iterations [{}]: zero guess".format(
                iterations, ksp_converged_reason_name(reason)))
            sol.zeroEntries()
            self._last_first_order_guess = dict(iterations=iterations, ratio=np.nan, used=False)
            return False
        residual = rhs.duplicate()
        A.mult(sol, residual)
        residual.aypx(-1.0, rhs)
        ratio = residual.norm() / rhs.norm()
        residual.destroy()
        walltime = perf_counter() - pre_time
        PETSc.Sys.Print("adj_ first-order guess: {} iterations, {:.1f} s; the exact system's initial residual is "
                        "{:.3e} of |rhs|".format(iterations, walltime, ratio))
        self._last_first_order_guess = dict(iterations=iterations, ratio=float(ratio), used=True, walltime=walltime)
        return True

    @staticmethod
    def _adjoint_state_key(u_vec, alpha, mesh_node_motions):
        """What the adjoint's matrix depends on: the state, alpha and the mesh motion."""
        return (np.array(u_vec, copy=True), float(alpha),
                None if mesh_node_motions is None else np.array(mesh_node_motions, copy=True))

    def _state_keys_equal(self, key0, key):
        u0, alpha0, motions0 = key0
        u, alpha, motions = key
        same = (alpha == alpha0 and np.array_equal(u0, u) and (motions0 is None) == (motions is None)
                and (motions is None or np.array_equal(motions0, motions)))
        return bool(self.mesh.mesh.comm.allreduce(int(same), op=MPI.MIN))

    def _adjoint_cache_matches(self, state_key):
        cache = self._adjoint_cache
        return cache is not None and self._state_keys_equal(cache['key'], state_key)

    def _release_adjoint_cache(self):
        """Free what the adjoint of the last state keeps: the transposed matrices
        and KSP, the cached jacvec operators and the derived fields' cached
        Jacobians."""
        cache, self._adjoint_cache = self._adjoint_cache, None
        freed = cache is not None
        if freed:
            self.sim_model.destroy_adjoint_operator(cache['ksp'])
            del cache
        if self._dRdx_cache is not None:
            destroy_chained_operator(self._dRdx_cache['op'])
            self._dRdx_cache = None
            freed = True
        if self._dRdalpha_cache is not None:
            self._dRdalpha_cache = None
        if self.sim_model.derived_fields:
            self.sim_model.release_derived_caches()
            freed = True
        if freed:
            self.sim_model._return_freed_memory()

    def set_adjoint_options(self, cell_centre_matrix=False, cache_jacobians=False, structured_gradient=False,
                            matrix_free=False, cache_jacvec_operators=False):
        """
        Switch the speed-ups of the adjoint of a model with a condensed Jacobian
        (all exact reformulations; each off by default). Frees every cache.
          cell_centre_matrix       build the constant cell-centre matrix dc/dx vectorized, once
          cache_jacobians          keep the Green-Gauss Jacobians dG/dU, dG/dalpha, dG/dx
                                   while the state, alpha and the mesh are unchanged
          structured_gradient      apply dG/dU as W (x) I plus a boundary part instead of an
                                   assembled matrix
          matrix_free              the adjoint operator without forming A or A_UG B
          cache_jacvec_operators   build the dR/dx operator and dR/dalpha once per design
        """
        self._release_adjoint_cache()
        rec = getattr(self.sim_model, "reconstruction", None)
        if rec is not None:
            rec.fast_cell_centre_jacobian = bool(cell_centre_matrix)
            rec.cache_jacobians = bool(cache_jacobians)
            rec.structured_state_jacobian = bool(structured_gradient)
        self.adjoint_matrix_free = bool(matrix_free)
        self.cache_jacvec_operators = bool(cache_jacvec_operators)

    def compute_jacvec_product(self, input_vals, outputs, d_inputs, d_outputs, d_residuals, mode):
        # Here we compute the product of the adjoint solution with d_residual
        pre_compute_jacvec_product_time = perf_counter()
        print("starting compute_jacvec_product in DG_windtunnel_model...")

        mesh_node_motions = input_vals['mesh_node_motions'] if 'mesh_node_motions' in input_vals else None
        shape_parameters = input_vals['shape_parameters'] if 'shape_parameters' in input_vals else None
        alpha = input_vals['alpha'][0]
        u_vec = outputs['u_vec'] # numpy array

        if mode == 'rev':
            # compute d_input = (dr_dinput^T)*d_residual

            # apply the input mesh node motions to the mesh of the sim model,
            # apply the angle of attack and compute the weak form
            if mesh_node_motions is not None:
                PETSc.Sys.Print("compute_jacvec_product, stepping into mesh.apply_node_motions")
                self.mesh.apply_node_motions(mesh_node_motions, design=shape_parameters)
            self.sim_model.set_angle_of_attack(alpha)
            self.sim_model.compute_weakform()

            if 'alpha' in d_inputs:
                # alpha is a single global scalar, so dR/d(alpha) is one column
                # and this reverse product is a plain dot product between vectors
                local_size = self.sim_model.u_vec.x.petsc_vec.local_size
                if self.cache_jacvec_operators and self.sim_model.derived_fields:
                    key = self._adjoint_state_key(u_vec, alpha, mesh_node_motions)
                    if self._dRdalpha_cache is None or not self._state_keys_equal(self._dRdalpha_cache[0], key):
                        dr_dalpha = self.sim_model.assemble_dRdalpha_vec(u_vec)
                        self._dRdalpha_cache = (key, np.array(dr_dalpha.getArray()[:local_size], copy=True))
                    dr_dalpha_local = self._dRdalpha_cache[1]
                else:
                    dr_dalpha = self.sim_model.assemble_dRdalpha_vec(u_vec)

                    # we first have to assert that the MPI objects are compatible
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
                cached_operator = self.cache_jacvec_operators and bool(self.sim_model.derived_fields)
                if cached_operator:
                    key = self._adjoint_state_key(u_vec, alpha, mesh_node_motions)
                    if self._dRdx_cache is None or not self._state_keys_equal(self._dRdx_cache['key'], key):
                        if self._dRdx_cache is not None:
                            destroy_chained_operator(self._dRdx_cache['op'])
                        self._dRdx_cache = {'key': key, 'op': self.sim_model.compute_dRdxmesh_mat(
                            u_vec, self.mesh.coordinate_space)}
                    dr_dxmesh = self._dRdx_cache['op']
                else:
                    dr_dxmesh = self.sim_model.compute_dRdxmesh_mat(u_vec, self.mesh.coordinate_space)

                if not hasattr(self, '_d_residual_jvp_petsc'):
                    self._d_residual_jvp_petsc = dr_dxmesh.createVecLeft()
                    self._d_input_jvp_petsc = dr_dxmesh.createVecRight()

                set_petsc_vec_array(self._d_residual_jvp_petsc, d_residuals['u_vec'])

                # compute d_input
                self._d_input_jvp_petsc.set(0.0)
                dr_dxmesh.multTranspose(self._d_residual_jvp_petsc, self._d_input_jvp_petsc)
                d_input = self._d_input_jvp_petsc
                if self.sim_model.derived_fields and not cached_operator:
                    # a fresh chained operator per call (not cached): free it now
                    destroy_chained_operator(dr_dxmesh)

                PETSc.Sys.Print("d_input norm: {}".format(d_input.norm()))

                d_input_cg1 = d_input.getArray().reshape(-1, self.mesh.mesh.geometry.dim)   # (N_CG1_dofs, gdim)
                d_input_geom = self.mesh.dof_to_node_map.T @ d_input_cg1  # (N_geom_nodes, gdim)
                d_inputs['mesh_node_motions'] = d_input_geom

                PETSc.garbage_cleanup()
        else:
            raise ValueError("Reverse mode required")
        
        PETSc.Sys.Print("DG_windtunnel_model compute_jacvec_product done")

        post_compute_jacvec_product_time = perf_counter()
        total_compute_jacvec_product_time = post_compute_jacvec_product_time - pre_compute_jacvec_product_time

        PETSc.Sys.Print("compute_jacvec_product total time: {}".format(total_compute_jacvec_product_time))
