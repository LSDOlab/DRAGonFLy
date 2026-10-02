import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

import csdl_alpha as csdl

from modopt import CSDLAlphaProblem
from modopt import COBYLA, SLSQP, PySLSQP

from dragonfly_sim.utils.mesh_manager_utils import wing_inner_bdry_function
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.postprocessor import DG_postprocessor

from dragonfly_sim.utils.ffd_dv_utils import (cp_dv_directions, build_cp_motion_dv, build_sectional_dv,
                                              spanwise_linear_bounds, print_cp_motion_bounds, ShapeDVSet)
from dragonfly_sim.core.shape_parameterization import SectionalShape


# Meshes are stored with Git LFS in the repository's meshes/ directory
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")


if __name__ == '__main__':
    # wing_vol_L3_xdmf_wallmerged.xdmf is the same mesh with its first wall layers merged (fewer cells,
    # aspect ratio max ~415 instead of ~113k); it is read the same way, with name="Grid".

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, os.path.join(MESH_DIR, "wing_vol_L3_xdmf.xdmf"), "r") as xdmf:
        mesh_from_file = xdmf.read_mesh(name="Grid")  # For the Simple Transonic Wing mesh, since this was generated externally


    # NOTE: In the simple transonic wing mesh: - x: LE-TE wing line, flow in positive x-direction
    #                                          - y: spanwise coordinate, wing tip in positive y-direction 
    #                                          - z: vertical coordinate, lift in positive z-direction

    # Polynomial order
    poly_o = 0
    gamma = 1.4

    # Inlet flow conditions
    rho_0 = 1.0
    M_0 = 0.8
    p_0 = 1.0
    attack = np.radians(2)

    # inflow_angle is derived from alpha inside define_inlet_outlet_conditions
    # now, so that alpha is the single source of truth for the freestream
    # direction -- it has to be, since alpha is a design variable and the
    # direction has to follow it on every evaluation.
    boundary_dict = {'inlet': {'rho': rho_0,
                               'M': M_0,
                               'p': p_0,
                               'alpha': attack},
                     'outlet': {'p': p_0}
                     }

    # Angle-of-attack design-variable bounds.
    attack_lower = np.radians(1.5)
    attack_upper = np.radians(2.5)

    ffd_bspline_deg = [2, 2, 1]
    ffd_shape = [3, 5, 2]
    ffd_center = [int(shape/2) for shape in ffd_shape]

    ffd_block_corner_list = [[(-0.1, -1e-8, 0.32), (5.1, -1e-8, -0.3)], [(7.4, 14.1, 0.32), (9.1, 14.1, -0.3)]]

    # Which control point coordinate directions are design variables, and the
    # bounds on each. Keys are spatial direction indices (0 = x chordwise,
    # 1 = y spanwise, 2 = z vertical on this mesh); a direction that is absent is
    # frozen at its baseline coordinate. Values may be a scalar symmetric
    # half-range, a (lower, upper) pair, or a callable(coords) -> (lower, upper).
    #
    # z uses spatially varying bounds: the FFD block tapers from a root chord of
    # ~5.2 to a tip chord of ~1.7, so a single scalar bound is a much larger
    # fraction of the local chord at the tip than at the root. Shrinking the bound
    # linearly along the span keeps the allowed deformation a roughly constant
    # fraction of the local chord.
    ffd_dv_spec = {
        # 1: spanwise_linear_bounds(base_bound=2.0, scale_at_root=0.0,
        #                           scale_at_ref=1.0, y_ref=14.1),
        2: spanwise_linear_bounds(base_bound=0.1, scale_at_root=1.0,
                                  scale_at_ref=1.0, y_ref=14.1),
    }
    ffd_coordinate_idxs_opt = cp_dv_directions(ffd_dv_spec)

    recorder = csdl.Recorder(inline=False)
    recorder.start()

    csdl_euler_model = DG_windtunnel_model(mesh_from_file, boundary_dict, ffd_shape,
                                           np.array([0.25, 0., 0.], dtype=np.double),
                                           mesh_inner_bdry_function=wing_inner_bdry_function,
                                           ffd_block_corner_list=ffd_block_corner_list,
                                           cp_coord_opt_idxs=ffd_coordinate_idxs_opt, 
                                           poly_order=poly_o, gamma=1.4, ffd_degree=ffd_bspline_deg,
                                           filename_suffix="opt_test_L3mesh_SLSQP_cpgrid=10x5x2_p=0_PODtest_M=0_85",
                                           asm_overlap=1, ilu_levels=1)
  
    csdl_euler_model.set_up_sim()

    csdl_coeff_model = DG_postprocessor(csdl_euler_model.mesh, csdl_euler_model.sim_model, csdl_euler_model.WALL_TAG, np.array([0.25, 0., 0.], dtype=np.double), p_inf_dim=101325.)

    ffd = csdl_euler_model.ffd

    # define design variable input
    shape_dvs = ShapeDVSet()
    cp_dv = build_cp_motion_dv(ffd.baseline_coefficients, ffd_dv_spec)
    assert np.array_equal(cp_dv.coord_idxs, ffd_coordinate_idxs_opt)
    cp_motions = shape_dvs.add('cp_motions', cp_dv)
    print_cp_motion_bounds(cp_dv, printer=PETSc.Sys.Print)

    # Sectional variables, one value per spanwise section of control points
    # (ffd_shape[1] of them, root first), or fewer for a B-spline profile over
    # the sections. Pin the root entry with (0., 0.) bounds where the symmetry
    # plane requires it (span; twist and chord keep the root in the plane
    # anyway). Enable by uncommenting:
    sectional = SectionalShape(ffd, principal_dim=1)
    # twist_bounds = (np.radians([0., -3., -3., -3., -3.]), np.radians([0., 3., 3., 3., 3.]))
    # twist = shape_dvs.add('twist', build_sectional_dv(ffd_shape[1], twist_bounds))
    # sectional.add('twist', twist, pivot=[0.25, 0.5])   # about the block's quarter chord
    # chord = shape_dvs.add('chord', build_sectional_dv(ffd_shape[1], 0.2))
    # sectional.add('chord', chord, pivot=[0., 0.5])     # leading edge fixed

    # Sectional layer on the constant baseline first, raw motions on top; see
    # SectionalShape for why that order is required.
    ffd_coefficients = sectional.apply(ffd.baseline_variable())
    ffd_coefficients = ffd.apply_cp_motions(ffd_coefficients, cp_motions, ffd_coordinate_idxs_opt)

    # Angle of attack as a design variable. Replicated (every rank holds the same
    # scalar), which is why DG_windtunnel_model.compute_jacvec_product allreduces
    # its gradient.

    alpha = csdl.Variable(name='alpha', value=np.array([attack]))
    alpha.set_as_design_variable(lower=attack_lower, upper=attack_upper,
                                 scaler=1./np.radians(1.))

    bdry_pts_motion_global = ffd.wall_displacement(ffd_coefficients)

    # All shape variables as one vector: the (derivative-free) link that keeps
    # every rank's reverse chain attached.
    shape_param_vector = shape_dvs.flat_variable()

    # These are the mesh node deformations, ordered according to the global mesh node ordering
    mesh_nodes_deformation = csdl_euler_model.mesh_warper.evaluate(bdry_pts_motion_global, shape_param_vector)

    u_vec = csdl_euler_model.evaluate(mesh_nodes_deformation, shape_param_vector, alpha)

    outputs = csdl_coeff_model.evaluate(u_vec, mesh_nodes_deformation, alpha, shape_param_vector)

    # c_l/c_d/c_m are MONITORED ONLY -- no gradient is supplied for them (see
    # DG_postprocessor.evaluate), so they must not become a constraint or
    # objective. L/D/M are the differentiated, nondimensional quantities used
    # below -- dimensional values are only ever printed to the console inside
    # DG_postprocessor.compute(), for monitoring.
    c_l, c_d, c_m = outputs.c_l, outputs.c_d, outputs.c_m
    L, D, M = outputs.L, outputs.D, outputs.M
    # Exact (tangent-solve) pitching-moment slope. MONITORED ONLY -- no gradient
    # is supplied for it, so it must not be made a constraint or objective.
    # See DG_postprocessor.evaluate for what constraining it would require.
    c_m_alpha = outputs.c_m_alpha

    D.set_as_objective()

    # Define lower limit constraint on lift
    with csdl.namespace('Constraint 2'):
        # NOTE: L is always the raw nondimensional pressure-force integral 
        # (the optimizer never sees a dimensional value, regardless of 
        # p_inf_dim/L_ref_dim -- those only scale the console printout in 
        # compute())
        L_constraint = L
        L_constraint.add_name('L_limit')
        L_constraint.set_as_constraint(lower=10.)

    # Geometric constraints on the deformed wall (see shape_parameterization.
    # WallGeometry). The half-wing volume is exact as long as the root section
    # stays on the y = 0 symmetry plane. Enable by uncommenting:
    # wall_geometry = csdl_euler_model.wall_geometry()
    # volume = wall_geometry.enclosed_measure(bdry_pts_motion_global)
    # volume.add_name('volume')
    # volume.set_as_constraint(lower=wall_geometry.baseline_measure)
    # thickness_stations = wall_geometry.add_thickness_stations(
    #     np.linspace(0.1, 0.9, 5), span_stations=np.linspace(0.5, 13.5, 5))
    # thickness_ratio = wall_geometry.thickness_ratio(ffd_coefficients, thickness_stations)
    # thickness_ratio.add_name('thickness_ratio')
    # thickness_ratio.set_as_constraint(lower=0.9)
    # planform_stations = wall_geometry.add_planform_stations(np.linspace(0., 13.5, 8))
    # planform = wall_geometry.planform(ffd_coefficients, planform_stations)
    # planform['area'].add_name('planform_area')
    # baseline_area = planform_stations['baseline']['area']
    # planform['area'].set_as_constraint(lower=0.99*baseline_area, upper=1.01*baseline_area)

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)

    # PETSc.Sys.Print("starting sim run...")
    # sim.run()
    # PETSc.Sys.Print("finished sim run")
    
    # test accuracy of simulation object
    print("Starting check_totals...")
    sim.check_totals(step_size=1e-6)

    # PETSc.Sys.Print("Defining optimization problem...")

    # prob = CSDLAlphaProblem(problem_name='shape_opt',simulator=sim)
    # # # optimizer = COBYLA(prob, solver_options={'maxiter':400, 'catol':1e-6, 'rhobeg': 0.025}, turn_off_outputs=True)

    # optimizer = SLSQP(prob, solver_options={'ftol':1e-8, 'maxiter':150})
    # # # optimizer = PySLSQP(prob, solver_options={'acc':1e-8, 'maxiter':100})

    # PETSc.Sys.Print("Solving optimization problem...")

    # optimizer.solve()

    # # save data store file
    # if mesh_from_file.comm.Get_rank() == 0:
    #     csdl_euler_model.data_store.write_store_to_numpy_file(save_filename="SLSQP_{}.npy".format(csdl_euler_model.filename_suffix))
    # csdl_euler_model.data_store.write_store_to_numpy_file(save_filename="SLSQP_L3mesh_cpgrid=10x5x2_p=0_test_M=0_85.npy")
    # optimizer.print_results()