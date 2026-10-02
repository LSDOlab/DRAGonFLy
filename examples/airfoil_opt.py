import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

import csdl_alpha as csdl

from modopt import CSDLAlphaProblem
from modopt import COBYLA, SLSQP, PySLSQP

from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function
from dragonfly_sim.core.shape_parameterization import SectionalShape
from dragonfly_sim.utils.ffd_dv_utils import cp_dv_directions, build_cp_motion_dv, build_sectional_dv, print_cp_motion_bounds, ShapeDVSet

from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.postprocessor import DG_postprocessor


# Meshes are stored with Git LFS in the repository's meshes/ directory
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")


if __name__ == '__main__':
# naca0012_euler_mesh_quad_v2
# naca0012_cgrid_mesh
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, os.path.join(MESH_DIR, "naca0012_euler_mesh_quad_v2.xdmf"), "r") as xdmf:
        mesh_from_file = xdmf.read_mesh()

    # Polynomial order
    poly_o = 0
    gamma = 1.4

    # Inlet flow conditions
    rho_0 = 1.0
    M_0 = 0.8
    p_0 = 1.0
    attack = np.radians(2)

    # inflow_angle is derived from alpha inside define_inlet_outlet_conditions,
    # so alpha is the single source of truth for the freestream direction.
    boundary_dict = {'inlet': {'rho': rho_0,
                               'M': M_0,
                               'p': p_0,
                               'alpha': attack},
                     'outlet': {'p': p_0}
                     }

    ffd_bspline_deg = [2, 2]
    ffd_shape = [5, 3]
    ffd_center = [int(shape/2) for shape in ffd_shape]

    # Which control point coordinate directions are design variables, and the
    # bounds on each. Keys are spatial direction indices (0 = x chordwise,
    # 1 = y vertical); a direction that is absent is frozen at its baseline
    # coordinate. Values may be a scalar symmetric half-range, a (lower, upper)
    # pair, or a callable(coords) -> (lower, upper) for bounds that vary over the
    # FFD block -- see ffd_dv_utils.
    ffd_dv_spec = {1: 0.01}
    ffd_coordinate_idxs_opt = cp_dv_directions(ffd_dv_spec)

    recorder = csdl.Recorder(inline=False)
    recorder.start()

    csdl_euler_model = DG_windtunnel_model(mesh_from_file, boundary_dict, ffd_shape,
                                           np.array([0.25, 0.], dtype=np.double),
                                           mesh_inner_bdry_function=airfoil_inner_bdry_function,
                                           cp_coord_opt_idxs=ffd_coordinate_idxs_opt,
                                           poly_order=poly_o, gamma=1.4, ffd_degree=ffd_bspline_deg,
                                           filename_suffix="derivatives_meshwarping_2D_SLSQP_quadgrid_M=0_8_p=1",
                                           asm_overlap=1, ilu_levels=1)

    csdl_euler_model.set_up_sim()

    csdl_coeff_model = DG_postprocessor(csdl_euler_model.mesh, csdl_euler_model.sim_model, csdl_euler_model.WALL_TAG, np.array([0.25, 0.], dtype=np.double), p_inf_dim=101325.)

    ffd = csdl_euler_model.ffd

    # Shape design variables
    shape_dvs = ShapeDVSet()

    # Raw control-point motions. The bounds and the scaler are per-entry arrays
    # shaped like cp_motions, so each optimized direction carries its own box.
    # The scaler makes every direction's box +/-1 in the optimizer's space; pass
    # scaler_mode=None to reproduce an unscaled run.
    cp_dv = build_cp_motion_dv(ffd.baseline_coefficients, ffd_dv_spec)
    assert np.array_equal(cp_dv.coord_idxs, ffd_coordinate_idxs_opt)
    cp_motions = shape_dvs.add('cp_motions', cp_dv)
    print_cp_motion_bounds(cp_dv, printer=PETSc.Sys.Print)

    # Sectional variables: one value per chordwise column of control points
    # (ffd_shape[0] of them), or fewer for a B-spline profile over the columns.
    # camber moves a column vertically; thickness stretches it (the change in
    # the column's vertical extent). Enable by uncommenting:
    sectional = SectionalShape(ffd, principal_dim=0)
    # camber = shape_dvs.add('camber', build_sectional_dv(ffd_shape[0], 0.01))
    # sectional.add('camber', camber)
    # thickness = shape_dvs.add('thickness', build_sectional_dv(ffd_shape[0], 0.01))
    # sectional.add('thickness', thickness)

    # Sectional layer on the constant baseline first, raw motions on top; see
    # SectionalShape for why that order is required.
    ffd_coefficients = sectional.apply(ffd.baseline_variable())
    ffd_coefficients = ffd.apply_cp_motions(ffd_coefficients, cp_motions, ffd_coordinate_idxs_opt)

    # Angle of attack as a design variable, same bounds convention as the 3D
    # driver. Replicated across ranks; see DG_windtunnel_model.compute_jacvec_product.
    alpha = csdl.Variable(name='alpha', value=np.array([attack]))
    alpha.set_as_design_variable(lower=np.radians(1.75), upper=np.radians(2.25),
                                 scaler=1./np.radians(1.))

    # evaluate mesh deformation
    bdry_pts_motion_global = ffd.wall_displacement(ffd_coefficients)

    # All shape variables as one vector: the (derivative-free) link that keeps
    # every rank's reverse chain attached.
    shape_param_vector = shape_dvs.flat_variable()
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
    # Exact dc_m/d(alpha) from a tangent solve. MONITORED ONLY -- no gradient is
    # supplied, so it must not become a constraint or objective.
    c_m_alpha = outputs.c_m_alpha

    # Named so the total-derivative probe below can identify it: D comes out of
    # CustomOperation.create_output, which does not set Variable.name, and an
    # unnamed objective prints as a bare 'objective'.
    D.add_name('drag')
    D.set_as_objective()

    with csdl.namespace('Constraint 2'):
        # NOTE: L is always the raw nondimensional pressure-force integral 
        # (the optimizer never sees a dimensional value, regardless of 
        # p_inf_dim/L_ref_dim -- those only scale the console printout in compute())
        L_constraint = L
        L_constraint.add_name('L_limit')
        L_constraint.set_as_constraint(lower=0.225)  # -1e-3, upper=0.35+1e-3) # constraint

    # Geometric constraints on the deformed wall (see shape_parameterization.
    # WallGeometry). Enable by uncommenting:
    # wall_geometry = csdl_euler_model.wall_geometry()
    # area = wall_geometry.enclosed_measure(bdry_pts_motion_global)
    # area.add_name('area')
    # area.set_as_constraint(lower=wall_geometry.baseline_measure)
    # thickness_stations = wall_geometry.add_thickness_stations(np.linspace(0.1, 0.9, 9))
    # thickness_ratio = wall_geometry.thickness_ratio(ffd_coefficients, thickness_stations)
    # thickness_ratio.add_name('thickness_ratio')
    # thickness_ratio.set_as_constraint(lower=0.9)

    recorder.stop()

    # sim = csdl.experimental.PySimulator(recorder)
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
   
    # test accuracy of coefficient postprocessing
    PETSc.Sys.Print("starting check_totals...")
    # sim.check_totals(ofs=c_l, wrts=u_vec_inputs, step_size=1e-2)
    sim.check_totals(step_size=1e-5)

    # PETSc.Sys.Print("Defining optimization problem...")

    # prob = CSDLAlphaProblem(problem_name='shape_opt',simulator=sim)
    # optimizer = SLSQP(prob, solver_options={'ftol':1e-8, 'maxiter':100})
    # # optimizer = PySLSQP(prob, solver_options={'acc':1e-8, 'maxiter':100})

    # PETSc.Sys.Print("Solving optimization problem...")

    # optimizer.solve()

    # # save data store file
    # csdl_euler_model.data_store.write_store_to_numpy_file(save_filename="meshwarping_2D_SLSQP_quadgrid_M=0_8_p=1.npy")
    # # csdl_euler_model.data_store.write_store_to_numpy_file(save_filename="COBYLA_modopt_test_2D.npy")
    # optimizer.print_results()
