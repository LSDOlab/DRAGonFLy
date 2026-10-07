import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

import csdl_alpha as csdl

from modopt import CSDLAlphaProblem
from modopt import COBYLA, SLSQP, PySLSQP

from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function
from dragonfly_sim.core.shape_design import (FFDShapeParameterization, ControlPointMotions,
                                             SectionalVariables)

from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.postprocessor import DG_postprocessor


# Meshes are read from the repository's meshes/ directory
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

    # The shape design variables, as layers on an FFD block around the airfoil
    # (see shape_design). Raw control-point motions: which coordinate directions
    # are design variables, and the bounds on each. Keys are spatial direction
    # indices (0 = x chordwise, 1 = y vertical); a direction that is absent is
    # frozen at its baseline coordinate. Values may be a scalar symmetric
    # half-range, a (lower, upper) pair, or a callable(coords) -> (lower, upper)
    # for bounds that vary over the FFD block -- see ffd_dv_utils. The scaler
    # makes every direction's box +/-1 in the optimizer's space.
    layers = [ControlPointMotions({1: 0.01})]
    # Sectional variables: one value per chordwise column of control points
    # (ffd_shape[0] of them). camber moves a column vertically; thickness
    # stretches it (the change in the column's vertical extent). They act on the
    # baseline before the raw motions (see shape_design). Enable by uncommenting:
    # layers.insert(0, SectionalVariables(0, {'camber': 0.01, 'thickness': 0.01}))
    shape = FFDShapeParameterization(ffd_shape, ffd_bspline_deg, layers=layers)

    recorder = csdl.Recorder(inline=False)
    recorder.start()

    csdl_euler_model = DG_windtunnel_model(mesh_from_file, boundary_dict, shape,
                                           np.array([0.25, 0.], dtype=np.double),
                                           mesh_inner_bdry_function=airfoil_inner_bdry_function,
                                           poly_order=poly_o, gamma=1.4,
                                           filename_suffix="derivatives_meshwarping_2D_SLSQP_quadgrid_M=0_8_p=1",
                                           asm_overlap=1, ilu_levels=1)

    csdl_euler_model.set_up_sim()

    csdl_coeff_model = DG_postprocessor(csdl_euler_model.mesh, csdl_euler_model.sim_model, csdl_euler_model.WALL_TAG, np.array([0.25, 0.], dtype=np.double), p_inf_dim=101325.)

    # Shape design variables first (all of them as one flat vector), then alpha
    shape_param_vector = shape.declare_design_variables()
    shape.print_design_variables(printer=PETSc.Sys.Print)

    # Angle of attack as a design variable, same bounds convention as the 3D
    # driver. Replicated across ranks; see DG_windtunnel_model.compute_jacvec_product.
    alpha = csdl.Variable(name='alpha', value=np.array([attack]))
    alpha.set_as_design_variable(lower=np.radians(1.75), upper=np.radians(2.25),
                                 scaler=1./np.radians(1.))

    # evaluate mesh deformation
    mesh_nodes_deformation = csdl_euler_model.deform_mesh()

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
    # wall_geometry = shape.geometry
    # area = wall_geometry.enclosed_measure(shape.wall_displacement())
    # area.add_name('area')
    # area.set_as_constraint(lower=wall_geometry.baseline_measure)
    # thickness_stations = wall_geometry.add_thickness_stations(np.linspace(0.1, 0.9, 9))
    # thickness_ratio = wall_geometry.thickness_ratio(shape.coefficients(), thickness_stations)
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
