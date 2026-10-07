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

from dragonfly_sim.utils.ffd_dv_utils import spanwise_linear_bounds
from dragonfly_sim.core.shape_design import (FFDShapeParameterization, WingShape,
                                             ControlPointMotions)


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
    ffd_block_corner_list = [[(-0.1, -1e-8, 0.32), (5.1, -1e-8, -0.3)], [(7.4, 14.1, 0.32), (9.1, 14.1, -0.3)]]

    # The shape design variables and their bounds (see shape_design.WingShape).
    # The baseline planform is measured from the mesh: quarter-chord sweep
    # 25.3 deg, AR 8.615, root chord 5.0, taper ratio 0.3. Planform variables
    # are absolute values starting at the baseline (their bounds must bracket
    # it); leave one out to keep it fixed. The semi-span follows from
    # s = AR * c_r * (1 + taper) / 4, so a larger AR at a fixed root chord
    # means a longer wing with more area. Section variables have one value per
    # FFD spanwise section (ffd_shape[1] of them, root first) and are deltas
    # from the baseline: thickness is the relative change of t/c, camber the
    # change of maximum camber over local chord (a parabolic camber line).
    # Bounds there may also be per-section arrays, e.g. to pin the root.
    wing = WingShape(
        planform={'sweep': np.radians((20., 30.)),     # quarter-chord sweep [rad]
                  'aspect_ratio': (7.5, 10.),
                  'root_chord': (4.5, 5.5),
                  'taper_ratio': (0.2, 0.4)},
        sections={'thickness': 0.1,
                  'camber': 0.01},
        sweep_chord_fraction=0.25,
        # the STW's trapezoid ends at y = 14; the wall's largest y (the default)
        # would include the rounded tip cap and shift AR and taper slightly
        semi_span=14.0)
    # Raw control-point motions can be added on top as a second layer, e.g.
    # ControlPointMotions({2: spanwise_linear_bounds(base_bound=0.1, scale_at_root=1.0,
    #                                                scale_at_ref=1.0, y_ref=14.1)})
    shape = FFDShapeParameterization(ffd_shape, ffd_bspline_deg, ffd_block_corner_list,
                                     layers=[wing])

    recorder = csdl.Recorder(inline=False)
    recorder.start()

    csdl_euler_model = DG_windtunnel_model(mesh_from_file, boundary_dict, shape,
                                           np.array([0.25, 0., 0.], dtype=np.double),
                                           mesh_inner_bdry_function=wing_inner_bdry_function,
                                           poly_order=poly_o, gamma=1.4,
                                           filename_suffix="wing_opt_L3mesh_p=0_M=0_8",
                                           asm_overlap=1, ilu_levels=1)
    csdl_euler_model.set_up_sim()

    csdl_coeff_model = DG_postprocessor(csdl_euler_model.mesh, csdl_euler_model.sim_model, csdl_euler_model.WALL_TAG, np.array([0.25, 0., 0.], dtype=np.double), p_inf_dim=101325.)

    # Shape design variables first (all of them as one flat vector), then alpha
    shape_param_vector = shape.declare_design_variables()
    shape.print_design_variables(printer=PETSc.Sys.Print)

    # Angle of attack as a design variable. Replicated (every rank holds the same
    # scalar), which is why DG_windtunnel_model.compute_jacvec_product allreduces
    # its gradient.

    alpha = csdl.Variable(name='alpha', value=np.array([attack]))
    alpha.set_as_design_variable(lower=attack_lower, upper=attack_upper,
                                 scaler=1./np.radians(1.))

    # These are the mesh node deformations, ordered according to the global mesh node ordering
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
    # Exact (tangent-solve) pitching-moment slope. MONITORED ONLY -- no gradient
    # is supplied for it, so it must not be made a constraint or objective.
    # See DG_postprocessor.evaluate for what constraining it would require.
    c_m_alpha = outputs.c_m_alpha

    D.add_name('drag')
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

    # Geometric constraints. The wing's planform quantities follow directly from
    # its variables (shape.outputs()['wing']: area, span, semi_span, tip_chord,
    # section_chords, ...); the deformed wall itself is measured by
    # shape.geometry (shape_parameterization.WallGeometry). The half-wing volume
    # is exact as long as the root section stays on the y = 0 symmetry plane.
    # Enable by uncommenting:
    # planform = shape.outputs()['wing']
    # planform['area'].add_name('planform_area')
    # baseline_area = wing.baseline['area']
    # planform['area'].set_as_constraint(lower=0.99*baseline_area, upper=1.01*baseline_area)
    # wall_geometry = shape.geometry
    # volume = wall_geometry.enclosed_measure(shape.wall_displacement())
    # volume.add_name('volume')
    # volume.set_as_constraint(lower=wall_geometry.baseline_measure)
    # thickness_stations = wall_geometry.add_thickness_stations(
    #     np.linspace(0.1, 0.9, 5), span_stations=np.linspace(0.5, 13.5, 5))
    # thickness_ratio = wall_geometry.thickness_ratio(shape.coefficients(), thickness_stations)
    # thickness_ratio.add_name('thickness_ratio')
    # thickness_ratio.set_as_constraint(lower=0.9)

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