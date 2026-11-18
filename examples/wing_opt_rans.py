"""
Fuel-burn minimization of the Simple Transonic Wing with the SA-neg RANS model,
with the planform (sweep, aspect ratio, span, taper) and section (thickness and
camber modes at three spanwise stations) design variables of wing_opt_euler.py
plus the angle of attack. The flight condition is cruise at 10,400 m (rho 0.3926
kg/m^3, T 220.55 K, p 24857 Pa) at M 0.77.

The fuel burn comes from STW_fuelburn (STW_opt.py): the wing mass (Raymer's
transport wing-weight equation, from the take-off mass, area, aspect ratio, root
t/c, taper and quarter-chord sweep) is coupled to the take-off mass through the
Breguet cruise and climb segments, which use the CFD lift and drag (plus a
fixed fuselage drag). The lift has to carry the mid-cruise weight
g sqrt(M_cruise_start M_landing).

The flow model is CompressibleRANSModel (p = 0, Green-Gauss gradients
condensed out of Newton, MUSCL, pseudo-transient continuation from the free
stream), passed to DG_windtunnel_model as model_class; the drag includes the
wall friction. The pitching moment may not fall below, and the root bending
moment may not exceed, their values on the undeformed wing. The wing wall is
adiabatic no-slip, the y = 0 symmetry plane slip. The L3 grid is wall-resolved (first cell ~9e-6, root chord 5); the
wall-merged XDMF variant coarsens exactly those layers, so it is not meant
for RANS.

    OMP_NUM_THREADS=1 mpirun -n 8 python wing_opt_rans.py
"""
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import csdl_alpha as csdl

from modopt import CSDLAlphaProblem
from modopt import COBYLA, SLSQP, PySLSQP

from dragonfly_sim.utils.mesh_manager_utils import wing_inner_bdry_function
from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.RANS_model import CompressibleRANSModel
from dragonfly_sim.core.postprocessor import DG_postprocessor

from dragonfly_sim.utils.ffd_dv_utils import spanwise_linear_bounds
from dragonfly_sim.core.shape_design import (FFDShapeParameterization, WingShape,
                                             ControlPointMotions, _section_mode_profiles)

from STW_opt import STW_fuelburn


# Meshes are read from the repository's meshes/ directory
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")


if __name__ == '__main__':
    # The Simple Transonic Wing's L3 grid: 12 structured blocks in CGNS, read with PyVista and merged
    # into one hexahedral dolfinx mesh in memory (181,152 cells). The boundary conditions in the file
    # are not used; the wall is found geometrically. wing_vol_L2.cgns and wing_vol_L1.cgns are the
    # finer levels. wing_vol_L3_xdmf_wallmerged.xdmf is the L3 mesh with its first wall layers merged
    # (fewer cells, aspect ratio max ~415 instead of ~113k); load it with mesh_name="Grid".
    mesh_from_file, mesh_info = load_dolfinx_mesh(os.path.join(MESH_DIR, "wing_vol_L3.cgns"),
                                                  MPI.COMM_WORLD, tdim=3)
    PETSc.Sys.Print("wing_vol_L3.cgns: {n_cells} cells, {n_points} nodes from {n_blocks} blocks"
                    .format(**mesh_info))


    # NOTE: In the simple transonic wing mesh: - x: LE-TE wing line, flow in positive x-direction
    #                                          - y: spanwise coordinate, wing tip in positive y-direction 
    #                                          - z: vertical coordinate, lift in positive z-direction

    # Polynomial order
    poly_o = 0
    gamma = 1.4

    # Flight condition: cruise at 10,400 m. The mesh is in metres (root chord
    # 5 m, semi-span 14 m, wing area 91 m^2).
    rho_dim = 0.3926     # [kg/m^3]
    T_dim = 220.55       # [K]
    p_dim = 24857.       # [Pa]
    M_0 = 0.77
    a_dim = np.sqrt(gamma * p_dim / rho_dim)      # speed of sound, ~297.7 m/s
    V_dim = M_0 * a_dim                           # cruise speed, ~229.2 m/s
    q_inf_dim = 0.5 * rho_dim * V_dim ** 2        # dynamic pressure, ~10.3 kPa

    # The flow solver works in nondimensional units, rho_inf = p_inf = 1 with
    # lengths in metres: only the Mach number carries the flight condition,
    # and the solver's force unit is p_dim * (1 m)^2.
    rho_0 = 1.0
    p_0 = 1.0
    attack = np.radians(2)

    # Reynolds number. The RANS model's Re is per unit length of the mesh
    # (mu_inf = rho_inf U_inf / Re), i.e. per metre: rho V / mu at the flight
    # condition, with mu from Sutherland's law (~1.443e-5 Pa s at 220.55 K),
    # so ~6.2e6 per metre (Re_mac ~2.2e7 on the baseline mean aerodynamic chord
    # of 3.564 m). It stays fixed per unit length when the planform changes:
    # the physical scale of the wing is what is held. Estimated first-cell y+
    # on the L3 grid (first cell ~9e-6 m): ~2.
    mu_dim = 1.458e-6 * T_dim ** 1.5 / (T_dim + 110.4)
    Re = rho_dim * V_dim / mu_dim
    PETSc.Sys.Print("Flight condition: M {:.3f}, V {:.3f} m/s, q {:.1f} Pa, Re {:.4e} per metre"
                    .format(M_0, V_dim, q_inf_dim, Re))

    # inflow_angle is derived from alpha inside define_farfield_conditions
    # now, so that alpha is the single source of truth for the freestream
    # direction -- it has to be, since alpha is a design variable and the
    # direction has to follow it on every evaluation.
    boundary_dict = {'rho': rho_0, 'M': M_0, 'p': p_0, 'alpha': attack}

    # Angle-of-attack design-variable bounds. The lift has to carry the
    # mid-cruise weight (~52 t, c_L ~0.54 at this q and 91 m^2), more than the
    # baseline's c_l at 2 deg (~0.43), hence the wide upper bound.
    attack_lower = np.radians(1.5)
    attack_upper = np.radians(5.)

    # FFD block: (chord, span, vertical). Three spanwise sections (root, mid-span, tip) are the
    # stations of the section variables; the chordwise basis is a quartic Bezier (5 control points),
    # which holds the thickness and camber modes below exactly.
    ffd_bspline_deg = [4, 2, 1]
    ffd_shape = [5, 3, 2]
    ffd_block_corner_list = [[(-0.1, -1e-8, 0.32), (5.1, -1e-8, -0.3)], [(7.4, 14.1, 0.32), (9.1, 14.1, -0.3)]]

    # The shape design variables and their bounds (see shape_design.WingShape).
    # The baseline planform is measured from the mesh: quarter-chord sweep
    # 25.3 deg, AR 8.615, span 28 (semi-span 14), root chord 5.0, taper ratio
    # 0.3. Planform variables are absolute values starting at the baseline
    # (their bounds must bracket it); leave one out to keep it fixed. AR, span
    # and root chord are tied by s = AR * c_r * (1 + taper) / 4: with span, AR
    # and taper as variables the root chord follows (the derived one; see
    # WingShape's `derived`). A taper change is exact at the three FFD
    # sections and blended between them (chord error < 1% for taper 0.2-0.5).
    #
    # Section variables are deltas from the baseline, with `modes` chordwise
    # values at each of the ffd_shape[1] = 3 spanwise sections (root first,
    # all modes of a section together: 3 x 4 thickness and 3 x 3 camber
    # values). With xi the local chord fraction and B_m the Bernstein
    # polynomials of degree modes - 1:
    #   thickness: relative change of t/c, sum_m tau_m B_m(xi) (0.1 in every
    #              mode: 10% thicker everywhere)
    #   camber:    change of camber over local chord, sum_m kappa_m 4 xi (1 - xi)
    #              B_m(xi) (leading and trailing edge fixed; kappa in every mode:
    #              a parabolic camber line of maximum kappa)
    # Bounds may also be per-section arrays, e.g. to pin the root:
    # {'bounds': (np.array([0., -0.1, -0.1]), np.array([0., 0.1, 0.1])), 'modes': 4}.
    wing = WingShape(
        planform={'sweep': np.radians((20., 30.)),     # quarter-chord sweep [rad]
                  'aspect_ratio': (7.5, 10.),
                  'span': (26., 30.),                  # full span; the model is the half wing
                  'taper_ratio': (0.2, 0.4)},
        sections={'thickness': {'bounds': 0.1, 'modes': 4},
                  'camber': {'bounds': 0.01, 'modes': 3}},
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

    csdl_flow_model = DG_windtunnel_model(mesh_from_file, boundary_dict, shape,
                                          mesh_inner_bdry_function=wing_inner_bdry_function,
                                          poly_order=poly_o, gamma=1.4,
                                          filename_suffix="wing_opt_rans_fuelburn_L3mesh_p=0_M=0_77",
                                          asm_overlap=1, ilu_levels=2,
                                          model_class=CompressibleRANSModel,
                                          model_kwargs={'Re': Re, 'T_inf_dim': T_dim})
    # Pseudo-transient continuation from the free stream needs many more steps
    # than a plain Newton solve (~60 for a 2D airfoil at M 0.7; more for a
    # transonic wing).
    csdl_flow_model.max_newton_iterations = 400
    # The residual tolerance stays at its default (1e-9). check_totals'
    # finite differences need a tightly converged flow, but on this wing the
    # Euler residual stalls at ~2.6e-10, and a solve that misses its tolerance
    # returns the sentinel drag (finite differences 0 for every variable).
    # Check where the RANS residual levels off before tightening it.
    # Memory (L3, p = 0, 4 ranks, measured 2026-10-07): the PTC solve peaks at
    # ~22 GB in total, but one assembly of the exact condensed Jacobian dR/dU
    # -- what the adjoint (check_totals, optimization) solves on -- went past
    # 33 GB, more than a 46 GB workstation had left. Plan the derivative runs
    # for a larger machine.
    # Adjoint linear solves: absolute residual tolerance 1e-9 (the default is 1e-11). The
    # check that rejects a degraded solve (linear_residual_accept_limit) keeps its ten-fold
    # margin above the tolerance. The transposed Jacobian and preconditioner built for a design
    # are reused by the adjoint solves of all its outputs (see DG_windtunnel_model).
    csdl_flow_model.linear_solver_atol = 1e-9
    csdl_flow_model.linear_residual_accept_limit = 1e-8
    # Adjoint speed-ups, measured in examples/benchmark_adjoint_options.py and benchmark_adjoint_ksp.py
    # (the derivative evaluation takes 498 s instead of 674 s on 6 ranks, with identical derivatives):
    # the derived-field Jacobians, the dR/dx operator and the cell-centre matrix are built once per
    # design, and the adjoint's Krylov method is DGMRES(200) with 30 deflated vectors instead of FGMRES(200).
    # They are applied after set_up_sim(), below.
    adjoint_ksp_options = {"ksp_type": "dgmres", "ksp_gmres_restart": 200, "ksp_dgmres_eigen": 30,
                           "ksp_dgmres_max_eigen": 60, "ksp_pc_side": "right",
                           "ksp_norm_type": "unpreconditioned"}
    # Free the solver's matrices after each forward solve and the adjoint's
    # after each design's adjoints, so the two never coexist (rebuilt when next needed).
    csdl_flow_model.release_solver_memory = True
    # No per-evaluation mesh_deformation file; the solution files are still written
    csdl_flow_model.write_mesh_deformation = False
    # One solution frame per design (CSDL evaluates each design twice)
    csdl_flow_model.write_once_per_design = True
    csdl_flow_model.set_up_sim()
    csdl_flow_model.set_adjoint_options(cell_centre_matrix=True, cache_jacobians=True, structured_gradient=True,
                                        cache_jacvec_operators=True)
    csdl_flow_model.adjoint_ksp_options = adjoint_ksp_options

    # Moments of the wall force (see DG_postprocessor), both constrained below:
    #   root_bending: about the x direction (chordwise axis) through the origin, on the
    #                 y = 0 symmetry plane at the root;
    #   pitch:        about the y direction (spanwise axis) through the quarter chord of the
    #                 root section, from the fitted baseline planform (x of the root
    #                 quarter chord, z of the chord plane at the root).
    # The points are fixed in space, also when a design variable moves the root.
    root_quarter_chord = (wing.baseline['x_ref_root'], 0., wing.baseline['z_chord_plane'][1])
    moments = [("root_bending", (0., 0., 0.), (1., 0., 0.)),
               ("pitch", root_quarter_chord, (0., 1., 0.))]
    PETSc.Sys.Print("Moment points: root bending about (0, 0, 0), pitch about the root quarter chord {}"
                    .format(np.round(root_quarter_chord, 6)))

    # D includes the wall friction (D_friction is its viscous part)
    csdl_coeff_model = DG_postprocessor(csdl_flow_model.mesh, csdl_flow_model.sim_model, csdl_flow_model.WALL_TAG, moments=moments, p_inf_dim=p_dim)

    # The moments of the undeformed wing at the initial flow conditions, from a forward solve on
    # the undeformed mesh: the limits of the moment constraints below (upper for root bending,
    # lower for pitching). The first
    # optimization evaluation is this same design, and starts from this converged state.
    u_baseline, baseline_converged, _ = csdl_flow_model.solve_forward(attack)
    if not baseline_converged:
        raise RuntimeError("the baseline flow solve did not converge: no baseline moments for the constraints")
    baseline_forces = csdl_coeff_model.forces_and_coefficients(u_baseline)
    baseline_moments = baseline_forces["moments"]
    PETSc.Sys.Print("Baseline moments (nondimensional): " + ", ".join(
        "{} {:.8f}".format(name, m["M"]) for name, m in baseline_moments.items()))

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
    mesh_nodes_deformation = csdl_flow_model.deform_mesh()

    u_vec = csdl_flow_model.evaluate(mesh_nodes_deformation, shape_param_vector, alpha)

    outputs = csdl_coeff_model.evaluate(u_vec, mesh_nodes_deformation, alpha, shape_param_vector)

    # c_l/c_d/c_m_<name> are MONITORED ONLY -- no gradient is supplied for them (see
    # DG_postprocessor.evaluate), so they must not become a constraint or
    # objective. L/D/M_<name> are the differentiated, nondimensional quantities used
    # below -- dimensional values are only ever printed to the console inside
    # DG_postprocessor.compute(), for monitoring.
    c_l, c_d = outputs.c_l, outputs.c_d
    L, D = outputs.L, outputs.D
    M_root_bending, M_pitch = outputs.M_root_bending, outputs.M_pitch

    # Fuel-burn model. Its inputs follow the current design variables: the
    # planform quantities (full wing area, aspect ratio, taper ratio and
    # quarter-chord sweep, as Raymer's wing-weight equation wants) from the
    # wing layer, and the root t/c from the root section's thickness modes,
    # the baseline t/c (0.121, RAE2822) scaled by the relative thickness change
    # at the maximum-thickness chord fraction (~0.38). The thickness variable
    # holds the root section's modes first.
    planform = shape.outputs()['wing']
    S_wings = planform['area']
    AR = planform['aspect_ratio']
    taper_ratio = planform['taper_ratio']
    sweep = planform['sweep']

    tc_root_baseline = 0.121
    n_thickness_modes = wing.section_spec['thickness']['modes']
    root_thickness_weights = _section_mode_profiles('thickness', n_thickness_modes, np.array([0.38]))[0]
    root_thickness_modes = wing.variables['thickness'][0:n_thickness_modes]
    tc_root = tc_root_baseline * (1. + csdl.sum(root_thickness_weights * root_thickness_modes))

    # Dimensional lift and drag [N] of both wings: the solver's force unit is
    # p_dim * (1 m)^2, and the model is the half wing.
    L_total = 2. * p_dim * L
    D_wings = 2. * p_dim * D

    fuelburn_model = STW_fuelburn(q_inf_dim, V_cruise=V_dim)
    TOGM, LGM, cruise_start_mass, fuel_burn = fuelburn_model.evaluate(
        S_wings, AR, tc_root, taper_ratio, sweep, L_total, D_wings)
    TOGM.add_name('take_off_gross_mass')
    LGM.add_name('landing_gross_mass')
    cruise_start_mass.add_name('cruise_start_mass')
    tc_root.add_name('tc_root')

    # Objective: the fuel burn mass [kg]. It is ~1e4 kg (the baseline value is
    # not known before the simulator runs), so the scaler brings it to O(1).
    D.add_name('drag')   # monitored only
    fuel_burn.add_name('fuel_burn')
    fuel_burn.set_as_objective(scaler=1e-4)

    # Lift constraint: the lift of both wings has to carry the mid-cruise
    # weight g sqrt(M_cruise_start M_landing), the geometric mean of the
    # cruise segment's start and end weights. As a ratio it is O(1) scaled.
    with csdl.namespace('Constraint 2'):
        W_cruise = fuelburn_model.g * csdl.sqrt(cruise_start_mass * LGM)
        L_constraint = L_total / W_cruise
        L_constraint.add_name('L_limit')
        L_constraint.set_as_constraint(lower=1.0)

    # Moment constraints: the root bending moment may not exceed, and the pitching moment may not
    # fall below, its value on the undeformed wing. M_<name> are the raw nondimensional moment integrals (like
    # L), the quantities to constrain; their coefficients are monitoring only.
    with csdl.namespace('Constraint 3'):
        M_root_bending.add_name('root_bending_limit')
        M_root_bending.set_as_constraint(upper=baseline_moments["root_bending"]["M"],
                                         scaler=1./abs(baseline_moments["root_bending"]["M"]))
        M_pitch.add_name('pitch_limit')
        M_pitch.set_as_constraint(lower=baseline_moments["pitch"]["M"],
                                  scaler=1./abs(baseline_moments["pitch"]["M"]))

    # Geometric constraints. The wing's planform quantities follow directly from
    # its variables (shape.outputs()['wing']: area, span, semi_span, root_chord,
    # tip_chord, section_chords, ...); the deformed wall itself is measured by
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
    # print("Starting check_totals...")
    # sim.check_totals(step_size=1e-6)

    PETSc.Sys.Print("Defining optimization problem...")

    prob = CSDLAlphaProblem(problem_name='shape_opt',simulator=sim)
    # optimizer = COBYLA(prob, solver_options={'maxiter':400, 'catol':1e-6, 'rhobeg': 0.025}, turn_off_outputs=True)

    # modopt writes its outputs into a timestamped folder that every rank would
    # create at once (a FileExistsError race): only rank 0 writes them
    optimizer = SLSQP(prob, solver_options={'ftol':1e-6, 'maxiter':150},
                      turn_off_outputs=mesh_from_file.comm.Get_rank() != 0)
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-8, 'maxiter':100})

    PETSc.Sys.Print("Solving optimization problem...")

    optimizer.solve()

    # save data store file
    if mesh_from_file.comm.Get_rank() == 0:
        csdl_flow_model.data_store.write_store_to_numpy_file(save_filename="SLSQP_{}.npy".format(csdl_flow_model.filename_suffix))
    optimizer.print_results()