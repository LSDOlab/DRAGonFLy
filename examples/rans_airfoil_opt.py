"""
Shape and angle-of-attack sensitivities of a turbulent (SA-neg RANS) NACA0012
through CSDL: the RANS counterpart of airfoil_opt.py.

The flow model is CompressibleRANSModel (p = 0, Green-Gauss gradients
condensed out of Newton, MUSCL), passed to DG_windtunnel_model as
model_class. Its adjoint, dR/dalpha and dR/dx_mesh include the chain terms
through the reconstructed gradient, the cell centres and the wall distance,
so check_totals compares exact analytic totals against finite differences.

Needs a wall-resolved mesh (first cell ~1e-5 chords), e.g. the coarse
9,480-cell C-grid in meshes/naca0012_cgrid_rans_coarse.xdmf:

    OMP_NUM_THREADS=1 mpirun -n 4 python rans_airfoil_opt.py
"""
import os
import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

import csdl_alpha as csdl

from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function
from dragonfly_sim.core.shape_design import FFDShapeParameterization, ControlPointMotions
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.postprocessor import DG_postprocessor
from dragonfly_sim.core.RANS_model import CompressibleRANSModel

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")


if __name__ == '__main__':
    mesh_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MESH_DIR, "naca0012_cgrid_rans_coarse.xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        mesh_from_file = xdmf.read_mesh()

    M_0, attack, Re = 0.7, np.radians(2.0), 6e6
    boundary_dict = {'inlet': {'rho': 1.0, 'M': M_0, 'p': 1.0, 'alpha': attack},
                     'outlet': {'p': 1.0}}

    ffd_bspline_deg = [2, 2]
    ffd_shape = [5, 3]
    # vertical motions of the control points, +/- 0.01 chords
    shape = FFDShapeParameterization(ffd_shape, ffd_bspline_deg,
                                     layers=[ControlPointMotions({1: 0.01})])

    recorder = csdl.Recorder(inline=False)
    recorder.start()

    windtunnel = DG_windtunnel_model(mesh_from_file, boundary_dict, shape,
                                     np.array([0.25, 0.], dtype=np.double),
                                     mesh_inner_bdry_function=airfoil_inner_bdry_function,
                                     poly_order=0, gamma=1.4,
                                     filename_suffix="rans_airfoil_check_totals",
                                     asm_overlap=1, ilu_levels=2,
                                     model_class=CompressibleRANSModel, model_kwargs={'Re': Re})
    # PTC needs more steps than a plain Newton solve: ~60 from the free
    # stream at M 0.7, ~20 from a nearby converged state
    windtunnel.max_newton_iterations = 400
    # converge the flow far enough that the finite differences of check_totals
    # resolve a 1e-6 step (with the default 1e-9 they measure solver noise)
    windtunnel.euler_residual_conv_limit = 1e-12
    windtunnel.set_up_sim()

    coefficients = DG_postprocessor(windtunnel.mesh, windtunnel.sim_model, windtunnel.WALL_TAG,
                                    np.array([0.25, 0.], dtype=np.double))
    shape_param_vector = shape.declare_design_variables()

    alpha = csdl.Variable(name='alpha', value=np.array([attack]))
    alpha.set_as_design_variable(lower=np.radians(1.75), upper=np.radians(2.25), scaler=1. / np.radians(1.))

    mesh_nodes_deformation = windtunnel.deform_mesh()

    u_vec = windtunnel.evaluate(mesh_nodes_deformation, shape_param_vector, alpha)
    outputs = coefficients.evaluate(u_vec, mesh_nodes_deformation, alpha, shape_param_vector)
    L, D = outputs.L, outputs.D
    D.add_name('drag')
    D.set_as_objective()
    L.add_name('L_limit')
    L.set_as_constraint(lower=0.2)

    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    PETSc.Sys.Print("starting check_totals...")
    # 1e-6, not airfoil_opt's 1e-5: the RANS forces bend sharply on the 1e-5
    # scale (forward and backward differences there differ by ~0.3%), so a
    # larger step measures finite-difference error, not the adjoint
    sim.check_totals(step_size=1e-6)
