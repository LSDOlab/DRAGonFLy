"""
Forward flow analysis of an airfoil with DG_windtunnel_model: no FFD block and
no mesh warper, so the mesh stays fixed and the only input is the angle of
attack.

Two ways to run it:

  * solve_forward(alpha): one flow solve without any CSDL graph;
  * a CSDL graph alpha -> u_vec -> L, D, M, which gives the adjoint
    derivatives with respect to alpha (checked against finite differences
    here with check_totals).

    OMP_NUM_THREADS=1 mpirun -n 4 python airfoil_analysis.py                 # Euler, M 0.8
    OMP_NUM_THREADS=1 mpirun -n 4 python airfoil_analysis.py --model rans    # SA-neg RANS, M 0.7
"""
import argparse
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

import csdl_alpha as csdl

from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.postprocessor import DG_postprocessor
from dragonfly_sim.core.RANS_model import CompressibleRANSModel

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("euler", "rans"), default="euler")
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--farfield", choices=("split", "riemann"), default="split")
    args = parser.parse_args()

    if args.model == "rans":
        mesh_file = args.mesh or os.path.join(MESH_DIR, "naca0012_cgrid_rans_coarse.xdmf")
        mach, model_kwargs = 0.7, dict(model_class=CompressibleRANSModel, model_kwargs={'Re': 6e6},
                                       asm_overlap=1, ilu_levels=2)
    else:
        mesh_file = args.mesh or os.path.join(MESH_DIR, "naca0012_euler_mesh_quad_v2.xdmf")
        mach, model_kwargs = 0.8, dict(asm_overlap=1, ilu_levels=1)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        mesh_from_file = xdmf.read_mesh()

    attack = np.radians(2.0)
    boundary_dict = {'inlet': {'rho': 1.0, 'M': mach, 'p': 1.0, 'alpha': attack},
                     'outlet': {'p': 1.0}}

    # No shape parameterization: no FFD block, no mesh warper -- a forward analysis.
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    windtunnel = DG_windtunnel_model(mesh_from_file, boundary_dict,
                                     mesh_inner_bdry_function=airfoil_inner_bdry_function,
                                     poly_order=0, filename_suffix="airfoil_analysis_" + args.model,
                                     farfield=args.farfield, **model_kwargs)
    windtunnel.euler_residual_conv_limit = 1e-12
    windtunnel.max_newton_iterations = 400
    windtunnel.set_up_sim()

    # (1) A plain forward solve and its forces, no CSDL involved
    u, converged, forces = windtunnel.solve_forward(attack)
    PETSc.Sys.Print("solve_forward: converged {}; ".format(converged)
                    + ", ".join("{} {:.6f}".format(k, v) for k, v in forces.items()))

    # (2) The CSDL graph with alpha as its only input
    coefficients = DG_postprocessor(windtunnel.mesh, windtunnel.sim_model, windtunnel.WALL_TAG,
                                    windtunnel.aero_center)
    alpha = csdl.Variable(name='alpha', value=np.array([attack]))
    alpha.set_as_design_variable(lower=np.radians(1.0), upper=np.radians(3.0))
    u_vec = windtunnel.evaluate(alpha=alpha)
    outputs = coefficients.evaluate(u_vec, alpha=alpha)
    outputs.D.add_name('drag')
    outputs.D.set_as_objective()
    outputs.L.add_name('lift')
    outputs.L.set_as_constraint(lower=0.0)
    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    sim.run()
    PETSc.Sys.Print("CSDL run: D {:.8f}, L {:.8f}".format(float(sim[outputs.D][0]), float(sim[outputs.L][0])))
    # checks the objective and constraint (D and L) against finite differences
    sim.check_totals(step_size=1e-6)
