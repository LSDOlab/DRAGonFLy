"""
Steady RANS (SA-neg) analysis of the BIG_FISH model in water, on the all-hex mesh of fish_unstructured.py
(meshes/big_fish_unstructured.msh), after examples/ONERA-OAT15A/oat15a_unsteady.py. Air-water model branch: the
water is a constant-density liquid (core/materials.py:ConstantDensity), always low-Mach preconditioned.

The mesh has the fish axis along z with the head at +z; the solver's free stream runs along +x (angle --yaw in
the x-z plane), so the mesh is rotated on loading: x' = -z, y' = y, z' = x. The head then points upstream (-x),
y stays the fish's height (160 mm) and z' its width (80 mm, the side-force direction). Lengths stay in metres, so
Re = U L_ref / nu with L_ref = 1 m (the solver's length unit), 5e6 at 5 m/s.

Stages (as oat15a_unsteady.py's start-up, here run to a steady state):

  1. Initial state (unless --restart): the first-order RANS model (SA-neg, no MUSCL) by pseudo-transient
     continuation (PTC) from the free stream, at most --init-steps steps.
  2. The second-order (MUSCL) RANS model from that state by PTC, to |R| <= --steady-tol or --max-steps steps.
     The pseudo steps run in chunks of --report-every with one CFL controller, so the CFL ramp continues
     across chunks; the forces are printed and appended to history.csv after every chunk.

Forces on the wall (pressure + viscous traction), in N with rho = 998.2 kg/m^3 and U = --speed: drag along the
free stream (and its friction part), side force (z'), vertical force (y'); C_D on the wetted area and on the
frontal area (the wall projected on the y'-z' plane).

Output, in a new folder under --out-dir:

  solution.bp           the steady fields (gauge pressure p' with C_p = 2 p', velocity, temperature, nu_tilde,
                        density) for ParaView, on the rotated mesh; with --write-every also intermediate frames
  history.csv           PTC step, |R|, drag/side/vertical force [N], friction drag [N], C_D (wetted, frontal)
  result.csv            the final state's forces and convergence
  checkpoint_final.npz  the solution (restart with --restart); checkpoint_latest.npz every --checkpoint-every
                        steps
  config.json           the run's arguments

Memory: the solver needs roughly 40-60 KB per cell (an 80k-cell test mesh peaked at 4.8 GiB on one rank, with
--linear-preconditioner asm; schur needs more), so the ~760k-cell mesh needs some 30-45 GiB -- more than a 16 GB
workstation. Run it on a larger machine, under the memory guard, with the conda env's own mpirun (the system
OpenMPI's does not start the env's MPICH ranks):

    cd examples/eel
    OMP_NUM_THREADS=1 python ../memory_guard.py --max-gib 48 -- $CONDA_PREFIX/bin/mpirun -n 4 python fish_steady.py

A smaller mesh for tests: python fish_unstructured.py --out fish_micro.msh --h-skin 0.12 --bl-t 0.008 --q-bl 8
--h-iface 0.3 --grow 1.5 --h-wake 1.0 --h-far 25 --sphere-level 1 (80k cells), then --mesh fish_micro.msh.
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))
from flow_analysis import build_model, WALL_TAG  # noqa: E402
from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh  # noqa: E402
from dragonfly_sim.utils.checkpoint import save_checkpoint, load_checkpoint  # noqa: E402
from dragonfly_sim.core.time_integration import CFLController  # noqa: E402

MESH = os.path.join(HERE, os.pardir, os.pardir, "meshes", "big_fish_unstructured.msh")
RESTART_KEYS = ("mesh", "speed", "Re", "yaw", "T_inf")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=MESH)
    p.add_argument("--wall-tag", default="body", help="physical group of the fish surface")
    p.add_argument("--speed", type=float, default=5.0, help="free-stream speed of the water [m/s]")
    p.add_argument("--nu", type=float, default=1.0e-6, help="kinematic viscosity [m^2/s] (for the default --Re)")
    p.add_argument("--Re", type=float, default=None, help="Reynolds number per metre (default: speed / nu)")
    p.add_argument("--T-inf", type=float, default=293.0, help="free-stream temperature [K]")
    p.add_argument("--yaw", type=float, default=0.0,
                   help="angle of the free stream to the fish axis, in the width (z') plane [deg]")
    # asm is a good alternative to schur that needs less memory
    p.add_argument("--linear-preconditioner", choices=("asm", "schur", "mg"), default="schur",
                   help="schur: pressure Schur field split with BoomerAMG (water); asm: ASM/ILU, less memory; "
                        "mg: hp-multigrid")
    p.add_argument("--jacobian-mode", choices=("compact_pc", "exact", "compact", "pressure_pc"),
                   default="compact_pc")
    p.add_argument("--ksp-option", action="append", default=[], metavar="NAME=VALUE",
                   help="extra PETSc option of the Newton Krylov solves, without its prefix; repeatable")
    p.add_argument("--direct", action="store_true",
                   help="MUMPS for the linear solves (far too much memory for the full mesh; small meshes only)")
    p.add_argument("--init-steps", type=int, default=200, help="first-order PTC steps of the initial state")
    p.add_argument("--init-tol", type=float, default=1e-6)
    p.add_argument("--cfl0", type=float, default=5.0)
    p.add_argument("--cfl-max", type=float, default=None, help="cap of the PTC CFL (default: the model's)")
    p.add_argument("--max-steps", type=int, default=1000, help="second-order PTC step budget")
    p.add_argument("--steady-tol", type=float, default=1e-9, help="stop at |R| below this")
    p.add_argument("--report-every", type=int, default=5, help="print the forces every N PTC steps")
    p.add_argument("--write-every", type=int, default=0, help="also a solution.bp frame every N steps (0: final only)")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--restart", default=None, help="checkpoint .npz to continue stage 2 from (skips stage 1)")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()
    if args.Re is None:
        args.Re = args.speed / args.nu
    args.mesh = os.path.abspath(args.mesh)
    return args


def rotate_to_flow_frame(msh):
    """Fish axis z (head +z) -> x (head -x, facing the +x free stream): x' = -z, y' = y, z' = x."""
    x = msh.geometry.x
    X = x.copy()
    x[:, 0], x[:, 2] = -X[:, 2], X[:, 0]


def model_options(args, muscl):
    return dict(model="sa", alpha_deg=args.yaw, Re=args.Re, T_inf=args.T_inf, muscl=muscl, fluid="water",
                water_speed=args.speed, wall_tags=[args.wall_tag], symmetry_plane=False, direct=args.direct,
                jacobian_mode=args.jacobian_mode)


def configure(model, args):
    model.report_positivity_limiter = False
    model.linear_preconditioner = args.linear_preconditioner
    model.newton_ksp_options = dict(o.split("=", 1) for o in args.ksp_option)
    model.use_ptc = True
    model.ptc_settings = {**model.ptc_settings, "cfl0": args.cfl0,
                          **({} if args.cfl_max is None else {"cfl_max": args.cfl_max})}


class BodyForces:
    """Forces on the wall from the model's wall force density (pressure + viscous traction): drag along the free
    stream, its friction part, side force (z' turned by the yaw) and vertical force (y'), in N, and C_D on the
    wetted and on the frontal area."""

    def __init__(self, model, mesh_obj, yaw_deg):
        msh = mesh_obj.mesh
        self.model, self.comm = model, msh.comm
        n, ds = mesh_obj.n, mesh_obj.form_manager.ds(WALL_TAG)
        drag = model._inflow_direction_numpy(np.radians(yaw_deg))
        side = np.cross(drag, [0.0, 1.0, 0.0])
        u = model.u_vec
        force = model.wall_force_density(u, n)
        friction = force - model.thermo(u).p * n
        self._forms = [dolfinx.fem.form(ufl.dot(f, ufl.as_vector(d)) * ds) for f, d in
                       ((force, drag), (friction, drag), (force, side), (force, [0.0, 1.0, 0.0]))]
        area = lambda e: self.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(e * ds)), MPI.SUM)
        self.S_wet = area(1.0)
        self.S_front = 0.5 * area(abs(ufl.dot(n, ufl.as_vector(drag))))
        dim = model.material.dimensional        # rho [kg/m^3], speed [m/s]: forces in rho U^2 m^2 -> N
        self.to_newton = dim["rho"] * dim["speed"] ** 2
        self.q = 0.5 * model.farfield_conditions["rho"] * model.farfield_conditions["speed"] ** 2

    def __call__(self):
        """(drag, friction drag, side, vertical) [N], C_D wetted, C_D frontal. Collective."""
        self.model.update_derived_fields()
        F = [self.comm.allreduce(dolfinx.fem.assemble_scalar(f), MPI.SUM) for f in self._forms]
        return (*[f * self.to_newton for f in F], F[0] / (self.q * self.S_wet), F[0] / (self.q * self.S_front))


def initial_state(msh, args):
    """Stage 1: first-order RANS by PTC from the free stream; returns the state per cell and frees the model."""
    _, model, _ = build_model(msh, **model_options(args, muscl=False))
    configure(model, args)
    model.max_newton_iterations = args.init_steps
    model.residual_conv_limit = args.init_tol
    model.set_up_solver()
    tic = time.perf_counter()
    r = model.solve_system().norm()
    PETSc.Sys.Print("[init] first-order RANS: {} PTC steps, |R| = {:.3e}, converged = {} [{:.1f} s]".format(
        model.time_integrator.last_steady_steps, r, model.solver_converged, time.perf_counter() - tic))
    dofs = model.functionspaces["V"].dofmap.list[:, 0]
    state = model.u_vec.x.array.reshape(-1, model.n_state)[dofs].copy()
    model.solver.destroy()
    del model
    gc.collect()
    PETSc.garbage_cleanup(msh.comm)
    return state


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    msh, _ = load_dolfinx_mesh(args.mesh, comm, tdim=3)
    rotate_to_flow_frame(msh)
    ncells = comm.allreduce(msh.topology.index_map(3).size_local, op=MPI.SUM)
    PETSc.Sys.Print("BIG_FISH steady RANS (SA-neg) in water: {} ({} cells), U = {:g} m/s, Re = {:.3g} per m, "
                    "yaw {:g} deg, {} ranks".format(os.path.basename(args.mesh), ncells, args.speed, args.Re,
                                                    args.yaw, comm.size))

    run_dir = None
    if comm.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        run_dir = tempfile.mkdtemp(prefix="fish_steady_U{:g}_yaw{:g}_{}_".format(
            args.speed, args.yaw, time.strftime("%Y%m%d-%H%M%S")), dir=args.out_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
    run_dir = comm.bcast(run_dir, root=0)
    config = {k: getattr(args, k) for k in RESTART_KEYS}

    state = None if args.restart else initial_state(msh, args)

    # stage 2: the MUSCL RANS model to a steady state
    mesh_obj, model, _ = build_model(msh, **model_options(args, muscl=True))
    configure(model, args)
    model.set_up_solver()
    PETSc.Sys.Print("second-order model: {} unknowns, linear preconditioner {}".format(
        model.u_vec.x.petsc_vec.getSize(), args.linear_preconditioner))
    if args.restart:
        meta = load_checkpoint(args.restart, [model.u_vec])
        bad = {k: (meta.get(k), config[k]) for k in RESTART_KEYS if meta.get(k) != config[k]}
        if bad:
            raise ValueError("--restart: settings differ from the checkpoint: {}".format(bad))
        PETSc.Sys.Print("Restarted from {}".format(args.restart))
    else:
        dofs = model.functionspaces["V"].dofmap.list
        model.u_vec.x.array.reshape(-1, model.n_state)[dofs] = state[:, None, :]
        model.u_vec.x.scatter_forward()

    forces = BodyForces(model, mesh_obj, args.yaw)
    PETSc.Sys.Print("wall: wetted area {:.4f} m^2, frontal area {:.5f} m^2".format(forces.S_wet, forces.S_front))
    model.define_sol_export_files(file_name=os.path.join(run_dir, "solution"), time_dependent=True)
    integrator = model.time_integrator
    ctrl = CFLController(**model.ptc_settings)
    header = "step,residual,drag_N,drag_friction_N,side_N,vertical_N,C_D_wetted,C_D_frontal"
    history = []

    def report(step, r):
        f = forces()
        history.append((step, r, *f))
        PETSc.Sys.Print("forces  step {:5d}  |R| {:.4e}  drag {:+.5f} N (friction {:+.5f} N)  side {:+.5f} N  "
                        "vertical {:+.5f} N  C_D wetted {:.6f}  frontal {:.5f}".format(step, r, *f))
        if comm.rank == 0:
            np.savetxt(os.path.join(run_dir, "history.csv"), np.array(history), delimiter=",", header=header,
                       comments="", fmt=["%d"] + ["%.10g"] * 7)

    def checkpoint(name, step):
        save_checkpoint(os.path.join(run_dir, name), [model.u_vec], {**config, "steady": True, "step": step})

    step, written = 0, None
    model._ensure_residual_vector()
    r = model._assemble_physical_residual()
    report(step, r)
    tic = time.perf_counter()
    converged = r < args.steady_tol
    while not converged and step < args.max_steps:
        chunk = min(args.report_every, args.max_steps - step)
        r = integrator.solve_steady(controller=ctrl, max_steps=chunk, tol=args.steady_tol).norm()
        done = integrator.last_steady_steps
        step += done
        converged = model.solver_converged
        report(step, r)
        if not np.isfinite(r):
            PETSc.Sys.Print("STOP: non-finite residual")
            break
        if done < chunk and not converged:
            PETSc.Sys.Print("STOP: the CFL controller gave up (|R| = {:.3e})".format(r))
            break
        if args.write_every and step % args.write_every < chunk:
            model.write_solution_output(None, time=float(step))
            written = step
        if args.checkpoint_every and step % args.checkpoint_every < chunk:
            checkpoint("checkpoint_latest.npz", step)
    wall_s = time.perf_counter() - tic

    if written != step:
        model.write_solution_output(None, time=float(step))
    checkpoint("checkpoint_final.npz", step)
    drag, drag_f, side, vert, cd_w, cd_f = history[-1][2:]
    if comm.rank == 0:
        np.savetxt(os.path.join(run_dir, "result.csv"),
                   np.array([[float(converged), step, r, wall_s, drag, drag_f, side, vert, cd_w, cd_f]]),
                   delimiter=",", comments="",
                   header="converged,steps,residual,wall_s,drag_N,drag_friction_N,side_N,vertical_N,"
                          "C_D_wetted,C_D_frontal")
    PETSc.Sys.Print("RESULT converged={} steps={} |R|={:.3e}  drag {:.5f} N (friction {:.5f} N, pressure {:.5f} N)  "
                    "side {:+.5f} N  vertical {:+.5f} N  C_D wetted {:.6f}  frontal {:.5f}  [{:.1f} s]".format(
                        converged, step, r, drag, drag_f, drag - drag_f, side, vert, cd_w, cd_f, wall_s))
    PETSc.Sys.Print("Done. Output in {}".format(run_dir))


if __name__ == "__main__":
    main()
